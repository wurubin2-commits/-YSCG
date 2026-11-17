"""本地 JSONL 宿主替身：端到端演练 workspace v2 协议（不调用真实模型）。

用法（仓库根目录）：
  python tests/local_host.py --dataset A --rounds 3
  python tests/local_host.py --dataset all --rounds 2

替身职责与平台一致：
- 挂载父产物（官方 baseline checkpoint），核对摘要；
- 提供公开 train/validation 投影路径；
- 回应 history/model/candidate/state RPC（model 用固定 JSON 顾问替身）；
- candidate 时独立测量验证指标（用官方 metrics 函数）；
- 轮间传递被选中的产物与状态；
- 训练结束后用 predict 子进程逐行预测并核对协议输出。
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
BASELINES = ROOT / "models" / "工业基准模型"
sys.path.insert(0, str(SUBMISSION))

from baselines.data import read_dataset  # noqa: E402
from baselines.model import Model  # noqa: E402
from baselines.train import metrics  # noqa: E402

PROTOCOL = "contest.harness.workspace.v2"
PROMPT = "从官方基线与公开训练、验证数据出发，在授权预算内自主提出假设并续训。"


def digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_receipt(directory, artifact_id):
    directory = Path(directory)
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            files.append({"path": path.relative_to(directory).as_posix(),
                          "sha256": digest(path), "size_bytes": path.stat().st_size})
    descriptor = {"checkpoint_paths": ["checkpoint.npz"],
                  "code_paths": [f["path"] for f in files if f["path"] in ("training.json", "history.jsonl")],
                  "predict_argv": ["python3", "/app/harness.py", "predict"]}
    encoded = json.dumps({"files": files, "artifact": descriptor}, sort_keys=True, separators=(",", ":")).encode()
    return {"artifact_id": artifact_id, "artifact_sha256": "sha256:" + hashlib.sha256(encoded).hexdigest(),
            "artifact": descriptor, "files": files}


def stub_advisor(messages, options):
    """固定顾问替身：返回合法 JSON 建议（白名单内）。"""
    return {"request_id": f"call-{threading.get_ident()}-{options.get('max_tokens')}",
            "id": "chatcmpl-stub",
            "choices": [{"message": {"role": "assistant", "content":
                '{"learning_rate":0.001,"batch_size":128,"epochs":2,"reason":"本地替身：保守小步续训"}'}}]}


class LocalHost:
    def __init__(self, dataset, round_index, parent_dir, parent_receipt, artifacts, history,
                 state_dir, work, stub=stub_advisor):
        self.dataset = dataset
        self.round_index = round_index
        self.parent_dir = Path(parent_dir)
        self.parent_receipt = parent_receipt
        self.artifacts = artifacts
        self.history = history
        self.state_dir = state_dir
        self.work = Path(work)
        self.stub = stub
        self.record = None
        self.failure = None
        self.candidate_dir = None
        self.candidate_summary = None
        self.saved_state_dir = None
        self.model_calls = 0
        self._seq = 0

    def start_frame(self):
        task = "classification" if self.dataset == "A" else "regression"
        metric = "macro_f1" if self.dataset == "A" else "rmse"
        return {
            "protocol": PROTOCOL, "run_id": "local-run", "dataset_key": self.dataset,
            "round_index": self.round_index, "stage_id": f"local-round-{self.round_index}",
            "prompt": PROMPT, "prompt_version": "industrial-workspace-flexible-v3",
            "rounds_per_dataset": 10,
            "parent_checkpoint_sha256": digest(self.parent_dir / "checkpoint.npz"),
            "task_contract": {"dataset_key": self.dataset, "task": task, "metric": metric,
                              "metric_direction": "higher" if task == "classification" else "lower",
                              "public_projection": {
                                  "train_path": str(BASELINES / self.dataset.lower() / "public" / "train.jsonl"),
                                  "validation_path": str(BASELINES / self.dataset.lower() / "public" / "validation.jsonl"),
                                  "row_fields": ["id", "inputs", "answer"], "source_metadata": False}},
            "model_name": "base:local-stub", "parent_artifact": self.parent_receipt,
            "artifacts": self.artifacts,
            "state_sha256": digest(self.state_dir / "memory.json") if self.state_dir and (self.state_dir / "memory.json").is_file() else "",
            "state_path": str(self.state_dir) if self.state_dir else None,
            "max_output_tokens": 1024,
            "resource_limits": {"candidate_upload_bytes_per_round": 64 * 1024 * 1024},
            "current_validation": {}, "wall_seconds": 600,
        }

    def _materialize(self, files, target):
        target = Path(target)
        target.mkdir(parents=True, exist_ok=False)
        for item in files:
            data = base64.b64decode(item["data_base64"])
            assert hashlib.sha256(data).hexdigest() == item["sha256"].removeprefix("sha256:") if "sha256" in item else True
            path = target / item["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return target

    def _measure(self, candidate_dir):
        model = Model.load(Path(candidate_dir) / "checkpoint.npz")
        config_classes = model.meta["classes"]
        val = read_dataset(BASELINES / self.dataset.lower() / "public" / "validation.jsonl",
                           model.meta["task"], config_classes, model.meta["paths"])
        return {"metrics": metrics(model, val)}

    def serve(self, process):
        def run():
            try:
                self._serve(process)
            except Exception as exc:
                self.failure = exc
                try:
                    process.kill()
                except OSError:
                    pass
        thread = threading.Thread(target=run)
        thread.start()
        return thread

    def _serve(self, process):
        process.stdin.write(json.dumps(self.start_frame(), ensure_ascii=False) + "\n")
        process.stdin.flush()
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("participant closed stdout")
            frame = json.loads(line)
            kind = frame.get("type")
            if kind == "round_failed":
                raise RuntimeError("participant failed: " + str(frame.get("error")))
            if kind == "round_complete":
                self.record = frame["record"]
                process.stdin.write(json.dumps({"type": "round_ack", "accepted": True}) + "\n")
                process.stdin.flush()
                process.stdin.close()
                return
            assert frame.get("id") == self._seq, f"rpc sequence mismatch: {frame}"
            self._seq += 1
            if kind == "history_request":
                reply = {"type": "history_response", "id": frame["id"], "records": self.history}
            elif kind == "model_request":
                self.model_calls += 1
                response = self.stub(frame["messages"], {k: v for k, v in frame.items() if k not in ("type", "id", "messages")})
                call_id = response.pop("request_id") + f"-{self.model_calls}"
                reply = {"type": "model_response", "id": frame["id"], "call_id": call_id, "response": response}
            elif kind == "state_request":
                self.saved_state_dir = self._materialize(frame["files"], self.work / f"state-store-r{self.round_index}")
                reply = {"type": "state_response", "id": frame["id"], "state_sha256": "sha256:" + "11" * 32}
            elif kind == "candidate_request":
                assert frame["parent_checkpoint_sha256"] == digest(self.parent_dir / "checkpoint.npz"), "parent binding changed"
                self.candidate_dir = self._materialize(frame["files"], self.work / f"artifact-r{self.round_index}")
                self.candidate_summary = self._measure(self.candidate_dir)
                reply = {"type": "candidate_response", "id": frame["id"],
                         "artifact_id": f"artifact-{self.dataset}-r{self.round_index}",
                         "summary": self.candidate_summary}
            else:
                raise RuntimeError("unexpected frame " + str(kind))
            process.stdin.write(json.dumps(reply, ensure_ascii=False) + "\n")
            process.stdin.flush()


def _install_local_mounts(parent_dir):
    """Windows 本地替身：把 SDK 固定的 /previous 挂载根重定向到临时目录。

    仅测试进程生效；镜像内不携带此补丁，平台真实挂载 /previous。
    """
    from contest_harness import round as round_module

    original_init = round_module.ParentView.__init__

    def patched(self, frame):
        original_init(self, frame)
        self.root = Path(parent_dir)

    round_module.ParentView.__init__ = patched


def run_round(dataset, round_index, parent_dir, parent_receipt, artifacts, history, state_dir, root):
    work = root / f"work-{dataset}-r{round_index}"
    work.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SUBMISSION)
    env["CONTEST_SUBMISSION"] = "contest_submission"
    env["CONTEST_WORK"] = str(work)
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # 通过 sitecustomize 在子进程早期安装挂载补丁
    shim = root / "shim"
    shim.mkdir(exist_ok=True)
    (shim / "sitecustomize.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "from contest_harness import round as _r\n"
        "_orig = _r.ParentView.__init__\n"
        "def _patched(self, frame):\n"
        "    _orig(self, frame)\n"
        "    self.root = Path(os.environ['LOCAL_PARENT_MOUNT'])\n"
        "_r.ParentView.__init__ = _patched\n",
        encoding="utf-8",
    )
    env["PYTHONPATH"] = str(shim) + os.pathsep + str(SUBMISSION)
    env["LOCAL_PARENT_MOUNT"] = str(parent_dir)
    process = subprocess.Popen(
        [sys.executable, str(SUBMISSION / "harness.py"), "train"],
        cwd=SUBMISSION, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8",
    )
    host = LocalHost(dataset, round_index, parent_dir, parent_receipt, artifacts, history, state_dir, work)
    thread = host.serve(process)
    stderr = process.stderr.read()
    code = process.wait(timeout=600)
    thread.join(timeout=10)
    if code != 0 or host.failure is not None:
        print(stderr, file=sys.stderr)
        raise RuntimeError(f"round {round_index} failed: exit={code} host={host.failure!r}")
    assert host.record["model_call_ids"], "model_call_ids must be non-empty"
    assert host.model_calls >= 1
    return host, stderr


def run_predict(dataset, artifact_dir, receipt, n_rows=20):
    """按平台方式启动 predict：只挂载选中产物，逐行请求并校验输出。"""
    val_path = BASELINES / dataset.lower() / "public" / "validation.jsonl"
    rows = [json.loads(line) for line in val_path.read_text(encoding="utf-8").splitlines() if line.strip()][:n_rows]
    task = "classification" if dataset == "A" else "regression"
    context = {"protocol": "contest.predict-context.v1", "dataset_key": dataset, "task": task,
               "artifact": receipt["artifact"], "artifact_id": receipt["artifact_id"]}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SUBMISSION)
    env["CONTEST_SUBMISSION"] = "contest_submission"
    env["CONTEST_ARTIFACT_PATH"] = str(Path(artifact_dir) / "checkpoint.npz")
    env["CONTEST_ARTIFACT_MANIFEST"] = json.dumps(receipt["files"])
    env["CONTEST_PREDICTION_CONTEXT"] = json.dumps(context, ensure_ascii=False)
    env["CONTEST_DATASET_KEY"] = dataset
    env["CONTEST_TASK"] = task
    env["PYTHONUTF8"] = "1"
    # SDK predict 的 model_dir 固定 /model；本地替身通过 monkeypatch 不行（子进程），
    # 因此预测检查点路径完全依赖 CONTEST_ARTIFACT_PATH（与平台一致）。
    payload = "".join(json.dumps({"task": task, "inputs": row["inputs"]}, ensure_ascii=False) + "\n" for row in rows)
    process = subprocess.run(
        [sys.executable, str(SUBMISSION / "harness.py"), "predict"],
        input=payload, env=env, text=True, encoding="utf-8", capture_output=True, timeout=120,
    )
    assert process.returncode == 0, process.stderr
    lines = [json.loads(line) for line in process.stdout.splitlines()]
    assert len(lines) == len(rows), f"expected {len(rows)} predictions, got {len(lines)}"
    for body in lines:
        outputs = body["outputs"]
        if task == "classification":
            assert isinstance(outputs, list) and outputs
            tops = [o["score"] for o in outputs]
            assert sum(s == max(tops) for s in tops) == 1
        else:
            assert outputs["value"] >= 0
    return lines


def exercise(dataset, rounds, root):
    print(f"=== dataset {dataset}: {rounds} rounds ===")
    baseline_dir = BASELINES / dataset.lower()
    parent_dir = root / f"parent-{dataset}-r0"
    parent_dir.mkdir()
    shutil.copyfile(baseline_dir / "checkpoint.npz", parent_dir / "checkpoint.npz")
    receipt = artifact_receipt(parent_dir, f"baseline-{dataset}")
    artifacts = [receipt]
    history = []
    state_dir = None
    selected_dir = parent_dir
    selected_receipt = receipt
    for round_index in range(1, rounds + 1):
        host, stderr = run_round(dataset, round_index, selected_dir, selected_receipt, artifacts, history, state_dir, root)
        record = host.record
        print(f"round {round_index}: decision-selected={record['selected_artifact_id']} "
              f"calls={record['model_call_ids']} history={record['history_ids']}")
        if host.candidate_dir is not None:
            new_receipt = artifact_receipt(host.candidate_dir, f"artifact-{dataset}-r{round_index}")
            artifacts.append(new_receipt)
            if record["selected_artifact_id"] == new_receipt["artifact_id"]:
                selected_dir, selected_receipt = host.candidate_dir, new_receipt
        history.append({"id": f"record-{dataset}-r{round_index}", "round_index": round_index,
                        "hypothesis": record["hypothesis"], "change_summary": record["change_summary"],
                        "validation_summary": host.candidate_summary or {}})
        if host.saved_state_dir is not None:
            state_dir = host.saved_state_dir
    preds = run_predict(dataset, selected_dir, selected_receipt)
    print(f"predict OK: {len(preds)} rows, first={json.dumps(preds[0], ensure_ascii=False)[:160]}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["A", "B", "C", "all"])
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    datasets = ["A", "B", "C"] if args.dataset == "all" else [args.dataset]
    with tempfile.TemporaryDirectory(prefix="yscg-host-") as tmp:
        root = Path(tmp)
        for dataset in datasets:
            exercise(dataset, args.rounds, root)
    print("ALL LOCAL HOST CHECKS PASSED")


if __name__ == "__main__":
    main()