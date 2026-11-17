"""本地演练：用 FakeRnd 模拟平台，验证 contest_submission 骨架的协议流程。

覆盖：
- train：读 JSONL、至少一次真实模型调用、submit_candidate、save_state、finish
- predict：分类返回内层 payload、回归返回 {"value": ...}
"""
import importlib.util
import json
from pathlib import Path

SUB_PATH = Path(__file__).resolve().parents[1] / "submission" / "contest_submission.py"


def _load_submission():
    spec = importlib.util.spec_from_file_location("contest_submission_smoke", SUB_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeReceipt:
    def __init__(self, text):
        self.text = text
        self.call_id = "call-1"


class FakeParent:
    checkpoint_sha256 = None


class FakeRnd:
    def __init__(self, train_path, val_path, round_index=1):
        self.task = {
            "task_type": "classification",
            "public_projection": {"train": str(train_path), "validation": str(val_path)},
        }
        self.prompt = "prompt"
        self.prompt_version = "v"
        self.dataset_key = "A"
        self.round_index = round_index
        self.model_name = "fake-model"
        self.max_output_tokens = 128
        self.limits = {}
        self.parent = FakeParent()
        self.call_ids = []
        self.finished = None
        self.submitted = None
        self.saved_state = None

    def model(self, messages, max_tokens=512, temperature=0):
        self.call_ids.append("call-1")
        return FakeReceipt("假设：保持多数类，观察漂移。")

    def submit_candidate(self, candidate_dir, **kwargs):
        self.submitted = {"artifact_id": "art-1", "candidate_dir": candidate_dir, **kwargs}
        return {"artifact_id": "art-1", "summary": {"metric": 0.5}}

    def state_workspace(self, path):
        return path

    def save_state(self, state):
        self.saved_state = state

    def history(self):
        return []

    def finish(self, **kwargs):
        self.finished = kwargs


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def _prepare(tmp_path, monkeypatch, task_type="classification", answers=None):
    workspace = tmp_path / "workspace"
    data_dir = tmp_path / "data"
    workspace.mkdir()
    data_dir.mkdir()

    answers = answers or ["a", "b"] * 5
    rows = [
        {"id": f"t{i}", "inputs": {"x": i}, "answer": answers[i % len(answers)]}
        for i in range(10)
    ]
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "validation.jsonl"
    _write_jsonl(train_path, rows)
    _write_jsonl(val_path, rows[:1])

    monkeypatch.setenv("CONTEST_WORKSPACE", str(workspace))
    monkeypatch.setenv("CONTEST_DATA_DIR", str(data_dir))

    mod = _load_submission()
    fake = FakeRnd(train_path, val_path)
    fake.task["task_type"] = task_type
    return mod, fake, workspace


def test_train_classification_and_predict(tmp_path, monkeypatch):
    mod, fake, workspace = _prepare(tmp_path, monkeypatch, task_type="classification")

    mod.train(fake)

    # 协议硬约束：至少一次模型调用，且 finish 带上非空 model_call_ids
    assert fake.call_ids == ["call-1"]
    assert fake.finished is not None
    assert fake.finished["model_call_ids"] == ["call-1"]
    assert fake.finished["selected_artifact_id"] == "art-1"
    assert fake.saved_state is not None

    ckpt = workspace / "candidates" / "r1" / "checkpoint.npz"
    assert ckpt.exists()
    state_json = workspace / "state" / "state.json"
    assert state_json.exists()

    out = mod.predict({"task": "classification"}, model_dir=str(workspace / "candidates" / "r1"))
    assert isinstance(out, list) and out
    assert set(out[0]) == {"label", "score"}
    assert 0.0 <= out[0]["score"] <= 1.0


def test_train_regression_and_predict(tmp_path, monkeypatch):
    mod, fake, workspace = _prepare(
        tmp_path, monkeypatch, task_type="regression", answers=[1.0, 3.0] * 5
    )

    mod.train(fake)

    out = mod.predict({"task": "regression"}, model_dir=str(workspace / "candidates" / "r1"))
    assert isinstance(out, dict) and "value" in out
    assert out["value"] >= 0.0
