# /app/contest_submission.py
# 最小可提交骨架：协议优先，YSCG 逻辑后续替换 _train_simple / predict / 决策函数
import importlib.util
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# 只声明镜像实际支持的任务；如果只支持一个任务，删掉另一个
TASKS = ["classification", "regression"]
SUPPORTED_TASKS = TASKS

# 协议硬约束：stdout 只能写协议 JSONL，日志一律走 stderr
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# 容器内默认 /workspace、/data；本地演练可用环境变量覆盖
WORKSPACE_ROOT = Path(os.environ.get("CONTEST_WORKSPACE", "/workspace"))
DATA_DIR = Path(os.environ.get("CONTEST_DATA_DIR", "/data"))


def describe():
    """给 harness 读取的能力声明（不打印到 stdout，由 harness 负责输出协议行）。"""
    return {
        "protocol": "contest.harness.workspace.v2",
        "model_access": "host_gateway_rpc",
        "tasks": list(TASKS),
    }


# -------------------------
# 基础工具
# -------------------------
def _truncate_bytes(s: str, limit: int = 4000) -> str:
    b = str(s).encode("utf-8")
    if len(b) <= limit:
        return str(s)
    return b[:limit].decode("utf-8", "ignore")


def _scalar_str(a) -> str:
    try:
        return str(a.item())
    except Exception:
        return str(a)


def _task_get(rnd, key, default=None):
    t = getattr(rnd, "task", None)
    if isinstance(t, dict):
        return t.get(key, default)
    if t is not None:
        try:
            return t[key]
        except Exception:
            pass
        return getattr(t, key, default)
    return default


def _task_type_from_rnd(rnd) -> str:
    for key in ("task_type", "task", "type"):
        v = _task_get(rnd, key)
        if v in ("classification", "regression"):
            return v
    env = os.environ.get("CONTEST_TASK")
    if env in ("classification", "regression"):
        return env
    raise ValueError("无法从 rnd.task / CONTEST_TASK 确定任务类型")


def _public_paths(rnd):
    pp = _task_get(rnd, "public_projection", None)
    train = None
    val = None
    if isinstance(pp, dict):
        train = (
            pp.get("training")
            or pp.get("train")
            or pp.get("training_path")
            or pp.get("train_path")
        )
        val = (
            pp.get("validation")
            or pp.get("valid")
            or pp.get("validation_path")
        )
    return (
        Path(train) if train else DATA_DIR / "train.jsonl",
        Path(val) if val else DATA_DIR / "validation.jsonl",
    )


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# -------------------------
# 父检查点 / 简单续训
# -------------------------
def _load_parent_state(rnd) -> dict:
    """只通过 rnd.parent.path 读父检查点，不搜 /app，不改 /previous。"""
    parent = getattr(rnd, "parent", None)
    if not parent:
        return {}
    if not getattr(parent, "checkpoint_sha256", None):
        return {}

    try:
        p = parent.path("checkpoint.npz")
    except Exception as e:
        logging.info("父检查点 checkpoint.npz 不可用：%s", e)
        return {}

    if not os.path.exists(p):
        return {}

    try:
        with np.load(p, allow_pickle=True) as z:
            tt = _scalar_str(z["task_type"]) if "task_type" in z else ""
            if tt == "classification":
                labels = [str(x) for x in z["labels"].tolist()] if "labels" in z else []
                counts = [int(x) for x in z["counts"].tolist()] if "counts" in z else []
                return {"task_type": tt, "counts": dict(zip(labels, counts))}
            if tt == "regression":
                return {
                    "task_type": tt,
                    "reg_mean": float(z["reg_mean"]) if "reg_mean" in z else 0.0,
                    "reg_n": int(z["reg_n"]) if "reg_n" in z else 0,
                }
    except Exception as e:
        logging.warning("读取父检查点失败，按首轮/空状态处理：%s", e)
    return {}


def _train_simple(parent_state: dict, rows: list, task_type: str) -> dict:
    """占位续训：分类学标签计数，回归学非负均值。

    替换点：YSCG 的漂移自适应/增量更新/锚点 floor/保护性记忆在这里接。
    """
    if task_type == "classification":
        counts = {str(k): int(v) for k, v in parent_state.get("counts", {}).items()}
        for r in rows:
            if "answer" not in r:
                continue
            ans = str(r["answer"])
            counts[ans] = counts.get(ans, 0) + 1
        return {"task_type": "classification", "counts": counts}

    m = float(parent_state.get("reg_mean", 0.0))
    n = int(parent_state.get("reg_n", 0))
    for r in rows:
        if "answer" not in r:
            continue
        try:
            v = float(r["answer"])
        except Exception:
            continue
        if v < 0:
            continue
        n += 1
        m += (v - m) / n
    return {"task_type": "regression", "reg_mean": max(0.0, m), "reg_n": n}


def _save_checkpoint(path: Path, state: dict) -> None:
    if state["task_type"] == "classification":
        labels = sorted(state["counts"].keys())
        counts = [state["counts"][lab] for lab in labels]
        np.savez(
            path,
            task_type=np.array("classification"),
            labels=np.array(labels, dtype=str),
            counts=np.array(counts, dtype=np.int64),
        )
    else:
        np.savez(
            path,
            task_type=np.array("regression"),
            reg_mean=np.float64(state["reg_mean"]),
            reg_n=np.int64(state["reg_n"]),
        )


# -------------------------
# 每轮强制真实模型调用
# -------------------------
def _call_llm_once(rnd, task_type: str, rows: list, parent_state: dict) -> str:
    parent_brief = json.dumps(parent_state, ensure_ascii=False)[:800]
    messages = [
        {
            "role": "system",
            "content": (
                "你是漂移自适应训练助手的假设生成器。"
                "只输出简短假设和相对父模型的改动，不要输出协议帧，不要输出 JSONL。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"任务类型：{task_type}\n"
                f"dataset_key：{getattr(rnd, 'dataset_key', '?')}\n"
                f"round_index：{getattr(rnd, 'round_index', '?')}\n"
                f"公开训练行数：{len(rows)}\n"
                f"父状态摘要：{parent_brief}\n"
                "请给出本轮续训假设，<=200 字，并说明相对父模型实际要改什么。"
            ),
        },
    ]

    mt = getattr(rnd, "max_output_tokens", 512) or 512

    try:
        mt = int(mt)
    except Exception:
        mt = 512

    receipt = rnd.model(messages, max_tokens=min(512, max(1, mt)), temperature=0)
    text = getattr(receipt, "text", None) or getattr(receipt, "response", None) or ""
    return str(text)


def _ensure_one_model_call(rnd, task_type: str, rows: list, parent_state: dict) -> str:
    """硬约束：本轮至少要有一条真实模型调用回执，否则 finish 无效。"""
    text = _call_llm_once(rnd, task_type, rows, parent_state)
    if not (getattr(rnd, "call_ids", None) or []):
        logging.warning("首次模型调用未记录 call_id，重试一次")
        text = _call_llm_once(rnd, task_type, rows, parent_state)
    if not (getattr(rnd, "call_ids", None) or []):
        logging.error("仍未获得模型调用回执；finish 的 model_call_ids 将为空，可能被判无效")
    return text


# -------------------------
# 一轮训练
# -------------------------
def train(rnd):
    round_idx = int(getattr(rnd, "round_index", 1) or 1)
    task_type = _task_type_from_rnd(rnd)
    train_path, _val_path = _public_paths(rnd)

    rows = list(_iter_jsonl(train_path))
    logging.info("round=%s task=%s train_rows=%s", round_idx, task_type, len(rows))

    parent_state = _load_parent_state(rnd)

    # 硬约束：每轮至少一次真实模型调用；调用号会进入 rnd.call_ids
    llm_text = _ensure_one_model_call(rnd, task_type, rows, parent_state)

    hypothesis = _truncate_bytes(
        llm_text.strip() or f"基于父模型续训，轮次 {round_idx}",
        4000,
    )
    change_summary = _truncate_bytes(
        (
            f"使用公开训练集 {len(rows)} 行续训；"
            f"父状态={json.dumps(parent_state, ensure_ascii=False)[:300]}；"
            f"LLM假设={llm_text[:300]}"
        ),
        4000,
    )

    state = _train_simple(parent_state, rows, task_type)

    # 候选目录：/workspace 下新目录，只放普通文件（≤64 个 / 合计 64MiB）
    candidate_dir = WORKSPACE_ROOT / "candidates" / f"r{round_idx}"
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    _save_checkpoint(candidate_dir / "checkpoint.npz", state)

    # 候选内保留一份代码副本；预测仍走 /app/harness.py predict
    shutil.copy2(Path(__file__).resolve(), candidate_dir / "contest_submission.py")

    configuration = {
        "round_index": round_idx,
        "task_type": task_type,
        "n_train": len(rows),
        "parent_task_type": parent_state.get("task_type"),
    }

    submitted = rnd.submit_candidate(
        str(candidate_dir),
        hypothesis=hypothesis,
        change_summary=change_summary,
        configuration=configuration,
        seed=round_idx,
        checkpoint_paths=["checkpoint.npz"],
        code_paths=["contest_submission.py"],
        predict_argv=["python3", "/app/harness.py", "predict"],
    )
    logging.info("submitted artifact_id=%s", submitted.get("artifact_id"))

    # 跨轮状态：把 YSCG 的锚点 floor / 保护性记忆 / 策略参数放这里
    state_obj = rnd.state_workspace(str(WORKSPACE_ROOT / "state"))
    state_dir = Path(state_obj)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_payload = {
        "round_index": round_idx,
        "task_type": task_type,
        "counts": state.get("counts", {}),
        "reg_mean": state.get("reg_mean"),
        "reg_n": state.get("reg_n"),
        "anchor_floor": None,          # TODO: YSCG 锚点 floor
        "protective_memory": [],       # TODO: YSCG 保护性记忆
        "policy_params": {},           # TODO: YSCG 策略参数
        "last_hypothesis": hypothesis,
        "last_artifact_id": submitted.get("artifact_id"),
    }
    (state_dir / "state.json").write_text(
        json.dumps(state_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    rnd.save_state(state_obj)

    history = rnd.history() if hasattr(rnd, "history") else []
    history_ids = [h.get("id") for h in history if isinstance(h, dict) and h.get("id")]

    # 默认选新候选；替换点：用平台返回测量 + rnd.history() 做准入/回滚决策
    reason = _truncate_bytes(
        "默认选择本轮新候选。平台返回测量摘要="
        + json.dumps(submitted, ensure_ascii=False)[:1000],
        4000,
    )

    rnd.finish(
        selected_artifact_id=submitted["artifact_id"],
        hypothesis=hypothesis,
        change_summary=change_summary,
        reason=reason,
        configuration=configuration,
        seed=round_idx,
        history_ids=history_ids,
        model_call_ids=list(getattr(rnd, "call_ids", []) or []),
    )


# -------------------------
# 官方 CPU 基线模型（vendor：baseline_model.py / baseline_data.py）
# 三个官方基线 checkpoint（A 分类 / B、C 回归）可直接被 harness 调用
# -------------------------
_VENDOR_CACHE: dict = {}
_MODEL_CACHE: dict = {}


def _get_baseline_model(ckpt_path):
    """按（路径, mtime, size）缓存已加载的基线模型，避免逐行预测时重复加载。"""
    try:
        st = os.stat(ckpt_path)
        key = (str(ckpt_path), st.st_mtime_ns, st.st_size)
    except Exception:
        key = (str(ckpt_path), None, None)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    bm = _vendor("baseline_model")
    if bm is None:
        return None
    model = bm.Model.load(str(ckpt_path))
    _MODEL_CACHE[key] = model
    return model


def _vendor(name: str):
    if name in _VENDOR_CACHE:
        return _VENDOR_CACHE[name]
    try:
        p = Path(__file__).resolve().parent / f"{name}.py"
        if not p.exists():
            _VENDOR_CACHE[name] = None
            return None
        spec = importlib.util.spec_from_file_location(f"yscg_vendor_{name}", str(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _VENDOR_CACHE[name] = mod
        return mod
    except Exception as e:
        logging.warning("vendor %s 加载失败: %s", name, e)
        _VENDOR_CACHE[name] = None
        return None


def _is_official_baseline_checkpoint(ckpt_path) -> bool:
    """官方基线 checkpoint 是 npz（含 metadata / w1），与本骨架自定义格式不同。"""
    try:
        with np.load(ckpt_path, allow_pickle=False) as z:
            return "metadata" in z.files and "w1" in z.files
    except Exception:
        return False


def _predict_official_baseline(ckpt_path, inputs):
    bd = _vendor("baseline_data")
    if bd is None:
        return None
    model = _get_baseline_model(ckpt_path)
    if model is None:
        return None
    paths, values = bd.flatten(inputs)
    if paths != model.meta["paths"]:
        raise ValueError("prediction feature schema differs from checkpoint")
    out = model.predict(np.asarray([values], dtype=np.float64))[0]
    if model.meta["task"] == "regression":
        return {"value": float(out)}
    return [{"label": str(lab), "score": float(score)}
            for lab, score in zip(model.meta["classes"], out)]


# -------------------------
# 预测：严格一行进一行出由 harness 负责，这里只返回本行内层 payload
# 协议：harness 会把返回值包成 {"outputs": <返回值>}
# -------------------------
def _fallback_output(task_type: str):
    if task_type == "regression":
        return {"value": 0.0}
    return [{"label": "unknown", "score": 1.0}]


def predict(sample, model_dir=None):
    if model_dir is None:
        model_dir = os.environ.get("CONTEST_MODEL_DIR", "/model")

    task_type = None
    if isinstance(sample, dict):
        task_type = sample.get("task")
    else:
        task_type = getattr(sample, "task", None)
    if task_type not in ("classification", "regression"):
        task_type = os.environ.get("CONTEST_TASK", "classification")

    ckpt = os.environ.get("CONTEST_ARTIFACT_PATH") or str(Path(model_dir) / "checkpoint.npz")

    if not os.path.exists(ckpt):
        return _fallback_output(task_type)

    # 优先：官方 CPU 基线检查点（三个初始工业模型 A/B/C）
    if _is_official_baseline_checkpoint(ckpt):
        inputs = sample.get("inputs") if isinstance(sample, dict) else getattr(sample, "inputs", None)
        if inputs is not None:
            try:
                out = _predict_official_baseline(ckpt, inputs)
                if out is not None:
                    return out
            except Exception as e:
                logging.warning("官方基线预测失败，回退：%s", e)

    try:
        with np.load(ckpt, allow_pickle=True) as z:
            tt = _scalar_str(z["task_type"]) if "task_type" in z else task_type

            if tt == "classification":
                labels = [str(x) for x in z["labels"].tolist()] if "labels" in z else []
                counts = [int(x) for x in z["counts"].tolist()] if "counts" in z else []
                if labels and counts and len(labels) == len(counts):
                    best = max(range(len(labels)), key=lambda i: counts[i])
                    # 只输出一个最高分标签，保证标签不重复、最高分唯一
                    return [{"label": labels[best], "score": 1.0}]
                return _fallback_output("classification")

            mean = float(z["reg_mean"]) if "reg_mean" in z else 0.0
            return {"value": float(max(0.0, mean))}

    except Exception as e:
        logging.warning("预测加载检查点失败，返回 fallback：%s", e)

    return _fallback_output(task_type)
