"""验证 harness 的 predict 能直接调用三个官方 CPU 基线模型（A/B/C）。

对每个基线：取 public/validation 的第一条真实样本，调用 contest_submission.predict，
再与官方 baselines/model.py 的 Model.predict 结果逐项比对。
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SUB_PATH = ROOT / "submission" / "contest_submission.py"
BASE = ROOT / "baselines"

PROFILES = [
    ("a", "A", "classification"),
    ("b", "B", "regression"),
    ("c", "C", "regression"),
]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("folder,key,task", PROFILES)
def test_harness_predict_calls_official_baseline(folder, key, task, monkeypatch):
    monkeypatch.delenv("CONTEST_ARTIFACT_PATH", raising=False)
    pkg = BASE / folder
    ckpt = pkg / "checkpoint.npz"
    assert ckpt.exists(), f"缺少基线检查点: {ckpt}"

    with open(pkg / "public" / "validation.jsonl", encoding="utf-8") as f:
        line = next(ln for ln in f if ln.strip())
    sample = json.loads(line)
    inputs = sample["inputs"]

    sub = _load(SUB_PATH, "contest_submission_baseline_test")
    out = sub.predict({"task": task, "inputs": inputs}, model_dir=str(pkg))

    bm = _load(pkg / "baselines" / "model.py", f"ref_model_{key}")
    bd = _load(pkg / "baselines" / "data.py", f"ref_data_{key}")
    model = bm.Model.load(str(ckpt))
    paths, values = bd.flatten(inputs)
    assert paths == model.meta["paths"]
    ref = model.predict(np.asarray([values], dtype=np.float64))[0]

    if task == "classification":
        labels = model.meta["classes"]
        ref_list = [{"label": c, "score": float(p)} for c, p in zip(labels, ref)]
        assert isinstance(out, list) and len(out) == len(ref_list)
        for got, want in zip(out, ref_list):
            assert got["label"] == want["label"]
            assert abs(got["score"] - want["score"]) < 1e-9
    else:
        assert isinstance(out, dict) and "value" in out
        assert abs(out["value"] - float(ref)) < 1e-9
