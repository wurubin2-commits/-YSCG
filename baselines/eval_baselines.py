"""评估三个官方 CPU 基线模型（A/B/C）在各自 public/validation 上的指标。

用法：
  python baselines/eval_baselines.py

说明：
- 直接调用官方 `baselines/model.py` 的 Model.load + predict（与平台 predict 协议同源）；
- A/B 为分类（macro-F1 + accuracy），C 为回归（RMSE + MAE）；
- 输出 baselines/baseline_report.json。
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROFILES = [
    ("a", "A", "classification"),
    ("b", "B", "regression"),
    ("c", "C", "regression"),
]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


def main():
    model_mod = _load_module(ROOT / "a" / "baselines" / "model.py", "baseline_model")
    data_mod = _load_module(ROOT / "a" / "baselines" / "data.py", "baseline_data")

    rows = []
    for folder, key, task in PROFILES:
        pkg = ROOT / folder
        ckpt = pkg / "checkpoint.npz"
        val = pkg / "public" / "validation.jsonl"
        info = json.loads((pkg / "baseline-info.json").read_text(encoding="utf-8"))

        model = model_mod.Model.load(str(ckpt))
        assert model.meta["task"] == task, (key, model.meta["task"], task)

        # 校验检查点摘要与 baseline-info 一致
        digest = model_mod.checkpoint_digest(str(ckpt))
        digest_ok = digest == info.get("checkpoint_sha256")

        ds = data_mod.read_dataset(
            str(val), task,
            classes=model.meta["classes"],
            expected_paths=model.meta["paths"],
        )
        preds = model.predict(ds["x"])

        if task == "classification":
            y_pred = preds.argmax(axis=1)
            metrics = {
                "accuracy": float((y_pred == ds["y"]).mean()),
                "macro_f1": macro_f1(ds["y"], y_pred, len(model.meta["classes"])),
                "n": int(len(ds["y"])),
            }
        else:
            y = ds["y"].astype(np.float64)
            p = preds.astype(np.float64)
            metrics = {
                "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
                "mae": float(np.mean(np.abs(p - y))),
                "n": int(len(y)),
            }

        row = {
            "profile": key,
            "dataset_key": info.get("dataset_key", key),
            "task": task,
            "baseline_id": info.get("id"),
            "parameter_count": info.get("parameter_count"),
            "checkpoint_sha256": digest,
            "checkpoint_sha256_match": bool(digest_ok),
            "metrics": metrics,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    report = {"baselines": rows}
    (ROOT / "baseline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== 基线汇总 ===")
    print(f"{'profile':<8}{'task':<16}{'metrics'}")
    for r in rows:
        print(f"{r['profile']:<8}{r['task']:<16}{json.dumps(r['metrics'], ensure_ascii=False)}")
    print("\nreport ->", ROOT / "baseline_report.json")


if __name__ == "__main__":
    main()
