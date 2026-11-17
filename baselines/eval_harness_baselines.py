"""用 harness 的 predict 逐行调用三个官方基线模型，并在 public/validation 上算基线分数。

与 eval_baselines.py 的区别：
- eval_baselines.py 直接调官方 Model.predict；
- 本脚本走 contest_submission.predict（即 harness 的预测入口），证明 harness 能真正调用三个基线。
结果写入 baselines/harness_baseline_report.json。
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SUB_PATH = ROOT.parent / "submission" / "contest_submission.py"
PROFILES = [
    ("a", "A", "classification"),
    ("b", "B", "regression"),
    ("c", "C", "regression"),
]


def _load_sub():
    spec = importlib.util.spec_from_file_location("contest_submission_eval", str(SUB_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _macro_f1(y_true, y_pred, labels):
    f1s = []
    for lab in labels:
        tp = int(((y_pred == lab) & (y_true == lab)).sum())
        fp = int(((y_pred == lab) & (y_true != lab)).sum())
        fn = int(((y_pred != lab) & (y_true == lab)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


def main():
    import os
    os.environ.pop("CONTEST_ARTIFACT_PATH", None)
    sub = _load_sub()

    rows = []
    for folder, key, task in PROFILES:
        pkg = ROOT / folder
        with open(pkg / "public" / "validation.jsonl", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]

        y_true, y_pred, values, errors = [], [], [], 0
        for ln in lines:
            row = json.loads(ln)
            try:
                out = sub.predict({"task": task, "inputs": row["inputs"]}, model_dir=str(pkg))
            except Exception:
                errors += 1
                continue
            if task == "classification":
                label = max(out, key=lambda d: d["score"])["label"]
                y_true.append(str(row["answer"]))
                y_pred.append(label)
            else:
                values.append(float(out["value"]))
                y_true.append(float(row["answer"]))

        if task == "classification":
            labels = sorted(set(y_true))
            yt = np.asarray(y_true)
            yp = np.asarray(y_pred)
            metrics = {
                "accuracy": float((yt == yp).mean()),
                "macro_f1": _macro_f1(yt, yp, labels),
                "n": int(len(yt)),
                "errors": errors,
            }
        else:
            yt = np.asarray(y_true, dtype=np.float64)
            vp = np.asarray(values, dtype=np.float64)
            metrics = {
                "rmse": float(np.sqrt(np.mean((vp - yt) ** 2))),
                "mae": float(np.mean(np.abs(vp - yt))),
                "n": int(len(yt)),
                "errors": errors,
            }

        row_out = {"profile": key, "task": task, "metrics": metrics}
        rows.append(row_out)
        print(json.dumps(row_out, ensure_ascii=False))

    (ROOT / "harness_baseline_report.json").write_text(
        json.dumps({"baselines": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== harness 调用基线汇总 ===")
    for r in rows:
        print(f"{r['profile']}  {r['task']:<16}{json.dumps(r['metrics'], ensure_ascii=False)}")
    print("\nreport ->", ROOT / "harness_baseline_report.json")


if __name__ == "__main__":
    main()
