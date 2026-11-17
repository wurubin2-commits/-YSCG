"""用 harness 从三个官方 CPU 基线（A/B/C）续训一轮，并把实验记录上传 SwanLab。

- 训练：调用官方 `baselines/train.py` 的 `train_run(..., resume=基线checkpoint)`，保留
  Adam / 标准化 / 目标缩放状态；
- 记录：SwanLab（project=ZYYQ, workspace=07240115），逐 epoch 记 loss + 验证指标，
  结束记 before/after 与 gain；
- 产物：artifacts/harness-round/{A,B,C}/checkpoint.npz + history.jsonl + training.json。

用法：
  python baselines/harness_train_baselines.py --epochs 5 --mode online
"""
import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT.parent / "artifacts" / "harness-round"
PROFILES = [
    ("a", "A", "classification"),
    ("b", "B", "regression"),
    ("c", "C", "regression"),
]
SWANLAB_PROJECT = "ZYYQ"
SWANLAB_WORKSPACE = "07240115"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _log_history(swanlab, history_path: Path, task: str):
    for line in history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        step = int(rec.get("epoch", 0))
        data = {"train/loss": float(rec["loss"])}
        for k, v in (rec.get("validation") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                data[f"val/{k}"] = float(v)
        swanlab.log(data, step=step)


def main():
    parser = argparse.ArgumentParser(description="harness 续训官方基线并上传 SwanLab")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--mode", default="online", choices=["online", "local", "offline", "disabled"])
    parser.add_argument("--only", default=None, help="只跑某个数据集：A/B/C")
    args = parser.parse_args()

    import swanlab  # venv 已通过 .pth 桥接系统 site-packages

    results = []
    for folder, key, task in PROFILES:
        if args.only and key != args.only:
            continue
        pkg = ROOT / folder
        # train.py 的回退导入是 `from model import ...`，需要把 baselines/ 加进 sys.path
        if str(pkg / "baselines") not in sys.path:
            sys.path.insert(0, str(pkg / "baselines"))
        train_mod = _load(pkg / "baselines" / "train.py", f"train_{key}")
        config = json.loads((pkg / "baselines" / "configs" / f"{key}.json").read_text(encoding="utf-8"))
        info = json.loads((pkg / "baseline-info.json").read_text(encoding="utf-8"))

        out = ART / key
        if out.exists():
            shutil.rmtree(out)

        run = swanlab.init(
            project=SWANLAB_PROJECT,
            workspace=SWANLAB_WORKSPACE,
            experiment_name=f"harness-round-{key}",
            description=(
                f"harness 从官方基线 {info.get('id')} 续训 {args.epochs} epoch；"
                f"task={task}"
            ),
            config={
                "dataset_key": key,
                "task": task,
                "baseline_id": info.get("id"),
                "parent_checkpoint_sha256": info.get("checkpoint_sha256"),
                "epochs": args.epochs,
                "hidden": config.get("hidden"),
                "learning_rate": config.get("learning_rate"),
                "batch_size": config.get("batch_size"),
                "seed": config.get("seed"),
            },
            mode=args.mode,
        )
        print(f"[swanlab] {key} mode={args.mode}")

        report = train_mod.train_run(
            key,
            str(pkg / "public" / "train.jsonl"),
            str(pkg / "public" / "validation.jsonl"),
            str(out),
            resume=str(pkg / "checkpoint.npz"),
            epochs=args.epochs,
            stage_id=f"harness-round-{key}",
            label_provenance=str(pkg / "labels-provenance.json") if key == "B" else None,
        )

        _log_history(swanlab, out / "history.jsonl", task)

        before = report.get("validation_before", {})
        after = report.get("validation_after", {})
        final = {}
        for k in sorted(set(before) | set(after)):
            b, a = before.get(k), after.get(k)
            if isinstance(b, (int, float)) and isinstance(a, (int, float)):
                final[f"final/{k}_before"] = float(b)
                final[f"final/{k}_after"] = float(a)
                # 分类：gain=after-before；回归：gain=before-after（越大越好）
                final[f"gain/{k}"] = float(a - b) if task == "classification" else float(b - a)
        final["final/train_rows"] = int(report.get("train_rows", 0))
        final["final/validation_rows"] = int(report.get("validation_rows", 0))
        final["final/parameter_count"] = int(report.get("parameter_count", 0))
        final["final/training_seconds"] = float(report.get("training_seconds", 0.0))
        swanlab.log(final, step=int(args.epochs) + 1)
        swanlab.finish()

        results.append({
            "profile": key,
            "task": task,
            "baseline_id": info.get("id"),
            "before": before,
            "after": after,
            "gain": {k: final.get(f"gain/{k}") for k in sorted(set(before) | set(after))},
            "train_rows": report.get("train_rows"),
            "validation_rows": report.get("validation_rows"),
            "parameter_count": report.get("parameter_count"),
            "training_seconds": report.get("training_seconds"),
            "checkpoint": str(out / "checkpoint.npz"),
        })

    (ROOT / "harness_train_report.json").write_text(
        json.dumps({"rounds": results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("\nreport ->", ROOT / "harness_train_report.json")


if __name__ == "__main__":
    main()
