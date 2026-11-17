"""10 轮自进化实验：每轮从父模型续训 → 验证 → 准入判断（变好才替换，否则回滚）。

- 三个数据集 A/B/C 各自跑 10 轮，共享同一套准入规则；
- 每轮记录：父模型指标 / 候选指标 / 最佳指标 / 是否接受（accept）/ 训练损失；
- SwanLab：每个数据集一个实验（experiment_name=harness-evolve-{A,B,C}），轮次为 step。

用法：
  python baselines/harness_rounds.py --rounds 10 --epochs 2 --mode online
"""
import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT.parent / "artifacts" / "rounds"
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


def _primary(task, metrics):
    return metrics.get("macro_f1") if task == "classification" else metrics.get("rmse")


def _better(task, after, before):
    """候选是否优于父模型（严格改善）。"""
    if task == "classification":
        return after >= before + 1e-6
    # 回归：RMSE 越低越好
    return after <= before - 1e-6


def main():
    parser = argparse.ArgumentParser(description="10 轮自进化实验")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--mode", default="online", choices=["online", "local", "offline", "disabled"])
    parser.add_argument("--only", default=None, help="只跑某个数据集 A/B/C")
    args = parser.parse_args()

    import swanlab

    all_results = []
    for folder, key, task in PROFILES:
        if args.only and key != args.only:
            continue
        pkg = ROOT / folder
        if str(pkg / "baselines") not in sys.path:
            sys.path.insert(0, str(pkg / "baselines"))
        train_mod = _load(pkg / "baselines" / "train.py", f"train_{key}_rounds")
        config = json.loads((pkg / "baselines" / "configs" / f"{key}.json").read_text(encoding="utf-8"))
        info = json.loads((pkg / "baseline-info.json").read_text(encoding="utf-8"))

        base_dir = ART / key
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        run = swanlab.init(
            project=SWANLAB_PROJECT,
            workspace=SWANLAB_WORKSPACE,
            experiment_name=f"harness-evolve-{key}",
            description=f"10 轮自进化（每轮 {args.epochs} epoch，准入=严格改善才替换）",
            config={
                "dataset_key": key,
                "task": task,
                "baseline_id": info.get("id"),
                "rounds": args.rounds,
                "epochs_per_round": args.epochs,
                "admission": "strict_improvement",
            },
            mode=args.mode,
        )
        print(f"[swanlab] {key} mode={args.mode}")

        parent_ckpt = str(pkg / "checkpoint.npz")
        # 父模型当前指标（官方基线在 validation 上的分数）
        from_ref = train_mod.metrics(
            train_mod.Model.load(parent_ckpt),
            train_mod.read_dataset(
                str(pkg / "public" / "validation.jsonl"), task,
                classes=config.get("classes") or None,
            ),
        ) if hasattr(train_mod, "metrics") else None

        parent_metrics = from_ref or {}
        best_metric = _primary(task, parent_metrics)

        records = []
        accepted = 0
        for r in range(1, args.rounds + 1):
            out = base_dir / f"r{r:02d}"
            report = train_mod.train_run(
                key,
                str(pkg / "public" / "train.jsonl"),
                str(pkg / "public" / "validation.jsonl"),
                str(out),
                resume=parent_ckpt,
                epochs=args.epochs,
                stage_id=f"round-{r:02d}",
                label_provenance=str(pkg / "labels-provenance.json") if key == "B" else None,
            )
            before = report.get("validation_before", {})
            after = report.get("validation_after", {})
            m_before = _primary(task, before)
            m_after = _primary(task, after)
            ok = _better(task, m_after, m_before)

            if ok:
                parent_ckpt = str(out / "checkpoint.npz")
                parent_metrics = after
                accepted += 1
                best_metric = m_after

            loss = None
            hist = out / "history.jsonl"
            if hist.exists():
                lines = [ln for ln in hist.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if lines:
                    loss = json.loads(lines[-1]).get("loss")

            data = {
                "round/primary_before": m_before,
                "round/primary_after": m_after,
                "round/primary_gain": (m_after - m_before) if task == "classification" else (m_before - m_after),
                "round/best": best_metric,
                "round/accepted": 1 if ok else 0,
            }
            if task == "classification":
                data["round/accuracy_after"] = after.get("accuracy")
            else:
                data["round/mae_after"] = after.get("mae")
            if loss is not None:
                data["train/loss"] = loss
            swanlab.log(data, step=r)

            rec = {
                "round": r,
                "before": before,
                "after": after,
                "primary_before": m_before,
                "primary_after": m_after,
                "accepted": bool(ok),
                "best": best_metric,
            }
            records.append(rec)
            print(f"[{key}] round {r:02d}: before={m_before:.4f} after={m_after:.4f} "
                  f"accepted={ok} best={best_metric:.4f}")

        swanlab.log({
            "final/best_primary": best_metric,
            "final/baseline_primary": _primary(task, from_ref or {}),
            "final/accepted_rounds": accepted,
            "final/rounds": args.rounds,
        }, step=args.rounds + 1)
        swanlab.finish()

        all_results.append({
            "profile": key,
            "task": task,
            "baseline_id": info.get("id"),
            "baseline_primary": _primary(task, from_ref or {}),
            "best_primary": best_metric,
            "accepted_rounds": accepted,
            "rounds": records,
        })

    (ROOT / "harness_rounds_report.json").write_text(
        json.dumps({"experiments": all_results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nreport ->", ROOT / "harness_rounds_report.json")


if __name__ == "__main__":
    main()
