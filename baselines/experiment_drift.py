"""B 漂移 10 轮：按时间分批模拟漂移，PSI 监测 → 超阈值才重训 → 锚点保护。

流程（每轮）：
  1. 取该轮的新批次数据（按样本 id 时间顺序切 10 段）；
  2. 计算输入 PSI（相对参考批次，无监督，不用标签）；
  3. 漂移超阈值 → 用「最近窗口」从 champion 续训；否则保留父模型（省成本）；
  4. 候选在验证集评估，优于 champion 才接受，否则回滚；
  5. 记录 drift_score / 决策 / 指标，上传 SwanLab。

用法：
  python baselines/experiment_drift.py --rounds 10 --psi-threshold 0.15 --mode online
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ART = ROOT.parent / "artifacts" / "drift"
PROFILES = [("a", "A", "classification"), ("b", "B", "regression"), ("c", "C", "regression")]
SWANLAB_PROJECT = "ZYYQ"
SWANLAB_WORKSPACE = "07240115"

CHAMPIONS = {
    "A": ROOT.parent / "artifacts" / "rounds" / "A" / "r04" / "checkpoint.npz",
    "B": ROOT.parent / "artifacts" / "rounds" / "B" / "r02" / "checkpoint.npz",
    "C": ROOT.parent / "artifacts" / "rounds" / "C" / "r02" / "checkpoint.npz",
}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def psi_per_feature(ref, cur, bins=10, eps=1e-6):
    out = []
    for j in range(ref.shape[1]):
        r, c = ref[:, j], cur[:, j]
        edges = np.quantile(r, np.linspace(0.0, 1.0, bins + 1))
        edges = np.unique(np.concatenate(([-np.inf], edges, [np.inf])))
        if len(edges) < 3:
            out.append(0.0)
            continue
        rp = np.histogram(r, bins=edges)[0].astype(float)
        cp = np.histogram(c, bins=edges)[0].astype(float)
        rp /= max(rp.sum(), 1.0)
        cp /= max(cp.sum(), 1.0)
        rp = np.clip(rp, eps, None)
        cp = np.clip(cp, eps, None)
        out.append(float(np.sum((cp - rp) * np.log(cp / rp))))
    return np.array(out)


def _primary(task, m):
    return m.get("macro_f1") if task == "classification" else m.get("rmse")


def _better(task, after, before, min_gain=1e-4):
    if task == "classification":
        return after >= before + min_gain
    return after <= before - min_gain


def main():
    parser = argparse.ArgumentParser(description="B 漂移 10 轮")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--psi-threshold", type=float, default=0.15)
    parser.add_argument("--mode", default="online", choices=["online", "local", "offline", "disabled"])
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    import swanlab
    trainer = _load(ROOT / "diverse_trainer.py", "diverse_trainer")

    all_results = []
    for folder, key, task in PROFILES:
        if args.only and key != args.only:
            continue
        pkg = ROOT / folder
        model_mod = _load(pkg / "baselines" / "model.py", f"model_{key}_drift")
        data_mod = _load(pkg / "baselines" / "data.py", f"data_{key}_drift")
        info = json.loads((pkg / "baseline-info.json").read_text(encoding="utf-8"))

        champ = CHAMPIONS[key]
        if not Path(champ).exists():
            champ = pkg / "checkpoint.npz"
        champion = model_mod.Model.load(str(champ))
        champ_path = str(champ)

        train_ds = data_mod.read_dataset(
            str(pkg / "public" / "train.jsonl"), task,
            classes=champion.meta["classes"] or None, expected_paths=champion.meta["paths"],
        )
        val_ds = data_mod.read_dataset(
            str(pkg / "public" / "validation.jsonl"), task,
            classes=champion.meta["classes"] or None, expected_paths=champion.meta["paths"],
        )
        order = np.argsort(np.asarray(train_ds["ids"]))
        xs, ys = train_ds["x"][order], train_ds["y"][order]
        chunks = np.array_split(np.arange(len(xs)), args.rounds)
        ref = xs[chunks[0]]

        champion_metrics = trainer.evaluate(champion, val_ds["x"], val_ds["y"])
        start_primary = _primary(task, champion_metrics)
        best = _primary(task, champion_metrics)

        base_dir = ART / key
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "tried_config_hashes": [], "rejected_config_hashes": [], "candidate_hashes": [],
            "anchor_floor": {"primary": best}, "protective_memory": [], "policy_params": {},
            "last_reject_reason": "", "drift_score": 0.0,
        }

        run = swanlab.init(
            project=SWANLAB_PROJECT, workspace=SWANLAB_WORKSPACE,
            experiment_name=f"harness-drift-{key}",
            description=f"B 漂移 {args.rounds} 轮（PSI 监测 → 超阈值重训 → 锚点保护）",
            config={"dataset_key": key, "task": task, "baseline_id": info.get("id"),
                    "rounds": args.rounds, "psi_threshold": args.psi_threshold,
                    "champion_start": start_primary},
            mode=args.mode,
        )
        print(f"[swanlab] {key} mode={args.mode} (champion={_primary(task, champion_metrics):.4f})")

        records = []
        retrains = 0
        for r in range(1, args.rounds + 1):
            idx = chunks[r - 1]
            drift = float(np.mean(psi_per_feature(ref, xs[idx])))
            state["drift_score"] = drift
            triggered = drift > args.psi_threshold
            action = "retrain" if triggered else "conservative"

            m_before = _primary(task, champion_metrics)
            accept = False
            m_after = m_before
            reason = "无显著漂移，保留父模型（省成本）"
            llm_id = f"local-advisor-drift-{key}-{r:02d}"

            if triggered:
                retrains += 1
                window = np.concatenate(chunks[max(0, r - 3):r])
                cand = model_mod.Model.load(champ_path)
                cand.meta["learning_rate"] = 0.0005
                trainer.fit_config(cand, xs[window], ys[window], {"lr": 0.0005, "epochs": 1})
                out_dir = base_dir / f"r{r:02d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                ckpt = out_dir / "checkpoint.npz"
                if ckpt.exists():
                    ckpt.unlink()
                cand.save(ckpt)
                cand_hash = _sha(ckpt)
                state["candidate_hashes"].append(cand_hash)
                cand_metrics = trainer.evaluate(cand, val_ds["x"], val_ds["y"])
                m_after = _primary(task, cand_metrics)
                accept = _better(task, m_after, m_before)
                if accept:
                    champion = cand
                    champion_metrics = cand_metrics
                    champ_path = str(ckpt)
                    best = m_after
                    reason = f"漂移 {drift:.3f}>{args.psi_threshold}，重训改善 {m_after - m_before:+.4f}，接受"
                else:
                    state["rejected_config_hashes"].append(cand_hash)
                    reason = f"漂移 {drift:.3f}>{args.psi_threshold}，重训未改善（{m_after:.4f} vs {m_before:.4f}），回滚保留父模型"
                state["last_reject_reason"] = "" if accept else reason
                state["anchor_floor"]["primary"] = best

            data = {
                "drift/psi_mean": drift,
                "drift/triggered": int(triggered),
                "round/action": 1 if triggered else 0,
                "round/primary_before": m_before,
                "round/primary_after": m_after,
                "round/best": best,
                "round/accepted": 1 if accept else 0,
                "round/anchor_floor": best,
                "round/anchor_distance": (m_after - best) if task == "classification" else (best - m_after),
                "round/retrain_count": retrains,
            }
            swanlab.log(data, step=r)

            records.append({
                "round": r, "drift_score": drift, "triggered": bool(triggered),
                "action": action, "llm_call_id": llm_id,
                "primary_before": m_before, "primary_after": m_after,
                "accepted": bool(accept), "best": best, "reason": reason,
            })
            print(f"[{key}] r{r:02d} PSI={drift:.3f} {'DRIFT' if triggered else 'no-drift'} "
                  f"{m_before:.4f}->{m_after:.4f} accept={accept}")

        swanlab.log({
            "final/best": best,
            "final/champion_start": start_primary,
            "final/retrain_rounds": retrains,
            "final/accepted_rounds": sum(1 for x in records if x["accepted"]),
            "final/rounds": len(records),
        }, step=args.rounds + 1)
        swanlab.finish()

        all_results.append({
            "profile": key, "task": task, "baseline_id": info.get("id"),
            "champion_start": start_primary,
            "best": best, "retrain_rounds": retrains,
            "accepted_rounds": sum(1 for x in records if x["accepted"]),
            "drift_rounds": sum(1 for x in records if x["triggered"]),
            "state": state, "rounds": records,
        })

    (ROOT / "drift_report.json").write_text(
        json.dumps({"experiments": all_results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nreport ->", ROOT / "drift_report.json")


if __name__ == "__main__":
    main()
