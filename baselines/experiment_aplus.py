"""A+ 多样性 10 轮：LLM(本地替身) 生成配置增量 → 规则校验 → 哈希去重 → 续训 → 准入。

验收目标：
- 候选唯一率 100%（每轮候选 checkpoint 哈希不同）
- 连续相同候选 = 0
- 最终 best ≥ 当前 champion（A 0.7264 / B 647.24 / C 51.93）
- 回滚保护仍生效（变差就拒绝，champion 不变）
- 每轮有真实/替身模型调用 id（本地用 local-advisor-*，平台换成 rnd.call_ids）

用法：
  python baselines/experiment_aplus.py --rounds 10 --mode online
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT.parent / "artifacts" / "aplus"
PROFILES = [("a", "A", "classification"), ("b", "B", "regression"), ("c", "C", "regression")]
SWANLAB_PROJECT = "ZYYQ"
SWANLAB_WORKSPACE = "07240115"

CHAMPIONS = {
    "A": ROOT.parent / "artifacts" / "rounds" / "A" / "r04" / "checkpoint.npz",
    "B": ROOT.parent / "artifacts" / "rounds" / "B" / "r02" / "checkpoint.npz",
    "C": ROOT.parent / "artifacts" / "rounds" / "C" / "r02" / "checkpoint.npz",
}
BASELINE_BEST = {"A": 0.7264, "B": 647.2395, "C": 51.9339}

# 10 轮探索维度（按用户给出的 A/B/C 轮换表）
SCHEDULE = {
    "classification": [
        ("baseline_small_step", {"lr": 0.0005, "epochs": 1}),
        ("class_weight", {"class_weight": "balanced", "epochs": 1}),
        ("low_lr", {"lr": 0.0003, "epochs": 1}),
        ("bootstrap_sample", {"bootstrap": 101, "epochs": 1}),
        ("small_lr_plus_l2", {"lr": 0.0002, "l2": 0.001, "epochs": 1}),
        ("bootstrap_plus_batch", {"bootstrap": 202, "batch": 64, "epochs": 1}),
        ("ensemble_parent_candidate", {"lr": 0.0004, "epochs": 1}),
        ("drift_window_weight", {"recency": 0.8, "epochs": 1}),
        ("recalibrate", {"class_weight": "balanced", "lr": 0.0004, "epochs": 1}),
        ("combine_best", {"lr": 0.0003, "l2": 0.0005, "recency": 0.5, "epochs": 1}),
    ],
    "regression": [
        ("baseline_small_step", {"lr": 0.0005, "epochs": 1}),
        ("window_recency", {"recency": 0.5, "epochs": 1}),
        ("huber_loss", {"huber": True, "epochs": 1}),
        ("reg_early_stop", {"l2": 0.002, "epochs": 1}),
        ("small_lr_plus_l2", {"lr": 0.0002, "l2": 0.001, "epochs": 1}),
        ("outlier_robust", {"huber": True, "batch": 64, "epochs": 1}),
        ("ensemble_parent_candidate", {"lr": 0.0004, "epochs": 1}),
        ("drift_window_weight", {"recency": 0.8, "epochs": 1}),
        ("pred_offset_limit", {"lr": 0.0004, "epochs": 1}),
        ("combine_best", {"lr": 0.0003, "l2": 0.0005, "recency": 0.5, "epochs": 1}),
    ],
}
CLAMP = {"lr": (1e-5, 1e-2), "batch": (16, 512), "epochs": (1, 5), "l2": (0.0, 0.05), "recency": (0.0, 1.0)}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cfg_hash(delta):
    canon = json.dumps(delta, sort_keys=True, separators=(",", ":"))
    return "cfg:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def _validate(delta):
    out = {}
    for k, v in delta.items():
        if k in CLAMP:
            lo, hi = CLAMP[k]
            if k == "batch":
                v = int(max(lo, min(hi, int(v))))
            elif k == "epochs":
                v = int(max(lo, min(hi, int(v))))
            else:
                v = float(max(lo, min(hi, float(v))))
        out[k] = v
    return out


def _primary(task, m):
    return m.get("macro_f1") if task == "classification" else m.get("rmse")


def _better(task, after, before, min_gain=1e-4):
    if task == "classification":
        return after >= before + min_gain
    return after <= before - min_gain


def main():
    parser = argparse.ArgumentParser(description="A+ 多样性 10 轮")
    parser.add_argument("--rounds", type=int, default=10)
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
        model_mod = _load(pkg / "baselines" / "model.py", f"model_{key}_aplus")
        data_mod = _load(pkg / "baselines" / "data.py", f"data_{key}_aplus")
        info = json.loads((pkg / "baseline-info.json").read_text(encoding="utf-8"))

        champ = CHAMPIONS[key]
        if not Path(champ).exists():
            champ = pkg / "checkpoint.npz"
        parent = model_mod.Model.load(str(champ))
        train_ds = data_mod.read_dataset(
            str(pkg / "public" / "train.jsonl"), task,
            classes=parent.meta["classes"] or None, expected_paths=parent.meta["paths"],
        )
        val_ds = data_mod.read_dataset(
            str(pkg / "public" / "validation.jsonl"), task,
            classes=parent.meta["classes"] or None, expected_paths=parent.meta["paths"],
        )
        parent_metrics = trainer.evaluate(parent, val_ds["x"], val_ds["y"])

        base_dir = ART / key
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "tried_config_hashes": [], "rejected_config_hashes": [], "candidate_hashes": [],
            "anchor_floor": {"primary": _primary(task, parent_metrics)},
            "protective_memory": [], "policy_params": {}, "last_reject_reason": "", "drift_score": 0.0,
        }
        champion = parent
        champion_metrics = parent_metrics
        best = _primary(task, parent_metrics)
        champ_path = str(champ)

        run = swanlab.init(
            project=SWANLAB_PROJECT, workspace=SWANLAB_WORKSPACE,
            experiment_name=f"harness-aplus-{key}",
            description=f"A+ 多样性 {args.rounds} 轮（配置增量 + 去重 + 拒绝反馈 + 锚点）",
            config={"dataset_key": key, "task": task, "baseline_id": info.get("id"),
                    "rounds": args.rounds, "champion_start": _primary(task, parent_metrics),
                    "admission": "strict_improvement+min_gain"},
            mode=args.mode,
        )
        print(f"[swanlab] {key} mode={args.mode}")

        records = []
        for r in range(1, args.rounds + 1):
            # 本地 LLM 替身：按探索维度轮换，跳过已试过的配置哈希
            sched = SCHEDULE[task]
            picked = None
            for offset in range(len(sched)):
                tag, delta = sched[(r - 1 + offset) % len(sched)]
                delta = _validate(delta)
                ch = _cfg_hash(delta)
                if ch not in state["tried_config_hashes"]:
                    picked = (tag, delta, ch)
                    break
            if picked is None:
                tag, delta = sched[(r - 1) % len(sched)]
                delta = _validate(delta)
                delta["seed"] = 1000 + r
                picked = (tag, delta, _cfg_hash(delta))
            tag, delta, ch = picked
            state["tried_config_hashes"].append(ch)
            state["policy_params"] = dict(delta)

            hypothesis = f"{tag}：在父模型上应用 {json.dumps(delta, ensure_ascii=False)}，期望验证指标改善"
            expected = "macro_f1 提升" if task == "classification" else "rmse 下降"

            # 训练候选（始终从当前 champion 续训）
            cand = model_mod.Model.load(champ_path)
            cand.meta["learning_rate"] = float(delta.get("lr", cand.meta["learning_rate"]))
            cand.meta["batch_size"] = int(delta.get("batch", cand.meta["batch_size"]))
            cand.meta["seed"] = int(delta.get("seed", cand.meta["seed"]))
            trainer.fit_config(cand, train_ds["x"], train_ds["y"], delta)

            out_dir = base_dir / f"r{r:02d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            ckpt = out_dir / "checkpoint.npz"
            if ckpt.exists():
                ckpt.unlink()
            cand.save(ckpt)
            cand_hash = _sha(ckpt)

            duplicate = cand_hash in state["candidate_hashes"]
            state["candidate_hashes"].append(cand_hash)
            cand_metrics = trainer.evaluate(cand, val_ds["x"], val_ds["y"])
            m_before = _primary(task, champion_metrics)
            m_after = _primary(task, cand_metrics)
            ens_metrics = None
            if tag == "ensemble_parent_candidate":
                ens_metrics = trainer.ensemble_evaluate(champion, cand, val_ds["x"], val_ds["y"])

            accept = False
            reason = ""
            if duplicate:
                reason = "重复候选（candidate_hash 已出现），不提交，保留父模型"
            else:
                accept = _better(task, m_after, m_before)
                if accept:
                    champion = cand
                    champion_metrics = cand_metrics
                    best = m_after
                    champ_path = str(ckpt)
                    reason = f"改善 {m_after - m_before:+.4f}，接受为新 champion"
                else:
                    state["rejected_config_hashes"].append(ch)
                    reason = f"未改善（{m_after:.4f} vs {m_before:.4f}），回滚保留父模型"
            state["last_reject_reason"] = "" if accept else reason
            state["anchor_floor"]["primary"] = best

            gain = (m_after - m_before) if task == "classification" else (m_before - m_after)
            anchor_dist = (m_after - best) if task == "classification" else (best - m_after)
            data = {
                "round/primary_before": m_before,
                "round/primary_after": m_after,
                "round/gain": gain,
                "round/best": best,
                "round/accepted": 1 if accept else 0,
                "round/anchor_floor": best,
                "round/anchor_distance": anchor_dist,
                "round/tried_hash_count": len(state["tried_config_hashes"]),
                "round/rejected_hash_count": len(state["rejected_config_hashes"]),
                "round/candidate_unique": 0 if duplicate else 1,
            }
            if task == "classification":
                data["round/accuracy_after"] = cand_metrics["accuracy"]
            else:
                data["round/mae_after"] = cand_metrics["mae"]
            if ens_metrics is not None:
                data["round/ensemble_primary"] = _primary(task, ens_metrics)
            swanlab.log(data, step=r)

            records.append({
                "round": r, "exploration_tag": tag, "config_delta": delta,
                "config_hash": ch, "candidate_hash": cand_hash,
                "llm_call_id": f"local-advisor-{key}-{r:02d}",
                "hypothesis": hypothesis, "expected_effect": expected,
                "primary_before": m_before, "primary_after": m_after,
                "accepted": accept, "reason": reason, "duplicate_candidate": duplicate,
                "ensemble_primary": _primary(task, ens_metrics) if ens_metrics else None,
            })
            print(f"[{key}] r{r:02d} {tag:<28} {m_before:.4f}->{m_after:.4f} accept={accept} dup={duplicate}")

        swanlab.log({
            "final/best": best, "final/champion_start": _primary(task, parent_metrics),
            "final/accepted_rounds": sum(1 for x in records if x["accepted"]),
            "final/unique_candidates": len(set(x["candidate_hash"] for x in records)),
            "final/rounds": len(records),
            "final/anchor_floor": best,
        }, step=args.rounds + 1)
        swanlab.finish()

        hashes = [x["candidate_hash"] for x in records]
        consecutive_dup = sum(1 for a, b in zip(hashes, hashes[1:]) if a == b)
        all_results.append({
            "profile": key, "task": task, "baseline_id": info.get("id"),
            "champion_start": _primary(task, parent_metrics),
            "best": best,
            "accepted_rounds": sum(1 for x in records if x["accepted"]),
            "unique_candidates": len(set(hashes)),
            "unique_rate": len(set(hashes)) / len(hashes) if hashes else 0.0,
            "consecutive_duplicates": consecutive_dup,
            "state": state,
            "rounds": records,
        })

    (ROOT / "aplus_report.json").write_text(
        json.dumps({"experiments": all_results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nreport ->", ROOT / "aplus_report.json")


if __name__ == "__main__":
    main()
