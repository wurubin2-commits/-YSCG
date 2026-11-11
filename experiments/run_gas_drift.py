"""Gas sensor drift 自进化闭环 + SwanLab 记录（赛题硬门槛：公开记录 + ≥3 次迭代）。

用法：
  python experiments/run_gas_drift.py --batches 20                # 默认 online，上传到 ZYYQ
  python experiments/run_gas_drift.py --batches 20 --mode local   # 仅本地调试（不联网）
  python experiments/run_gas_drift.py --batches 20 --mode offline # 离线保存，之后 swanlab sync

SwanLab 唯一记录地址：https://swanlab.cn/@07240115/ZYYQ
  -> project="ZYYQ", workspace="07240115"

说明：评测流标签总可得（use_free_labels），不消耗标注配额；模型优先 LightGBM，
缺库时自动退化为 sklearn RandomForest(n_jobs=1)，超参搜索优先 bayes_opt、缺库退化为随机搜索。
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# --- SwanLab 导入：venv 缺 swanlab 时自动桥接系统 Python 的 site-packages ---
try:
    import swanlab
except ImportError:  # pragma: no cover - 环境相关
    _SITE = os.environ.get(
        "YSCG_SWANLAB_SITE",
        r"C:\Users\Lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages",
    )
    if os.path.isdir(_SITE) and _SITE not in sys.path:
        sys.path.append(_SITE)
    import swanlab

from yscg.config import SystemConfig  # noqa: E402
from yscg.data import gas_drift_stream  # noqa: E402
from yscg.drift import Attributor, DriftMonitor  # noqa: E402
from yscg.eval import HeldOutSet  # noqa: E402
from yscg.guardrails import (  # noqa: E402
    AdmissionGate,
    AnchorSet,
    BudgetManager,
    MajorityPredictor,
    RollbackGuard,
    ValidatorProbe,
)
from yscg.memory import ProtectiveMemory  # noqa: E402
from yscg.models import ModelRegistry, create_model  # noqa: E402
from yscg.models.factory import _HAS_LGBM  # noqa: E402
from yscg.pipeline import SelfEvolutionLoop  # noqa: E402
from yscg.strategy import MockAdvisor, RulePolicy  # noqa: E402

# ---- SwanLab 唯一记录地址（旧地址已删除）----
SWANLAB_WORKSPACE = "07240115"
SWANLAB_PROJECT = "ZYYQ"
SWANLAB_URL = "https://swanlab.cn/@07240115/ZYYQ"


def main():
    parser = argparse.ArgumentParser(description="Gas sensor drift 自进化闭环 + SwanLab")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "gas" / "Dataset"))
    parser.add_argument("--batches", type=int, default=20, help="最多处理的流批次数")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hpo-trials", type=int, default=3)
    parser.add_argument("--max-retrains", type=int, default=6)
    parser.add_argument("--mode", choices=["online", "local", "offline", "disabled"],
                        default="online", help="online=上传 swanlab.cn（正式记录）")
    parser.add_argument("--experiment", default=None, help="实验名（默认按模型后端自动生成）")
    args = parser.parse_args()

    backend = "LightGBM" if _HAS_LGBM else "RandomForest"
    experiment_name = args.experiment or f"Gas-{backend}-测试"
    print(f"[model] backend = {backend}")
    if not _HAS_LGBM:
        print("[warn] 未安装 lightgbm，当前使用 sklearn RandomForest 兜底（装上 lightgbm 会自动切回）")

    # ===== 1. SwanLab 初始化（放在最前面，实验开始即记录）=====
    run = swanlab.init(
        project=SWANLAB_PROJECT,      # ZYYQ
        workspace=SWANLAB_WORKSPACE,  # 07240115
        experiment_name=experiment_name,
        description=f"Gas sensor drift 自进化闭环；记录地址 {SWANLAB_URL}",
        config={"model": backend, "dataset": "Gas Drift", "seed": args.seed},
        mode=args.mode,
    )
    print(f"[swanlab] project={SWANLAB_PROJECT} workspace={SWANLAB_WORKSPACE} "
          f"mode={args.mode} experiment={experiment_name}")

    try:
        # ===== 2. 组装数据与闭环（原流程）=====
        stream, spec = gas_drift_stream(args.data, batch_size=args.batch_size)
        cfg = SystemConfig(model_type="lgbm", n_batches=min(stream.n_batches, args.batches),
                           batch_size=args.batch_size, seed=args.seed, annotation_budget=0)
        cfg.strategy.window_size = 5
        cfg.strategy.hpo_trials = args.hpo_trials
        cfg.max_retrains = args.max_retrains

        init_X, init_y = stream.get_init_data()
        val_X, val_y = stream.get_val_data()
        held_X, held_y = stream.get_heldout_data()

        art = PROJECT_ROOT / "artifacts" / "gas-drift"
        reg = ModelRegistry(art, "lgbm")
        budget = BudgetManager(0, cfg.max_retrains, cfg.strategy.hpo_trials * cfg.max_retrains)
        monitor = DriftMonitor(cfg.strategy)
        anchor = AnchorSet(eps=cfg.strategy.anchor_eps)
        anchor.add_slice("val", val_X, val_y)
        memory = ProtectiveMemory(cfg.strategy.memory_capacity, cfg.strategy.memory_min_conf)
        gate = AdmissionGate(cfg.strategy, anchor, memory)
        guard = RollbackGuard(reg, probe_X=val_X[:50])
        good = create_model("lgbm", seed=cfg.seed)
        good.fit(init_X, init_y, n_estimators=60)
        probe = ValidatorProbe(good, MajorityPredictor(init_y), (val_X, val_y))
        heldout = HeldOutSet(held_X, held_y)

        # ===== 3. 每一步把指标传给 SwanLab =====
        def log_fn(step, data):
            numeric = {k: v for k, v in data.items() if isinstance(v, (int, float))}
            if numeric:
                swanlab.log(numeric, step=step)

        loop = SelfEvolutionLoop(
            cfg, stream, reg, budget, RulePolicy(cfg.strategy), MockAdvisor(),
            Attributor(), monitor, log_fn=log_fn,
            admission_gate=gate, rollback_guard=guard,
            validator_probe=probe, memory=memory, heldout=heldout,
        )
        summary = loop.run()

        # 汇总也记一条（step 在最后）
        swanlab.log({k: v for k, v in summary.items() if isinstance(v, (int, float))},
                    step=cfg.n_batches)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        print(f"registry versions: {len(reg.history())} | qualified: {len(reg.qualified_versions())}")
    finally:
        # ===== 4. 收尾 =====
        swanlab.finish()


if __name__ == "__main__":
    main()
