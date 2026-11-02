"""端到端仿真入口：漂移流全闭环 + SwanLab 实验记录 + 准入决策 + 对照试验。

用法：
  python experiments/run_simulation.py --model lgbm --batches 70
  python experiments/run_simulation.py --model mlp --batches 70
  # 静态基线对照（部署后不更新的模型）
  python experiments/run_simulation.py --static
  # 重复运行 N 次，报告增益的均值/方差与置信区间（对齐「对照试验」要求）
  python experiments/run_simulation.py --repeats 5
  # 对照试验：对照组消耗相同训练算力/评估调用但不自我修改
  python experiments/run_simulation.py --repeats 5 --control
  # 已注册 SwanLab 云端账号后：--mode online
  # 企业赛接 PyTRiO（OpenAI 兼容接口）：
  #   set YSCG_LLM_BASE_URL=... & set YSCG_LLM_API_KEY=... & set YSCG_LLM_MODEL=Qwen3.8-27B
  python experiments/run_simulation.py --advisor openai
"""
import argparse
import copy
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import swanlab  # noqa: E402

from yscg.config import SystemConfig  # noqa: E402
from yscg.data import TabularDriftSimulator  # noqa: E402
from yscg.drift import Attributor, DriftMonitor  # noqa: E402
from yscg.eval import HeldOutSet, repeated_gain_report  # noqa: E402
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
from yscg.pipeline import SelfEvolutionLoop  # noqa: E402
from yscg.strategy import MockAdvisor, RulePolicy, StaticPolicy, create_advisor  # noqa: E402


def build_advisor(advisor_type: str):
    if advisor_type == "mock":
        return create_advisor("mock")
    return create_advisor(
        "openai",
        base_url=os.environ.get("YSCG_LLM_BASE_URL", ""),
        api_key=os.environ.get("YSCG_LLM_API_KEY", ""),
        model=os.environ.get("YSCG_LLM_MODEL", "Qwen3.8-27B"),
    )


def build_components(cfg: SystemConfig, args, artifacts: Path) -> dict:
    """组装流/注册表/预算/策略/准入门/回滚护栏/验证器/留出集。"""
    stream = TabularDriftSimulator(
        n_batches=cfg.n_batches,
        batch_size=cfg.batch_size,
        init_samples=3000,
        val_samples=1000,
        heldout_samples=500,
        annotation_budget=cfg.annotation_budget,
        seed=cfg.seed,
        scenario=args.scenario,
    )
    registry = ModelRegistry(artifacts, cfg.model_type)
    budget = BudgetManager(
        cfg.annotation_budget,
        cfg.max_retrains,
        cfg.strategy.hpo_trials * cfg.max_retrains,
    )
    policy = StaticPolicy(cfg.strategy) if args.static else RulePolicy(cfg.strategy)
    advisor = MockAdvisor() if args.static else build_advisor(args.advisor)
    attributor = Attributor()
    monitor = DriftMonitor(cfg.strategy)

    init_X, init_y = stream.get_init_data()
    val_X, val_y = stream.get_val_data()
    held_X, held_y = stream.get_heldout_data()

    # 锚点集：冻结留出切片（防能力退化，锚点分数只升不降）
    anchor = AnchorSet(eps=cfg.strategy.anchor_eps)
    anchor.add_slice("val", val_X, val_y, weight=1.0)

    # 保护性记忆（记忆检查数据源，容量受限、分层）
    memory = ProtectiveMemory(cfg.strategy.memory_capacity, cfg.strategy.memory_min_conf)

    # 准入门：增益 → 锚点 → 记忆
    admission_gate = AdmissionGate(cfg.strategy, anchor, memory)

    # 回滚护栏：旧版本完整保留 + 回滚预演（回滚不是免费的）
    rollback_guard = RollbackGuard(registry, probe_X=val_X[:200])

    # 验证器自检探针：已知好/坏候选，验证验证器能否正确排序
    good_model = create_model(cfg.model_type, seed=cfg.seed)
    if cfg.model_type == "lgbm":
        good_model.fit(init_X, init_y, n_estimators=max(50, cfg.strategy.light_boost_rounds))
    else:
        good_model.fit(init_X, init_y, epochs=cfg.strategy.mlp_epochs_light)
    bad_model = MajorityPredictor(init_y)
    validator_probe = ValidatorProbe(good_model, bad_model, (val_X, val_y))

    # 冻结留出集（最终评判标准，迭代过程不可访问）
    heldout = HeldOutSet(held_X, held_y)

    return dict(
        stream=stream, registry=registry, budget=budget, policy=policy,
        advisor=advisor, attributor=attributor, monitor=monitor,
        admission_gate=admission_gate, rollback_guard=rollback_guard,
        validator_probe=validator_probe, memory=memory, heldout=heldout,
    )


def run_one(cfg: SystemConfig, args, arm: str, seed: int, control_mode: bool = False) -> dict:
    """以给定 seed 运行一次完整闭环，返回 summary。对照组消耗相同算力但不自我修改。"""
    cfg = replace(cfg, seed=seed, strategy=copy.deepcopy(cfg.strategy))
    suffix = f"-ctrl" if control_mode else ""
    artifacts = PROJECT_ROOT / "artifacts" / f"sim-{arm}{suffix}-s{seed}"
    comp = build_components(cfg, args, artifacts)

    run = swanlab.init(
        project=args.project,
        experiment_name=f"sim-{arm}{suffix}-s{seed}",
        description=(
            "对照组：消耗相同训练算力/评估调用但不自我修改"
            if control_mode
            else (
                "静态基线对照：初始模型部署后不更新" if args.static
                else f"漂移流闭环仿真（{args.scenario}）：监测—归因—决策—优化—准入上线/回滚"
            )
        ),
        config=cfg.flat(),
        mode=args.mode,
    )

    side_log = artifacts / "metrics.jsonl"
    side_log.parent.mkdir(parents=True, exist_ok=True)
    side_lines = []

    def log_fn(step, data):
        swanlab.log(data, step=step)
        side_lines.append({"step": step, **data})

    loop = SelfEvolutionLoop(
        cfg, comp["stream"], comp["registry"], comp["budget"], comp["policy"],
        comp["advisor"], comp["attributor"], comp["monitor"], log_fn=log_fn,
        admission_gate=comp["admission_gate"], rollback_guard=comp["rollback_guard"],
        validator_probe=comp["validator_probe"], memory=comp["memory"],
        heldout=comp["heldout"], control_mode=control_mode,
    )
    summary = loop.run()
    side_log.write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in side_lines), encoding="utf-8"
    )
    swanlab.finish()
    return summary


def main():
    parser = argparse.ArgumentParser(description="工业模型自进化系统闭环仿真")
    parser.add_argument("--model", choices=["lgbm", "mlp"], default="lgbm")
    parser.add_argument("--batches", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=600, help="标注总配额")
    parser.add_argument(
        "--scenario",
        choices=["transient", "persistent"],
        default="transient",
        help="transient=漂移后恢复(考验回滚保护)；persistent=漂移持续(考验自适应)",
    )
    parser.add_argument("--advisor", choices=["mock", "openai"], default="mock")
    parser.add_argument("--mode", choices=["local", "online", "disabled"], default="local")
    parser.add_argument("--static", action="store_true", help="静态基线对照：部署后不更新的模型")
    parser.add_argument("--repeats", type=int, default=1, help="重复运行次数（报告方差与置信区间）")
    parser.add_argument("--control", action="store_true",
                        help="对照组：消耗相同训练算力/评估调用但不自我修改")
    parser.add_argument("--project", default="yscg")
    args = parser.parse_args()

    arm = f"{args.model}{'-static' if args.static else ''}{'-persist' if args.scenario == 'persistent' else ''}"
    base_cfg = SystemConfig(
        model_type=args.model,
        n_batches=args.batches,
        batch_size=args.batch_size,
        seed=args.seed,
        annotation_budget=args.budget,
        project=args.project,
    )

    if args.repeats <= 1 and not args.control:
        summary = run_one(base_cfg, args, arm, args.seed, control_mode=False)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("registry history:")
        for rec in comp_history(arm, args.seed, args.model):
            print(f"  {rec['id']:<4} {rec['tag']:<18} metrics={rec['metrics']} admission={rec.get('admission')}")
        return

    # 重复运行 / 对照试验：报告增益的均值/方差随重复次数的关系与置信区间
    exp_scores: list[float] = []
    ctrl_scores: list[float] = []
    for i in range(args.repeats):
        seed = args.seed + i
        s = run_one(base_cfg, args, arm, seed, control_mode=False)
        exp_scores.append(float(s["summary/heldout_acc"]))
        if args.control:
            c = run_one(base_cfg, args, arm, seed, control_mode=True)
            ctrl_scores.append(float(c["summary/heldout_acc"]))

    report = repeated_gain_report(exp_scores, ctrl_scores if args.control else None)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def comp_history(arm: str, seed: int, model_type: str) -> list:
    artifacts = PROJECT_ROOT / "artifacts" / f"sim-{arm}-s{seed}"
    reg = ModelRegistry(artifacts, model_type)
    return reg.history() if (artifacts / "registry.json").exists() else []


if __name__ == "__main__":
    main()
