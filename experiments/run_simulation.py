"""端到端仿真入口：漂移流全闭环 + SwanLab 实验记录。

用法：
  python experiments/run_simulation.py --model lgbm --batches 70
  python experiments/run_simulation.py --model mlp --batches 70
  # 已注册 SwanLab 云端账号后：--mode online（实验自动同步云端）
  # 企业赛接 PyTRiO（OpenAI 兼容接口）：
  #   set YSCG_LLM_BASE_URL=... & set YSCG_LLM_API_KEY=... & set YSCG_LLM_MODEL=Qwen3.8-27B
  python experiments/run_simulation.py --advisor openai
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import swanlab  # noqa: E402

from yscg.config import SystemConfig  # noqa: E402
from yscg.data import TabularDriftSimulator  # noqa: E402
from yscg.drift import Attributor, DriftMonitor  # noqa: E402
from yscg.guardrails import BudgetManager  # noqa: E402
from yscg.models import ModelRegistry  # noqa: E402
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
    parser.add_argument("--project", default="yscg")
    args = parser.parse_args()

    arm = f"{args.model}{'-static' if args.static else ''}{'-persist' if args.scenario == 'persistent' else ''}"
    cfg = SystemConfig(
        model_type=args.model,
        n_batches=args.batches,
        batch_size=args.batch_size,
        seed=args.seed,
        annotation_budget=args.budget,
        project=args.project,
    )
    stream = TabularDriftSimulator(
        n_batches=cfg.n_batches,
        batch_size=cfg.batch_size,
        init_samples=3000,
        val_samples=1000,
        annotation_budget=cfg.annotation_budget,
        seed=cfg.seed,
        scenario=args.scenario,
    )
    artifacts = PROJECT_ROOT / "artifacts" / f"sim-{arm}"
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

    run = swanlab.init(
        project=args.project,
        experiment_name=f"sim-{arm}",
        description=(
            "静态基线对照：初始模型部署后不更新"
            if args.static
            else f"漂移流闭环仿真（{args.scenario}）：漂移监测—诊断归因—策略决策—实验优化—评估上线/回滚"
        ),
        config=cfg.flat(),
        mode=args.mode,
    )

    # 侧记录：与分析脚本/报告使用（SwanLab 为正式实验记录）
    side_log = artifacts / "metrics.jsonl"
    side_log.parent.mkdir(parents=True, exist_ok=True)
    side_lines = []

    def log_fn(step, data):
        swanlab.log(data, step=step)
        side_lines.append({"step": step, **data})

    loop = SelfEvolutionLoop(
        cfg, stream, registry, budget, policy, advisor, attributor, monitor,
        log_fn=log_fn,
    )
    summary = loop.run()
    side_log.write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in side_lines), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("registry history:")
    for rec in registry.history():
        print(f"  {rec['id']:<4} {rec['tag']:<18} metrics={rec['metrics']}")
    swanlab.finish()


if __name__ == "__main__":
    main()
