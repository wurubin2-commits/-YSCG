"""规则策略引擎：漂移报告的兜底决策。

核心思想（成本收益）：漂移本身不触发动作，只有"漂移 + 模型受损"才值得花
算力与标注预算。决策优先级：
  1. 窗口准确率新鲜且模型崩坏 → 紧急全量重训；
  2. 窗口准确率新鲜且健康 → 不动作（分布变了但模型没事，省预算）；
  3. 窗口准确率过期/未知 → 依据漂移严重度：轻度适配 / 全量重训 / 不动作。

LLM 顾问仅提供策略参数调整建议，动作决策始终由规则引擎给出，
保证系统在无 LLM / LLM 异常时依然稳定运行（基础门槛：无人干预跑完全程）。
"""
from enum import Enum

from ..drift.monitor import DriftReport


class Action(str, Enum):
    NOOP = "noop"
    LIGHT_ADAPT = "light_adapt"
    FULL_RETRAIN = "full_retrain"
    ROLLBACK = "rollback"


ACTION_CODE = {Action.NOOP: 0, Action.LIGHT_ADAPT: 1, Action.FULL_RETRAIN: 2, Action.ROLLBACK: 3}


class RulePolicy:
    def __init__(self, strategy):
        self.strategy = strategy

    def decide(
        self,
        report: DriftReport,
        champion_window_acc: float | None,
        static_baseline_acc: float,
        acc_age: int | None = None,
    ) -> Action:
        s = self.strategy
        acc_fresh = (
            champion_window_acc is not None
            and (acc_age is None or acc_age <= s.stale_after_batches)
        )
        if acc_fresh:
            if champion_window_acc < static_baseline_acc - 0.10:
                return Action.FULL_RETRAIN  # 冠军在流上崩坏 → 紧急重训
            if champion_window_acc >= static_baseline_acc - s.health_tolerance:
                return Action.NOOP  # 漂移但模型健康 → 省预算
            if report.severity == "mild":
                return Action.LIGHT_ADAPT
            if report.severity == "severe":
                return Action.FULL_RETRAIN
            return Action.NOOP
        if report.severity == "severe":
            return Action.FULL_RETRAIN
        if report.severity == "mild":
            return Action.LIGHT_ADAPT
        return Action.NOOP


class StaticPolicy(RulePolicy):
    """静态基线对照：永不采取任何动作（部署后不更新的模型）。"""

    def decide(self, report, champion_window_acc, static_baseline_acc, acc_age=None):
        return Action.NOOP
