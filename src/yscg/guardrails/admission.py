"""准入决策：增益 → 锚点 → 记忆 三层门 + 验证器自检。

多层判断：
- G1 增益检查（Gain）：候选在评估窗口上须显著优于冠军。
- G2 锚点检查（Anchor）：候选锚点分数不退化、floor 只升不降。
- G3 记忆检查（Memory）：候选在分层回归样本上的回归计数 <= 容差。
- 验证器自检（ValidatorProbe）：用已知好/坏候选验证「验证器能否给出正确排序」，
  若失效则拒绝本轮所有候选，而非假装决策机制仍在有效运行。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..memory import ProtectiveMemory, RegressionSampler, regression_count
from .anchor import AnchorSet, accuracy


class Gate(str, Enum):
    PASS = "pass"
    GAIN = "gain"
    ANCHOR = "anchor"
    MEMORY = "memory"
    VALIDATOR = "validator"
    ROLLBACK = "rollback"


@dataclass
class AdmissionResult:
    admitted: bool
    gate: Gate
    gain: float = 0.0
    anchor_score: float = 0.0
    regression_count: int = 0
    reason: str = ""

    def to_log(self) -> dict:
        return {
            "admission/admitted": 1 if self.admitted else 0,
            "admission/gate": self.gate.value,
            "admission/gain": self.gain,
            "admission/anchor_score": self.anchor_score,
            "admission/regression_count": self.regression_count,
        }


class AdmissionGate:
    def __init__(
        self,
        strategy,
        anchor: AnchorSet,
        memory: ProtectiveMemory,
        metric=accuracy,
    ):
        self.strategy = strategy
        self.anchor = anchor
        self.memory = memory
        self.metric = metric

    def _gain(self, challenger, champion, Xp, yp) -> float:
        if Xp is None or yp is None or len(yp) == 0:
            return 0.0
        c = self.metric(yp, challenger.predict_proba(Xp))
        m = self.metric(yp, champion.predict_proba(Xp))
        return float(c - m)

    def evaluate(
        self,
        challenger,
        champion,
        eval_pool,
        validator_healthy: bool = True,
        round_key: object | None = None,
    ) -> AdmissionResult:
        # 验证器失效 → 拒绝本轮所有候选，不假装仍在有效决策。
        if not validator_healthy:
            return AdmissionResult(False, Gate.VALIDATOR, reason="验证器自检失败：拒绝本轮所有候选")

        Xp, yp = (None, None) if eval_pool is None else eval_pool
        gain = self._gain(challenger, champion, Xp, yp)

        # G1 增益
        if gain < self.strategy.promote_min_gain:
            return AdmissionResult(False, Gate.GAIN, gain=gain, reason=f"增益不足 {gain:.4f}")

        # G2 锚点
        if not self.anchor.is_empty():
            scores = self.anchor.score(challenger, self.metric)
            comp = self.anchor.composite(scores)
            ok, viol = self.anchor.check(scores)
        else:
            scores, comp, ok, viol = {}, 0.0, True, []
        if not ok:
            return AdmissionResult(
                False, Gate.ANCHOR, gain=gain, anchor_score=comp, reason=f"锚点退化: {viol}"
            )

        # G3 记忆（分层回归样本，同一轮复用）
        sampler = RegressionSampler(self.memory, min_quota=self.strategy.memory_min_quota)
        R = sampler.sample(self.strategy.reg_sample_size, round_key=round_key)
        hard, _soft = regression_count(challenger, R)
        tol = max(1, int(self.strategy.reg_tolerance_frac * max(len(R), 1)))
        if hard > tol:
            return AdmissionResult(
                False, Gate.MEMORY, gain=gain, anchor_score=comp,
                regression_count=hard, reason=f"回归计数 {hard} > 容差 {tol}",
            )

        return AdmissionResult(
            True, Gate.PASS, gain=gain, anchor_score=comp,
            regression_count=hard, reason="通过全部准入门",
        )


class ValidatorProbe:
    """验证器自检：用已知好/坏候选的排序正确性判断验证器是否失效。

    good 模型应显著优于 bad 模型；若验证器连这对已知样本都无法正确排序，
    说明验证器（或底层指标/锚点/记忆计算）已失效，本轮应拒绝所有候选。
    """

    def __init__(self, good_model, bad_model, eval_pool, metric=accuracy, anchor_weight: float = 0.5):
        self.good = good_model
        self.bad = bad_model
        self.pool = eval_pool  # (X, y)
        self.metric = metric
        self.anchor_weight = float(anchor_weight)

    def _candidate_score(self, cand, champion, gate: AdmissionGate) -> float:
        X, y = self.pool
        gain = gate._gain(cand, champion, X, y)
        anchor = gate.anchor.composite(gate.anchor.score(cand, gate.metric))
        return float(gain) + self.anchor_weight * float(anchor)

    def healthy(self, gate: AdmissionGate, champion) -> bool:
        gs = self._candidate_score(self.good, champion, gate)
        bs = self._candidate_score(self.bad, champion, gate)
        return gs > bs


class MajorityPredictor:
    """已知「坏」候选：永远预测多数类，用作验证器自检的对照。"""

    def __init__(self, y_train: np.ndarray, n_classes: int | None = None):
        n_classes = int(np.max(y_train)) + 1 if n_classes is None else int(n_classes)
        counts = np.bincount(y_train.astype(int), minlength=n_classes).astype(float)
        self.priors = counts / max(counts.sum(), 1e-12)
        self.n_classes = n_classes

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self.priors, (len(X), 1))
