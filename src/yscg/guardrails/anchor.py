"""锚点集：冻结留出切片 + 单调不减的能力下限（Anchor Gate）。

设计要点（对齐「锚点检查」）：
- 锚点 = 一组冻结、不可变、绝不参与训练的评测切片，跨版本可比。
- 锚点分数 floor 是单调不减的棘轮：只有通过全部门并真正上线的版本才允许抬升，
  因此「锚点分数持续增长」，防止能力退化。
- 逐维不退化：候选在每一维上的分数 >= floor - eps，吸收随机噪声但拦截真实退化。
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float((probs.argmax(axis=1) == y_true).mean())


class AnchorSet:
    def __init__(self, eps: float = 0.02):
        self.eps = float(eps)
        self._slices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._weights: dict[str, float] = {}
        self._floor: dict[str, float] = {}

    def add_slice(self, name: str, X: np.ndarray, y: np.ndarray, weight: float = 1.0) -> None:
        self._slices[name] = (np.asarray(X), np.asarray(y))
        self._weights[name] = float(weight)

    @property
    def names(self) -> list[str]:
        return list(self._slices)

    def is_empty(self) -> bool:
        return len(self._slices) == 0

    def score(self, model, metric: Callable = accuracy) -> dict[str, float]:
        return {
            name: metric(y, model.predict_proba(X))
            for name, (X, y) in self._slices.items()
        }

    def composite(self, scores: dict[str, float]) -> float:
        if not scores:
            return 0.0
        w = sum(self._weights.get(n, 1.0) for n in scores)
        if w <= 0:
            return 0.0
        return float(sum(self._weights.get(n, 1.0) * scores[n] for n in scores) / w)

    def check(self, scores: dict[str, float]) -> tuple[bool, list[str]]:
        """逐维不退化：score >= floor - eps。返回 (是否通过, 违反维度)。"""
        violations = []
        for name, s in scores.items():
            floor = self._floor.get(name, -np.inf)
            if s < floor - self.eps:
                violations.append(name)
        return len(violations) == 0, violations

    def ratchet(self, scores: dict[str, float]) -> None:
        """准入后抬升 floor（只升不降）。"""
        for name, s in scores.items():
            self._floor[name] = max(self._floor.get(name, -np.inf), s)

    def floor(self) -> dict[str, float]:
        return dict(self._floor)
