"""漂移监测：特征层 PSI + 模型层预测置信度，事件后支持参考窗口 rebase。

PSI 分箱使用"参考+当前"合并分位数边界，且 bin 数随样本量自适应
（小批量时 PSI 的卡方噪声底 ≈ (bins-1)/n，bin 过多会淹没真实漂移）。
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

MIN_BINS = 5
MAX_BINS = 10
EPS = 1e-6


@dataclass
class DriftReport:
    psi_mean: float
    psi_per_feature: dict
    n_drifted: int
    conf_mean: float
    conf_drop: float
    severity: str  # none | mild | severe

    def to_log(self) -> dict:
        return {
            "drift/psi_mean": self.psi_mean,
            "drift/n_features_drifted": self.n_drifted,
            "drift/conf_mean": self.conf_mean,
            "drift/conf_drop": self.conf_drop,
            "drift/severity": {"none": 0, "mild": 1, "severe": 2}[self.severity],
        }


def _psi(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (p.sum() + EPS) + EPS
    q = q / (q.sum() + EPS) + EPS
    return float(((p - q) * np.log(p / q)).sum())


def _psi_combined(ref_col: np.ndarray, cur_col: np.ndarray) -> float:
    if ref_col.std() < EPS or cur_col.std() < EPS:
        return 0.0
    n_bins = int(np.clip(len(cur_col) // 50, MIN_BINS, MAX_BINS))
    edges = np.quantile(
        np.concatenate([ref_col, cur_col]), np.linspace(0, 1, n_bins + 1)
    )
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref_col, bins=edges)[0].astype(float)
    q = np.histogram(cur_col, bins=edges)[0].astype(float)
    return _psi(p, q)


class DriftMonitor:
    def __init__(self, strategy):
        self.strategy = strategy
        self._ref_X: np.ndarray | None = None
        self._baseline_conf: float | None = None
        self._history: deque = deque(maxlen=strategy.window_size)

    def set_reference(self, X: np.ndarray, probs: np.ndarray | None = None):
        """以给定数据（通常为流的最初若干批）建立参考分布。"""
        self._ref_X = X.copy()
        if probs is not None:
            self._baseline_conf = float(np.max(probs, axis=1).mean())
        self._history.clear()

    def observe(self, X: np.ndarray, probs: np.ndarray):
        self._history.append((X.copy(), probs.copy()))

    def check(self, X: np.ndarray, probs: np.ndarray) -> DriftReport:
        psi_per_feature = {}
        if self._ref_X is not None:
            for j in range(X.shape[1]):
                psi_per_feature[j] = _psi_combined(self._ref_X[:, j], X[:, j])
        psi_mean = float(np.mean(list(psi_per_feature.values()))) if psi_per_feature else 0.0
        n_drifted = sum(1 for v in psi_per_feature.values() if v > 0.2)

        conf_mean = float(np.max(probs, axis=1).mean())
        conf_drop = 0.0
        if self._baseline_conf is not None and self._baseline_conf > EPS:
            conf_drop = max(0.0, (self._baseline_conf - conf_mean) / self._baseline_conf)

        s = self.strategy
        if psi_mean >= s.psi_severe or conf_drop >= 2 * s.conf_drop_frac:
            severity = "severe"
        elif psi_mean >= s.psi_mild or conf_drop >= s.conf_drop_frac:
            severity = "mild"
        else:
            severity = "none"
        return DriftReport(psi_mean, psi_per_feature, n_drifted, conf_mean, conf_drop, severity)

    def rebase(self):
        """漂移事件处理后重设参考窗口，避免永久报警。"""
        if not self._history:
            return
        X = np.concatenate([h[0] for h in self._history], axis=0)
        probs = np.concatenate([h[1] for h in self._history], axis=0)
        self.set_reference(X, probs)
