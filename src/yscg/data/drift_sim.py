"""表格漂移仿真器：官方数据包发布前用于开发与验证闭环。

三段漂移设计（scenario="transient"，考验回滚保护）：
  批次 0-29   无漂移（基线分布）
  批次 30-49  协变量漂移：特征 0-3 均值线性上移至 1.2σ
  批次 50-69  协变量+概念漂移：特征 0-3 上移 2.5σ、4-7 上移 1.5σ，
              且 x0+x1 超过阈值区域的样本标签翻转
  批次 70+    恢复：分布与标签回归基线（考验系统能否再适应回来）

scenario="persistent"：漂移持续到流结束（无恢复段），考验系统能否适应新工况。
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.datasets import make_classification

from .stream import DataStream


@dataclass
class DriftPhase:
    start: int
    end: int  # exclusive
    shift: dict = field(default_factory=dict)  # feature -> sigma 倍数
    label_flip: bool = False


class TabularDriftSimulator(DataStream):
    def __init__(
        self,
        n_features: int = 20,
        n_informative: int = 8,
        n_batches: int = 70,
        batch_size: int = 100,
        init_samples: int = 3000,
        val_samples: int = 1000,
        annotation_budget: int = 300,
        seed: int = 42,
        scenario: str = "transient",
    ):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.annotation_budget = annotation_budget
        self._annotated = 0
        self._batch_idx = -1

        rng = np.random.default_rng(seed)
        pool_size = n_batches * batch_size
        X, y = make_classification(
            n_samples=pool_size + init_samples + val_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=0,
            n_classes=2,
            flip_y=0.02,
            random_state=seed,
        )
        self._pool = X[:pool_size].astype(np.float32)
        self._pool_y = y[:pool_size]
        self._init = X[pool_size : pool_size + init_samples].astype(np.float32)
        self._init_y = y[pool_size : pool_size + init_samples]
        self._val = X[pool_size + init_samples :].astype(np.float32)
        self._val_y = y[pool_size + init_samples :]

        self._std = self._pool.std(axis=0)
        self._flip_threshold = float(
            np.quantile(self._pool[:, 0] + self._pool[:, 1], 0.7)
        )
        if scenario == "persistent":
            self.phases = [
                DriftPhase(0, 30),
                DriftPhase(30, 50, shift={0: 1.2, 1: 1.2, 2: 1.2, 3: 1.2}),
                DriftPhase(
                    50, n_batches,
                    shift={0: 2.5, 1: 2.5, 2: 2.5, 3: 2.5, 4: 1.5, 5: 1.5, 6: 1.5, 7: 1.5},
                    label_flip=True,
                ),
            ]
        else:
            self.phases = [
                DriftPhase(0, 30),
                DriftPhase(30, 50, shift={0: 1.2, 1: 1.2, 2: 1.2, 3: 1.2}),
                DriftPhase(
                    50, 70,
                    shift={0: 2.5, 1: 2.5, 2: 2.5, 3: 2.5, 4: 1.5, 5: 1.5, 6: 1.5, 7: 1.5},
                    label_flip=True,
                ),
                DriftPhase(70, n_batches),
            ]
        self._rng = rng

    # ---- 数据访问 ----
    def get_init_data(self):
        return self._init, self._init_y

    def get_val_data(self):
        return self._val, self._val_y

    def current_batch(self) -> np.ndarray:
        return self._current_X

    # ---- DataStream 接口 ----
    def next_batch(self) -> np.ndarray:
        self._batch_idx += 1
        i = self._batch_idx
        if i >= self.n_batches:
            raise StopIteration
        X = self._pool[i * self.batch_size : (i + 1) * self.batch_size].copy()
        phase = self._phase_at(i)
        if phase is not None:
            ramp = 1.0
            if i < phase.end - 5:  # 渐变段：前 5 批线性增强，之后全量
                ramp = max(0.2, (i - phase.start + 1) / 5.0)
            for feat, sigma in phase.shift.items():
                X[:, feat] += sigma * self._std[feat] * ramp
            X += self._rng.normal(0, 0.05 * self._std, size=X.shape).astype(np.float32)
        self._current_X = X.astype(np.float32)
        self._current_y = self._batch_y(i, X)
        return self._current_X

    def annotate(self, indices: np.ndarray) -> np.ndarray:
        n = len(indices)
        if self._annotated + n > self.annotation_budget:
            raise ValueError("标注配额不足")
        self._annotated += n
        return self._current_y[indices]

    @property
    def annotation_used(self) -> int:
        return self._annotated

    @property
    def exhausted(self) -> bool:
        return self._batch_idx >= self.n_batches - 1

    def truth(self) -> np.ndarray | None:
        return self._current_y

    # ---- 内部 ----
    def _phase_at(self, batch_idx: int):
        for p in self.phases:
            if p.start <= batch_idx < p.end:
                return p
        return None

    def _batch_y(self, batch_idx: int, X: np.ndarray) -> np.ndarray:
        start = batch_idx * self.batch_size
        y = self._pool_y[start : start + self.batch_size].copy()
        phase = self._phase_at(batch_idx)
        if phase is not None and phase.label_flip:
            mask = X[:, 0] + X[:, 1] > self._flip_threshold
            y[mask] = 1 - y[mask]
        return y
