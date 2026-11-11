"""数据流接口：无标签批次流 + 标注配额；以及赛题实况的「有标签流」。"""
from abc import ABC, abstractmethod

import numpy as np


class DataStream(ABC):
    """隐藏测试流的统一接口（旧版：无标签 + 标注配额）。

    与真实评测一致：流只提供无标签样本；标签需通过 annotate() 消耗
    标注配额获取。系统须自行决定何时、对哪些样本花费配额。
    """

    @abstractmethod
    def next_batch(self) -> np.ndarray:
        """返回下一个批次的无标签特征矩阵 (batch, n_features)。"""

    @abstractmethod
    def annotate(self, indices: np.ndarray) -> np.ndarray:
        """标注当前批次中指定下标的样本，返回标签。"""

    @property
    @abstractmethod
    def annotation_used(self) -> int:
        """已消耗的标注配额。"""

    @property
    @abstractmethod
    def exhausted(self) -> bool:
        """流是否结束。"""

    def truth(self) -> np.ndarray | None:
        """当前批次真值标签（仅本地仿真评估用，不消耗标注配额）。

        真实评测中平台掌握隐藏标签、自行计分，系统无此接口。
        """
        return None


class LabeledDataStream(DataStream):
    """有标签流：标签总可得（对齐赛题 B 包规则），不消耗任何配额。

    - next_batch() 返回无标签特征批次；
    - truth() 返回当前批次标签（系统可直接用于评估与准入）；
    - annotate() 兼容旧接口，直接返回 truth 子集，但不扣配额；
    - get_init_data / get_val_data / get_heldout_data 提供训练/验证/冻结留出。
    """

    def __init__(
        self,
        init_X: np.ndarray,
        init_y: np.ndarray,
        val_X: np.ndarray,
        val_y: np.ndarray,
        stream_X: np.ndarray,
        stream_y: np.ndarray,
        heldout_X: np.ndarray | None = None,
        heldout_y: np.ndarray | None = None,
        batch_size: int = 100,
    ):
        self._init_X = np.asarray(init_X)
        self._init_y = np.asarray(init_y)
        self._val_X = np.asarray(val_X)
        self._val_y = np.asarray(val_y)
        self._stream_X = np.asarray(stream_X)
        self._stream_y = np.asarray(stream_y)
        self._held_X = np.asarray(heldout_X if heldout_X is not None else val_X)
        self._held_y = np.asarray(heldout_y if heldout_y is not None else val_y)
        self.batch_size = int(batch_size)
        self._idx = 0
        self._current_X = None
        self._current_y = None

    # ---- DataStream 接口 ----
    def next_batch(self) -> np.ndarray:
        if self._idx >= self.n_batches:
            raise StopIteration
        a = self._idx * self.batch_size
        b = min(a + self.batch_size, len(self._stream_X))
        self._current_X = self._stream_X[a:b]
        self._current_y = self._stream_y[a:b]
        self._idx += 1
        return self._current_X

    def annotate(self, indices: np.ndarray) -> np.ndarray:
        return self._current_y[np.asarray(indices)]

    @property
    def annotation_used(self) -> int:
        return 0  # 标签总可得，不耗配额

    @property
    def exhausted(self) -> bool:
        return self._idx >= self.n_batches

    @property
    def n_batches(self) -> int:
        if len(self._stream_X) == 0:
            return 0
        return int(np.ceil(len(self._stream_X) / self.batch_size))

    def truth(self) -> np.ndarray | None:
        return self._current_y

    # ---- 数据切分 ----
    def current_batch(self) -> np.ndarray:
        return self._current_X

    def get_init_data(self):
        return self._init_X, self._init_y

    def get_val_data(self):
        return self._val_X, self._val_y

    def get_heldout_data(self):
        return self._held_X, self._held_y
