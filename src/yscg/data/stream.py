"""数据流接口：无标签批次流 + 标注配额。"""
from abc import ABC, abstractmethod

import numpy as np


class DataStream(ABC):
    """隐藏测试流的统一接口。

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
