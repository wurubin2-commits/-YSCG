"""任务类型与指标层：统一「越高越好」的评分口径，覆盖分类与回归。

对齐赛题三类任务：
- Gas sensor drift：多分类（6 类气体）
- CMAPSS / XJTU-SY 轴承：RUL 回归
- 也兼容二元分类

约定：score(y_true, pred) 越大越好，这样上游 gain = score(candidate) - score(champion)
可统一比较；回归内部用负 MAE 作为 score，并单独报告 MAE/RMSE/R2 原始值。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_cls"
    MULTICLASS_CLASSIFICATION = "multiclass_cls"
    REGRESSION = "regression"

    @property
    def is_classification(self) -> bool:
        return self in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)


# ---------------- 指标实现（纯 numpy）----------------

def _to_labels(pred: np.ndarray) -> np.ndarray:
    if pred.ndim == 2:  # 概率矩阵 -> 标签
        return pred.argmax(axis=1)
    return pred


def accuracy(y_true, pred) -> float:
    y = np.asarray(y_true)
    p = _to_labels(np.asarray(pred))
    return float((p == y).mean()) if len(y) else 0.0


def macro_f1(y_true, pred) -> float:
    y = np.asarray(y_true).astype(int)
    p = _to_labels(np.asarray(pred)).astype(int)
    classes = np.unique(y)
    f1s = []
    for c in classes:
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def mae(y_true, pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float).reshape(-1)
    return float(np.mean(np.abs(p - y))) if len(y) else 0.0


def rmse(y_true, pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((p - y) ** 2))) if len(y) else 0.0


def r2(y_true, pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float).reshape(-1)
    if len(y) < 2:
        return 0.0
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


@dataclass
class TaskSpec:
    """一个任务的类型与评分口径。score 统一「越高越好」。"""

    name: str
    task_type: TaskType
    n_classes: int | None = None
    metric: str = "auto"  # auto | accuracy | macro_f1 | mae | rmse | r2
    extra: dict = field(default_factory=dict)

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    def score(self, y_true, pred) -> float:
        """统一评分（越高越好）。"""
        if not self.is_regression:
            return accuracy(y_true, pred) if self.metric in ("auto", "accuracy") else macro_f1(y_true, pred)
        # 回归：score = -MAE（越高越好）
        return -mae(y_true, pred)

    def report(self, y_true, pred) -> dict:
        """原始指标报告（便于 SwanLab 记录）。"""
        if not self.is_regression:
            return {"accuracy": accuracy(y_true, pred), "macro_f1": macro_f1(y_true, pred)}
        return {"mae": mae(y_true, pred), "rmse": rmse(y_true, pred), "r2": r2(y_true, pred)}


def detect_task_type(y) -> TaskType:
    """根据标签推断任务类型（整数少量取值→分类，否则回归）。"""
    a = np.asarray(y)
    if a.ndim != 1:
        return TaskType.REGRESSION
    if np.issubdtype(a.dtype, np.integer):
        uniq = np.unique(a)
        if len(uniq) <= 2:
            return TaskType.BINARY_CLASSIFICATION
        if len(uniq) <= 50:
            return TaskType.MULTICLASS_CLASSIFICATION
    return TaskType.REGRESSION


def make_task_spec(name: str, y: np.ndarray, n_classes: int | None = None) -> TaskSpec:
    """从标签自动构造 TaskSpec。"""
    tt = detect_task_type(y)
    if n_classes is None and tt.is_classification:
        n_classes = int(np.unique(y).size)
    return TaskSpec(name=name, task_type=tt, n_classes=n_classes)
