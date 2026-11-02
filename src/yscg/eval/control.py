"""留出集、对照组与重复运行统计。

对齐需求：
- 冻结留出任务（HeldOutSet）：迭代过程不可访问，仅作最终评判标准。
- 对照试验：对照组与实验组从完全相同初始工况出发，消耗相同量的资源
  （本实现对齐「训练算力 = 模型拟合次数」与「评估调用 = 标注条数」，
  LLM 顾问为 mock 时 Token 两侧均为 0，天然对齐），但不进行自我修改
  （见 pipeline.loop 的 control_mode）。
- 重复运行统计：报告增益均值/方差与重复次数的关系，以及该次数下增益估计的
  置信区间（t 分布）。
"""
from __future__ import annotations

import numpy as np


def _t_critical(n: int, alpha: float) -> float:
    """t 分布临界值；无 scipy 时退化为正态近似（z 值）。"""
    try:
        from scipy import stats
        return float(stats.t.ppf(1 - alpha / 2, n - 1))
    except Exception:  # noqa: BLE001 - 无 scipy 时降级
        z = {0.10: 1.645, 0.05: 1.96, 0.01: 2.576}
        return z.get(alpha, 1.96)


def accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float((probs.argmax(axis=1) == y_true).mean())


class HeldOutSet:
    """冻结留出任务：构造后不可变，迭代过程无法访问，仅最终评判。"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def evaluate(self, model, metric=accuracy) -> float:
        return float(metric(self.y, model.predict_proba(self.X)))

    def __len__(self) -> int:
        return len(self.y)


def running_stats(gains: list[float]) -> list[dict]:
    """n=1..N 的累计均值/标准差，展示「观测方差随重复次数的关系」。"""
    out: list[dict] = []
    acc: list[float] = []
    for g in gains:
        acc.append(float(g))
        m = float(np.mean(acc))
        s = float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0
        out.append({"n": len(acc), "mean": m, "std": s})
    return out


def bootstrap_gain(gains: list[float], alpha: float = 0.05) -> dict:
    """在 N 次重复下，增益估计的均值、标准差与置信区间。"""
    g = np.asarray(gains, dtype=float)
    n = len(g)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    mean = float(g.mean())
    std = float(g.std(ddof=1)) if n > 1 else 0.0
    if n >= 2:
        t = _t_critical(n, alpha)
        se = std / np.sqrt(n)
        ci_low, ci_high = mean - t * se, mean + t * se
    else:
        ci_low = ci_high = mean
    return {"n": n, "mean": mean, "std": std, "ci_low": float(ci_low), "ci_high": float(ci_high)}


def repeated_gain_report(exp_scores: list[float], ctrl_scores: list[float] | None) -> dict:
    """汇总实验组/对照组/增益的统计报告。"""
    gains = (
        [e - c for e, c in zip(exp_scores, ctrl_scores)]
        if ctrl_scores is not None
        else list(exp_scores)
    )
    return {
        "per_seed_experiment": [float(x) for x in exp_scores],
        "per_seed_control": [float(x) for x in ctrl_scores] if ctrl_scores is not None else None,
        "gains": [float(x) for x in gains],
        "running": running_stats(gains),
        "summary": bootstrap_gain(gains),
    }
