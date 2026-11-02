"""留出集 + 对照组 + 重复运行统计（方差与置信区间）。"""
from .control import (
    HeldOutSet,
    bootstrap_gain,
    running_stats,
    repeated_gain_report,
)

__all__ = ["HeldOutSet", "bootstrap_gain", "running_stats", "repeated_gain_report"]
