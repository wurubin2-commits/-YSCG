"""回滚护栏：旧版本完整保留 + 回滚路径预演验证（Rollback Guard）。

设计要点（对齐「回滚不是免费的」）：
- 回滚不是免费的：只有「旧版本被完整保留」且「回滚路径经过验证」的候选，
  才允许进入无人工干预的上线环节。
- 完整保留：冠军模型文件须存在且非空（版本仓库持久化权重）。
- 回滚路径验证：重载持久化的冠军模型，并与内存中的冠军在探针上比对预测，
  一致才视为「可安全回滚」。二者任一失败则拒绝上线，避免「上去了却退不回来」。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RollbackCheck:
    ok: bool
    reason: str = ""


class RollbackGuard:
    def __init__(self, registry, probe_X: np.ndarray | None = None, atol: float = 1e-5):
        self.registry = registry
        self.probe_X = None if probe_X is None else np.asarray(probe_X)
        self.atol = float(atol)

    def preflight(self, champion_obj) -> RollbackCheck:
        """上线前校验：当前冠军可被完整重载，且重载后预测一致。"""
        rec = self.registry.champion_record()
        if rec is None:
            return RollbackCheck(False, "无冠军记录，回滚不可用")

        f = self.registry.models_dir / rec["file"]
        if not f.exists() or f.stat().st_size == 0:
            return RollbackCheck(False, f"冠军模型文件缺失或为空: {rec['file']}")

        try:
            loaded = self.registry.load_champion()
        except Exception as e:  # noqa: BLE001 - 回滚预演必须吞异常并如实上报
            return RollbackCheck(False, f"冠军模型重载失败: {e}")
        if loaded is None:
            return RollbackCheck(False, "冠军模型重载为空")

        if self.probe_X is not None and len(self.probe_X) > 0:
            try:
                a = champion_obj.predict_proba(self.probe_X)
                b = loaded.predict_proba(self.probe_X)
            except Exception as e:  # noqa: BLE001
                return RollbackCheck(False, f"回滚预演预测失败: {e}")
            if not np.allclose(a, b, atol=self.atol):
                return RollbackCheck(False, "回滚预演不一致：重载模型预测偏离")

        return RollbackCheck(True, "旧版本完整保留，回滚路径验证通过")
