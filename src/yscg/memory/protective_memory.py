"""保护性记忆：容量受限、分层、只存「已验证正确」的样本，并支持分层回归采样。

设计要点（对齐稳定性规则中的「记忆检查」）：
- 容量上限：ProtectiveMemory 有总量上限，超出时去冗余淘汰（保留多样性）。
- 入内存条件：只有「分类正确 且 置信度 >= memory_min_conf」的已标注样本才进入，
  即「历史上已被验证为正确」的行为/案例。
- 分层键 stratum = "<场景>.<难度>.<指标面>"，覆盖不同指标、场景、难度层级，
  避免回归样本只看单一维度。
- 回归样本 = 保护记忆的分层采样（每层保底配额 + 剩余按占比），供记忆检查复用。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MemoryRecord:
    x: np.ndarray        # 特征向量（已验证正确）
    y: int               # golden 标签
    gold_conf: float     # 验证时的正确类置信度
    stratum: str         # "<场景>.<难度>.<指标面>"
    batch: int           # 加入时的批次序号
    version: str         # 验证它的模型版本


def difficulty_label(conf: float) -> str:
    if conf < 0.7:
        return "hard"
    if conf < 0.9:
        return "medium"
    return "easy"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class ProtectiveMemory:
    def __init__(self, capacity: int = 1000, min_conf: float = 0.8):
        self.capacity = int(capacity)
        self.min_conf = float(min_conf)
        self._records: list[MemoryRecord] = []

    def maybe_add(
        self,
        X: np.ndarray,
        y: np.ndarray,
        probs: np.ndarray,
        scenario: str,
        batch: int,
        version: str,
        facet: str = "correctness",
    ) -> int:
        """只把已验证正确且高置信的样本纳入保护记录，返回新增条数。"""
        conf = np.max(probs, axis=1)
        pred = probs.argmax(axis=1)
        added = 0
        for i in range(len(X)):
            if int(pred[i]) == int(y[i]) and float(conf[i]) >= self.min_conf:
                rec = MemoryRecord(
                    x=np.asarray(X[i], dtype=np.float32).copy(),
                    y=int(y[i]),
                    gold_conf=float(conf[i]),
                    stratum=f"{scenario}.{difficulty_label(float(conf[i]))}.{facet}",
                    batch=int(batch),
                    version=str(version),
                )
                self._insert(rec)
                added += 1
        return added

    def _insert(self, rec: MemoryRecord) -> None:
        self._records.append(rec)
        while len(self._records) > self.capacity and len(self._records) > 1:
            self._evict()

    def _evict(self) -> None:
        """去冗余优先：淘汰与新样本最相似的旧记录；无法判定时 FIFO。"""
        new = self._records[-1]
        best_i, best_sim = 0, -1.0
        for i in range(len(self._records) - 1):
            sim = _cosine(new.x, self._records[i].x)
            if sim > best_sim:
                best_i, best_sim = i, sim
        if best_sim < 0:
            best_i = 0
        self._records.pop(best_i)

    def records(self) -> list[MemoryRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def strata_counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for r in self._records:
            c[r.stratum] = c.get(r.stratum, 0) + 1
        return c


class RegressionSampler:
    """保护记忆的分层采样：每层保底配额，剩余按占比，保证多维度覆盖。"""

    def __init__(self, memory: ProtectiveMemory, min_quota: int = 10, seed: int = 42):
        self.memory = memory
        self.min_quota = int(min_quota)
        self.rng = np.random.default_rng(seed)
        self._cache: dict[object, list[MemoryRecord]] = {}

    def sample(self, n: int, round_key: object | None = None) -> list[MemoryRecord]:
        # 「一次编译、多次复用」：同一轮多个候选共享同一份回归样本。
        if round_key is not None and round_key in self._cache:
            return list(self._cache[round_key])

        recs = self.memory.records()
        if not recs or n <= 0:
            sampled: list[MemoryRecord] = []
        else:
            n = min(int(n), len(recs))
            sampled = self._stratified(recs, n)

        if round_key is not None:
            self._cache[round_key] = list(sampled)
        return sampled

    def clear_cache(self) -> None:
        self._cache.clear()

    def _stratified(self, recs: list[MemoryRecord], n: int) -> list[MemoryRecord]:
        strata: dict[str, list[MemoryRecord]] = {}
        for r in recs:
            strata.setdefault(r.stratum, []).append(r)
        names = sorted(strata)
        groups = [strata[s] for s in names]

        quotas = [min(self.min_quota, len(g)) for g in groups]
        leftover = n - sum(quotas)
        if leftover > 0:
            rem = [len(g) - quotas[i] for i, g in enumerate(groups)]
            total_rem = sum(rem)
            if total_rem > 0:
                alloc = [int(leftover * r / total_rem) for r in rem]
                for _ in range(leftover - sum(alloc)):
                    i = max(range(len(groups)), key=lambda k: rem[k] - alloc[k])
                    alloc[i] += 1
                for i in range(len(groups)):
                    quotas[i] += alloc[i]

        out: list[MemoryRecord] = []
        for i, g in enumerate(groups):
            k = min(quotas[i], len(g))
            idx = self.rng.choice(len(g), size=k, replace=False)
            out.extend(g[j] for j in idx)
        return out


def regression_count(model, records: list[MemoryRecord]) -> tuple[int, float]:
    """在回归样本上统计：硬回归计数 + 软回归（置信度衰减）总和。

    硬回归：候选预测标签 != golden 标签。
    软回归：候选仍分类正确，但正确类置信度相对验证时下降的幅度之和。
    """
    if not records:
        return 0, 0.0
    X = np.stack([r.x for r in records])
    P = model.predict_proba(X)
    ys = np.array([r.y for r in records])
    pred = P.argmax(axis=1)
    hard = int((pred != ys).sum())

    gold = np.array([r.gold_conf for r in records])
    correct = pred == ys
    drop = np.maximum(0.0, gold - P[np.arange(len(ys)), ys])
    soft = float(drop[correct].sum())
    return hard, soft
