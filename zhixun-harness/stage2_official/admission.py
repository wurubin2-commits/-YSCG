"""可选准入护栏：增益门、跨轮锚点棘轮与回滚预演。

吸收队友 ZMC 分支 guardrails 的判定顺序（增益 → 锚点 → 可回滚），
适配官方 SDK 的轮次协议。默认开启：门槛要求封测性能不低于静态基线，
该护栏是这条线的专用防线。设置 ZHIXUN_ADMISSION=0 可回到 V3 基线行为。
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    enabled: bool
    min_gain: float
    eps: float
    rollback: bool


def config():
    return Config(
        enabled=os.environ.get("ZHIXUN_ADMISSION", "1") != "0",
        min_gain=float(os.environ.get("ZHIXUN_ADMISSION_MIN_GAIN") or 0.0),
        eps=float(os.environ.get("ZHIXUN_ADMISSION_EPS") or 0.0),
        rollback=os.environ.get("ZHIXUN_ADMISSION_ROLLBACK", "1") == "1",
    )


def signed_gain(direction, parent, candidate):
    if parent is None or candidate is None:
        return None
    return candidate - parent if direction == "higher" else parent - candidate


def score_floor(prior, direction):
    """已采用轮次的最佳公开主指标，作为只升不降的能力下限。"""
    values = [
        record.get("candidate_public_score")
        for record in prior
        if record.get("decision") == "adopted" and record.get("candidate_public_score") is not None
    ]
    if not values:
        return None
    return max(values) if direction == "higher" else min(values)


def slice_floors(prior):
    """按公开切片累积历史最好分；recall 取最高，mae 取最低。"""
    floors = {}
    for record in prior:
        if record.get("decision") != "adopted":
            continue
        for row in (record.get("local_train_receipt") or {}).get("stability", []):
            name, metric, after = row.get("slice"), row.get("metric"), row.get("after")
            if name is None or after is None:
                continue
            if metric == "recall":
                floors[name] = max(floors.get(name, -math.inf), after)
            else:
                floors[name] = min(floors.get(name, math.inf), after)
    return floors


def degraded_slices(stability, floors, eps):
    bad = []
    for row in stability:
        name, metric, after = row.get("slice"), row.get("metric"), row.get("after")
        if name not in floors or after is None:
            continue
        if metric == "recall":
            if after < floors[name] - eps:
                bad.append(name)
        elif after > floors[name] + eps:
            bad.append(name)
    return bad


def rollback_preflight(path, model, load):
    """旧版本必须能被完整重载，否则回滚不是免费的。"""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        return False, "parent checkpoint is missing, empty or a symlink"
    try:
        reloaded = load(path)
    except Exception as error:  # noqa: BLE001 - 预演必须吞异常并如实上报
        return False, f"parent checkpoint failed to reload: {type(error).__name__}"
    for name in ("task", "seed", "steps", "parameter_count"):
        if reloaded.meta.get(name) != model.meta.get(name):
            return False, f"reloaded parent differs on {name}"
    return True, "parent checkpoint reloads to the same state"


def evaluate(prior, stability, direction, parent_score, candidate_score, *, model=None, path=None, load=None):
    """返回 (是否准入, 失败的门, 原因)；关闭时始终准入。"""
    settings = config()
    if not settings.enabled:
        return True, "disabled", "admission gate is off"

    gain = signed_gain(direction, parent_score, candidate_score)
    if gain is None:
        return False, "gain", "no comparable public scores"
    if gain <= settings.min_gain:
        return False, "gain", f"gain {gain:.6g} does not exceed {settings.min_gain:g}"

    floor = score_floor(prior, direction)
    if floor is not None:
        if direction == "higher" and candidate_score < floor - settings.eps:
            return False, "anchor", f"public score {candidate_score:.6g} fell below floor {floor:.6g}"
        if direction == "lower" and candidate_score > floor + settings.eps:
            return False, "anchor", f"public score {candidate_score:.6g} rose above floor {floor:.6g}"

    bad = degraded_slices(stability or [], slice_floors(prior), settings.eps)
    if bad:
        return False, "anchor", "degraded public slices: " + ", ".join(bad)

    if settings.rollback and model is not None and path is not None and load is not None:
        ok, reason = rollback_preflight(path, model, load)
        if not ok:
            return False, "rollback", reason

    return True, "pass", "passed gain, anchor and rollback gates"
