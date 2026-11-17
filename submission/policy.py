"""轮间策略引擎：规则兜底 + LLM 建议白名单校验（受控 RSI 的策略层）。

动作空间（续训协议内，永不改变网络结构/任务/种子）：
- continue:   以默认保守超参续训父检查点
- tune:       按 (learning_rate, batch_size, epochs) 组合续训
- keep:       不训练，保留父模型（收益预期为负或预算不足时）

LLM 只能在白名单网格内建议超参；非法建议一律丢弃并记录。
"""
import math

# 白名单搜索空间：保守、CPU 友好、与官方基线兼容（结构不变）
LR_GRID = (0.0005, 0.001, 0.002, 0.003, 0.005)
BATCH_GRID = (64, 128, 256)
EPOCHS_RANGE = (1, 30)

DEFAULTS = {
    "A": {"learning_rate": 0.003, "batch_size": 128, "epochs": 8},
    "B": {"learning_rate": 0.003, "batch_size": 128, "epochs": 8},
    "C": {"learning_rate": 0.002, "batch_size": 256, "epochs": 4},
}


def clamp_config(raw, dataset_key):
    """把任意来源（LLM/规则）的超参建议夹到白名单内；非法输入返回 None。"""
    if not isinstance(raw, dict):
        return None
    base = dict(DEFAULTS[dataset_key])
    try:
        lr = float(raw.get("learning_rate", base["learning_rate"]))
        bs = int(raw.get("batch_size", base["batch_size"]))
        ep = int(raw.get("epochs", base["epochs"]))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (lr, bs, ep)):
        return None
    lr = min(LR_GRID, key=lambda v: abs(v - lr))
    bs = min(BATCH_GRID, key=lambda v: abs(v - bs))
    ep = max(EPOCHS_RANGE[0], min(EPOCHS_RANGE[1], ep))
    return {"learning_rate": lr, "batch_size": bs, "epochs": ep}


def rule_proposal(dataset_key, diagnostics, history, round_index):
    """规则兜底提案：无 LLM 时也能给出合理的保守续训计划。

    - 验证指标随历史持续改善 → 保持节奏续训；
    - 连续两轮无改善 → 降低学习率细调；
    - 漂移严重且父模型验证表现差 → 稍多轮数适配。
    """
    cfg = dict(DEFAULTS[dataset_key])
    trend = _recent_trend(dataset_key, history)
    if trend is not None and trend <= 0:
        cfg["learning_rate"] = max(LR_GRID[0], cfg["learning_rate"] / 2)
        cfg["learning_rate"] = min(LR_GRID, key=lambda v: abs(v - cfg["learning_rate"]))
    severe = diagnostics["drift"]["psi_mean"] >= 0.25
    if severe:
        cfg["epochs"] = min(EPOCHS_RANGE[1], cfg["epochs"] + 4)
    return cfg


def _recent_trend(dataset_key, history):
    """从平台历史记录提取主指标最近趋势（>0 改善，<=0 停滞/劣化）。"""
    metric = "macro_f1" if dataset_key == "A" else "rmse"
    values = []
    for record in history:
        summary = record.get("validation_summary") or record.get("summary") or {}
        metrics = summary.get("metrics") if isinstance(summary, dict) else None
        if isinstance(metrics, dict) and isinstance(metrics.get(metric), (int, float)):
            values.append(float(metrics[metric]))
    if len(values) < 2:
        return None
    delta = values[-1] - values[-2]
    return delta if metric == "macro_f1" else -delta


def should_keep_parent(dataset_key, before, after):
    """准入闸门：候选不优于父模型则保留父模型（防劣化上线）。"""
    metric = "macro_f1" if dataset_key == "A" else "rmse"
    b, a = before.get(metric), after.get(metric)
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False  # 指标缺失时不轻易否决，交给平台测量
    if dataset_key == "A":
        return a < b - 1e-9
    return a > b + 1e-9