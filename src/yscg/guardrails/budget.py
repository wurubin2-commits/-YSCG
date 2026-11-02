"""受控机制：预算管理（防无限迭代/超支）与策略白名单校验（LLM 建议须过此关）。"""

INT_FIELDS = {
    "light_boost_rounds", "full_boost_rounds", "mlp_epochs_light", "mlp_epochs_full",
    "hpo_trials", "annotate_per_drift", "retrain_annotate",
    "stale_after_batches", "cooldown_batches", "eval_window_pool",
}

STRATEGY_WHITELIST: dict[str, tuple[float, float]] = {
    "psi_mild": (0.0, 1.0),
    "psi_severe": (0.0, 1.0),
    "conf_drop_frac": (0.0, 1.0),
    "health_tolerance": (0.0, 0.3),
    "stale_after_batches": (5, 50),
    "cooldown_batches": (0, 10),
    "promote_min_gain": (0.0, 0.5),
    "promote_max_val_drop": (0.0, 0.5),
    "promote_relaxed_val_drop": (0.0, 0.5),
    "promote_full_abs_drop": (0.0, 0.5),
    "eval_window_pool": (1, 10),
    "light_boost_rounds": (10, 200),
    "full_boost_rounds": (50, 1000),
    "mlp_epochs_light": (1, 30),
    "mlp_epochs_full": (5, 100),
    "hpo_trials": (1, 30),
    "annotate_per_drift": (10, 500),
    "retrain_annotate": (10, 1000),
}


def validate_strategy_patch(patch: dict) -> dict:
    """过滤白名单外字段、裁剪越界值、整型字段取整。返回可安全应用的补丁。"""
    safe = {}
    for key, value in patch.items():
        if key not in STRATEGY_WHITELIST or not isinstance(value, (int, float)):
            continue
        low, high = STRATEGY_WHITELIST[key]
        value = float(min(max(value, low), high))
        if key in INT_FIELDS:
            value = int(round(value))
        safe[key] = value
    return safe


class BudgetExceeded(Exception):
    pass


class BudgetManager:
    def __init__(self, annotation_budget: int, max_retrains: int, max_hpo_trials: int):
        self.annotation_budget = annotation_budget
        self.max_retrains = max_retrains
        self.max_hpo_trials = max_hpo_trials
        self.annotations_used = 0
        self.retrains_used = 0
        self.hpo_trials_used = 0
        self.promotions = 0
        self.rollbacks = 0

    def authorize_annotation(self, n: int):
        if self.annotations_used + n > self.annotation_budget:
            raise BudgetExceeded(
                f"标注配额超限: {self.annotations_used + n} > {self.annotation_budget}"
            )
        self.annotations_used += n

    def authorize_retrain(self):
        if self.retrains_used + 1 > self.max_retrains:
            raise BudgetExceeded(f"重训次数超限: > {self.max_retrains}")
        self.retrains_used += 1

    def authorize_hpo_trials(self, n: int) -> int:
        allowed = min(n, self.max_hpo_trials - self.hpo_trials_used)
        self.hpo_trials_used += allowed
        return allowed

    def snapshot(self) -> dict:
        return {
            "annotations_used": self.annotations_used,
            "retrains_used": self.retrains_used,
            "hpo_trials_used": self.hpo_trials_used,
            "promotions": self.promotions,
            "rollbacks": self.rollbacks,
        }
