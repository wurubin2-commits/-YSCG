"""系统配置与策略参数。"""
from dataclasses import dataclass, field, asdict


@dataclass
class StrategyConfig:
    """策略层参数（策略库）。LLM 顾问仅可在白名单范围内修改这些值。"""

    window_size: int = 10            # 监测参考窗口批次数
    psi_mild: float = 0.10           # PSI 均值超过该值 → 轻度漂移
    psi_severe: float = 0.25         # PSI 均值超过该值 → 重度漂移
    conf_drop_frac: float = 0.15     # 预测置信度相对基线下降比例 → 模型层漂移
    health_tolerance: float = 0.03   # 窗口准确率不低于基线-容差 → 模型健康，忽略漂移
    stale_after_batches: int = 15    # 窗口准确率超过该批次数未更新 → 视为过期
    cooldown_batches: int = 3        # 漂移事件后的冷却批次数（防抖动）
    promote_min_gain: float = 0.02   # 挑战者窗口准确率需高于冠军的最小增益
    promote_max_val_drop: float = 0.02  # 轻量适配时挑战者验证集准确率允许的最大回退
    promote_relaxed_val_drop: float = 0.10  # 全量重训（严重漂移）时相对验证集的最大回退
    promote_full_abs_drop: float = 0.25  # 全量重训（严重漂移）时相对初始基线的绝对回退下限
    light_boost_rounds: int = 60     # 轻量适配训练规模
    full_boost_rounds: int = 300     # 全量重训训练规模
    mlp_epochs_light: int = 8
    mlp_epochs_full: int = 20
    hpo_trials: int = 6              # 全量重训时的贝叶斯优化试验数
    annotate_per_drift: int = 100    # 每次漂移事件用于评估的标注配额
    retrain_annotate: int = 100      # 每次重训新增的训练标注配额
    eval_window_pool: int = 3        # 窗口准确率在最近 K 个评估窗口上池化（降方差、防过拟合）
    rebase_after_event: bool = True  # 漂移事件后重设参考窗口
    # ---- 锚点检查（Anchor Gate）----
    anchor_eps: float = 0.02         # 锚点单维退化容差（吸收随机噪声，拦截真实退化）
    # ---- 记忆检查（Memory Gate）----
    memory_capacity: int = 1000      # 保护性记忆容量上限
    memory_min_conf: float = 0.8     # 入内存的最低验证置信度
    memory_min_quota: int = 10       # 分层回归采样每层最小配额（保证覆盖）
    reg_sample_size: int = 300       # 回归样本集大小
    reg_tolerance_frac: float = 0.05 # 回归计数容差（占回归样本集比例）


@dataclass
class SystemConfig:
    model_type: str = "lgbm"         # lgbm | mlp
    n_batches: int = 70
    batch_size: int = 100
    seed: int = 42
    annotation_budget: int = 300
    max_retrains: int = 5
    project: str = "ZYYQ"            # SwanLab 项目（唯一记录地址）
    workspace: str = "07240115"      # SwanLab 用户/组命名空间
    # 对齐赛题 B 包规则：评测流标签总可得，无需消耗标注配额（改为 CPU/迭代成本）
    use_free_labels: bool = True
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    def flat(self) -> dict:
        d = asdict(self)
        s = d.pop("strategy")
        return {**d, **{f"strategy.{k}": v for k, v in s.items()}}
