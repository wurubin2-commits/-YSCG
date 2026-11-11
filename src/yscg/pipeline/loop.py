"""闭环编排：漂移监测 → 诊断归因 → 策略决策 → 实验执行 → 评估上线/回滚。

系统契约（对齐赛题基础门槛）：
- 无人干预自主运行：任何异常（预算超限、LLM 失败）都降级为继续监测而非崩溃；
- 每次训练/评估动作都产出指标与版本记录（SwanLab + 模型注册表）；
- 动作决策由规则引擎兜底，LLM 仅贡献策略参数建议（受白名单约束）。
"""
import numpy as np

from ..config import SystemConfig
from ..drift import Attributor, DriftMonitor
from ..guardrails import (
    AdmissionGate,
    AdmissionResult,
    BudgetExceeded,
    BudgetManager,
    Gate,
    RollbackGuard,
    validate_strategy_patch,
)
from ..models import ModelRegistry, create_model
from ..strategy import ACTION_CODE, Action, RulePolicy

COST_INIT = 1.0
COST_LIGHT = 0.1
COST_FULL = 0.5
COST_HPO_TRIAL = 0.05


def _accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == y_true).mean())


def _entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-9, 1.0)
    return -(p * np.log(p)).sum(axis=1)


class SelfEvolutionLoop:
    def __init__(
        self,
        cfg: SystemConfig,
        stream,
        registry: ModelRegistry,
        budget: BudgetManager,
        policy: RulePolicy,
        advisor,
        attributor: Attributor,
        monitor: DriftMonitor,
        log_fn=None,
        admission_gate: AdmissionGate | None = None,
        rollback_guard: RollbackGuard | None = None,
        validator_probe=None,
        memory=None,
        heldout=None,
        control_mode: bool = False,
    ):
        self.cfg = cfg
        self.stream = stream
        self.registry = registry
        self.budget = budget
        self.policy = policy
        self.advisor = advisor
        self.attributor = attributor
        self.monitor = monitor
        self._log = log_fn or (lambda step, data: None)
        # 准入机制相关（可选，None 时回退到旧版增益逻辑，保持向后兼容）
        self.admission_gate = admission_gate
        self.rollback_guard = rollback_guard
        self.validator_probe = validator_probe
        self.memory = memory
        self.heldout = heldout
        self.control_mode = control_mode  # 对照组：消耗相同算力但不自我修改

        self.champion = None
        self.champion_window_acc: float | None = None
        self.val_acc: float | None = None
        self.baseline_acc: float | None = None
        self.cost_minutes = 0.0
        self._ann_X: list[np.ndarray] = []
        self._ann_y: list[np.ndarray] = []
        self._eval_windows: list[tuple[np.ndarray, np.ndarray]] = []
        self._patches_applied = 0
        self._decisions: dict[str, int] = {a.value: 0 for a in Action}
        self._acc_age: int | None = None  # 窗口准确率的"新鲜度"（批次数）
        self._cooldown = 0
        self._init_X = self._init_y = None
        self._val_X = self._val_y = None
        self._admission_stats: dict[str, int] = {"pass": 0, "fail": 0, "validator_fail": 0, "rollback_fail": 0}

    # ---------------- 主流程 ----------------
    def run(self) -> dict:
        self._init_X, self._init_y = self.stream.get_init_data()
        self._val_X, self._val_y = self.stream.get_val_data()
        self._train_initial_champion()

        warmup = self.cfg.strategy.window_size
        for step in range(warmup):
            try:
                X = self.stream.next_batch()
            except StopIteration:
                break
            probs = self.champion.predict_proba(X)
            self.monitor.observe(X, probs)
            self._log(step, self._step_log(step, None, Action.NOOP, probs))
        self.monitor.rebase()

        for step in range(warmup, self.cfg.n_batches):
            try:
                X = self.stream.next_batch()
            except StopIteration:
                break
            probs = self.champion.predict_proba(X)
            report = self.monitor.check(X, probs)
            self.monitor.observe(X, probs)

            if self._acc_age is not None:
                self._acc_age += 1

            if self._cooldown > 0:
                self._cooldown -= 1
                decision = Action.NOOP
            else:
                patch = self._consult_advisor(report)
                decision = self.policy.decide(
                    report, self.champion_window_acc, self.baseline_acc, self._acc_age
                )
                self._decisions[decision.value] += 1
                if decision in (Action.LIGHT_ADAPT, Action.FULL_RETRAIN):
                    try:
                        self._handle_drift(step, X, probs, decision, report.severity)
                    except BudgetExceeded:
                        self._log(step, {"system/budget_exceeded": 1})
                    self.monitor.rebase()
                    self._cooldown = self.cfg.strategy.cooldown_batches
                elif report.severity != "none":
                    # 漂移但模型健康：仅重设参考基线，零成本
                    self.monitor.rebase()
                    self._cooldown = self.cfg.strategy.cooldown_batches

            self._log(step, self._step_log(step, report, decision, probs))

        summary = self._final_summary()
        self._log(self.cfg.n_batches, summary)
        return summary

    # ---------------- 阶段实现 ----------------
    def _train_initial_champion(self):
        s = self.cfg.strategy
        model = create_model(self.cfg.model_type, seed=self.cfg.seed)
        if self.cfg.model_type == "lgbm":
            model.fit(self._init_X, self._init_y, n_estimators=s.full_boost_rounds)
        else:
            model.fit(self._init_X, self._init_y, epochs=s.mlp_epochs_full)
        self.champion = model
        self.val_acc = _accuracy(self._val_y, model.predict_proba(self._val_X))
        self.baseline_acc = self.val_acc
        self.cost_minutes += COST_INIT
        version_id = self.registry.register(model, {"val_acc": self.val_acc}, "init")
        self.registry.promote(version_id)
        # 初始化锚点 floor = 初始冠军分数，使「防退化」从基线起即生效
        if self.admission_gate is not None and not self.admission_gate.anchor.is_empty():
            self.admission_gate.anchor.ratchet(self.admission_gate.anchor.score(model))

    def _consult_advisor(self, report) -> dict:
        if self.advisor is None:
            return {}
        context = {
            "drift_report": report.__dict__,
            "attribution": self.attributor.attribute(report),
            "budget": self.budget.snapshot(),
            "champion": {
                "val_acc": self.val_acc,
                "window_acc": self.champion_window_acc,
            },
            "strategy": self.cfg.strategy.__dict__,
            "registry_versions": len(self.registry.history()),
        }
        suggestion = self.advisor.advise(context) or {}
        patch = validate_strategy_patch(suggestion.get("strategy_patch", {}))
        for key, value in patch.items():
            setattr(self.cfg.strategy, key, value)
        if patch:
            self._patches_applied += 1
        return patch

    def _handle_drift(self, step: int, X: np.ndarray, probs: np.ndarray, action: Action, scenario: str = "none"):
        s = self.cfg.strategy
        # 成本意识：轻度漂移只花一半配额，把预算留给后续可能更严重的漂移
        scale = 1 if action == Action.FULL_RETRAIN else 0.5
        n_before = len(self._ann_X)  # 按实际入库量判断是否有新标签（兼容免费标签）

        # 1) 评估窗口：随机采样标注（无偏）；窗口准确率在最近 K 个窗口上池化，
        #    避免单窗口样本的方差/运气让健康门永久误判（历史教训：卡在 1.0）
        exclude_idx = len(self._ann_X)  # 新评估窗口不得进入挑战者训练集（防评估泄漏）
        win_X, win_y, win_idx = self._annotate_random(X, int(s.annotate_per_drift * scale))
        if win_y is not None and len(win_y):
            self._eval_windows.append((win_X, win_y))
            self._eval_windows = self._eval_windows[-s.eval_window_pool:]
            pool_X, pool_y = self._eval_pool()
            self.champion_window_acc = _accuracy(pool_y, self.champion.predict_proba(pool_X))
            self._acc_age = 0

        # 2) 不确定性采样补充训练标注（与评估窗口不相交，防训练/评估重叠泄漏）
        self._annotate_uncertain(X, probs, int(s.retrain_annotate * scale), exclude=win_idx)
        if len(self._ann_X) == n_before:
            # 无新增标注 → 重训学不到新东西，跳过以节省重训/HPO 预算（防空转迭代）
            self._log(step, {"experiment/skipped_no_new_labels": 1})
            return
        X_tr, y_tr = self._build_training_set(
            exclude_idx if win_y is not None and len(win_y) else None
        )

        # 3) 训练挑战者（轻量直接训，全量带 HPO）
        self.budget.authorize_retrain()
        pool_X, pool_y = self._eval_pool()
        challenger, tag, trial_cost = self._train_challenger(X_tr, y_tr, action, pool_X, pool_y)

        # 4) 评估与对比：champion/challenger（同一池化评估窗口）
        val_acc_ch = _accuracy(self._val_y, challenger.predict_proba(self._val_X))
        win_acc_ch = (
            _accuracy(pool_y, challenger.predict_proba(pool_X))
            if pool_y is not None and len(pool_y)
            else val_acc_ch
        )
        win_acc_champ = self.champion_window_acc if self.champion_window_acc is not None else 0.0
        self.cost_minutes += COST_LIGHT if action == Action.LIGHT_ADAPT else COST_FULL
        self.cost_minutes += trial_cost * COST_HPO_TRIAL

        # 4.5) 更新保护性记忆：把本轮新标注且验证正确的样本纳入（记忆检查的数据源）
        self._update_memory(scenario, step, exclude_idx)

        # 5) 准入决策（多层门：增益/锚点/记忆 + 验证器自检 + 回滚护栏）
        admission, promoted = self._admit(
            challenger, tag, action, pool_X, pool_y,
            val_acc_ch, win_acc_ch, win_acc_champ, step,
        )
        self.registry.register(
            challenger, {"val_acc": val_acc_ch, "window_acc": win_acc_ch},
            tag, admission=("pass" if admission.admitted else "fail"),
        )

        if promoted:
            version_id = self.registry.history()[-1]["id"]
            self.registry.promote(version_id)
            self.champion = challenger
            self.val_acc = val_acc_ch
            self.champion_window_acc = win_acc_ch
            self.budget.promotions += 1
            # 锚点棘轮抬升（准入后才允许，保证锚点分数持续增长）
            if self.admission_gate is not None and not self.admission_gate.anchor.is_empty():
                self.admission_gate.anchor.ratchet(self.admission_gate.anchor.score(challenger))
        else:
            self.budget.rollbacks += 1

        self._log(step, {
            "experiment/challenger_val_acc": val_acc_ch,
            "experiment/challenger_window_acc": win_acc_ch,
            "experiment/champion_window_acc": win_acc_champ,
            "experiment/eval_pool_size": len(pool_y) if pool_y is not None else 0,
            "experiment/promoted": 1 if promoted else 0,
            **admission.to_log(),
        })

    def _admit(self, challenger, tag, action, pool_X, pool_y,
               val_acc_ch, win_acc_ch, win_acc_champ, step):
        """多层准入决策：增益/锚点/记忆 + 验证器自检 + 回滚护栏。返回 (result, promoted)。"""
        s = self.cfg.strategy

        # 验证器自检：失效则拒绝本轮所有候选，不假装决策机制仍有效
        validator_healthy = True
        if self.validator_probe is not None and self.admission_gate is not None:
            validator_healthy = self.validator_probe.healthy(self.admission_gate, self.champion)
            if not validator_healthy:
                self._admission_stats["validator_fail"] += 1
                self._log(step, {"validator/healthy": 0})
        else:
            self._log(step, {"validator/healthy": 1})

        if self.admission_gate is not None:
            admission = self.admission_gate.evaluate(
                challenger, self.champion, (pool_X, pool_y),
                validator_healthy=validator_healthy, round_key=step,
            )
        else:
            # 向后兼容：旧版增益 + 验证回退逻辑（无准入门时）
            if not validator_healthy:
                admission = AdmissionResult(False, Gate.VALIDATOR, reason="验证器自检失败")
            else:
                gain = win_acc_ch - win_acc_champ
                if action == Action.LIGHT_ADAPT:
                    val_ok = val_acc_ch >= self.val_acc - s.promote_max_val_drop
                else:
                    val_ok = val_acc_ch >= max(
                        self.val_acc - s.promote_relaxed_val_drop,
                        self.baseline_acc - s.promote_full_abs_drop,
                    )
                admitted = gain >= s.promote_min_gain and val_ok
                admission = AdmissionResult(
                    admitted, Gate.PASS if admitted else Gate.GAIN, gain=gain,
                    reason="旧版增益逻辑" if admitted else "增益不足或验证回退",
                )

        # 回滚护栏：只有旧版本完整保留且回滚路径经过验证才允许上线
        rollback_ok = True
        if admission.admitted and self.rollback_guard is not None:
            rc = self.rollback_guard.preflight(self.champion)
            if not rc.ok:
                rollback_ok = False
                admission = AdmissionResult(
                    False, Gate.ROLLBACK, gain=admission.gain,
                    anchor_score=admission.anchor_score,
                    regression_count=admission.regression_count, reason=rc.reason,
                )
                self._admission_stats["rollback_fail"] += 1

        if admission.admitted:
            self._admission_stats["pass"] += 1
        else:
            self._admission_stats["fail"] += 1

        # 对照组：消耗相同算力但不自我修改（永不 promote）
        promoted = admission.admitted and rollback_ok and not self.control_mode
        return admission, promoted

    def _update_memory(self, scenario: str, step: int, start_idx: int) -> None:
        """把本轮新标注且验证正确的样本写入保护性记忆。"""
        if self.memory is None or start_idx >= len(self._ann_X):
            return
        Xs = np.concatenate(self._ann_X[start_idx:], axis=0)
        ys = np.concatenate(self._ann_y[start_idx:], axis=0)
        probs = self.champion.predict_proba(Xs)
        rec = self.registry.champion_record()
        version = rec["id"] if rec else "init"
        self.memory.maybe_add(Xs, ys, probs, scenario, step, version)

    def _heldout_acc(self) -> float:
        """冻结留出集上的最终评判指标（迭代过程不可访问）。"""
        if self.heldout is None or self.champion is None:
            return -1.0
        return float(self.heldout.evaluate(self.champion))

    def _eval_pool(self):
        if not self._eval_windows:
            return None, None
        Xs = [w[0] for w in self._eval_windows]
        ys = [w[1] for w in self._eval_windows]
        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

    def _train_challenger(self, X_tr, y_tr, action: Action, pool_X, pool_y):
        s = self.cfg.strategy
        if action == Action.LIGHT_ADAPT:
            model = create_model(self.cfg.model_type, seed=self.cfg.seed)
            self._fit(model, X_tr, y_tr, light=True)
            return model, "light_adapt", 0

        trials = self.budget.authorize_hpo_trials(s.hpo_trials)
        if trials <= 0:
            model = create_model(self.cfg.model_type, seed=self.cfg.seed)
            self._fit(model, X_tr, y_tr, light=False)
            return model, "full_retrain", 0
        model, best = self._hpo(X_tr, y_tr, pool_X, pool_y, trials)
        return model, "full_retrain_hpo", trials

    def _fit(self, model, X, y, light: bool):
        s = self.cfg.strategy
        if self.cfg.model_type == "lgbm":
            rounds = s.light_boost_rounds if light else s.full_boost_rounds
            model.fit(X, y, n_estimators=rounds)
        else:
            epochs = s.mlp_epochs_light if light else s.mlp_epochs_full
            model.fit(X, y, epochs=epochs)

    def _hpo(self, X_tr, y_tr, pool_X, pool_y, trials: int):
        """超参优化：优先贝叶斯优化；bayes_opt 不可用时随机搜索兜底。

        目标为 (验证集 acc + 池化评估窗口 acc) / 2，兼顾新旧分布。
        """
        def score(model) -> float:
            va = _accuracy(self._val_y, model.predict_proba(self._val_X))
            wa = (
                _accuracy(pool_y, model.predict_proba(pool_X))
                if pool_y is not None and len(pool_y)
                else va
            )
            return 0.5 * (va + wa)

        if self.cfg.model_type == "lgbm":
            def objective(n_estimators, learning_rate):
                m = create_model("lgbm", seed=self.cfg.seed)
                m.fit(X_tr, y_tr, n_estimators=int(n_estimators), learning_rate=learning_rate)
                return score(m)

            pbounds = {"n_estimators": (50, 400), "learning_rate": (0.01, 0.2)}
        else:
            def objective(lr):
                m = create_model("mlp", seed=self.cfg.seed)
                m.fit(X_tr, y_tr, epochs=self.cfg.strategy.mlp_epochs_full, lr=lr)
                return score(m)

            pbounds = {"lr": (1e-3, 2e-2)}

        def _call_objective(cand: dict) -> float:
            if self.cfg.model_type == "lgbm":
                return objective(int(cand["n_estimators"]), float(cand["learning_rate"]))
            return objective(float(cand["lr"]))

        try:
            from bayes_opt import BayesianOptimization

            optimizer = BayesianOptimization(
                f=objective, pbounds=pbounds, random_state=self.cfg.seed, verbose=0,
            )
            optimizer.maximize(init_points=min(3, trials), n_iter=max(0, trials - 3))
            best = {k: float(v) for k, v in optimizer.max["params"].items()}
        except Exception:  # noqa: BLE001 - 无 bayes_opt 或小数据时随机搜索保底
            rng = np.random.default_rng(self.cfg.seed)
            best, best_score = None, -np.inf
            for _ in range(max(1, trials)):
                cand = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in pbounds.items()}
                s = _call_objective(cand)
                if s > best_score:
                    best_score, best = s, cand
            if best is None:
                best = {k: float((lo + hi) / 2.0) for k, (lo, hi) in pbounds.items()}

        model = create_model(self.cfg.model_type, seed=self.cfg.seed)
        if self.cfg.model_type == "lgbm":
            model.fit(X_tr, y_tr, n_estimators=int(best["n_estimators"]), learning_rate=best["learning_rate"])
        else:
            model.fit(X_tr, y_tr, epochs=self.cfg.strategy.mlp_epochs_full, lr=best["lr"])
        return model, best

    # ---------------- 数据与标注 ----------------
    def _annotate_random(self, X: np.ndarray, n: int):
        n = min(n, len(X))
        if n == 0:
            return None, None, None
        rng = np.random.default_rng(self.cfg.seed + self.budget.annotations_used)
        idx = rng.choice(len(X), size=n, replace=False)
        win_X, win_y, idx = self._pay_and_store(X, idx)
        if win_X is None:
            return None, None, None
        return win_X, win_y, idx

    def _annotate_uncertain(self, X: np.ndarray, probs: np.ndarray, n: int, exclude=None):
        n = min(n, len(X))
        if n == 0:
            return
        ent = _entropy(probs)
        if exclude is not None and len(exclude):
            ent = ent.copy()
            ent[exclude] = -np.inf
        idx = np.argsort(ent)[-n:]
        self._pay_and_store(X, idx)

    def _pay_and_store(self, X: np.ndarray, idx: np.ndarray):
        """标注并入库。

        对齐赛题 B 包规则：评测流标签总可得（use_free_labels=True），直接读 truth，
        不消耗任何配额；否则回退到旧的「标注配额」路径（本地仿真/兼容模式）。
        """
        idx = np.asarray(idx, dtype=int)
        truth = self.stream.truth()
        if self.cfg.use_free_labels and truth is not None:
            idx = idx[: len(truth)]
            if len(idx) == 0:
                return None, None, None
            y = truth[idx]
            self._ann_X.append(X[idx])
            self._ann_y.append(y)
            return X[idx], y, idx

        remaining = self.budget.annotation_budget - self.budget.annotations_used
        idx = idx[:remaining]
        if len(idx) == 0:
            return None, None, None
        self.budget.authorize_annotation(len(idx))
        y = self.stream.annotate(idx)
        self._ann_X.append(X[idx])
        self._ann_y.append(y)
        return X[idx], y, idx

    def _build_training_set(self, exclude_idx=None):
        parts_X = [self._init_X] + [
            x for i, x in enumerate(self._ann_X) if i != exclude_idx
        ]
        parts_y = [self._init_y] + [
            y for i, y in enumerate(self._ann_y) if i != exclude_idx
        ]
        return np.concatenate(parts_X, axis=0), np.concatenate(parts_y, axis=0)

    # ---------------- 日志与汇总 ----------------
    def _step_log(self, step: int, report, decision: Action, probs: np.ndarray) -> dict:
        data = {
            "policy/decision": ACTION_CODE[decision],
            "model/champion_val_acc": self.val_acc if self.val_acc is not None else -1,
            "model/champion_window_acc": self.champion_window_acc if self.champion_window_acc is not None else -1,
            "model/conf_mean": float(np.max(probs, axis=1).mean()),
            "advisor/patches_applied": self._patches_applied,
            **self.budget.snapshot(),
            "cost/gpu_minutes": self.cost_minutes,
        }
        if report is not None:
            data.update(report.to_log())
        truth = self.stream.truth()  # 仅本地仿真评估用，真实评测由平台计分
        if truth is not None and probs is not None:
            data["eval/stream_acc"] = _accuracy(truth, probs)
        return data

    def _final_summary(self) -> dict:
        decisions = {f"summary/decisions_{k}": v for k, v in self._decisions.items()}
        stream_acc = None
        truth = self.stream.truth()
        if truth is not None and self.champion is not None:
            stream_acc = _accuracy(truth, self.champion.predict_proba(self.stream.current_batch()))
        admission_stats = {f"summary/admission_{k}": v for k, v in self._admission_stats.items()}
        return {
            "summary/baseline_val_acc": self.baseline_acc,
            "summary/final_val_acc": self.val_acc,
            "summary/final_window_acc": self.champion_window_acc if self.champion_window_acc is not None else -1,
            "summary/stream_acc_final": stream_acc if stream_acc is not None else -1,
            "summary/heldout_acc": self._heldout_acc(),
            "summary/gain_over_baseline": (stream_acc - (self.baseline_acc or 0))
            if stream_acc is not None else -1,
            "summary/cost_minutes": self.cost_minutes,
            "summary/efficiency": self._efficiency(),
            "summary/qualified_versions": len(self.registry.qualified_versions()),
            **self.budget.snapshot(),
            **decisions,
            **admission_stats,
        }

    def _efficiency(self) -> float:
        """增益/成本 代理指标（对齐赛题客观分：单位资源消耗的性能增益）。"""
        if self.cost_minutes <= 0 or self.champion_window_acc is None:
            return -1.0
        gain = self.champion_window_acc - (self.baseline_acc or 0)
        return gain / self.cost_minutes
