"""冒烟测试：仿真器漂移、监测器、策略、注册表、模型工厂、全闭环。"""
import numpy as np
from sklearn.datasets import make_classification

from yscg.config import SystemConfig
from yscg.data import TabularDriftSimulator
from yscg.drift import Attributor, DriftMonitor
from yscg.drift.monitor import DriftReport
from yscg.guardrails import BudgetManager, validate_strategy_patch
from yscg.models import ModelRegistry, create_model
from yscg.pipeline import SelfEvolutionLoop
from yscg.strategy import Action, RulePolicy


def test_simulator_shapes_and_covariate_drift():
    sim = TabularDriftSimulator(n_batches=40, batch_size=100, seed=0)
    X0 = sim.next_batch()
    assert X0.shape == (100, 20)
    for _ in range(34):
        sim.next_batch()
    X35 = sim.next_batch()
    assert X35[:, 0].mean() - X0[:, 0].mean() > 0.5  # 特征 0 发生协变量漂移


def test_simulator_annotate_consumes_budget():
    sim = TabularDriftSimulator(n_batches=10, batch_size=50, annotation_budget=100, seed=1)
    sim.next_batch()
    y = sim.annotate(np.arange(30))
    assert y.shape == (30,)
    assert sim.annotation_used == 30


def test_monitor_detects_drift_and_rebase():
    cfg = SystemConfig(n_batches=40)
    monitor = DriftMonitor(cfg.strategy)
    sim = TabularDriftSimulator(n_batches=40, batch_size=100, seed=2)
    probs = np.full((100, 2), 0.9)

    X0 = sim.next_batch()
    monitor.set_reference(X0, probs)
    report_clean = monitor.check(X0, probs)
    assert report_clean.severity == "none"

    for _ in range(29):
        Xb = sim.next_batch()
        monitor.observe(Xb, probs)
    X30 = sim.next_batch()
    monitor.observe(X30, probs)
    report = monitor.check(X30, probs)
    assert report.severity in ("mild", "severe")
    assert report.n_drifted >= 1

    attributor = Attributor()
    attr = attributor.attribute(report)
    assert attr["top_features"][0]["feature"].startswith("f")
    monitor.rebase()
    assert monitor.check(X30, probs).severity == "none"  # rebase 后基线重置


def test_policy_decisions():
    cfg = SystemConfig()
    policy = RulePolicy(cfg.strategy)
    severe = DriftReport(0.5, {}, 3, 0.6, 0.3, "severe")
    mild = DriftReport(0.15, {}, 1, 0.8, 0.05, "mild")
    clean = DriftReport(0.02, {}, 0, 0.85, 0.0, "none")
    assert policy.decide(severe, None, 0.8) == Action.FULL_RETRAIN
    assert policy.decide(mild, None, 0.8) == Action.LIGHT_ADAPT
    assert policy.decide(clean, None, 0.8) == Action.NOOP
    assert policy.decide(clean, 0.65, 0.8) == Action.FULL_RETRAIN  # 冠军崩坏兜底


def test_registry_promote_and_reload(tmp_path):
    X, y = make_classification(n_samples=200, n_features=10, random_state=0)
    reg = ModelRegistry(tmp_path, "lgbm")
    m = create_model("lgbm", seed=0)
    m.fit(X, y, n_estimators=30)
    reg.register(m, {"acc": 0.9}, "init")
    reg.promote("v1")
    assert reg.champion_metrics()["acc"] == 0.9
    loaded = reg.load_champion()
    assert loaded.predict_proba(X[:3]).shape == (3, 2)


def test_mlp_smoke():
    X, y = make_classification(n_samples=300, n_features=10, n_classes=2, random_state=0)
    m = create_model("mlp", seed=0)
    m.fit(X, y, epochs=3)
    assert m.predict_proba(X[:5]).shape == (5, 2)


def test_validate_strategy_patch():
    patch = validate_strategy_patch({"psi_mild": 5.0, "hpo_trials": 3.7, "evil": 1})
    assert patch == {"psi_mild": 1.0, "hpo_trials": 4}


def test_full_loop_lgbm(tmp_path):
    cfg = SystemConfig(n_batches=45, batch_size=80, seed=7)
    sim = TabularDriftSimulator(
        n_batches=45, batch_size=80, seed=7,
        init_samples=2000, val_samples=500, annotation_budget=cfg.annotation_budget,
    )
    reg = ModelRegistry(tmp_path / "artifacts", "lgbm")
    budget = BudgetManager(
        cfg.annotation_budget, cfg.max_retrains, cfg.strategy.hpo_trials * cfg.max_retrains
    )
    loop = SelfEvolutionLoop(
        cfg, sim, reg, budget, RulePolicy(cfg.strategy), None,
        Attributor(), DriftMonitor(cfg.strategy),
    )
    summary = loop.run()
    assert summary["summary/final_window_acc"] >= 0
    assert reg.champion_record() is not None
    assert budget.retrains_used >= 1
    assert budget.annotations_used <= cfg.annotation_budget
