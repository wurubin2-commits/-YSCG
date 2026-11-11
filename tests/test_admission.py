"""准入决策机制测试：记忆/锚点/准入门/验证器自检/回滚护栏/全闭环。"""
import numpy as np
import pytest
from sklearn.datasets import make_classification

from yscg.config import SystemConfig
from yscg.data import TabularDriftSimulator
from yscg.drift import Attributor, DriftMonitor
from yscg.eval import HeldOutSet
from yscg.guardrails import (
    AdmissionGate,
    AnchorSet,
    BudgetManager,
    Gate,
    MajorityPredictor,
    RollbackGuard,
    ValidatorProbe,
)
from yscg.memory import MemoryRecord, ProtectiveMemory, RegressionSampler, regression_count
from yscg.models import ModelRegistry, create_model
from yscg.pipeline import SelfEvolutionLoop
from yscg.strategy import RulePolicy


def _data(n=300, nf=8, seed=0):
    X, y = make_classification(
        n_samples=n, n_features=nf, n_informative=5, n_redundant=0,
        n_classes=2, random_state=seed,
    )
    return X, y


def _lgbm(X, y, seed=0, rounds=50):
    m = create_model("lgbm", seed=seed)
    m.fit(X, y, n_estimators=rounds)
    return m


# ---- 保护性记忆 ----
def test_memory_capacity_and_correctness_filter():
    X, y = _data(n=100)
    model = _lgbm(X, y)
    probs = model.predict_proba(X)
    mem = ProtectiveMemory(capacity=20, min_conf=0.8)
    mem.maybe_add(X, y, probs, scenario="normal", batch=0, version="v1")
    assert len(mem) <= 20
    # 只收正确且高置信的样本
    for r in mem.records():
        assert r.gold_conf >= 0.8


def test_memory_rejects_wrong_predictions():
    X, y = _data(n=60)
    mem = ProtectiveMemory(capacity=100, min_conf=0.0)
    # 构造全错的预测（每行把 1-y 置 1.0，使 argmax 恒为 1-y）
    probs = np.zeros((len(X), 2))
    probs[np.arange(len(X)), 1 - y] = 1.0
    mem.maybe_add(X, y, probs, scenario="normal", batch=0, version="v1")
    assert len(mem) == 0  # 全部预测错误，不入内存


# ---- 分层回归采样 ----
def test_regression_sampler_stratified_coverage():
    X, y = _data(n=400)
    model = _lgbm(X, y)
    probs = model.predict_proba(X)
    mem = ProtectiveMemory(capacity=1000, min_conf=0.0)
    mem.maybe_add(X, y, probs, scenario="normal", batch=0, version="v1")
    mem.maybe_add(X, y, probs, scenario="severe", batch=1, version="v1")
    counts = mem.strata_counts()
    sampler = RegressionSampler(mem, min_quota=5, seed=1)
    R = sampler.sample(80)
    assert len(R) <= 80
    got = {}
    for r in R:
        got[r.stratum] = got.get(r.stratum, 0) + 1
    for s, c in counts.items():
        assert got.get(s, 0) >= min(5, c)


def test_regression_count_good_vs_bad():
    X, y = _data()
    model = _lgbm(X, y)
    probs = model.predict_proba(X)
    recs = [
        MemoryRecord(X[i].astype(np.float32), int(y[i]), float(probs[i].max()),
                     "normal.easy.correctness", 0, "v1")
        for i in range(100)
    ]
    hard_good, _ = regression_count(model, recs)
    hard_bad, _ = regression_count(MajorityPredictor(y), recs)
    assert hard_bad > hard_good


# ---- 锚点 ----
def test_anchor_ratchet_and_regression_check():
    X, y = _data()
    model = _lgbm(X, y)
    anchor = AnchorSet(eps=0.02)
    anchor.add_slice("val", X, y, 1.0)
    s = anchor.score(model)
    anchor.ratchet(s)
    assert anchor.floor()["val"] == pytest.approx(s["val"], abs=1e-6)
    assert anchor.check(anchor.score(model))[0] is True
    # 多数类「坏」模型应触发退化
    ok, viol = anchor.check(anchor.score(MajorityPredictor(y)))
    assert ok is False and "val" in viol


# ---- 准入门 ----
def test_admission_gate_gain_and_validator():
    X, y = _data()
    model = _lgbm(X, y)
    cfg = SystemConfig()
    anchor = AnchorSet(eps=cfg.strategy.anchor_eps)
    anchor.add_slice("val", X, y, 1.0)
    mem = ProtectiveMemory(capacity=100, min_conf=0.8)
    gate = AdmissionGate(cfg.strategy, anchor, mem)
    pool = (X, y)
    r = gate.evaluate(model, model, pool, validator_healthy=True)
    assert r.admitted is False and r.gate == Gate.GAIN  # 相同模型无增益
    r2 = gate.evaluate(model, model, pool, validator_healthy=False)
    assert r2.admitted is False and r2.gate == Gate.VALIDATOR  # 验证器失效拒绝所有候选


# ---- 验证器自检 ----
def test_validator_probe_detects_broken_ranking():
    X, y = _data()
    good = _lgbm(X, y)
    bad = MajorityPredictor(y)
    cfg = SystemConfig()
    anchor = AnchorSet(eps=cfg.strategy.anchor_eps)
    anchor.add_slice("val", X, y, 1.0)
    mem = ProtectiveMemory(capacity=100, min_conf=0.8)
    gate = AdmissionGate(cfg.strategy, anchor, mem)
    assert ValidatorProbe(good, bad, (X, y)).healthy(gate, good) is True
    # 交换好/坏 → 验证器应报告失效
    assert ValidatorProbe(bad, good, (X, y)).healthy(gate, good) is False


# ---- 回滚护栏 ----
def test_rollback_guard_requires_retention(tmp_path):
    X, y = _data(n=200)
    model = _lgbm(X, y)
    reg = ModelRegistry(tmp_path, "lgbm")
    reg.register(model, {"acc": 0.9}, "init")
    reg.promote("v1")
    guard = RollbackGuard(reg, probe_X=X[:20])
    assert guard.preflight(model).ok is True
    (tmp_path / "models" / "v1-init.joblib").unlink()  # 破坏完整保留
    assert guard.preflight(model).ok is False


# ---- 全闭环（含准入/记忆/锚点/回滚/留出集）----
def test_full_loop_with_admission(tmp_path):
    cfg = SystemConfig(n_batches=45, batch_size=80, seed=7)
    sim = TabularDriftSimulator(
        n_batches=45, batch_size=80, seed=7,
        init_samples=2000, val_samples=500, heldout_samples=300,
        annotation_budget=cfg.annotation_budget,
    )
    reg = ModelRegistry(tmp_path / "artifacts", "lgbm")
    budget = BudgetManager(
        cfg.annotation_budget, cfg.max_retrains, cfg.strategy.hpo_trials * cfg.max_retrains
    )
    val_X, val_y = sim.get_val_data()
    held_X, held_y = sim.get_heldout_data()
    anchor = AnchorSet(eps=cfg.strategy.anchor_eps)
    anchor.add_slice("val", val_X, val_y, 1.0)
    mem = ProtectiveMemory(cfg.strategy.memory_capacity, cfg.strategy.memory_min_conf)
    gate = AdmissionGate(cfg.strategy, anchor, mem)
    guard = RollbackGuard(reg, probe_X=val_X[:20])
    heldout = HeldOutSet(held_X, held_y)

    loop = SelfEvolutionLoop(
        cfg, sim, reg, budget, RulePolicy(cfg.strategy), None,
        Attributor(), DriftMonitor(cfg.strategy),
        admission_gate=gate, rollback_guard=guard, memory=mem, heldout=heldout,
    )
    summary = loop.run()
    assert 0.0 <= summary["summary/heldout_acc"] <= 1.0
    assert summary["summary/qualified_versions"] >= 0
    assert "summary/admission_pass" in summary
    assert budget.retrains_used >= 1
    assert budget.annotations_used <= cfg.annotation_budget
    # 记忆容量不超上限
    assert len(mem) <= cfg.strategy.memory_capacity
