"""Admission gate: gain, ratchet floor and rollback preflight."""

import pytest

from stage1_local.common import ROOT, TASKS
from stage2_official import admission


def enable(monkeypatch, **values):
    monkeypatch.setenv("ZHIXUN_ADMISSION", "1")
    for name, value in values.items():
        monkeypatch.setenv(name, str(value))
    return admission.config()


def record(decision, score, stability=None):
    return {"decision": decision, "candidate_public_score": score,
            "local_train_receipt": {"stability": stability or []}}


def test_enabled_by_default(monkeypatch):
    """门槛要求封测不低于静态基线，因此该护栏必须默认生效。"""
    monkeypatch.delenv("ZHIXUN_ADMISSION", raising=False)
    assert admission.config().enabled is True


def test_disabled_passes(monkeypatch):
    monkeypatch.setenv("ZHIXUN_ADMISSION", "0")
    ok, gate, reason = admission.evaluate([], [], "higher", 0.7, 0.6)
    assert (ok, gate) == (True, "disabled")
    assert "off" in reason


def test_gain_gate(monkeypatch):
    enable(monkeypatch, ZHIXUN_ADMISSION_MIN_GAIN=0.01)
    ok, gate, _ = admission.evaluate([], [], "higher", 0.70, 0.705)
    assert not ok and gate == "gain"
    ok, gate, _ = admission.evaluate([], [], "higher", 0.70, 0.72)
    assert ok and gate == "pass"


def test_anchor_floor_follows_the_measured_parent(monkeypatch):
    """历史 floor 高于当前父分时不下修会永久锁死；下修到父分后，超过父分的候选应被放行。"""
    enable(monkeypatch)
    prior = [record("adopted", 0.75)]
    ok, gate, _ = admission.evaluate(prior, [], "higher", 0.72, 0.73)
    assert ok and gate == "pass"
    ok, gate, _ = admission.evaluate(prior, [], "higher", 0.72, 0.76)
    assert ok and gate == "pass"


def test_anchor_floor_still_blocks_a_candidate_below_the_parent(monkeypatch):
    enable(monkeypatch)
    prior = [record("adopted", 0.75)]
    ok, gate, _ = admission.evaluate(prior, [], "higher", 0.76, 0.74)
    assert not ok and gate == "gain"


def test_anchor_floor_for_lower_is_better(monkeypatch):
    enable(monkeypatch)
    prior = [record("adopted", 50.0)]
    ok, gate, _ = admission.evaluate(prior, [], "lower", 52.0, 51.0)
    assert ok and gate == "pass"
    ok, gate, _ = admission.evaluate(prior, [], "lower", 52.0, 53.0)
    assert not ok and gate == "gain"


def test_slice_floor(monkeypatch):
    enable(monkeypatch)
    prior = [record("adopted", 0.75, [{"slice": "乙醛", "metric": "recall", "after": 0.9}])]
    ok, gate, reason = admission.evaluate(
        prior, [{"slice": "乙醛", "metric": "recall", "after": 0.85}], "higher", 0.74, 0.76)
    assert not ok and gate == "anchor" and "乙醛" in reason


@pytest.mark.parametrize("task", ["A", "B", "C"])
def test_rollback_preflight_accepts_declared_checkpoint(monkeypatch, task):
    path = ROOT / TASKS[task] / "cpu工业模型" / "checkpoint.npz"
    if not path.is_file():
        pytest.skip(f"{task} checkpoint is a local input and is absent")
    from stage2_official.trainer import baseline

    model = baseline(task, "model").Model.load(path)
    enable(monkeypatch, ZHIXUN_ADMISSION_ROLLBACK="1")
    ok, reason = admission.rollback_preflight(
        path, model, lambda item: baseline(task, "model").Model.load(item))
    assert ok, reason


def test_rollback_preflight_rejects_missing(monkeypatch, tmp_path):
    enable(monkeypatch)
    ok, reason = admission.rollback_preflight(tmp_path / "absent.npz", None, lambda item: None)
    assert not ok and "missing" in reason


def test_rollback_preflight_rejects_empty(monkeypatch, tmp_path):
    enable(monkeypatch)
    empty = tmp_path / "empty.npz"
    empty.write_bytes(b"")
    ok, reason = admission.rollback_preflight(empty, None, lambda item: None)
    assert not ok and "missing" in reason
