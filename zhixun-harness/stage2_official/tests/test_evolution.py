import pytest

from stage2_official import evolution
from stage2_official.evolution import (
    best_ever, brief, demoted_recipes, propose, rank, recipe_history, rejected_recipes,
    resolve_choice, structural_priority, to_experiment, top_trainable, trials,
)


def record(round_index, *, before, after, rate=0.0005, epochs=2, recipe="standard", decision="measured_keep",
           strength=None):
    config = {"action": "train", "rate": rate, "batch": 128, "epochs": epochs, "recipe": recipe}
    if strength is not None:
        config["l2sp_strength" if recipe == "l2sp" else "residual_mean_weight"] = strength
    return {"round": round_index, "configuration": config, "decision": decision,
            "parent_public_score": before, "candidate_public_score": after,
            "local_train_receipt": {"validation_before": {}, "validation_after": {}}}


def test_trials_ignores_keep_rounds_and_unmeasured_scores():
    prior = [
        {"round": 1, "decision": "model_keep", "configuration": {"action": "keep"},
         "parent_public_score": 681.0, "candidate_public_score": None},
        record(2, before=681.0, after=600.0),
    ]
    rows = trials(prior, "lower")
    assert len(rows) == 1
    assert rows[0]["recipe"] == "standard"
    assert rows[0]["gain"] == pytest.approx((681.0 - 600.0) / 681.0)


def test_best_ever_respects_metric_direction():
    prior = [record(1, before=700.0, after=650.0), record(2, before=650.0, after=640.0)]
    score, item = best_ever(prior, "lower")
    assert score == 640.0 and item["round"] == 2
    prior = [{"round": 1, "candidate_public_score": 0.71}, {"round": 2, "candidate_public_score": 0.73}]
    score, item = best_ever(prior, "higher")
    assert score == 0.73 and item["round"] == 2
    assert best_ever([], "lower") == (None, None)


def test_structural_priority_prefers_the_attributed_recipe():
    ordered = structural_priority("A", ("standard", "class_balance", "l2", "sensor_mask_v1"),
                                  [], "higher", cause="sensor_channel")
    assert ordered[0] == "sensor_mask_v1"
    ordered = structural_priority("B", ("standard", "huber", "l2", "b_residual_v1"),
                                  [record(1, before=681.0, after=650.0, recipe="b_residual_v1")],
                                  "lower", cause="target_shift")
    assert ordered[0] == "b_residual_v1"
    assert structural_priority("A", ("standard",), [], "higher") == []


def test_recipe_history_reports_only_structural_outcomes():
    prior = [record(1, before=681.0, after=650.0, recipe="huber"),
             record(2, before=650.0, after=660.0, recipe="huber"),
             record(3, before=660.0, after=640.0, recipe="standard")]
    history = recipe_history(prior, "lower")
    assert set(history) == {"huber"}
    assert history["huber"] == pytest.approx(((681.0 - 650.0) / 681.0 + (650.0 - 660.0) / 650.0) / 2)


def test_propose_opens_below_the_parent_learning_rate():
    candidates = propose({"task": "B", "direction": "lower", "parent_rate": 0.003, "parent_batch": 64,
                          "remaining": 4, "allowed_recipes": ("standard",), "prior": [],
                          "stalled": 0, "cause": "none"})
    first = candidates[0]
    assert first["mutation"] == "probe_step"
    assert first["params"]["epochs"] == 1
    assert first["params"]["rate"] == pytest.approx(0.001)
    assert candidates[-1]["mutation"] == "keep"


def test_propose_continues_while_the_step_pays_and_halves_it_after_a_loss():
    winning = propose({"task": "B", "direction": "lower", "parent_rate": 0.003, "parent_batch": 128,
                       "remaining": 4, "allowed_recipes": ("standard",), "prior": [
                           record(1, before=681.0, after=600.0, rate=0.001, epochs=1)], "stalled": 0})
    assert winning[0]["mutation"] == "continue_step"
    assert winning[0]["params"]["rate"] == pytest.approx(0.0006)
    losing = propose({"task": "B", "direction": "lower", "parent_rate": 0.003, "parent_batch": 128,
                      "remaining": 4, "allowed_recipes": ("standard",), "prior": [
                          record(1, before=598.0, after=626.0, rate=0.001, epochs=1)], "stalled": 1})
    assert losing[0]["mutation"] == "safe_step"
    assert losing[0]["params"]["rate"] == pytest.approx(0.0005)


def test_propose_offers_the_structure_when_the_plateau_is_long():
    prior = [record(index, before=600.0, after=610.0) for index in (1, 2)]
    candidates = propose({"task": "B", "direction": "lower", "parent_rate": 0.0005, "parent_batch": 128,
                          "remaining": 4, "allowed_recipes": ("standard", "huber", "l2"),
                          "prior": prior, "stalled": 3, "cause": "none"})
    assert candidates[0]["mutation"] == "structural"
    assert any(item["mutation"] == "elite_replay" for item in candidates)
    assert sorted(item["id"] for item in candidates) == list(range(1, len(candidates) + 1))


def test_elite_replay_is_skipped_when_the_parent_is_the_elite():
    """父模型就是历史最优候选时，重放同一配置等于在已收敛模型上再训，应当跳过。"""
    prior = [record(1, before=681.0, after=650.0, decision="adopted"),
             record(2, before=650.0, after=655.0)]
    candidates = propose({"task": "B", "direction": "lower", "parent_rate": 0.003, "parent_batch": 128,
                          "remaining": 4, "allowed_recipes": ("standard", "huber", "l2"),
                          "prior": prior, "stalled": 2})
    assert not any(item["mutation"] == "elite_replay" for item in candidates)


def test_propose_skips_candidates_that_exceed_the_epoch_budget():
    candidates = propose({"task": "A", "direction": "higher", "parent_rate": 0.002, "parent_batch": 128,
                          "remaining": 0, "allowed_recipes": ("standard", "l2"), "prior": [], "stalled": 1})
    assert [item["mutation"] for item in candidates] == ["keep"]


def test_rank_moves_a_known_bad_recipe_down():
    candidates = propose({"task": "C", "direction": "lower", "parent_rate": 0.002, "parent_batch": 256,
                          "remaining": 4, "allowed_recipes": ("standard", "huber", "l2"), "prior": [],
                          "stalled": 3, "failed_recipes": ["huber"]})
    ordered = rank(candidates, {"direction": "lower", "prior": [], "stalled": 2,
                                "failed_recipes": ["huber"]})
    structural = next(item for item in ordered if item["mutation"] == "structural")
    assert structural["params"]["recipe"] != "huber"


def test_resolve_choice_reads_a_candidate_id():
    candidates = propose({"task": "B", "direction": "lower", "parent_rate": 0.0005, "parent_batch": 64,
                          "remaining": 4, "allowed_recipes": ("standard", "l2"), "prior": [], "stalled": 0})
    plan, chosen = resolve_choice('{"action":"train","candidate":1,"reason":"smaller step"}', candidates)
    assert plan.action == "train"
    assert plan.reason == "smaller step"
    assert chosen is candidates[0]
    assert resolve_choice('{"candidate":9}', candidates) is None
    assert resolve_choice("not json", candidates) is None
    assert resolve_choice('{"candidate":2}', []) is None


def test_resolve_choice_can_pick_keep():
    candidates = [{"id": 1, "mutation": "keep", "params": {"action": "keep"}, "why": "flat", "rank_score": 0.0}]
    plan, _ = resolve_choice('{"action":"keep","candidate":1,"reason":"no room"}', candidates)
    assert plan.action == "keep"


def test_top_trainable_refuses_a_losing_neighbourhood():
    keep = {"id": 1, "mutation": "keep", "params": {"action": "keep"}, "why": "flat", "rank_score": 0.0}
    losing = {"id": 2, "mutation": "safe_step", "params": {"rate": 0.0001, "batch": 128, "epochs": 1,
                                                           "recipe": "standard"},
              "why": "step", "rank_score": -0.03}
    assert top_trainable([keep, losing]) is None
    winning = dict(losing, rank_score=0.004)
    assert top_trainable([keep, winning]) is winning


def test_to_experiment_validates_the_candidate_payload():
    plan = to_experiment({"mutation": "structural",
                          "params": {"rate": 0.0002, "batch": 64, "epochs": 1, "recipe": "l2"},
                          "why": "objective change"})
    assert plan.action == "train" and plan.recipe == "l2"
    assert plan.reason == "objective change"
    keep = to_experiment({"mutation": "keep", "params": {"action": "keep"}, "why": "flat"})
    assert keep.action == "keep"


def test_brief_drops_the_internal_score():
    candidates = propose({"task": "A", "direction": "higher", "parent_rate": 0.002, "parent_batch": 128,
                          "remaining": 4, "allowed_recipes": ("standard", "l2"), "prior": [], "stalled": 0})
    view = brief(candidates)
    assert all(set(item) == {"id", "mutation", "params", "why"} for item in view)
    assert max(len(item["why"]) for item in view) <= 60


def blocked(round_index, *, before, after, recipe="huber", blocked_slices=("low_rul",),
            decision="stability_keep", rate=0.001):
    """A candidate that improved the headline metric but broke a protected slice."""
    record_row = record(round_index, before=before, after=after, rate=rate, recipe=recipe,
                        decision=decision)
    record_row["local_train_receipt"]["stability"] = [
        {"slice": name, "rows": 420, "metric": "mae", "before": 308.91, "after": 448.64,
         "passed": False}
        for name in blocked_slices]
    return record_row


def test_rejected_recipes_counts_a_slice_regression_not_only_a_losing_gain():
    # huber improved the headline RMSE by 85 but wrecked low_rul: refuted all the same.
    prior = [blocked(3, before=681.16, after=595.59)]
    assert rejected_recipes(prior, "lower") == ("huber",)
    assert trials(prior, "lower")[0]["blocked"] == ["low_rul"]
    assert trials(prior, "lower")[0]["blocked_rows"][0]["after"] == 448.64


def test_rejected_recipes_never_blacklists_a_plain_step():
    # A standard step that lost says the step was too long, not that the direction is wrong.
    prior = [record(2, before=681.16, after=706.79, decision="measured_keep")]
    assert rejected_recipes(prior, "lower") == ()


def test_a_passing_structural_recipe_is_not_rejected():
    paid = record(1, before=681.16, after=651.51, recipe="b_residual_v1", decision="adopted")
    paid["local_train_receipt"]["stability"] = [
        {"slice": "low_rul", "metric": "mae", "before": 308.91, "after": 300.0, "passed": True}]
    assert rejected_recipes([paid], "lower") == ()


def test_low_rul_regression_outranks_the_drift_keyword():
    ordered = structural_priority(
        "B", ("standard", "huber", "l2", "b_residual_v1", "l2sp"), [], "lower",
        cause="covariate_shift", failed_slices=("low_rul",))
    assert ordered[0] == "b_residual_v1"


def test_a_refuted_recipe_stops_being_proposed():
    prior = [blocked(3, before=681.16, after=595.59)]
    candidates = propose({"task": "B", "direction": "lower", "parent_rate": 0.002, "parent_batch": 128,
                          "remaining": 6, "allowed_recipes": ("standard", "huber", "l2",
                                                              "b_residual_v1"),
                          "prior": prior, "stalled": 1, "cause": "none",
                          "failed_recipes": rejected_recipes(prior, "lower")})
    recipes = [(item["params"] or {}).get("recipe") for item in candidates if item["mutation"] != "keep"]
    assert "huber" not in recipes
    assert any(name in ("b_residual_v1", "l2", "standard") for name in recipes)


def test_repeating_a_paid_configuration_does_not_refute_the_recipe():
    # Re-running an adopted configuration on a converged parent loses by construction; that
    # says nothing about the recipe, and blacklisting it would push the search onto worse ones.
    paid = record(1, before=681.16, after=651.51, recipe="b_residual_v1", decision="adopted",
                  strength=0.5)
    repeat = record(2, before=651.51, after=665.52, recipe="b_residual_v1", decision="measured_keep",
                    strength=0.5)
    assert rejected_recipes([paid, repeat], "lower") == ()
    different_scale = record(3, before=651.51, after=670.0, recipe="b_residual_v1",
                             decision="measured_keep", rate=0.0002, strength=0.5)
    assert rejected_recipes([paid, repeat, different_scale], "lower") == ("b_residual_v1",)


def test_structural_waits_for_a_second_plain_loss():
    # One losing step earns another annealing, not an objective change: structural can win the
    # headline metric while wrecking a protected slice, so it must queue behind cheap attempts.
    prior = [record(1, before=681.16, after=706.79, decision="measured_keep")]
    candidates = propose({"task": "B", "direction": "lower", "parent_rate": 0.003,
                          "parent_batch": 128, "remaining": 6,
                          "allowed_recipes": ("standard", "huber", "b_residual_v1"),
                          "prior": prior, "stalled": 0, "cause": "none"})
    assert "structural" not in [item["mutation"] for item in candidates]
    two_losses = [record(1, before=681.16, after=706.79, decision="measured_keep"),
                  record(2, before=706.79, after=731.0, decision="measured_keep")]
    candidates = propose({"task": "B", "direction": "lower", "parent_rate": 0.003,
                          "parent_batch": 128, "remaining": 6,
                          "allowed_recipes": ("standard", "huber", "b_residual_v1"),
                          "prior": two_losses, "stalled": 0, "cause": "none"})
    assert "structural" in [item["mutation"] for item in candidates]


def test_a_demoted_recipe_keeps_a_slot_but_never_leads():
    # class_balance paid at 0.001 then lost at the same 0.001: not refuted, but it must not
    # outrank `l2`, which is what actually kept paying on A.
    paid = record(1, before=0.7140, after=0.7278, recipe="class_balance", decision="adopted",
                  rate=0.001, epochs=1)
    repeat = record(2, before=0.7278, after=0.7267, recipe="class_balance",
                    decision="measured_keep", rate=0.001, epochs=1)
    assert rejected_recipes([paid, repeat], "higher") == ()
    assert demoted_recipes([paid, repeat], "higher") == ("class_balance",)
    ordered = structural_priority("A", ("standard", "class_balance", "l2"), [paid, repeat],
                                  "higher", cause="label_shift",
                                  demoted=demoted_recipes([paid, repeat], "higher"))
    assert ordered[0] == "l2"
    assert "class_balance" in ordered
