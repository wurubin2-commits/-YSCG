"""Bounded experiment choices and compact feedback for the official gateway."""

import hashlib
import json
import math
import os
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from stage2_local.planner import Proposal
from stage2_official.continuation import B_RESIDUAL_RECIPE, L2SP_MAX_STRENGTH, L2SP_RECIPE

RECIPES = {"A": ("standard", "class_balance", "l2"), "B": ("standard", "huber", "l2"),
           "C": ("standard", "huber", "l2")}
OUTPUT_TOKEN_LIMIT = 512
EPOCH_MAX = int(os.environ.get("ZHIXUN_EPOCH_MAX", "2"))
PROMPT = (
    'Decide one CPU experiment. Candidates are derived from measured evidence: answer '
    '{"action":"train","candidate":k,"reason":"..."} to pick one, or send a full train object '
    'to override. JSON only. Keep: {"action":"keep","reason":"..."}, only when every candidate '
    'would lose accuracy. train fields: reason, rate (0.00005–0.005), batch (32/64/128/256), '
    'epochs (1/' + str(EPOCH_MAX) + '). Reason 200 chars max.'
)


class Experiment(Proposal):
    reason: str = Field(min_length=1, max_length=200)
    recipe: Literal["standard", "class_balance", "huber", "l2", "l2sp", "sensor_mask_v1", "b_residual_v1",
                    "c_group_mse_v1"] = "standard"
    l2sp_strength: float | None = Field(default=None, gt=0.0, le=L2SP_MAX_STRENGTH, allow_inf_nan=False)
    residual_mean_weight: float | None = Field(default=None, gt=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_keep(self):
        if self.action == "keep" and self.recipe != "standard":
            raise ValueError("Keep cannot select a training recipe")
        if (self.recipe == L2SP_RECIPE) != (self.l2sp_strength is not None):
            raise ValueError("L2-SP proposals must specify their strength")
        if (self.recipe == B_RESIDUAL_RECIPE) != (self.residual_mean_weight is not None):
            raise ValueError("Residual proposals must specify their mean weight")
        return self


def effective_parameters(plan):
    return plan.model_dump(include={"recipe", "rate", "batch", "epochs", "l2sp_strength", "residual_mean_weight"},
                           exclude_none=True)


def c_half_rate(plan):
    """Preserve the advisor's action and budget while taking a smaller standard update."""
    if plan.action != "train" or plan.recipe != "standard":
        return plan
    return Experiment.model_validate({**plan.model_dump(), "rate": max(0.00005, plan.rate / 2)})


def experiment_key(task, parent, train, seed, parameters):
    payload = [task, parent, train, seed, parameters]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def recorded_parameters(record):
    config = record.get("configuration")
    if not isinstance(config, dict):
        return None
    # Older records used longer prose and omitted the default recipe.
    try:
        plan = Experiment.model_validate({**config, "reason": "Historical experiment"})
    except ValidationError:
        return None
    return effective_parameters(plan) if plan.action == "train" else None


def already_trained(prior, key):
    for record in prior:
        if not (record.get("local_train_receipt") or record.get("candidate_artifact")):
            continue
        parameters = recorded_parameters(record)
        receipt = record.get("local_train_receipt") or {}
        train = record.get("train_sha256") or receipt.get("train_sha256")
        if parameters and experiment_key(record["task"], record["parent_sha256"],
                                         train, record["seed"], parameters) == key:
            return True
    return False


def compact_score(score):
    return float(f"{score:.6g}") if type(score) in (int, float) and math.isfinite(score) else None


def history_feedback(prior, history, parent_sha, train_sha, round_index):
    """Choose up to three training outcomes, preferring the current parent and batch."""
    by_round = {record["round"]: record for record in prior}
    for record in history:
        index = record.get("round_index")
        if type(index) is int and 0 < index < round_index:
            by_round.setdefault(index, record)

    outcomes = []
    for index, record in by_round.items():
        parameters = recorded_parameters(record)
        decision = record.get("decision")
        if parameters is None or decision not in (
            None, "adopted", "measured_keep", "stability_keep", "candidate_failed", "training_failed",
        ):
            continue
        same_parent = record.get("parent_sha256") == parent_sha
        scope = ("same_batch" if same_parent and record.get("train_sha256") == train_sha
                 else "same_parent" if same_parent else "older")
        if parameters["recipe"] == "standard":
            del parameters["recipe"]
        outcome = {"round": index, "scope": scope, "params": parameters,
                   "outcome": decision or "unreported"}
        before, after = (compact_score(record.get(field)) for field in
                         ("parent_public_score", "candidate_public_score"))
        if before is not None and after is not None:
            outcome["scores"] = [before, after]
        outcomes.append(outcome)

    priority = {"same_batch": 0, "same_parent": 1, "older": 2}
    outcomes.sort(key=lambda item: (priority[item["scope"]], -item["round"]))
    chosen = outcomes[:1]
    for group in ({"adopted"}, {"measured_keep", "stability_keep", "candidate_failed", "training_failed"}):
        match = next((item for item in outcomes if item["outcome"] in group), None)
        if match is not None and match not in chosen:
            chosen.append(match)
    return (chosen + [item for item in outcomes if item not in chosen])[:3]
