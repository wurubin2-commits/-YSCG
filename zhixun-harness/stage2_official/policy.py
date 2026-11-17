"""Bounded experiment choices and compact feedback for the official gateway."""

import hashlib
import json
import math
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from stage2_local.planner import Proposal

RECIPES = {"A": ("standard", "class_balance", "l2"), "B": ("standard", "huber", "l2"),
           "C": ("standard", "huber", "l2")}
OUTPUT_TOKEN_LIMIT = 192
PROMPT = (
    'Choose one CPU experiment or keep. Use prior results to avoid repeats and aim for at least '
    'three measured trials when justified. JSON only. Keep: {"action":"keep","reason":"..."}; '
    'train: action, reason, rate (0.00005–0.005), batch (32/64/128/256), epochs (1/2). '
    'Use a short reason (100 chars max).'
)


class Experiment(Proposal):
    reason: str = Field(min_length=1, max_length=100)
    recipe: Literal["standard", "class_balance", "huber", "l2"] = "standard"

    @model_validator(mode="after")
    def check_keep(self):
        if self.action == "keep" and self.recipe != "standard":
            raise ValueError("Keep cannot select a training recipe")
        return self


def effective_parameters(plan):
    return plan.model_dump(include={"recipe", "rate", "batch", "epochs"})


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
