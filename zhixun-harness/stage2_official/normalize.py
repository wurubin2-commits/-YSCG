"""Normalize model output into a legal experiment, and fall back to a rule plan.

A model may return a rate, batch or epoch outside the accepted schema. Rejecting
the whole round wastes its only training slot, so out-of-range values are clamped
into the legal range and the original value is preserved for later auditing.
"""

import json
import math
import os

from pydantic import ValidationError

from stage2_official.continuation import B_RESIDUAL_RECIPE, L2SP_MAX_STRENGTH, L2SP_RECIPE
from stage2_official.policy import Experiment

RATE_MIN = 0.00005
RATE_MAX = 0.005
BATCHES = (32, 64, 128, 256)
EPOCH_MIN = 1
EPOCH_MAX = int(os.environ.get("ZHIXUN_EPOCH_MAX", "2"))
REASON_MAX = 200
L2SP_DEFAULT = 1.0
RESIDUAL_DEFAULT = 0.5

# Conservative, CPU-friendly starting points when the model supplies no usable value.
DEFAULTS = {
    "A": {"rate": 0.003, "batch": 128},
    "B": {"rate": 0.003, "batch": 128},
    "C": {"rate": 0.002, "batch": 256},
}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _nearest(value, choices):
    return min(choices, key=lambda item: abs(item - value))


def _extract_object(text):
    """Recover the outermost JSON object from a fenced or chatty reply."""
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


def _parse(raw):
    if not isinstance(raw, str):
        return raw
    stripped = raw.strip()
    for candidate in (stripped, _extract_object(stripped)):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def normalize_proposal(raw, allowed_recipes, task):
    """Return (plan, report). The plan is None only when the action itself is unusable."""
    body = _parse(raw)
    if not isinstance(body, dict):
        return None, {"source": "unparseable", "clamped": {}}

    report = {"source": "model", "clamped": {}}

    def changed(field, before, after):
        report["clamped"][field] = {"from": before, "to": after}

    action = body.get("action")
    if action not in ("train", "keep"):
        return None, {"source": "invalid_action", "clamped": {}, "raw_action": action}

    reason = body.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "Bounded plan after proposal normalization"
        changed("reason", body.get("reason"), reason)
    elif len(reason) > REASON_MAX:
        reason = reason.strip()[:REASON_MAX]
        changed("reason", {"length": len(body["reason"])}, {"length": REASON_MAX})
    reason = reason.strip()

    if action == "keep":
        for field in ("rate", "batch", "epochs", "l2sp_strength", "residual_mean_weight"):
            if body.get(field) is not None:
                changed(field, body.get(field), None)
        if body.get("recipe") not in (None, "standard"):
            changed("recipe", body.get("recipe"), "standard")
        if report["clamped"]:
            report["source"] = "normalized"
        try:
            return Experiment(action="keep", reason=reason), report
        except ValidationError:
            return None, {"source": "invalid_schema", "clamped": report["clamped"]}

    base = DEFAULTS.get(task, DEFAULTS["B"])
    recipe = body.get("recipe") or "standard"
    if recipe not in allowed_recipes:
        changed("recipe", recipe, "standard")
        recipe = "standard"

    rate_raw = body.get("rate")
    if _finite(rate_raw):
        rate = float(min(max(float(rate_raw), RATE_MIN), RATE_MAX))
    else:
        rate = base["rate"]
    if rate != rate_raw:
        changed("rate", rate_raw, rate)

    batch_raw = body.get("batch")
    if _finite(batch_raw):
        batch = int(_nearest(float(batch_raw), BATCHES))
    else:
        batch = base["batch"]
    if batch != batch_raw:
        changed("batch", batch_raw, batch)

    epochs_raw = body.get("epochs")
    if _finite(epochs_raw):
        epochs = int(min(max(round(float(epochs_raw)), EPOCH_MIN), EPOCH_MAX))
    else:
        epochs = EPOCH_MIN
    if epochs != epochs_raw:
        changed("epochs", epochs_raw, epochs)

    extra = {}
    l2sp_raw = body.get("l2sp_strength")
    if recipe == L2SP_RECIPE:
        if _finite(l2sp_raw) and 0.0 < float(l2sp_raw) <= L2SP_MAX_STRENGTH:
            extra["l2sp_strength"] = float(l2sp_raw)
        else:
            extra["l2sp_strength"] = L2SP_DEFAULT
            changed("l2sp_strength", l2sp_raw, L2SP_DEFAULT)
    elif l2sp_raw is not None:
        changed("l2sp_strength", l2sp_raw, None)

    residual_raw = body.get("residual_mean_weight")
    if recipe == B_RESIDUAL_RECIPE:
        if _finite(residual_raw) and 0.0 < float(residual_raw) <= 1.0:
            extra["residual_mean_weight"] = float(residual_raw)
        else:
            extra["residual_mean_weight"] = RESIDUAL_DEFAULT
            changed("residual_mean_weight", residual_raw, RESIDUAL_DEFAULT)
    elif residual_raw is not None:
        changed("residual_mean_weight", residual_raw, None)

    payload = {"action": "train", "reason": reason, "rate": rate, "batch": batch,
               "epochs": epochs, "recipe": recipe, **extra}
    try:
        plan = Experiment.model_validate(payload)
    except ValidationError:
        return None, {"source": "invalid_schema", "clamped": report["clamped"]}
    if report["clamped"]:
        report["source"] = "normalized"
    return plan, report


def rule_plan(task, prior, remaining):
    """Return (plan, source) when the model gave nothing usable, else (None, None).

    The rule stops proposing once three measured trials exist and the last two
    gains are non-positive, so an exhausted search is not forced to keep training.
    """
    if remaining <= 0:
        return None, None
    gains = [item["gain"] for item in prior if _finite(item.get("gain"))]
    trials = sum(1 for item in prior if _finite(item.get("candidate_public_score")))
    if trials >= 3 and len(gains) >= 2 and gains[-1] <= 0 and gains[-2] <= 0:
        return None, None

    base = DEFAULTS.get(task, DEFAULTS["B"])
    rate = base["rate"]
    source = "rule_default"
    if gains and gains[-1] <= 0:
        rate = max(RATE_MIN, rate / 2)
        source = "rule_half_rate"
    last = prior[-1].get("configuration") if prior else None
    if isinstance(last, dict) and last.get("rate") == rate and last.get("batch") == base["batch"]:
        rate = max(RATE_MIN, rate / 2)
        source = "rule_half_rate_shift"
    try:
        plan = Experiment(action="train", reason=f"Rule fallback: {source}",
                          rate=rate, batch=base["batch"], epochs=EPOCH_MIN)
    except ValidationError:
        return None, None
    return plan, source
