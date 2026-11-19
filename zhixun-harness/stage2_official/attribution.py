"""Attribute drift evidence to one bounded cause and a matching recipe.

The five-step loop needs a cause, not just a score: covariate PSI alone cannot
tell an operator which sensor block moved, whether a class went missing, or
whether one operating condition now dominates the batch. This module turns the
summaries produced by diagnostics.diagnose and contest_submission.summarize into
a named cause plus a short list of recipes that already exist in continuation.py.

Pure functions over plain mappings: no I/O, no model access, no randomness.
"""

SENSOR_FEATURES = 8
SENSOR_COUNT = 16
# One sensor block must own the drifting columns, not a scatter across channels.
CHANNEL_MIN_HITS = 3
# A class holding less than this share of the batch cannot be learned from it.
MIN_CLASS_SHARE = 0.05
MISSING_CLASS_ALARM = 2
# Regression targets moving this far from the parent are a distribution shift.
TARGET_MEAN_SHIFT = 0.20
# One operating condition holding this share means the batch left the others.
CONDITION_DOMINANCE = 0.60
PSI_ALARM = 0.5


def sensor_channels(top_features):
    """Count how often each gas sensor block owns a drifting feature column."""
    hits = {}
    for item in top_features or []:
        index = item.get("index")
        if type(index) is not int or not 0 <= index < SENSOR_FEATURES * SENSOR_COUNT:
            return None
        channel = index // SENSOR_FEATURES
        hits[channel] = hits.get(channel, 0) + 1
    return hits or None


def class_pressure(labels, classes):
    """Report missing or under-represented classes in one batch."""
    labels = labels or {}
    total = sum(count for count in labels.values() if type(count) is int and count > 0)
    if total <= 0:
        return None
    missing = [name for name in classes if labels.get(name, 0) <= 0]
    smallest = min((labels.get(name, 0) / total) for name in classes) if classes else min(labels.values()) / total
    return {"rows": total, "missing": missing, "smallest_share": round(float(smallest), 6)}


def target_shift(batch, parent_target_mean):
    """Relative movement of the batch target mean against the parent checkpoint."""
    current, parent = batch.get("target_mean"), parent_target_mean
    if type(current) not in (int, float) or type(parent) not in (int, float):
        return None
    if not current == current or not parent == parent:
        return None
    return abs(float(current) - float(parent)) / max(abs(float(parent)), 1e-9)


def dominant_condition(counts):
    """Share held by the largest operating-condition group."""
    counts = counts or {}
    total = sum(value for value in counts.values() if type(value) is int and value > 0)
    if total <= 0:
        return None
    return max(counts.values()) / total


def attribute(task, batch, *, classes=(), drift=None, parent_target_mean=None, condition_counts=None):
    """Return the cause, the supporting evidence and the recipes it justifies."""
    evidence = {"task": task, "psi_max": (drift or {}).get("psi_max"),
                "drifted_features": (drift or {}).get("n_features_drifted")}
    none = {"cause": "none", "channels": [], "recipes": [], "evidence": evidence}

    if task == "A":
        pressure = class_pressure(batch.get("labels"), classes)
        if pressure is not None:
            evidence["missing_classes"] = pressure["missing"]
            evidence["smallest_class_share"] = pressure["smallest_share"]
        hits = sensor_channels((drift or {}).get("top_features"))
        if hits is not None:
            ordered = sorted(hits.items(), key=lambda item: (-item[1], item[0]))
            evidence["channel_hits"] = [{"channel": channel, "hits": count} for channel, count in ordered]
            if ordered[0][1] >= CHANNEL_MIN_HITS:
                channels = [channel for channel, count in ordered if count >= CHANNEL_MIN_HITS]
                return {"cause": "sensor_channel", "channels": sorted(channels),
                        "recipes": ["sensor_mask_v1"], "evidence": evidence}
        if pressure is not None and (len(pressure["missing"]) >= MISSING_CLASS_ALARM
                                     or pressure["smallest_share"] < MIN_CLASS_SHARE):
            return {"cause": "label_shift", "channels": [], "recipes": ["class_balance"], "evidence": evidence}
        if (drift or {}).get("n_features_drifted"):
            return {"cause": "covariate_shift", "channels": [], "recipes": [], "evidence": evidence}
        return none

    if task == "B":
        shift = target_shift(batch, parent_target_mean)
        if shift is not None:
            evidence["target_mean_shift"] = round(shift, 6)
            if shift >= TARGET_MEAN_SHIFT:
                return {"cause": "target_shift", "channels": [], "recipes": ["b_residual_v1"],
                        "evidence": evidence}
        if ((drift or {}).get("psi_max") or 0.0) >= PSI_ALARM:
            return {"cause": "covariate_shift", "channels": [], "recipes": [], "evidence": evidence}
        return none

    if task == "C":
        share = dominant_condition(condition_counts)
        if share is not None:
            evidence["dominant_condition_share"] = round(float(share), 6)
            evidence["condition_counts"] = dict(condition_counts)
            if share >= CONDITION_DOMINANCE and len([v for v in condition_counts.values() if v > 0]) >= 2:
                return {"cause": "operating_condition", "channels": [], "recipes": ["c_group_mse_v1"],
                        "evidence": evidence}
        shift = target_shift(batch, parent_target_mean)
        if shift is not None:
            evidence["target_mean_shift"] = round(shift, 6)
            if shift >= TARGET_MEAN_SHIFT:
                return {"cause": "target_shift", "channels": [], "recipes": [], "evidence": evidence}
        if ((drift or {}).get("psi_max") or 0.0) >= PSI_ALARM:
            return {"cause": "covariate_shift", "channels": [], "recipes": [], "evidence": evidence}
        return none

    return none


def merge_recipes(recipes, attributed):
    """Append attributed recipes that the caller already allows, without reordering."""
    result = tuple(recipes)
    for recipe in attributed:
        if recipe not in result:
            result += (recipe,)
    return result
