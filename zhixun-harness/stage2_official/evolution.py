"""Evidence-driven variation for the closed evolution loop.

Ten real rounds showed the bottleneck is not the model's ability to read evidence
but the variation operator. A nearly converged parent loses validation accuracy
when it is trained again at a normal learning rate (B round 6: 598.24 -> 626.22),
and four to five of ten rounds ended in keep. Half the budget therefore produced
nothing. This module turns measured evidence into a small ordered set of mutations
so the model only has to rank candidates instead of inventing hyper-parameters.

Pure functions over plain mappings: no I/O, no model access, no randomness.
"""

import math

from stage2_official.continuation import B_RESIDUAL_RECIPE, L2SP_MAX_STRENGTH, L2SP_RECIPE
from stage2_official.policy import Experiment

RATE_MIN = 0.00005
RATE_MAX = 0.005
BATCHES = (32, 64, 128, 256)
# A converged parent tolerates only a small step; above this ratio training is a net loss.
SAFE_STEP_RATIO = 0.25
# Structural recipes cost the same as a plain step but change what the loss optimizes.
STRUCTURAL = {
    "A": ("class_balance", "sensor_mask_v1", "l2"),
    "B": ("b_residual_v1", "l2sp", "huber", "l2"),
    "C": ("c_group_mse_v1", "huber", "l2"),
}
# Evidence keywords that justify one structural recipe over another.
CAUSE_RECIPE = {
    "sensor_channel": "sensor_mask_v1",
    "label_shift": "class_balance",
    "target_shift": {"A": "class_balance", "B": B_RESIDUAL_RECIPE, "C": "huber"},
    "operating_condition": "c_group_mse_v1",
    "covariate_shift": {"A": "sensor_mask_v1", "B": B_RESIDUAL_RECIPE, "C": "c_group_mse_v1"},
}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _ratio(direction, before, after):
    """Relative improvement of a candidate against its parent, sign-normalized."""
    if not (_finite(before) and _finite(after)) or before == 0:
        return None
    delta = (after - before) if direction == "higher" else (before - after)
    return delta / abs(before)


def _blocked_rows(item):
    """Protected slices this trial broke, with the numbers that broke them."""
    rows = (item.get("local_train_receipt") or {}).get("stability") or []
    return [{"slice": row.get("slice"), "metric": row.get("metric"),
             "before": row.get("before"), "after": row.get("after")}
            for row in rows if row.get("passed") is False]


def trials(prior, direction):
    """Measured training attempts, newest last, with their own gain and step size."""
    rows = []
    for item in prior or []:
        config = item.get("configuration")
        if not isinstance(config, dict) or config.get("action", "train") == "keep":
            continue
        if not _finite(item.get("parent_public_score")):
            continue
        rate, epochs = config.get("rate"), config.get("epochs")
        blocked = _blocked_rows(item)
        rows.append({
            "round": item.get("round"),
            "recipe": config.get("recipe") or "standard",
            "rate": rate if _finite(rate) else None,
            "batch": config.get("batch"),
            "epochs": epochs if _finite(epochs) else None,
            "gain": _ratio(direction, item.get("parent_public_score"), item.get("candidate_public_score")),
            "decision": item.get("decision"),
            "before": item.get("parent_public_score"),
            "after": item.get("candidate_public_score"),
            "strength": config.get("l2sp_strength") or config.get("residual_mean_weight"),
            "blocked": [row["slice"] for row in blocked],
            "blocked_rows": blocked,
        })
    return rows


def rejected_recipes(prior, direction):
    """Structural recipes this task already has evidence against.

    Adoption is the only success signal. A candidate can improve the headline metric and still
    be refused, because a protected slice regressed — that is precisely what a loss trading the
    tail for the bulk looks like. Reading failure as "gain <= 0" misses those rejections and the
    same doomed recipe gets proposed round after round, so both signals count here.

    One exception matters: re-running an already adopted configuration on a converged parent
    loses by construction, and that loss says nothing about the recipe. A recipe already paid
    at exactly this rate/batch/strength is therefore not refuted by that repeat.
    """
    rejected = []
    paid = {}
    for item in trials(prior, direction):
        if item["recipe"] == "standard":
            continue
        scale = (item["rate"], item["batch"], item["strength"])
        if item["decision"] == "adopted":
            paid.setdefault(item["recipe"], set()).add(scale)
            continue
        lost = _finite(item["gain"]) and item["gain"] <= 0
        repeat = scale in paid.get(item["recipe"], ())
        if (item["blocked"] or (lost and not repeat)) and item["recipe"] not in rejected:
            rejected.append(item["recipe"])
    return tuple(rejected)



def best_ever(prior, direction):
    """The best measured candidate so far: the elite this search can fall back to."""
    scored = [(item.get("candidate_public_score"), item) for item in (prior or [])
              if _finite(item.get("candidate_public_score"))]
    if not scored:
        return None, None
    score, record = (max(scored, key=lambda pair: pair[0]) if direction == "higher"
                     else min(scored, key=lambda pair: pair[0]))
    return score, record


def recipe_history(prior, direction):
    """Mean measured gain of each structural recipe on this task."""
    scores = {}
    for item in trials(prior, direction):
        if item["recipe"] == "standard" or not _finite(item["gain"]):
            continue
        scores.setdefault(item["recipe"], []).append(item["gain"])
    return {name: sum(values) / len(values) for name, values in scores.items()}


def demoted_recipes(prior, direction):
    """Recipes whose only evidence is a repeat of a configuration that already paid.

    Such a loss must not blacklist the recipe — re-running an adopted configuration on a
    converged parent loses by construction — but it must not outrank untested and profitable
    directions either. Demote, do not delete.
    """
    paid, repeated = {}, []
    for item in trials(prior, direction):
        if item["recipe"] == "standard":
            continue
        scale = (item["rate"], item["batch"], item["strength"])
        if item["decision"] == "adopted":
            paid.setdefault(item["recipe"], set()).add(scale)
        elif _finite(item["gain"]) and item["gain"] <= 0 and scale in paid.get(item["recipe"], ()):
            if item["recipe"] not in repeated:
                repeated.append(item["recipe"])
    return tuple(repeated)


def structural_priority(task, allowed, prior, direction, cause=None, failed_slices=(), failed=(),
                        demoted=()):
    """Order the structural recipes this batch's evidence actually justifies."""
    available = [name for name in STRUCTURAL.get(task, ()) if name in allowed]
    measured = recipe_history(prior, direction)
    # A recipe already measured as a loss is not a fresh direction; keep it only as a last resort.
    fresh = [name for name in available if name not in failed]
    if fresh:
        available = fresh
    hinted = CAUSE_RECIPE.get(cause or "")
    if isinstance(hinted, dict):
        hinted = hinted.get(task)
    if "low_rul" in failed_slices and task == "B":
        # Short-RUL regression is the signature of a loss that buys the bulk with the tail;
        # `huber` did exactly that on B. The residual-weighted recipe targets the tail directly,
        # so it outranks whatever the drift keyword suggested.
        hinted = B_RESIDUAL_RECIPE
    ordered = []

    def push(name, allow_demoted=False):
        if not name or name not in available or name in ordered:
            return
        # A demoted direction never leads while anything fresher exists.
        if name in demoted and not allow_demoted and any(other not in demoted for other in available):
            return
        ordered.append(name)

    if failed_slices:
        # A protected slice that regressed wants a loss that protects that slice.
        push(hinted)
        for name in available:
            if name not in ("l2",):
                push(name)
    push(hinted)
    for name in sorted(available, key=lambda name: (name in demoted, -measured.get(name, 0.0))):
        if measured.get(name, 0.0) >= 0.0:
            push(name)
    for name in available:
        push(name, allow_demoted=True)
    return ordered


def _params(rate, batch, epochs, recipe, strength=None):
    payload = {"rate": round(float(rate), 6), "batch": int(batch), "epochs": int(epochs), "recipe": recipe}
    if strength is not None and recipe == L2SP_RECIPE:
        payload["l2sp_strength"] = round(float(strength), 4)
    if strength is not None and recipe == B_RESIDUAL_RECIPE:
        payload["residual_mean_weight"] = round(float(strength), 4)
    return payload


def propose(context):
    """Ordered mutations for one round plus the keep fallback.

    The parent arrives already trained (15-20 epochs at 0.002-0.003), so a step at its own
    rate mostly re-fits it. The operator therefore opens at a third of that rate, keeps the
    scale while it pays, halves it after a loss, and only changes the objective once the
    plain step stops paying. Keeping is never proposed before the budget runs out: a
    measured candidate that loses costs compute but not score, because the parent stays.
    """
    task = context["task"]
    direction = context["direction"]
    prior = context.get("prior") or []
    remaining = context.get("remaining") or 0
    allowed = tuple(context.get("allowed_recipes") or ("standard",))
    parent_rate = float(context.get("parent_rate") or 0.002)
    parent_batch = int(context.get("parent_batch") or 128)
    stalled = int(context.get("stalled") or 0)
    measured = [item for item in trials(prior, direction) if _finite(item["gain"])]
    last = measured[-1] if measured else None
    losses = 0
    for item in reversed(measured):
        if item["gain"] <= 0:
            losses += 1
        else:
            break
    probe_rate = max(RATE_MIN, min(parent_rate / 3.0, RATE_MAX))
    step = max(1, int(context.get("round_epochs") or 1))
    recipes = structural_priority(task, allowed, prior, direction,
                                  cause=context.get("cause"), failed_slices=context.get("failed_slices") or (),
                                  failed=tuple(context.get("failed_recipes") or ()),
                                  demoted=tuple(context.get("demoted_recipes") or ()))
    out = []

    def add(name, params, why, expected="improve"):
        if params["epochs"] > remaining:
            return
        out.append({"id": len(out) + 1, "mutation": name, "params": params,
                    "why": why, "expected": expected})

    def add_structural():
        # Risk order: a plain step only changes the scale, a structural recipe changes what the
        # loss optimizes and can win the headline metric while wrecking a protected slice. A
        # single losing step is therefore not enough to buy that risk — the step anneals twice
        # first. An attributed cause or an already-broken slice is real evidence and may skip
        # the queue.
        cause = context.get("cause") or "none"
        proven = cause != "none" or bool(context.get("failed_slices"))
        if not (recipes and (proven or losses >= 2 or stalled >= 2)):
            return
        recipe = recipes[0]
        strength = (1.0 if recipe == L2SP_RECIPE else 0.5 if recipe == B_RESIDUAL_RECIPE else None)
        # The rate stays at the probe scale. A structural experiment is paid for by the objective,
        # and the parent changes every adopted round, so the same configuration on a new parent is
        # a legitimate new experiment (l2 at 0.001 paid three rounds running on A). Shrinking the
        # step to dodge an apparent duplicate weakens exactly what is being tested.
        rate = probe_rate
        add("structural", _params(rate, parent_batch, step, recipe, strength),
            (f"Change the objective to {recipe}: it targets the attributed cause"
             if cause != "none" else
             f"Change the objective to {recipe}: the plain step has stopped paying"),
            "structural")

    if last is None:
        add("probe_step", _params(probe_rate, parent_batch, step, "standard"),
            "No measurement yet: probe below the parent learning rate", "probe")
        add_structural()
    elif last["gain"] > 0:
        # Repeating the exact scale that worked overshoots on the next step (B round 2 lost
        # 40 RMSE that way), so the scale decays gently while the direction still pays.
        rate = max(RATE_MIN, (last["rate"] or probe_rate) * 0.6)
        add("continue_step", _params(rate, parent_batch, step, "standard"),
            "Last step improved accuracy: continue at a slightly smaller scale", "continue")
        add_structural()
    else:
        # Step size anneals with the loss streak, not with a single bad round: one loss
        # usually means the step was too long. A structural attempt also has to give the
        # plain step another turn, otherwise a losing recipe blocks the useful direction.
        rate = max(RATE_MIN, probe_rate * (0.5 ** min(losses, 3)))
        last_structural = last["recipe"] != "standard"
        if (stalled >= 1 or losses >= 1) and not last_structural:
            add_structural()
            add("safe_step", _params(rate, parent_batch, step, "standard"),
                "Last step lost accuracy: halve it before trying again", "small")
        else:
            add("safe_step", _params(rate, parent_batch, step, "standard"),
                "Last step lost accuracy: halve it before trying again", "small")
            add_structural()
    best_score, best = best_ever(prior, direction)
    # Replaying the elite configuration only pays when the parent is not already that elite:
    # re-training a converged parent at its own scale is the exact move that lost 28 RMSE on B.
    adopted_rounds = [item.get("round") for item in prior if item.get("decision") == "adopted"]
    elite_is_parent = (best is not None and best.get("decision") == "adopted"
                       and adopted_rounds and best.get("round") == max(adopted_rounds))
    if best is not None and stalled >= 2 and not elite_is_parent:
        config = best.get("configuration") or {}
        if config.get("rate"):
            add("elite_replay", _params(config["rate"], config.get("batch") or parent_batch, step,
                                        config.get("recipe") or "standard",
                                        config.get("l2sp_strength") or config.get("residual_mean_weight")),
                f"Replay round {best.get('round')}'s best measured configuration on the current parent",
                "rollback")
    out.append({"id": len(out) + 1, "mutation": "keep", "params": {"action": "keep"},
                "why": "Only when the budget is gone or no attempt is worth its compute",
                "expected": "flat"})
    return rank(out, context)


def rank(candidates, context):
    """Order the list: keep last, a known-losing recipe behind its peers."""
    failed = set(context.get("failed_recipes") or ())
    rows = []
    for position, item in enumerate(candidates):
        if item["mutation"] == "keep":
            item["rank_score"] = -1.0
            rows.append((2, position, item))
            continue
        recipe = (item["params"] or {}).get("recipe")
        # A recipe already measured as a loss is still worth its compute only behind the rest.
        # The default objective is never downgraded: its step is adjusted instead.
        losing = recipe in failed and recipe != "standard"
        item["rank_score"] = -0.02 if losing else 0.004
        rows.append((1 if losing else 0, position, item))
    return [item for _, _, item in sorted(rows, key=lambda row: (row[0], row[1]))]


def top_trainable(candidates, min_score=-0.005):
    """The best candidate worth spending a round on, or None when keeping is cheaper."""
    for item in candidates:
        if item["mutation"] == "keep":
            continue
        if item.get("rank_score", 0.0) >= min_score:
            return item
    return None


def to_experiment(candidate):
    """Turn one candidate into a validated experiment proposal."""
    if candidate["params"].get("action") == "keep":
        return Experiment(action="keep", reason=candidate["why"][:200])
    return Experiment(action="train", reason=candidate["why"][:200], **candidate["params"])


def resolve_choice(text, candidates):
    """Accept {"candidate": k} as a decision; the model no longer retypes parameters."""
    if not candidates or not isinstance(text, str):
        return None
    body = None
    start, end = text.find("{"), text.rfind("}")
    for piece in (text.strip(), text[start:end + 1] if start >= 0 and end > start else ""):
        if not piece:
            continue
        try:
            import json
            body = json.loads(piece)
            break
        except (ValueError, TypeError):
            continue
    if not isinstance(body, dict):
        return None
    index = body.get("candidate", body.get("choice"))
    if type(index) is not int or not 1 <= index <= len(candidates):
        return None
    chosen = candidates[index - 1]
    try:
        plan = to_experiment(chosen)
    except Exception:
        return None
    reason = body.get("reason")
    if isinstance(reason, str) and reason.strip():
        plan.reason = reason.strip()[:200]
    return plan, chosen


def brief(candidates):
    """Prompt-sized view: ids, mutations and parameters, one short clause each."""
    return [{"id": item["id"], "mutation": item["mutation"], "params": item["params"],
             "why": item["why"][:60]} for item in candidates]
