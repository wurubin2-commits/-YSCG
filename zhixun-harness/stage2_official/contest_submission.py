"""Contest workspace v2 callbacks. The host owns the data, model gateway and score."""

from collections import Counter
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from tempfile import mkdtemp

from contest_harness.errors import HarnessError
from pydantic import ValidationError

from stage1_local.common import ROOT, sha256
from stage2_official import admission, attribution, evolution
from stage2_official.continuation import B_RESIDUAL_RECIPE, SENSOR_MASK_RECIPE
from stage2_official.normalize import normalize_proposal, rule_plan
from stage2_official.operating_conditions import C_GROUP_RECIPE
from stage2_official.policy import (
    Experiment, OUTPUT_TOKEN_LIMIT, PROMPT, RECIPES, already_trained, c_half_rate, compact_score,
    effective_parameters, experiment_key, history_feedback,
)
from stage2_official.tool_layer import TOOL_SPECS, ToolBox, calls_of, latest_receipt, run_tool_rounds
from stage2_official.trainer import baseline

TASKS = ("classification", "regression")
CONTRACTS = {"A": ("classification", "macro_f1", "higher"),
             "B": ("regression", "rmse", "lower"), "C": ("regression", "rmse", "lower")}
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
# The whole-round budget stays at 12 by default; the official continuation example spends
# five epochs per round instead, which needs both a larger total and a larger per-round step.
MAX_EPOCHS = int(os.environ.get("ZHIXUN_MAX_EPOCHS", "12"))
ROUND_EPOCHS = int(os.environ.get("ZHIXUN_ROUND_EPOCHS", "1"))


def read_object(path):
    return json.loads(path.read_text(encoding="utf-8"))


def parent_checkpoint(rnd):
    names = rnd.parent.descriptor["checkpoint_paths"]
    expected = rnd.parent.checkpoint_sha256.removeprefix("sha256:")
    for name in names:
        path = rnd.parent.path(name)
        if path.suffix == ".npz" and sha256(path).removeprefix("sha256:") == expected:
            return path
    raise HarnessError("PARENT_DIGEST_MISMATCH", "No declared matching NumPy checkpoint")


def summarize(rows, kind):
    # The official projection may omit meta. Never infer grouping from anonymous IDs.
    seen, labels = set(), Counter()
    target_total = 0.0
    for row in rows:
        key, answer = row["id"], row.get("answer")
        if not key or key in seen:
            raise ValueError("Empty or repeated training ID")
        seen.add(key)
        if kind == "classification":
            if not isinstance(answer, str) or not answer:
                raise ValueError("Expected a public class label")
            labels[answer] += 1
        else:
            if number(answer) is None or answer < 0:
                raise ValueError("Expected finite nonnegative public RUL")
            target_total += answer
    if not seen:
        raise ValueError("Empty training projection")
    return {"rows": len(seen), **({"labels": dict(labels)} if kind == "classification"
                                 else {"target_mean": target_total / len(seen)})}


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def official_score(summary, metric):
    if isinstance(summary, dict):
        values = summary.get("metrics", summary)
        if isinstance(values, dict):
            return number(values.get(metric))
    return None


def condition_counts(path, model, task):
    """Operating-condition mix of C training rows; None when the columns are absent."""
    try:
        from stage2_official.operating_conditions import OTHER, operating_groups
        data = baseline(task, "data").read_dataset(path, model.meta["task"], model.meta["classes"],
                                                   model.meta["paths"])
        groups = operating_groups(data["x"], model.meta["paths"])
        counts = [0] * (OTHER + 1)
        for value in groups.tolist():
            counts[value] += 1
        return {str(index): count for index, count in enumerate(counts) if count > 0}
    except Exception:  # noqa: BLE001 - a diagnostic must never block the training round
        return None


def assess(rnd, report, submitted, metric, direction):
    parent = official_score(rnd.frame.get("current_validation"), metric)
    candidate = official_score(submitted.get("summary"), metric)
    local_parent = number(report["validation_before"][metric])
    local_candidate = number(report["validation_after"][metric])
    if parent is None and candidate is not None and local_candidate is not None:
        # Same implementation, different BLAS order: 1e-8 relative tolerance never matches on
        # a platform whose float summation differs from the local one, which silently turned
        # every decision into "no comparable public scores". A 0.1% band still proves the two
        # measurements come from the same weights without rejecting real agreement.
        tolerance = max(1e-9, abs(candidate) * 1e-3)
        if math.isclose(candidate, local_candidate, rel_tol=1e-3, abs_tol=tolerance):
            parent = local_parent
            source = "platform candidate cross-checked against the same local public validation"
        else:
            source = "missing comparable parent score"
    else:
        source = "same-round platform measurement"
    if parent is None or candidate is None:
        return False, parent, candidate, "no comparable public scores"
    better = candidate > parent + 1e-12 if direction == "higher" else candidate < parent - 1e-12
    return better, parent, candidate, source


def scratch_workspace():
    """Pick a writable scratch root; the graded root filesystem is read-only."""
    for candidate in (os.environ.get("CONTEST_WORK"), "/workspace", "/tmp"):
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_dir() and os.access(path, os.W_OK):
            return path
    raise ValueError("No writable scratch directory is available")


def train(rnd):
    started = time.monotonic()
    task = rnd.dataset_key
    kind, metric, direction = CONTRACTS[task]
    if (rnd.task.get("task"), rnd.task.get("metric"), rnd.task.get("metric_direction")) != (kind, metric, direction):
        raise ValueError("Unexpected task contract")
    workspace = scratch_workspace()
    directory = Path(mkdtemp(prefix=f"{task}-{rnd.round_index:02d}-", dir=workspace))
    path = parent_checkpoint(rnd)
    model = baseline(task, "model").Model.load(path)
    if model.meta["profile"] != task or model.meta["task"] != kind:
        raise ValueError("Parent checkpoint does not match the task")
    unit = {"B": "minutes", "C": "cycles"}.get(task)
    if unit and (rnd.task.get("target_unit", unit) != unit or model.meta.get("target_unit", unit) != unit):
        raise ValueError("Unexpected regression unit")
    if task == "B" and model.meta.get("target_unit") != unit:
        raise ValueError("B must use the bearing minutes checkpoint")
    seed = model.meta["seed"]
    history = rnd.history()
    state = rnd.state_workspace(directory / "state")
    evidence = state / "rounds.json"
    prior = read_object(evidence) if evidence.exists() else []
    if (not isinstance(prior, list) or any(item["task"] != task or item["round"] >= rnd.round_index for item in prior)):
        raise ValueError("Unexpected state for this task and round")
    remaining = max(0, MAX_EPOCHS - sum(item["epochs_attempted"] for item in prior))
    research = os.environ.get("ZHIXUN_RESEARCH_MODE") == "1"
    # Structural recipes change what the loss optimizes, which is the only lever that moves
    # validation behaviour inside a ten-round budget; a plain step mostly re-tunes a
    # converged parent. They stay available by default, and the attribution step narrows
    # the list to the causes this batch justifies.
    recipes = RECIPES[task] if os.environ.get("ZHIXUN_NARROW_RECIPES") != "1" else ("standard",)
    l2sp_enabled = task == "B" and os.environ.get("ZHIXUN_ENABLE_L2SP") == "1"
    sensor_mask_enabled = task == "A" and os.environ.get("ZHIXUN_ENABLE_SENSOR_MASK") == "1"
    b_residual_enabled = task == "B" and os.environ.get("ZHIXUN_ENABLE_B_RESIDUAL") == "1"
    c_group_mse_enabled = task == "C" and os.environ.get("ZHIXUN_ENABLE_C_GROUP_MSE") == "1"
    if l2sp_enabled:
        recipes += ("l2sp",)
    if sensor_mask_enabled:
        recipes += (SENSOR_MASK_RECIPE,)
    if b_residual_enabled:
        recipes += (B_RESIDUAL_RECIPE,)
    if c_group_mse_enabled:
        recipes += (C_GROUP_RECIPE,)
    rows = rnd.data.training_rows()
    diagnose_on = os.environ.get("ZHIXUN_DIAGNOSE") == "1"
    train_features, sample = None, None
    if research:
        from stage2_official.drift import sample_rows
        rows, sample = sample_rows(rows)
    elif diagnose_on:
        from stage2_official.diagnostics import collect_features
        rows, train_features = collect_features(rows, baseline(task, "data").flatten)
    summary_started = time.monotonic()
    batch = summarize(rows, kind)
    current_distribution = shift = None
    if research:
        from stage2_official.drift import drift_score, histogram
        current_distribution = histogram(sample, model, baseline(task, "data").flatten)
        shift = drift_score(prior[-1].get("feature_histogram") if prior else None, current_distribution)
        if diagnose_on:
            # sample is filled only while summarize consumed the stream.
            flatten_inputs = baseline(task, "data").flatten
            train_features = []
            for inputs in sample:
                try:
                    train_features.append(flatten_inputs(inputs)[1])
                except (TypeError, ValueError):
                    train_features = None
                    break
    input_drift = None
    if diagnose_on and train_features:
        try:
            from stage2_official.diagnostics import diagnose, validation_matrix
            input_drift = diagnose(train_features, validation_matrix(task, rnd.data.validation_path, model))
        except (ValueError, OSError, KeyError):
            input_drift = None
    diagnosis = None
    if os.environ.get("ZHIXUN_ATTRIBUTION", "1") != "0":
        counts = condition_counts(rnd.data.train_path, model, task) if task == "C" else None
        diagnosis = attribution.attribute(task, batch, classes=model.meta.get("classes") or (),
                                          drift=input_drift, parent_target_mean=model.meta.get("target_mean"),
                                          condition_counts=counts)
        # Only recipes the attributed cause justifies are added; order is preserved.
        recipes = attribution.merge_recipes(recipes, diagnosis["recipes"])
    # Consecutive rounds since the last adopted candidate: the search has stopped paying off.
    stalled = 0
    for item in reversed(prior):
        if item.get("decision") == "adopted":
            break
        stalled += 1
    parent_sha = sha256(path)
    train_sha, validation_sha = sha256(rnd.data.train_path), sha256(rnd.data.validation_path)
    parent_score = official_score(rnd.frame.get("current_validation"), metric)
    if parent_score is None:
        # Reuse a score only for the exact same checkpoint and validation projection.
        for item in reversed(prior):
            if item.get("validation_sha256") == validation_sha and item["selected_artifact"] == rnd.parent.artifact_id:
                parent_score = item.get("candidate_public_score" if item["decision"] == "adopted"
                                        else "parent_public_score")
                if parent_score is not None:
                    break
    prompt = {"task": task, "metric": metric, "direction": direction, "round": rnd.round_index,
              "parent_score": compact_score(parent_score),
              "parent_params": {"rate": model.meta["learning_rate"], "batch": model.meta["batch_size"]},
              "measured_trials": sum(item.get("candidate_public_score") is not None for item in prior),
              "stalled_rounds": stalled,
              "batch": {**batch}, "remaining_epochs": remaining,
              "results": history_feedback(prior, history, parent_sha, train_sha, rnd.round_index)}
    if unit:
        prompt["unit"] = unit
        prompt["batch"]["target_mean"] = compact_score(batch["target_mean"])
    instructions = PROMPT + " Scores are [parent,candidate]. Never repeat same_batch parameters."
    # The mutation operator, not the model's imagination, decides what is worth trying. A
    # converged parent loses accuracy under a normal step, and a ten-round budget only pays
    # for structural changes: so the evidence is turned into an ordered short list here.
    measured_trials = evolution.trials(prior, direction)
    # Adoption is the only success signal. A recipe that lost the headline metric, or that won it
    # while breaking a protected slice, is refuted on this task; treating only "gain <= 0" as
    # failure lets a rejected recipe be re-proposed until the budget runs out.
    failed_recipes = evolution.rejected_recipes(prior, direction)
    # A recipe whose only loss was a repeat of an adopted configuration is still usable; it just
    # stops leading the queue, because the profitable direction deserves the round.
    demoted_recipes = evolution.demoted_recipes(prior, direction)
    failed_slices = []
    blocked_evidence = []
    # A keep round trains nothing, so its record carries no stability receipt. Walk back to
    # the most recent candidate instead of only reading the immediate predecessor, matching
    # how latest_receipt() resolves tool evidence.
    for item in reversed(prior):
        stability = (item.get("local_train_receipt") or {}).get("stability") or []
        if stability:
            failed_slices = [row.get("slice") for row in stability if row.get("passed") is False]
            blocked_evidence = [{"round": item.get("round"), "slice": row.get("slice"),
                                 "metric": row.get("metric"),
                                 "before": compact_score(row.get("before")),
                                 "after": compact_score(row.get("after"))}
                                for row in stability if row.get("passed") is False]
            break
    # The step scale must not follow the parent's own rate, because every candidate writes
    # its rate back into the checkpoint meta: anchoring on it makes the probe collapse to
    # the floor round by round. Anchor on the first observed parent instead.
    base_rate = next((item.get("parent_learning_rate") for item in prior
                      if number(item.get("parent_learning_rate")) is not None), None)
    candidates = evolution.propose({
        "task": task, "direction": direction, "parent_rate": base_rate or model.meta["learning_rate"],
        "round_epochs": ROUND_EPOCHS,
        "parent_batch": model.meta["batch_size"], "remaining": remaining,
        "allowed_recipes": recipes, "prior": prior, "stalled": stalled,
        "cause": (diagnosis or {}).get("cause"), "failed_slices": failed_slices,
        "failed_recipes": failed_recipes, "demoted_recipes": demoted_recipes})
    prompt["candidates"] = evolution.brief(candidates)
    if candidates:
        instructions += (" Answer with a candidate id when one of them is justified; keeping"
                         " without a measured reason spends the round for nothing.")
    # A refusal is evidence, but only if the model can see it. Without these the same objective
    # gets proposed again: the headline metric improved, so nothing in the summary says "no".
    if failed_recipes:
        prompt["failed_recipes"] = list(failed_recipes)
        instructions += (" Recipes in failed_recipes were already refuted on this task: do not"
                         " choose them again, whichever step size the candidate carries.")
    if blocked_evidence:
        prompt["blocked_slices"] = blocked_evidence
        instructions += (" blocked_slices lists the protected slice that regressed despite a better"
                         " headline score: change the objective to protect it, do not repeat it.")
    floors = admission.slice_floors(prior)
    if floors:
        prompt["slice_floors"] = {name: compact_score(value) for name, value in floors.items()}
    # Token cost is not the binding constraint on this line: the harness exists so a real model
    # can look at its own last training receipt before deciding. The read tools therefore ride on
    # every round that has evidence, not only on rounds where the operator ran out of plans.
    last_outcome = prior[-1].get("decision") if prior else None
    # The current round has not trained yet: the informative receipt and loss curve belong to
    # the previous trial. A keep round trains nothing, so the latest receipt can be several
    # rounds back; latest_receipt() walks back to the most recent candidate.
    receipt_dir = latest_receipt(directory, rnd.round_index)
    evidence_available = receipt_dir is not None
    tools_wanted = evidence_available
    if tools_wanted:
        # Only the read tools are offered: the final JSON proposal already carries the
        # hyper-parameters, so a separate write_config round would only cost another sample.
        instructions += (" Read the evidence you need with read_metrics or read_log_tail before"
                         " deciding, then give the JSON decision.")
    if stalled >= 2:
        # A plateau means another step at the same scale buys nothing. Keeping is not a
        # direction change: it leaves the score exactly where it is, so ask for a real one.
        instructions += (f" The last {stalled} rounds produced no improvement: the parent is likely"
                         " converged and another step at the same scale buys nothing. Change the"
                         " objective or the step size; keep only if the evidence says every"
                         " candidate would lose accuracy.")
    if research:
        prompt.update(allowed_recipes=recipes, batch_shift=shift)
        instructions += " Train may include an allowed recipe."
    if l2sp_enabled:
        prompt["allowed_recipes"] = recipes
        instructions += " Recipe l2sp requires l2sp_strength in (0,10]."
    if sensor_mask_enabled:
        prompt["allowed_recipes"] = recipes
        instructions += " Train may use recipe sensor_mask_v1."
    if b_residual_enabled:
        prompt["allowed_recipes"] = recipes
        instructions += " Recipe b_residual_v1 requires residual_mean_weight in (0,1]."
    if c_group_mse_enabled:
        prompt["allowed_recipes"] = recipes
        instructions += " Train may use recipe c_group_mse_v1."
    if diagnosis is not None and diagnosis["recipes"]:
        prompt["attribution"] = {"cause": diagnosis["cause"], "channels": diagnosis["channels"],
                                 "evidence": {key: value for key, value in diagnosis["evidence"].items()
                                              if key != "condition_counts"}}
        prompt["allowed_recipes"] = recipes
        instructions += f" Attribution of this batch suggests {', '.join(diagnosis['recipes'])}."
    messages = [{"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, separators=(",", ":"),
                                                       allow_nan=False)}]
    summary_seconds = time.monotonic() - summary_started
    token_limit = min(OUTPUT_TOKEN_LIMIT, rnd.max_output_tokens)
    model_started = time.monotonic()
    toolbox = ToolBox(receipt_dir if evidence_available else directory, write_root=directory)
    holder = {"receipt": None}

    # Reasoning models spend the whole budget on a draft before writing any answer. The SDK whitelists
    # this option, so the decision stays a direct answer instead of a truncated thought process.
    # Tools ride on the same call: the model may read metrics or log tails and write one bounded
    # config patch before it answers. A host that rejects the option falls back to a plain call.
    read_specs = [spec for spec in TOOL_SPECS if spec["function"]["name"] != "write_config"]
    def invoke(history):
        options = dict(max_tokens=token_limit, temperature=0, enable_thinking=False)
        if tools_wanted:
            options.update(tools=read_specs, tool_choice="auto")
        try:
            receipt = rnd.model(history, **options)
        except HarnessError:
            receipt = rnd.model(history, max_tokens=token_limit, temperature=0, enable_thinking=False)
        holder["receipt"] = receipt
        return {"text": getattr(receipt, "text", "") or "", "tool_calls": calls_of(receipt)}

    def finish(history):
        # The model spent the whole tool budget without answering: force one tool-free call
        # so the round still produces a parseable decision instead of an empty string.
        try:
            receipt = rnd.model(history, max_tokens=token_limit, temperature=0, enable_thinking=False)
        except HarnessError:
            return {"text": "", "tool_calls": []}
        holder["receipt"] = receipt
        return {"text": getattr(receipt, "text", "") or "", "tool_calls": []}

    reply_text, tool_trace = run_tool_rounds(invoke, messages, toolbox, finish=finish)
    receipt = holder["receipt"]
    model_seconds = time.monotonic() - model_started

    selected = rnd.parent.artifact_id
    hypothesis, change = "Check whether a bounded update is justified", "Parent checkpoint retained"
    status, spend, config = "invalid_suggestion", 0, {}
    report, candidate_id, candidate_score = None, None, None
    admission_record = None
    # Only the host's same-round score is official decision evidence.
    parent_score = official_score(rnd.frame.get("current_validation"), metric)
    key = proposed_rate = None
    chosen = None
    trial = directory / "trial"
    reason = "The model returned a suggestion outside the accepted schema."
    normalization = None
    choice = evolution.resolve_choice(reply_text or "", candidates)
    if choice is not None:
        proposal, chosen = choice
        normalization = {"source": "candidate", "clamped": {}, "raw_action": None}
    else:
        try:
            proposal = Experiment.model_validate_json(reply_text or "")
        except ValidationError:
            proposal, normalization = normalize_proposal(reply_text or "", recipes, task)
    if proposal is None:
        # A reply outside the schema is not a reason to fall back to the rule table: the
        # evidence-backed candidate is already the better plan, and spending a second
        # sample to re-ask would cost tokens without changing what the evidence says.
        fallback = evolution.top_trainable(candidates)
        if fallback is not None:
            proposal, chosen = evolution.to_experiment(fallback), fallback
            normalization = {"source": "candidate_fallback", "clamped": {}, "raw_action": None}
        else:
            proposal, source = rule_plan(task, prior, remaining)
            if proposal is not None:
                normalization = {"source": source, "clamped": {}, "raw_action": None}
            else:
                reason = "The model returned an unusable suggestion and no rule plan was available."
    elif proposal.action == "keep":
        # A keep on a plateau spends the only budget that could still move the score, so the
        # evidence-backed candidate wins unless the round is genuinely out of room.
        forced = (evolution.top_trainable(candidates)
                  if remaining > 0 and (stalled >= 2 or last_outcome in ("candidate_failed", "training_failed"))
                  else None)
        if forced is not None:
            proposal, chosen = evolution.to_experiment(forced), forced
            normalization = {"source": "candidate_forced", "clamped": {}, "raw_action": "keep"}
        else:
            replacement, source = rule_plan(task, prior, remaining)
            if replacement is not None:
                normalization = {"source": source, "clamped": {}, "raw_action": "keep"}
                proposal = replacement
    if normalization is not None and normalization.get("source") == "model":
        normalization = None
    if proposal is not None and proposal.action == "train" and proposal.recipe not in recipes:
        # A recipe the operator did not offer cannot be trained. Take the best offered plan
        # instead of burning the round on an invalid proposal.
        fallback = evolution.top_trainable(candidates)
        if fallback is not None:
            proposal, chosen = evolution.to_experiment(fallback), fallback
            normalization = {"source": "candidate_fallback", "clamped": {}, "raw_action": "invalid_recipe"}
    if proposal is not None:
        if task == "C" and os.environ.get("ZHIXUN_C_HALF_RATE") == "1":
            proposed_rate = proposal.rate
            proposal = c_half_rate(proposal)
        hypothesis = proposal.reason
        config = proposal.model_dump(exclude_none=True)
        if proposal.action == "train":
            key = experiment_key(task, parent_sha, train_sha, seed, effective_parameters(proposal))
        if proposal.action == "keep":
            status, reason = "model_keep", "The model advised retaining the current parent."
        elif proposal.recipe not in recipes:
            status, reason = "invalid_suggestion", "The proposed recipe is unavailable for this task."
        elif already_trained(prior, key):
            status, reason = "duplicate_keep", "The same parent, training data, seed and effective parameters already ran."
        elif proposal.epochs > remaining:
            status, reason = "budget_keep", "The requested epochs exceed the remaining allocation."
        else:
            available = min(60.0, (rnd.wall_seconds or 60.0) - (time.monotonic() - started) - 5.0)
            if available <= 0:
                status, reason = "time_keep", "The round is too close to its time limit to attempt training."
            else:
                spend = proposal.epochs
                try:
                    command = [sys.executable, "-B", "-m", "stage2_official.trainer", task,
                               str(rnd.data.train_path), str(rnd.data.validation_path), str(path), str(trial),
                               f"{task}-{rnd.round_index}", str(proposal.rate), str(proposal.batch), str(spend),
                               "--recipe", proposal.recipe]
                    if proposal.l2sp_strength is not None:
                        command.extend(["--l2sp-strength", str(proposal.l2sp_strength)])
                    if proposal.residual_mean_weight is not None:
                        command.extend(["--residual-mean-weight", str(proposal.residual_mean_weight)])
                    subprocess.run(command,
                                   stdin=subprocess.DEVNULL, stdout=sys.stderr, stderr=sys.stderr,
                                   cwd=ROOT, timeout=available, check=True)
                    report = read_object(trial / "training.json")
                    if report["parent_checkpoint_sha256"].removeprefix("sha256:") != parent_sha.removeprefix("sha256:"):
                        raise ValueError("Wrong parent in trainer receipt")
                    if (report["checkpoint_sha256"] != sha256(trial / "checkpoint.npz")
                            or report["final_steps"] <= model.meta["steps"]
                            or report["train_sha256"] != train_sha
                            or report["validation_sha256"] != validation_sha):
                        raise ValueError("The trainer produced an invalid continuation")
                except (subprocess.SubprocessError, OSError, ValueError) as error:
                    status, reason = "training_failed", f"Continuation failed: {type(error).__name__}."
                    report = None
                if report is not None:
                    change = (f"Ran {spend} CPU epoch(s) with {proposal.recipe}, "
                              f"learning rate {proposal.rate}, batch {proposal.batch}.")
                    try:
                        submitted = rnd.submit_candidate(trial, hypothesis=hypothesis, change_summary=change,
                                                         configuration=config, seed=seed,
                                                         checkpoint_paths=["checkpoint.npz"],
                                                         predict_argv=PREDICT_ARGV)
                    except HarnessError as error:
                        status, reason = "candidate_failed", f"Candidate evaluation failed: {error.code}."
                    else:
                        candidate_id = submitted["artifact_id"]
                        better, parent_score, candidate_score, source = assess(rnd, report, submitted, metric, direction)
                        blocked = [row["slice"] for row in report.get("stability", []) if not row["passed"]]
                        admitted, gate, gate_reason = admission.evaluate(
                            prior, report.get("stability", []), direction, parent_score, candidate_score,
                            model=model, path=path,
                            load=lambda item: baseline(task, "model").Model.load(item))
                        if better and blocked:
                            status = "stability_keep"
                            reason = "Parent retained after degradation on public slices: " + ", ".join(blocked)
                        elif better and not admitted:
                            status = "measured_keep"
                            reason = f"Admission gate '{gate}' kept the parent: {gate_reason}."
                            admission_record = {"gate": gate, "reason": gate_reason, "direction": direction}
                        elif better:
                            selected = candidate_id
                            status, reason = "adopted", f"Candidate adopted based on {source}."
                            if gate != "disabled":
                                admission_record = {"gate": gate, "reason": gate_reason, "direction": direction}
                        else:
                            status, reason = "measured_keep", f"Parent retained based on {source}."

    usage = (getattr(receipt, "response", None) or {}).get("usage") or {}
    token_counts = {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if type(usage.get(key)) is int}
    record = {"task": task, "round": rnd.round_index, "batch": batch, "decision": status,
              "batch_shift": shift, "feature_histogram": current_distribution,
              "input_drift": input_drift, "attribution": diagnosis,
              "reason": reason, "hypothesis": hypothesis, "configuration": config, "seed": seed,
              "parent_artifact": rnd.parent.artifact_id, "candidate_artifact": candidate_id, "selected_artifact": selected,
              "parent_sha256": parent_sha, "candidate_sha256": report["checkpoint_sha256"] if report else None,
              "experiment_key": key, "train_sha256": train_sha, "validation_sha256": validation_sha,
              "metric": metric, "parent_public_score": parent_score, "candidate_public_score": candidate_score,
              "parent_learning_rate": model.meta["learning_rate"],
              "gain": ((candidate_score - parent_score) * (1 if direction == "higher" else -1)
                       if parent_score is not None and candidate_score is not None else None),
              "local_train_receipt": report, "epochs_attempted": spend, "model": rnd.model_name,
              "model_call_ids": list(rnd.call_ids), "history_ids": list(rnd.history_ids), "usage": token_counts,
              "prompt_bytes": sum(len(item["content"].encode("utf-8")) for item in messages),
              "output_token_limit": token_limit, "reasoning_disabled": True, "stalled_rounds": stalled,
              "tool_trace": tool_trace,
              "evolution": {"candidates": [{"id": item["id"], "mutation": item["mutation"],
                                            "params": item["params"]} for item in candidates],
                            "chosen": chosen["mutation"] if chosen else None,
                            "forced": bool(normalization and normalization.get("source") in
                                           ("candidate_forced", "candidate_fallback")),
                            "failed_slices": failed_slices, "failed_recipes": failed_recipes},
              "feedback_rounds": [row["round"] for row in prompt["results"]],
              "summary_seconds": round(summary_seconds, 4), "model_seconds": round(model_seconds, 4),
              "elapsed_seconds": round(time.monotonic() - started, 4)}
    if admission_record is not None:
        record["admission"] = admission_record
    if normalization is not None:
        record["proposal_normalization"] = {key: normalization[key] for key in
                                            ("source", "clamped", "raw_action") if key in normalization}
    if proposed_rate is not None and proposed_rate != config.get("rate"):
        record["proposed_learning_rate"] = proposed_rate
    for item in prior:
        item.pop("feature_histogram", None)
    evidence.write_text(json.dumps([*prior, record], ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                        encoding="utf-8")
    rnd.save_state(state)
    rnd.finish(selected_artifact_id=selected, hypothesis=hypothesis, change_summary=change, reason=reason,
               configuration=config, seed=seed, history_ids=list(rnd.history_ids), model_call_ids=list(rnd.call_ids))


@lru_cache(maxsize=2)
def loaded(path, dataset_key):
    if dataset_key is None:
        import numpy as np

        with np.load(path, allow_pickle=False) as arrays:
            dataset_key = json.loads(str(arrays["metadata"]))["profile"]
    if dataset_key not in CONTRACTS:
        raise ValueError("Unknown dataset for selected checkpoint")
    model = baseline(dataset_key, "model").Model.load(path)
    if model.meta["profile"] != dataset_key or model.meta["task"] != CONTRACTS[dataset_key][0]:
        raise ValueError("Selected checkpoint does not match the prediction context")
    return dataset_key, model


def predict(sample, model_dir):
    root = Path(model_dir).resolve()
    path = Path(os.environ.get("CONTEST_ARTIFACT_PATH", str(root / "checkpoint.npz")))
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Selected checkpoint must be mounted inside the model directory")
    task, model = loaded(path.resolve(), sample.dataset_key)
    if sample.task != CONTRACTS[task][0]:
        raise ValueError("Selected checkpoint does not match requested task")
    return baseline(task, "predict").predict(model, sample.inputs)
