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
from stage2_official.continuation import B_RESIDUAL_RECIPE, SENSOR_MASK_RECIPE
from stage2_official.operating_conditions import C_GROUP_RECIPE
from stage2_official.policy import (
    Experiment, OUTPUT_TOKEN_LIMIT, PROMPT, RECIPES, already_trained, c_half_rate, compact_score,
    effective_parameters, experiment_key, history_feedback,
)
from stage2_official.trainer import baseline

TASKS = ("classification", "regression")
CONTRACTS = {"A": ("classification", "macro_f1", "higher"),
             "B": ("regression", "rmse", "lower"), "C": ("regression", "rmse", "lower")}
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
MAX_EPOCHS = 12


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


def assess(rnd, report, submitted, metric, direction):
    parent = official_score(rnd.frame.get("current_validation"), metric)
    candidate = official_score(submitted.get("summary"), metric)
    local_parent = number(report["validation_before"][metric])
    local_candidate = number(report["validation_after"][metric])
    if parent is None and candidate is not None and local_candidate is not None:
        if math.isclose(candidate, local_candidate, rel_tol=1e-8, abs_tol=1e-8):
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


def train(rnd):
    started = time.monotonic()
    task = rnd.dataset_key
    kind, metric, direction = CONTRACTS[task]
    if (rnd.task.get("task"), rnd.task.get("metric"), rnd.task.get("metric_direction")) != (kind, metric, direction):
        raise ValueError("Unexpected task contract")
    workspace = Path(os.environ.get("CONTEST_WORK", "/workspace"))
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
    recipes = RECIPES[task] if research else ("standard",)
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
    if research:
        from stage2_official.drift import drift_score, histogram, sample_rows
        rows, sample = sample_rows(rows)
    summary_started = time.monotonic()
    batch = summarize(rows, kind)
    current_distribution = shift = None
    if research:
        current_distribution = histogram(sample, model, baseline(task, "data").flatten)
        shift = drift_score(prior[-1].get("feature_histogram") if prior else None, current_distribution)
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
              "batch": {**batch}, "remaining_epochs": remaining,
              "results": history_feedback(prior, history, parent_sha, train_sha, rnd.round_index)}
    if unit:
        prompt["unit"] = unit
        prompt["batch"]["target_mean"] = compact_score(batch["target_mean"])
    instructions = PROMPT + " Scores are [parent,candidate]. Never repeat same_batch parameters."
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
    messages = [{"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, separators=(",", ":"),
                                                       allow_nan=False)}]
    summary_seconds = time.monotonic() - summary_started
    token_limit = min(OUTPUT_TOKEN_LIMIT, rnd.max_output_tokens)
    model_started = time.monotonic()
    reply = rnd.model(messages, max_tokens=token_limit, temperature=0)
    model_seconds = time.monotonic() - model_started

    selected = rnd.parent.artifact_id
    hypothesis, change = "Check whether a bounded update is justified", "Parent checkpoint retained"
    status, spend, config = "invalid_suggestion", 0, {}
    report, candidate_id, candidate_score = None, None, None
    # Only the host's same-round score is official decision evidence.
    parent_score = official_score(rnd.frame.get("current_validation"), metric)
    key = proposed_rate = None
    trial = directory / "trial"
    reason = "The model returned a suggestion outside the accepted schema."
    try:
        proposal = Experiment.model_validate_json(reply.text or "")
    except ValidationError:
        proposal = None
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
                        if better and blocked:
                            status = "stability_keep"
                            reason = "Parent retained after degradation on public slices: " + ", ".join(blocked)
                        elif better:
                            selected = candidate_id
                            status, reason = "adopted", f"Candidate adopted based on {source}."
                        else:
                            status, reason = "measured_keep", f"Parent retained based on {source}."

    usage = reply.response.get("usage") or {}
    token_counts = {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if type(usage.get(key)) is int}
    record = {"task": task, "round": rnd.round_index, "batch": batch, "decision": status,
              "batch_shift": shift, "feature_histogram": current_distribution,
              "reason": reason, "hypothesis": hypothesis, "configuration": config, "seed": seed,
              "parent_artifact": rnd.parent.artifact_id, "candidate_artifact": candidate_id, "selected_artifact": selected,
              "parent_sha256": parent_sha, "candidate_sha256": report["checkpoint_sha256"] if report else None,
              "experiment_key": key, "train_sha256": train_sha, "validation_sha256": validation_sha,
              "metric": metric, "parent_public_score": parent_score, "candidate_public_score": candidate_score,
              "gain": ((candidate_score - parent_score) * (1 if direction == "higher" else -1)
                       if parent_score is not None and candidate_score is not None else None),
              "local_train_receipt": report, "epochs_attempted": spend, "model": rnd.model_name,
              "model_call_ids": list(rnd.call_ids), "history_ids": list(rnd.history_ids), "usage": token_counts,
              "prompt_bytes": sum(len(item["content"].encode("utf-8")) for item in messages),
              "output_token_limit": token_limit, "feedback_rounds": [row["round"] for row in prompt["results"]],
              "summary_seconds": round(summary_seconds, 4), "model_seconds": round(model_seconds, 4),
              "elapsed_seconds": round(time.monotonic() - started, 4)}
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
