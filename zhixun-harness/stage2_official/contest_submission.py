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
from stage2_local.planner import Proposal
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
    if task == "B" and model.meta.get("target_unit") != "minutes":
        raise ValueError("B must use the bearing minutes checkpoint")
    seed = model.meta["seed"]
    history = rnd.history()
    state = rnd.state_workspace(directory / "state")
    evidence = state / "rounds.json"
    prior = read_object(evidence) if evidence.exists() else []
    if (not isinstance(prior, list) or any(item["task"] != task or item["round"] >= rnd.round_index for item in prior)):
        raise ValueError("Unexpected state for this task and round")
    remaining = MAX_EPOCHS - sum(item["epochs_attempted"] for item in prior)
    batch = summarize(rnd.data.training_rows(), kind)
    prompt = {"task": task, "metric": metric, "direction": direction, "round": rnd.round_index,
              "batch": batch, "remaining_epochs": remaining, "parent_configuration": {
                  key: model.meta[key] for key in ("batch_size", "learning_rate", "epochs")},
              "history": [{k: item[k] for k in ("id", "hypothesis", "reason") if k in item} for item in history[-5:]],
              "prior_batches": [{"batch": item["batch"], "decision": item["decision"]} for item in prior[-3:]]}
    reply = rnd.model([
        {"role": "system", "content": "Analyze this public batch and prior attempts. Propose one useful CPU "
         "continuation or keep action. Prefer testing justified bounded hypotheses early so the system can "
         "compare at least three experiments. Output only JSON matching this schema: "
         + json.dumps(Proposal.model_json_schema())},
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, allow_nan=False)},
    ], max_tokens=min(512, rnd.max_output_tokens), temperature=0)

    selected = rnd.parent.artifact_id
    hypothesis, change = "Check whether a bounded update is justified", "Parent checkpoint retained"
    status, spend, config = "invalid_suggestion", 0, {}
    report, candidate_id, parent_score, candidate_score = None, None, None, None
    trial = directory / "trial"
    reason = "The model returned a suggestion outside the accepted schema."
    try:
        proposal = Proposal.model_validate_json(reply.text or "")
    except ValidationError:
        proposal = None
    if proposal is not None:
        hypothesis = proposal.reason
        config = proposal.model_dump(exclude_none=True)
        if proposal.action == "keep":
            status, reason = "model_keep", "The model advised retaining the current parent."
        elif proposal.epochs > remaining:
            status, reason = "budget_keep", "The requested epochs exceed the remaining allocation."
        else:
            available = min(60.0, (rnd.wall_seconds or 60.0) - (time.monotonic() - started) - 5.0)
            if available <= 0:
                status, reason = "time_keep", "The round is too close to its time limit to attempt training."
            else:
                spend = proposal.epochs
                try:
                    subprocess.run([sys.executable, "-B", "-m", "stage2_official.trainer", task,
                                    str(rnd.data.train_path), str(rnd.data.validation_path), str(path), str(trial),
                                    f"{task}-{rnd.round_index}", str(proposal.rate), str(proposal.batch), str(spend)],
                                   stdin=subprocess.DEVNULL, stdout=sys.stderr, stderr=sys.stderr,
                                   cwd=ROOT, timeout=available, check=True)
                    report = read_object(trial / "training.json")
                    if report["parent_checkpoint_sha256"].removeprefix("sha256:") != sha256(path).removeprefix("sha256:"):
                        raise ValueError("Wrong parent in trainer receipt")
                    if (report["checkpoint_sha256"] != sha256(trial / "checkpoint.npz")
                            or report["final_steps"] <= model.meta["steps"]):
                        raise ValueError("The trainer produced an invalid continuation")
                except (subprocess.SubprocessError, OSError, ValueError) as error:
                    status, reason = "training_failed", f"Continuation failed: {type(error).__name__}."
                    report = None
                if report is not None:
                    change = f"Ran {spend} CPU epoch(s), learning rate {proposal.rate}, batch {proposal.batch}."
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
                        if better:
                            selected = candidate_id
                        status = "adopted" if better else "measured_keep"
                        reason = f"{'Candidate adopted' if better else 'Parent retained'} based on {source}."

    usage = reply.response.get("usage") or {}
    token_counts = {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if type(usage.get(key)) is int}
    record = {"task": task, "round": rnd.round_index, "batch": batch, "decision": status,
              "reason": reason, "hypothesis": hypothesis, "configuration": config, "seed": seed,
              "parent_artifact": rnd.parent.artifact_id, "candidate_artifact": candidate_id, "selected_artifact": selected,
              "parent_sha256": sha256(path), "candidate_sha256": report["checkpoint_sha256"] if report else None,
              "metric": metric, "parent_public_score": parent_score, "candidate_public_score": candidate_score,
              "local_train_receipt": report, "epochs_attempted": spend, "model": rnd.model_name,
              "model_call_ids": list(rnd.call_ids), "history_ids": list(rnd.history_ids), "usage": token_counts,
              "elapsed_seconds": round(time.monotonic() - started, 4)}
    evidence.write_text(json.dumps([*prior, record], ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
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
