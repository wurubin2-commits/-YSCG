"""Run the organizer SDK against a simulated host with real public CPU training."""

import argparse
import base64
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

from stage1_local.common import ROOT, package, sha256, write_json
from stage2_official.tool_layer import MAX_TOOL_ROUNDS
from stage2_official.trainer import baseline

DIRECTORY = Path(__file__).parent
CONTRACTS = {"A": ("classification", "macro_f1", "higher"),
             "B": ("regression", "rmse", "lower"), "C": ("regression", "rmse", "lower")}
PREDICTION = ["python3", "/app/harness.py", "predict"]
L2SP_TRIAL_STRENGTH = 1.0
# Set by --live: a real model answers the decision requests instead of the scripted proposals.
ADVISOR = None


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def projection(path, destination, rounds):
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    for i in range(rounds):
        target = destination / f"train-{i+1:02d}.jsonl"
        with target.open("x", encoding="utf-8") as out:
            for row in rows[i*len(rows)//rounds:(i+1)*len(rows)//rounds]:
                print(json.dumps({k: row[k] for k in ("id", "inputs", "answer")}, ensure_ascii=False), file=out)
        yield target


def save_files(directory, files):
    directory.mkdir()
    for item in files:
        name = item["path"]
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()) or path == directory:
            raise ValueError("Unsafe simulated upload path")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(base64.b64decode(item["data_base64"], validate=True))


def send(process, msg):
    process.stdin.write(json.dumps(msg, ensure_ascii=False, allow_nan=False) + "\n")
    process.stdin.flush()


def read(process):
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("SDK process closed its protocol stream")
    return json.loads(line)


@contextmanager
def process(command, folder, *, image, mounts=(), variables=None):
    env = {name: value for name, value in os.environ.items() if name.upper() in {
        "PATH", "SYSTEMROOT", "WINDIR", "HOME", "USERPROFILE", "TEMP", "TMP", "TMPDIR",
        "SYSTEMDRIVE", "APPDATA", "LOCALAPPDATA",
    } or name.startswith("ZHIXUN_")}
    controlled = {"PYTHONUTF8": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                  "MKL_NUM_THREADS": "1", "BASELINE_CPU_THREADS": "1",
                  **{name: value for name, value in os.environ.items() if name.startswith("ZHIXUN_")},
                  **(variables or {})}
    if image:
        argv = ["docker", "run", "--rm", "-i", "--read-only", "--network", "none", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--cpus", "4", "--memory", "8g", "--memory-swap", "8g",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m"]
        if hasattr(os, "getuid"):
            argv.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        for name, value in controlled.items():
            argv.extend(["-e", f"{name}={value}"])
        for source, target, readonly in mounts:
            argv.extend(["--mount", f"type=bind,source={source},target={target}" + (",readonly" if readonly else "")])
        argv += [image, command]
    else:
        env.update(controlled, PYTHONPATH=os.pathsep.join([str(ROOT), str(DIRECTORY / "sdk")]),
                   CONTEST_SUBMISSION="stage2_official.contest_submission")
        argv = [sys.executable, "-B", "-m", "stage2_official.local_entry", command]
    with (folder / f"{command}.stderr.log").open("x", encoding="utf-8") as stderr:
        with subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=stderr, encoding="utf-8", text=True) as child:
            watchdog = threading.Timer(110, child.kill)
            watchdog.start()
            try:
                yield child
                if child.poll() is None and not child.stdin.closed:
                    child.stdin.close()
                if child.wait(timeout=15):
                    raise RuntimeError(f"{command} failed. See {folder / (command + '.stderr.log')}")
            finally:
                watchdog.cancel()
                if child.poll() is None:
                    child.kill()
                    child.wait()


def descriptor(path):
    digest = sha256(path)
    return {"artifact_id": "cpu-" + digest[-20:], "artifact_sha256": digest,
            "artifact": {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [], "predict_argv": PREDICTION},
            "files": [{"path": "checkpoint.npz", "sha256": digest, "size_bytes": path.stat().st_size}]}


def local_scores(task, validation, checkpoint):
    model = baseline(task, "model").Model.load(checkpoint)
    data = baseline(task, "data").read_dataset(validation, model.meta["task"], model.meta["classes"], model.meta["paths"])
    return baseline(task, "train").metrics(model, data)


def run_round(task, idx, directory, train, validation, parent, history, prior_state, image, scenario):
    kind, metric, direction = CONTRACTS[task]
    artifact = descriptor(parent)
    checkpoint = artifact["files"][0]["sha256"]
    parent_scores = local_scores(task, validation, parent)
    work = directory / "workspace"
    work.mkdir()
    state_sha = sha256(prior_state / "rounds.json") if prior_state else ""
    start = {"protocol": "contest.harness.workspace.v2", "run_id": "offline-check",
             "dataset_key": task, "round_index": idx, "prompt": "Use the current public projection and selected parent.",
             "prompt_version": "industrial-workspace-flexible-v3", "task_contract": {
                 "task": kind, "metric": metric, "metric_direction": direction, "public_projection": {
                     "train_path": "/data/train.jsonl" if image else str(train),
                     "validation_path": "/data/validation.jsonl" if image else str(validation)}},
             "model_name": "mock:offline", "max_output_tokens": 512, "parent_checkpoint_sha256": checkpoint,
             "parent_artifact": artifact, "artifacts": [artifact], "current_validation": {"metrics": parent_scores},
             "state_path": ("/state" if image else str(prior_state)) if prior_state else None,
             "state_sha256": state_sha, "wall_seconds": 90}
    if scenario == "wrong_parent":
        start["parent_artifact"]["files"][0]["sha256"] = "sha256:" + "0" * 64
    if scenario == "time_budget":
        start["wall_seconds"] = 4
    mounts = [(train, "/data/train.jsonl", True), (validation, "/data/validation.jsonl", True),
              (parent.parent, "/previous", True), (work, "/workspace", False)]
    if prior_state:
        mounts.append((prior_state, "/state", True))
    variables = ({} if image else {"CONTEST_WORK": str(work), "ZHIXUN_TEST_PREVIOUS": str(parent.parent)})
    if scenario == "recipes":
        variables["ZHIXUN_RESEARCH_MODE"] = "1"
    if scenario == "l2sp":
        variables["ZHIXUN_ENABLE_L2SP"] = "1"
    if scenario == "c_half_rate":
        variables["ZHIXUN_C_HALF_RATE"] = "1"
    if scenario == "sensor_mask":
        variables["ZHIXUN_ENABLE_SENSOR_MASK"] = "1"
    if scenario == "b_residual":
        variables["ZHIXUN_ENABLE_B_RESIDUAL"] = "1"
    if scenario == "c_group_mse":
        variables["ZHIXUN_ENABLE_C_GROUP_MSE"] = "1"
    trace, model_ids = [], []
    requests = []
    candidate, final_state, decision = None, None, None
    with process("train", directory, image=image, mounts=mounts, variables=variables) as child:
        send(child, start)
        seq = 0
        while True:
            frame = read(child)
            protocol = frame["type"]
            trace.append(protocol)
            if protocol == "round_complete":
                decision = frame["record"]
                if not model_ids or decision["model_call_ids"] != model_ids:
                    raise AssertionError("Missing or mismatched model call IDs")
                if decision["history_ids"] != [item["id"] for item in history]:
                    raise AssertionError("Wrong history references")
                if final_state is None:
                    raise AssertionError("Missing state upload")
                accepted = {artifact["artifact_id"]: parent}
                if candidate is not None:
                    accepted[descriptor(candidate)["artifact_id"]] = candidate
                selected = accepted[decision["selected_artifact_id"]]
                send(child, {"type": "round_ack", "accepted": True})
                if child.stdout.readline():
                    raise AssertionError("Protocol message after acknowledgment")
                break
            if protocol == "round_failed":
                raise RuntimeError(f"{task} round {idx}: {frame['error']}; see {directory / 'train.stderr.log'}")
            if seq != frame["id"]:
                raise AssertionError("Invalid frame sequence")
            reply = {"id": seq}
            seq += 1
            if protocol == "history_request":
                reply.update(type="history_response", records=history)
            elif protocol == "model_request":
                if frame["max_tokens"] > 512:
                    raise AssertionError("Unbounded mock request")
                if frame.get("enable_thinking") is not False:
                    raise AssertionError("The advisor must disable reasoning to answer inside the budget")
                requests.append({"prompt_bytes": sum(len(m["content"].encode("utf-8")) for m in frame["messages"]),
                                 "output_token_limit": frame["max_tokens"]})
                if scenario == "missing_call":
                    model_id = ""
                else:
                    model_id = f"local-{task}-{idx}-{len(model_ids)+1}"
                    model_ids.append(model_id)
                if ADVISOR is not None:
                    # Real model: the harness decides on what it actually answered. Tool
                    # specifications ride on the frame exactly as the gateway would send them.
                    answer = ADVISOR.ask(frame["messages"], frame["max_tokens"],
                                         tools=frame.get("tools"))
                    content = answer["text"]
                    message = {"role": "assistant", "content": content}
                    if answer["tool_calls"]:
                        message["tool_calls"] = answer["tool_calls"]
                else:
                    proposal = {"action": "train", "reason": f"Test bounded continuation on public batch {idx}",
                                "rate": (0.001, 0.0005, 0.00075)[(idx - 1) % 3],
                                "batch": 256 if task == "C" else 128, "epochs": 1}
                    if scenario in ("duplicate", "fixed", "repeat_pairs"):
                        proposal["rate"] = 0.001
                    if scenario == "duplicate":
                        if idx > 1:
                            context = json.loads(frame["messages"][-1]["content"])
                            previous = context["results"][0]
                            if previous["scores"][0] != previous["scores"][1] or previous["params"]["rate"] != 0.001:
                                raise AssertionError("The advisor did not receive the measured experiment")
                    if scenario == "recipes":
                        variants = ("class_balance", "l2", "standard") if task == "A" else ("huber", "l2", "standard")
                        proposal["recipe"] = variants[(idx - 1) % len(variants)]
                    if scenario == "l2sp":
                        proposal.update(recipe="l2sp", l2sp_strength=L2SP_TRIAL_STRENGTH)
                    if scenario == "sensor_mask":
                        proposal["recipe"] = "sensor_mask_v1"
                    if scenario == "b_residual":
                        proposal.update(recipe="b_residual_v1", residual_mean_weight=0.5)
                    if scenario == "c_group_mse":
                        proposal["recipe"] = "c_group_mse_v1"
                    if scenario == "evolution":
                        # The scripted host always takes the top evidence-backed candidate, so
                        # this measures the mutation operator with no model judgement in it.
                        content = json.dumps({"action": "train", "candidate": 1,
                                              "reason": "Take the first evidence-backed candidate"})
                    elif scenario == "invalid_plan":
                        content = "malformed mock result"
                    else:
                        content = json.dumps(proposal)
                    message = {"role": "assistant", "content": content}
                reply.update(type="model_response", call_id=model_id, response={
                    "choices": [{"message": message}]})
            elif protocol == "candidate_request":
                if frame["parent_checkpoint_sha256"] != checkpoint or frame["parent_artifact_id"] != artifact["artifact_id"]:
                    raise AssertionError("Wrong candidate parent")
                if frame["artifact"]["predict_argv"] != PREDICTION:
                    raise AssertionError("Wrong prediction command")
                save_files(directory / "candidate", frame["files"])
                candidate = directory / "candidate/checkpoint.npz"
                measured = local_scores(task, validation, candidate)
                if scenario in ("reject_candidate", "duplicate", "repeat_pairs"):
                    measured[metric] = parent_scores[metric]
                reply.update(type="candidate_response", artifact_id=descriptor(candidate)["artifact_id"],
                             summary={"metrics": measured})
            elif protocol == "state_request":
                if frame["parent_state_sha256"] != state_sha:
                    raise AssertionError("Wrong predecessor state")
                final_state = directory / "state"
                save_files(final_state, frame["files"])
                reply.update(type="state_response", state_sha256=sha256(final_state / "rounds.json"))
            else:
                raise AssertionError(f"Unexpected protocol message {protocol}")
            send(child, reply)
    write_json(directory / "trace.json", trace)
    write_json(directory / "requests.json", requests)
    write_json(directory / "decision.json", decision)
    return selected, final_state, {"id": f"{task}-{idx}", "round_index": idx, **decision}


def run_task(task, args, destination):
    started = time.monotonic()
    origin = package(task)
    destination.mkdir()
    reference = load(origin / "training.json")
    parent = origin / "checkpoint.npz"
    train, validation = origin / "public/train.jsonl", origin / "public/validation.jsonl"
    for source, key in ((parent, "checkpoint_sha256"), (train, "train_sha256"), (validation, "validation_sha256")):
        if sha256(source) != reference[key]:
            raise ValueError(f"Original {source.name} differs from the training receipt")
    # A mount directory contains only this checkpoint.
    starting_parent = destination / "baseline"
    starting_parent.mkdir()
    (starting_parent / parent.name).write_bytes(parent.read_bytes())
    parent = starting_parent / parent.name
    prior, history = None, []
    count = 1 if args.scenario == "duplicate" else args.rounds
    if args.scenario == "repeat_pairs":
        count = (args.rounds + 1) // 2
    batches = list(projection(train, destination, count))
    if args.scenario == "duplicate":
        batches *= args.rounds
    if args.scenario == "repeat_pairs":
        batches = [batch for batch in batches for _ in range(2)][:args.rounds]
    for idx, batch in enumerate(batches, 1):
        directory = destination / f"round-{idx:02d}"
        directory.mkdir()
        if args.scenario == "invalid_training":
            with batch.open(encoding="utf-8") as stream:
                invalid = json.loads(next(stream))
            invalid["inputs"] = {}
            batch = directory / "bad-schema.jsonl"
            with batch.open("x", encoding="utf-8") as stream:
                print(json.dumps(invalid), file=stream)
        parent, prior, entry = run_round(task, idx, directory, batch, validation, parent, history,
                                        prior, args.image, args.scenario)
        history.append(entry)
        print(f"{task} {idx}/{args.rounds}: {entry['selected_artifact_id']}", flush=True)
    kind, metric, _ = CONTRACTS[task]
    art = descriptor(parent)
    context = {"protocol": "contest.predict-context.v1", "dataset_key": task, "task": kind,
               "artifact": art["artifact"], "artifact_id": art["artifact_id"]}
    variables = {"CONTEST_PREDICTION_CONTEXT": json.dumps(context),
                 "CONTEST_ARTIFACT_PATH": "/model/checkpoint.npz" if args.image else str(parent)}
    if not args.image:
        variables["ZHIXUN_TEST_MODEL"] = str(parent.parent)
    with validation.open(encoding="utf-8") as stream:
        inputs = [json.loads(next(stream))["inputs"] for _ in range(2)]
    with process("predict", destination, image=args.image, variables=variables,
                 mounts=[(parent.parent, "/model", True)]) as child:
        predictions = []
        for sample in inputs:
            send(child, {"task": kind, "inputs": sample})
            predictions.append(read(child))
        child.stdin.close()
        if child.stdout.readline():
            raise AssertionError("Predict emitted too many responses")
    records = load(prior / "rounds.json")
    if args.scenario in ("normal", "recipes", "fixed", "l2sp", "c_half_rate", "evolution") and args.rounds >= 3 and sum(
        r["candidate_artifact"] is not None for r in records
    ) < 3:
        raise AssertionError("Expected at least three measured trials")
    if args.scenario == "recipes" and any(
        not r["local_train_receipt"] or r["local_train_receipt"]["recipe"] != r["configuration"]["recipe"]
        or not r["local_train_receipt"]["stability"] for r in records
    ):
        raise AssertionError("Missing recipe or stability evidence")
    if args.scenario == "l2sp" and any(
        not r["candidate_artifact"] or not r["local_train_receipt"]
        or r["local_train_receipt"].get("l2sp_strength") != L2SP_TRIAL_STRENGTH for r in records
    ):
        raise AssertionError("Missing measured L2-SP candidate or strength")
    if args.scenario == "c_half_rate" and any(
        not r["local_train_receipt"] or r["configuration"]["rate"] != r.get("proposed_learning_rate", 0) / 2
        or r["local_train_receipt"]["learning_rate"] != r["configuration"]["rate"] for r in records
    ):
        raise AssertionError("Missing measured C trial or effective learning rate")
    if args.scenario == "sensor_mask" and any(
        not r["candidate_artifact"] or not (r["local_train_receipt"] or {}).get("sensor_mask") for r in records
    ):
        raise AssertionError("Missing measured A sensor masking candidate")
    if args.scenario == "b_residual" and any(
        not r["candidate_artifact"] or (r["local_train_receipt"] or {}).get("residual_mean_weight") != 0.5
        for r in records
    ):
        raise AssertionError("Missing measured B residual-loss candidate or weight")
    if args.scenario == "c_group_mse" and any(
        not r["candidate_artifact"] or not (r["local_train_receipt"] or {}).get("group_mse") for r in records
    ):
        raise AssertionError("Missing measured C condition-loss candidate or group counts")
    expected_keeps = {"reject_candidate": "measured_keep",
                      "invalid_training": "training_failed", "time_budget": "time_keep"}
    if args.scenario in expected_keeps and {r["decision"] for r in records} != {expected_keeps[args.scenario]}:
        raise AssertionError(f"Expected the parent to be kept after {args.scenario}")
    if args.scenario == "invalid_plan" and not all(r.get("proposal_normalization") for r in records):
        raise AssertionError("Expected every malformed proposal to be replaced by a rule plan")
    if args.scenario == "duplicate" and [r["decision"] for r in records] != (
        ["measured_keep"] + ["duplicate_keep"] * (args.rounds - 1)
    ):
        raise AssertionError("An identical measured experiment was repeated")
    traces = [load(directory / "trace.json") for directory in sorted(destination.glob("round-*"))]
    requests = [item for directory in sorted(destination.glob("round-*"))
                for item in load(directory / "requests.json")]
    # A decision may now spend several model calls on its tool loop (plus one forced answer),
    # so the contract is a bounded call count, not exactly one request per round.
    if any(not 1 <= trace.count("model_request") <= MAX_TOOL_ROUNDS + 1
           or trace.count("candidate_request") > 1 for trace in traces):
        raise AssertionError("Expected a bounded tool loop and at most one candidate per round")
    return {"rounds": len(records), "model_calls": sum(len(item["model_call_ids"]) for item in records),
            "tool_calls": sum(len(item.get("tool_trace") or []) for item in records),
            "prompt_bytes": sum(item["prompt_bytes"] for item in requests),
            "max_output_tokens_per_call": max(item["output_token_limit"] for item in requests),
            "training_attempts": sum(item["epochs_attempted"] > 0 for item in records),
            "candidate_submissions": sum(trace.count("candidate_request") for trace in traces),
            "duplicates_skipped": sum(item["decision"] == "duplicate_keep" for item in records),
            "trials": sum(item["candidate_artifact"] is not None for item in records),
            "epochs_attempted": sum(item["epochs_attempted"] for item in records),
            "round_seconds": round(sum(item["elapsed_seconds"] for item in records), 2),
            "statuses": dict(Counter(item["decision"] for item in records)), "predictions": predictions,
            "mutations": dict(Counter(item["evolution"]["chosen"] for item in records if item.get("evolution"))),
            "metric": metric, "original_public": reference["validation_after"][metric],
            "selected_public": local_scores(task, validation, parent)[metric],
            "wall_seconds": round(time.monotonic() - started, 2), "selected_sha256": sha256(parent)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--task", choices=("all", *CONTRACTS), default="all")
    parser.add_argument("--rounds", choices=range(1, 11), default=10, type=int)
    parser.add_argument("--image", help="Official-entry image to use instead of the host Python")
    parser.add_argument("--live", metavar="BASE_MODEL",
                        help="Ask a real PyTRiO model instead of the scripted proposals, e.g. Qwen/Qwen3.8-27B")
    parser.add_argument("--scenario", choices=("normal", "invalid_plan", "reject_candidate", "invalid_training",
                                               "time_budget", "missing_call", "wrong_parent", "duplicate",
                                               "recipes", "fixed", "repeat_pairs", "l2sp", "c_half_rate",
                                               "sensor_mask", "b_residual", "c_group_mse", "evolution"), default="normal")
    args = parser.parse_args()
    if args.scenario == "l2sp" and args.task != "B":
        parser.error("The L2-SP trial requires --task B")
    if args.scenario == "c_half_rate" and args.task != "C":
        parser.error("The half-rate trial requires --task C")
    if args.scenario == "sensor_mask" and args.task != "A":
        parser.error("The sensor masking trial requires --task A")
    if args.scenario == "b_residual" and args.task != "B":
        parser.error("The residual-loss trial requires --task B")
    if args.scenario == "c_group_mse" and args.task != "C":
        parser.error("The condition-loss trial requires --task C")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}", args.name):
        parser.error("Invalid run name")
    output = DIRECTORY / "runs" / args.name
    output.mkdir(parents=True)
    with process("describe", output, image=args.image) as child:
        spec = read(child)
        if (spec["protocol"], spec["model_access"], spec["sdk_version"], set(spec["tasks"])) != (
            "contest.harness.workspace.v2", "host_gateway_rpc", "1.0.2", {"classification", "regression"}
        ):
            raise AssertionError("Unexpected official describe contract")
    global ADVISOR
    if args.live:
        from stage2_official.live_advisor import LiveAdvisor
        ADVISOR = LiveAdvisor(args.live)
    tasks = CONTRACTS if args.task == "all" else (args.task,)
    summary = {task: run_task(task, args, output / task) for task in tasks}
    write_json(output / "summary.json", {
        "mode": "mock_host" if ADVISOR is None else f"live:{args.live}",
        "scenario": args.scenario, "describe": spec, "tasks": summary,
        "advisor": ADVISOR.report() if ADVISOR is not None else None})
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
