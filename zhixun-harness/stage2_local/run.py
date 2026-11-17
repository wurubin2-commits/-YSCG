"""Run the offline multi-round harness on the three public industrial tasks."""

import argparse
import asyncio
from contextlib import closing
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time

import psutil
from pydantic import ValidationError
from pydantic_ai import models
from pydantic_ai.exceptions import AgentRunError

from stage1_local.common import ROOT, package, sha256
from stage2_local.batches import checked_sources, public_batches, read_json
from stage2_local.planner import Proposal, simulated, suggest
from stage2_local.worker import atomic_json, deny_network

RUNS = ROOT / "stage2_local/runs"
TASKS = {"A": ("macro_f1", 1, "score"), "B": ("rmse", -1, "minutes"), "C": ("rmse", -1, "cycles")}
MAX_EPOCHS = 12
WORKER_SECONDS = 60
WORKER_MIB = 1024
SESSION_SECONDS = 900


def relative(path):
    return path.relative_to(ROOT).as_posix()


def process_tree(pid):
    try:
        parent = psutil.Process(pid)
        return [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def worker(job, deadline, *, timeout=WORKER_SECONDS):
    env = {key: value for key, value in os.environ.items() if key.upper() in {
        "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR", "SYSTEMDRIVE", "USERPROFILE", "HOME",
        "APPDATA", "LOCALAPPDATA", "LANG", "LC_ALL",
    }}
    env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", BASELINE_CPU_THREADS="1")
    started, peak, status = time.monotonic(), 0, None
    command = [sys.executable, "-B", "-m", "stage2_local.worker", str(job), str(os.getpid()),
               str(psutil.Process().create_time())]
    with (job.parent / "worker.log").open("ab") as log:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL) as child:
            try:
                while child.poll() is None:
                    memory = 0
                    for member in process_tree(child.pid):
                        try:
                            memory += member.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    peak = max(peak, memory)
                    if memory > WORKER_MIB * 1024**2 or psutil.virtual_memory().available < 256 * 1024**2:
                        status = "memory_limit"
                    elif time.monotonic() > min(started + timeout, deadline):
                        status = "timeout"
                    if status:
                        break
                    time.sleep(0.1)
            finally:
                if child.poll() is None:
                    for member in reversed(process_tree(child.pid)):
                        try:
                            member.kill()
                        except psutil.NoSuchProcess:
                            pass
                    child.wait()
    return {"status": status or ("worker_failed" if child.returncode else "ok"), "exit_code": child.returncode,
            "seconds": round(time.monotonic() - started, 4), "peak_mib": round(peak / 1024**2, 2)}


def read_result(folder):
    result = read_json(folder / "result.json")
    if result["job_sha256"] != sha256(folder / "job.json"):
        raise ValueError("Worker result is for a different job")
    return result


def reference(root, task, receipt, deadline):
    folder = root / "reference"
    folder.mkdir(exist_ok=True)
    model = {"path": relative(package(task) / "checkpoint.npz"), "sha256": receipt["checkpoint_sha256"]}
    job = {"task": task, "parent": model, "validation_sha256": receipt["validation_sha256"], "plan": None}
    path = folder / "job.json"
    if not path.exists():
        atomic_json(path, job)
    elif read_json(path) != job:
        raise ValueError("Initial checkpoint or validation source changed")
    if not (folder / "result.json").is_file():
        if time.monotonic() >= deadline or worker(path, deadline)["status"] != "ok":
            raise RuntimeError(f"Reference evaluation failed. See {folder / 'worker.log'}")
    scores = read_result(folder)["before"]
    verify_scores(scores, receipt["validation_after"])
    return {**model, "scores": scores}


def verify_scores(actual, expected):
    if actual.keys() != expected.keys() or any(
        not math.isclose(value, expected[key], rel_tol=1e-9, abs_tol=1e-9) for key, value in actual.items()
    ):
        raise ValueError("Saved scores disagree with the current model on public validation")


def choose(task, folder, current, result):
    verify_scores(result["before"], current["scores"])
    candidate = result["candidate"]
    path = folder / "candidate/checkpoint.npz"
    if candidate["path"] != relative(path) or path.is_symlink() or sha256(path) != candidate["sha256"]:
        raise ValueError("Invalid candidate file")
    after = candidate["scores"]
    if after.keys() != current["scores"].keys() or any(not math.isfinite(value) for value in after.values()):
        raise ValueError("Invalid candidate scores")
    metric, direction, _ = TASKS[task]
    gain = direction * (after[metric] - current["scores"][metric])
    chosen = candidate if gain > 1e-12 else current
    return {"status": "accepted" if gain > 1e-12 else "kept", "selected": chosen, "gain": gain,
            "candidate": candidate, "optimizer_steps": [result["initial_steps"], result["final_steps"]],
            "continuation_seconds": result["continuation_seconds"]}


def batch_shift(batch, previous):
    if previous is None:
        return 0.0
    if "class_counts" in batch:
        a, b = batch["class_counts"], previous["class_counts"]
        return 0.5 * sum(abs(a.get(k, 0) / batch["rows"] - b.get(k, 0) / previous["rows"]) for k in sorted(a.keys() | b.keys()))
    return abs(batch["target_mean"] - previous["target_mean"]) / max(abs(previous["target_mean"]), 1)


def make_plan(strategy, agent, briefing):
    if strategy == "fixed":
        return Proposal(action="train", rate=0.001, batch=briefing["batch_size"], epochs=1,
                        reason="Use the fixed continuation settings for this comparison.")
    return suggest(agent, briefing)


def run_round(db, task, strategy, number, batch_path, batch, history, current, receipt, root, agent, deadline):
    key = (task, strategy, number)
    folder = root / strategy / f"round-{number:02d}"
    row = db.execute("SELECT intent, decision FROM rounds WHERE task=? AND strategy=? AND number=?", key).fetchone()
    if sha256(ROOT / current["path"]) != current["sha256"]:
        raise ValueError("The selected checkpoint has changed")
    if row and row[1]:
        saved = json.loads(row[1])
        if saved["batch"] != batch or saved["parent"] != current:
            raise ValueError("Saved round has a different source batch or parent")
        selected = saved["selected"]
        if sha256(ROOT / selected["path"]) != selected["sha256"]:
            raise ValueError("Previously selected checkpoint has changed")
        return saved

    folder.mkdir(parents=True, exist_ok=True)
    decision = {"round": number, "batch": batch, "parent": current, "selected": current, "gain": None,
                "runtime": None, "candidate": None}
    job_path = folder / "job.json"
    if row:
        intent = json.loads(row[0])
        if intent["job"]["parent"] != current or intent["job"]["train_sha256"] != batch["sha256"]:
            raise ValueError("In-flight round has a different batch or parent")
        if read_json(job_path) != intent["job"]:
            raise ValueError("Reserved job has changed")
        decision.update(plan=intent["job"]["plan"], epochs_reserved=intent["epochs"], status="interrupted")
    else:
        if time.monotonic() >= deadline:
            raise TimeoutError("This invocation reached its time budget. Resume with the same run name.")
        remaining = MAX_EPOCHS - sum(item["epochs_reserved"] for item in history)
        briefing = {"task": task, "round": number, "batch_size": receipt["batch_size"],
                    "batch": batch, "shift": batch_shift(batch, history[-1]["batch"] if history else None),
                    "metric": TASKS[task][0], "higher_is_better": TASKS[task][1] == 1,
                    "current_scores": current["scores"], "remaining_epochs": remaining,
                    "history": [{k: item[k] for k in ("round", "plan", "status", "gain")} for item in history]}
        atomic_json(folder / "briefing.json", briefing)
        decision.update(plan=None, epochs_reserved=0, status="invalid_plan")
        try:
            plan = make_plan(strategy, agent, briefing)
        except (ValidationError, AgentRunError) as error:
            decision["error_type"] = type(error).__name__
        else:
            decision["plan"] = plan.model_dump()
            if plan.action == "keep":
                decision["status"] = "policy_keep"
            elif plan.epochs > remaining:
                decision["status"] = "epoch_budget"
            else:
                job = {"task": task, "parent": current, "plan": decision["plan"], "train": relative(batch_path),
                       "train_sha256": batch["sha256"], "validation_sha256": receipt["validation_sha256"]}
                atomic_json(job_path, job)
                with db:
                    db.execute("INSERT INTO rounds VALUES (?,?,?,?,NULL)", (*key, json.dumps({"job": job, "epochs": plan.epochs})))
                decision["epochs_reserved"] = plan.epochs
                try:
                    runtime = worker(job_path, deadline)
                except OSError as error:
                    runtime = {"status": "launch_failed", "error_type": type(error).__name__}
                decision.update(runtime=runtime, status=runtime["status"])

    if decision["status"] in {"ok", "interrupted"}:
        if (folder / "result.json").is_file():
            try:
                decision.update(choose(task, folder, current, read_result(folder)))
            except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
                decision.update(status="invalid_candidate", error_type=type(error).__name__, error=str(error))
        elif decision["status"] == "ok":
            decision["status"] = "missing_result"
    with db:
        db.execute("""INSERT INTO rounds VALUES (?,?,?,NULL,?)
                      ON CONFLICT(task,strategy,number) DO UPDATE SET decision=excluded.decision""",
                   (*key, json.dumps(decision, allow_nan=False)))
    print(f"{task} {strategy} {number:02d}: {decision['status']}", flush=True)
    return decision


def run_task(db, root, task, count, strategies, agent, deadline):
    from stage2_local.tracking import record_history

    receipt = checked_sources(task)
    root = root / task
    batches = root / "batches"
    batches.mkdir(parents=True, exist_ok=True)
    original = reference(root, task, receipt, deadline)
    histories = {strategy: [] for strategy in strategies}
    with public_batches(task, count, receipt, batches) as stream:
        for number, path, batch in stream:
            for strategy, history in histories.items():
                current = history[-1]["selected"] if history else original
                history.append(run_round(db, task, strategy, number, path, batch, history, current, receipt,
                                         root, agent, deadline))
    metric, _, unit = TASKS[task]
    summary = {"metric": metric, "unit": unit, "validation_sha256": receipt["validation_sha256"], "original": original}
    for strategy, history in histories.items():
        strategy_root = root / strategy
        export = strategy_root / "history.json"
        if not export.exists():
            atomic_json(export, history)
        elif history != read_json(export):
            raise ValueError("Exported history does not match persisted decisions")
        measured = [round_["runtime"] for round_ in history if round_["runtime"]]
        summary[strategy] = {"selected": history[-1]["selected"], "rounds": count,
                             "evaluated": sum(item["candidate"] is not None for item in history),
                             "accepted": sum(item["status"] == "accepted" for item in history),
                             "worker_seconds": round(sum(m.get("seconds", 0) for m in measured), 4),
                             "peak_worker_mib": max((m.get("peak_mib", 0) for m in measured), default=0),
                             "swanlab": record_history(strategy_root, task, strategy, metric, history)}
    return summary


def run(args):
    deadline = time.monotonic() + SESSION_SECONDS
    tasks = list(TASKS) if args.task == "all" else [args.task]
    strategies = ["fixed", "adaptive"] if args.strategy == "both" else [args.strategy]
    config = {"tasks": tasks, "rounds": args.rounds, "strategies": strategies, "mode": "offline_simulation",
              "max_epochs_per_strategy": MAX_EPOCHS, "worker_timeout": WORKER_SECONDS}
    root = RUNS / args.name
    if root.resolve() != root:
        raise ValueError("Run folder must be inside the workspace without redirects")
    root.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(root / "lock.sqlite", timeout=0)) as lock:
        lock.execute("BEGIN EXCLUSIVE")
        cfg = root / "config.json"
        if not cfg.exists():
            atomic_json(cfg, config)
        elif read_json(cfg) != config:
            raise ValueError("Resume with the original choices for task, round count, and strategy")
        with closing(sqlite3.connect(root / "rounds.sqlite")) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS rounds (
                task TEXT, strategy TEXT, number INTEGER, intent TEXT, decision TEXT,
                PRIMARY KEY(task,strategy,number))""")
            policy = simulated()
            report = {"config": config, "tasks": {
                task: run_task(db, root, task, args.rounds, strategies, policy, deadline) for task in tasks}}
            path = root / "summary.json"
            if not path.exists():
                atomic_json(path, report)
            elif read_json(path) != report:
                raise ValueError("Saved summary differs from the verified run")
    print(path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--task", default="all", choices=("all", *TASKS))
    parser.add_argument("--rounds", default=10, type=int, choices=range(1, 11))
    parser.add_argument("--strategy", default="both", choices=("both", "fixed", "adaptive"))
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,47}", args.name):
        parser.error("Use a simple run name")
    if sys.version_info[:2] != (3, 12) or version("numpy") != "1.26.4" or version("pydantic-ai-slim") != "2.68.0":
        parser.error("Use the project's Python 3.12 environment and pinned requirements")
    with asyncio.Runner() as runner:
        runner.get_loop()  # Initialize Windows' internal loopback before enabling the guard.
        deny_network()
        models.ALLOW_MODEL_REQUESTS = False
        run(args)


if __name__ == "__main__":
    main()
