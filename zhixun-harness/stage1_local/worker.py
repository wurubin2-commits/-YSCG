"""Read-only original checkpoints; one small candidate and offline evidence per task."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from common import ROOT, package, sha256, write_json


def group_key(code, meta):
    if code == "A":
        return str(meta["source_batch"])
    if code == "B":
        return meta["bearing"]
    return f"{meta['source_subset']}:{meta['unit']}"


def run(code, output, epochs, learning_rate):
    started = time.monotonic()
    # The training worker cannot make Python socket connections. Package downloads
    # happen separately, before starting this offline process.
    blocked_network_events = []

    def offline_guard(event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
            blocked_network_events.append(event)
            raise PermissionError("Stage 1 is offline; network calls are disabled")

    sys.addaudithook(offline_guard)
    os.environ["BASELINE_CPU_THREADS"] = "1"
    os.environ["SWANLAB_MODE"] = "offline"
    os.environ["SWANLAB_ROOT"] = str(output / "swanlab_settings")
    os.chdir(output)
    sys.path.insert(0, str(package(code)))
    from baselines.model import Model
    from baselines.data import read_dataset
    from baselines.predict import predict
    from baselines.train import metrics, train_run
    import numpy as np
    import swanlab
    import psutil

    origin = package(code)
    receipt = json.loads((origin / "training.json").read_text(encoding="utf-8"))
    checkpoint = origin / "checkpoint.npz"
    baseline = Model.load(checkpoint)
    profile = json.loads((origin / "baselines/configs" / f"{code}.json").read_text())
    expected_task = "classification" if code == "A" else "regression"
    if baseline.meta["profile"] != code or baseline.meta["task"] != expected_task:
        raise ValueError("Wrong package/profile/checkpoint for task")
    if code == "B" and baseline.meta.get("target_unit") != "minutes":
        raise ValueError("Bearing checkpoint must predict minutes")

    checks = {"checkpoint": sha256(checkpoint) == receipt["checkpoint_sha256"]}
    checks.update({name: sha256(origin / "baselines" / name) == expected
                   for name, expected in receipt["source_sha256"].items()})
    splits, ids, groups = {}, {}, {}
    for split in ("train", "validation"):
        path = origin / "public" / f"{split}.jsonl"
        checks[split] = sha256(path) == receipt[f"{split}_sha256"]
        ids[split], groups[split] = set(), set()
        rows = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["id"] in ids[split]:
                    raise ValueError("Duplicate sample ID")
                ids[split].add(row["id"])
                groups[split].add(group_key(code, row["meta"]))
                rows += 1
        splits[split] = {"rows": rows, "groups": len(groups[split]), "sha256": sha256(path)}
    if not all(checks.values()):
        raise ValueError(f"Source differs from provided training receipt: {checks}")
    if ids["train"] & ids["validation"] or groups["train"] & groups["validation"]:
        raise ValueError("Training/validation IDs or groups overlap")

    validation = read_dataset(origin / "public/validation.jsonl", baseline.meta["task"],
                              baseline.meta["classes"], baseline.meta["paths"])
    before = metrics(baseline, validation)
    if not all(np.isclose(v, receipt["validation_after"][k], rtol=1e-9, atol=1e-9)
               for k, v in before.items()):
        raise ValueError("Baseline metrics do not reproduce supplied receipt")

    config = {
        "task": code, "purpose": "local_smoke_only", "official_scoring_ready": False,
        "epochs": epochs, "learning_rate": learning_rate, "seed": baseline.meta["seed"],
        "batch_size": baseline.meta["batch_size"], "parent_sha256": sha256(checkpoint),
        "train_sha256": splits["train"]["sha256"],
        "validation_sha256": splits["validation"]["sha256"],
        "hypothesis": "A small continuation at a fixed lower learning rate verifies the training path; improvement is not guaranteed.",
        "numpy": np.__version__, "swanlab": swanlab.__version__, "cpu_threads": 1,
    }
    write_json(output / "baseline.json", {"metrics": before, "checks": checks, "splits": splits,
                                          "meta": baseline.meta, "config": config})
    # Explicit settings also suppress inherited webhooks and account configuration.
    settings = swanlab.Settings(
        mode="offline", interactive=False, api_key=None,
        probe=swanlab.Settings.Probe(hardware=False, runtime=False, requirements=False,
                                    git=False, conda=False, monitor=False, swanlab=True),
        integration=swanlab.Settings.Integration(
            webhook=swanlab.Settings.Integration.Webhook(url="", value="")),
    )
    swanlab.init(project="zhixun-stage1-local", experiment_name=f"{code}-{output.parent.name}",
                 mode="offline", logdir=str(output / "swanlab"), config=config, settings=settings)
    try:
        swanlab.log({f"validation/{k}": v for k, v in before.items()}, step=0)
        train_started = time.perf_counter()
        report = train_run(code, origin / "public/train.jsonl", origin / "public/validation.jsonl",
                           output / "candidate", resume=checkpoint, epochs=epochs,
                           learning_rate=learning_rate, stage_id=f"stage1-{code}")
        train_call_seconds = time.perf_counter() - train_started
        candidate = Model.load(output / "candidate/checkpoint.npz")
        after = metrics(candidate, validation)
        if report["parent_checkpoint_sha256"] != sha256(checkpoint):
            raise ValueError("Parent checkpoint mismatch")
        if report["initial_steps"] != baseline.meta["steps"] or report["final_steps"] <= report["initial_steps"]:
            raise ValueError("Optimizer steps did not continue")
        if candidate.meta["epochs"] != baseline.meta["epochs"] + epochs:
            raise ValueError("Epoch counter did not continue")
        if not np.array_equal(candidate.mean, baseline.mean) or not np.array_equal(candidate.scale, baseline.scale):
            raise ValueError("Input normalization changed")
        for key in ("target_mean", "target_scale", "seed", "hidden", "classes", "paths"):
            if candidate.meta[key] != baseline.meta[key]:
                raise ValueError(f"Continuation changed {key}")
        if not all(np.isclose(value, report["validation_after"][key], rtol=1e-12, atol=1e-12)
                   for key, value in after.items()):
            raise ValueError("Reloaded candidate differs from training report")
        with (output / "candidate/history.jsonl").open(encoding="utf-8") as stream:
            history = [json.loads(line) for line in stream if line.strip()]
        for index, row in enumerate(history, 1):
            swanlab.log({"train/loss": row["loss"], "train/cumulative_epoch": row["epoch"],
                         **{f"validation/{k}": v for k, v in row["validation"].items()}}, step=index)
        metric = "macro_f1" if code == "A" else "rmse"
        gain = after[metric] - before[metric] if code == "A" else before[metric] - after[metric]
        accepted = math.isfinite(gain) and gain > 1e-12
        selected = output / "candidate/checkpoint.npz" if accepted else checkpoint
        with (origin / "public/validation.jsonl").open(encoding="utf-8") as stream:
            sample = json.loads(next(stream))
        selected_model = candidate if accepted else baseline
        prediction = predict(selected_model, sample["inputs"])
        write_json(output / "prediction.json", {"sample_id": sample["id"], "response": prediction,
                                                 "checkpoint_sha256": sha256(selected)})
        result = {"task": code, "status": "completed", "baseline": before, "candidate": after,
                  "primary_metric": metric, "unit": "score" if code == "A" else ("minutes" if code == "B" else "cycles"),
                  "improvement": gain, "decision": "accept_candidate" if accepted else "keep_baseline",
                  "reason": "Primary public validation metric strictly improved" if accepted else "No primary public validation improvement",
                  "selected_checkpoint": str(selected.relative_to(ROOT)), "selected_sha256": sha256(selected),
                  "parent_checkpoint_sha256": sha256(checkpoint), "epochs_added": epochs,
                  "initial_steps": report["initial_steps"], "final_steps": report["final_steps"],
                  "training_seconds": report["training_seconds"], "total_seconds": time.monotonic() - started,
                  "train_call_seconds_including_io": train_call_seconds,
                  "worker_peak_wset_MiB": getattr(psutil.Process().memory_info(), "peak_wset", 0) / 1024**2,
                  "original_hashes_match": all(checks.values()), "normalization_preserved": True,
                  "blocked_network_events": blocked_network_events, "swanlab_mode": "offline",
                  "official_scoring_ready": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(output / "result.json", result)
        swanlab.log({"decision/accepted": int(accepted), "decision/improvement": gain,
                     "cost/training_seconds": report["training_seconds"]}, step=epochs + 1)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        write_json(output / "failure.json", {"error_type": type(exc).__name__, "error": str(exc)})
        swanlab.log({"status/failed": 1}, step=epochs + 2)
        raise
    finally:
        swanlab.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list("ABC"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    if not 0 < args.learning_rate <= 0.01:
        parser.error("learning rate must be in (0, 0.01]")
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / "stage1_local/runs") or args.output.exists():
        parser.error("output must be a new directory within stage1_local/runs")
    args.output.mkdir(parents=True)
    try:
        run(args.task, args.output, args.epochs, args.learning_rate)
    except Exception:
        (args.output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
