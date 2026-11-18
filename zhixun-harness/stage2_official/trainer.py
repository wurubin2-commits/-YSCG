"""Run the corresponding organizer CPU baseline in a bounded subprocess."""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from stage1_local.common import TASKS


def baseline(task, module):
    return importlib.import_module(f"{TASKS[task]}.cpu工业模型.baselines.{module}")


def train_candidate(args, output):
    train_module = baseline(args.task, "train")
    if args.recipe == "standard":
        train_module.train_run(args.task, args.train, args.validation, output, resume=args.parent,
                               epochs=args.epochs, learning_rate=args.rate, batch_size=args.batch,
                               stage_id=args.stage)
        if os.environ.get("ZHIXUN_RESEARCH_MODE") != "1":
            return
    from stage1_local.common import sha256
    from stage2_official.continuation import (
        B_RESIDUAL_RECIPE, GAS_SENSOR_FEATURES, GAS_SENSORS, SENSOR_MASK_RECIPE, fit, stability,
    )
    from stage2_official.operating_conditions import C_GROUP_RECIPE

    model_type = baseline(args.task, "model").Model
    read_dataset = baseline(args.task, "data").read_dataset
    original = model_type.load(args.parent)
    meta = original.meta
    validation = read_dataset(args.validation, meta["task"], meta["classes"], meta["paths"])
    condition_report = {}
    if args.recipe == "standard":
        candidate = model_type.load(output / "checkpoint.npz")
        report = json.loads((output / "training.json").read_text(encoding="utf-8"))
    else:
        train = read_dataset(args.train, meta["task"], meta["classes"], meta["paths"])
        if set(train["ids"]) & set(validation["ids"]):
            raise ValueError("Training and validation IDs overlap")
        candidate = model_type.load(args.parent)
        candidate.meta.update(learning_rate=args.rate, batch_size=args.batch)
        candidate.validate()
        started = time.monotonic()
        fit(candidate, train["x"], train["y"], args.epochs, args.recipe, l2sp_strength=args.l2sp_strength,
            residual_mean_weight=args.residual_mean_weight, group_mse_report=condition_report)
        report = {"parent_checkpoint_sha256": sha256(Path(args.parent)), "initial_steps": meta["steps"],
                  "initial_epochs": meta["epochs"], "final_steps": candidate.meta["steps"],
                  "final_epochs": candidate.meta["epochs"], "training_seconds": time.monotonic() - started,
                  "train_sha256": train["sha256"], "validation_sha256": validation["sha256"],
                  "train_rows": len(train["y"]), "validation_rows": len(validation["y"]),
                  "validation_before": train_module.metrics(original, validation),
                  "validation_after": train_module.metrics(candidate, validation)}
    report["recipe"] = args.recipe
    if args.l2sp_strength is not None:
        report["l2sp_strength"] = args.l2sp_strength
    if args.recipe == B_RESIDUAL_RECIPE:
        report["residual_mean_weight"] = args.residual_mean_weight
    if args.recipe == C_GROUP_RECIPE:
        report["group_mse"] = condition_report
    if args.recipe == SENSOR_MASK_RECIPE:
        report["sensor_mask"] = {"row_fraction": 0.25, "sensors_masked_per_row": 1,
                                 "sensor_count": GAS_SENSORS, "features_per_sensor": GAS_SENSOR_FEATURES}
    if (args.recipe not in ("standard", "l2sp", SENSOR_MASK_RECIPE, B_RESIDUAL_RECIPE, C_GROUP_RECIPE)
            or os.environ.get("ZHIXUN_RESEARCH_MODE") == "1"):
        report["stability"] = stability(original, candidate, validation)
    if args.recipe == "standard":
        (output / "training.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    else:
        with tempfile.TemporaryDirectory(prefix=".recipe-", dir=output.parent) as temporary:
            staging = Path(temporary) / "result"
            staging.mkdir()
            candidate.save(staging / "checkpoint.npz")
            report["checkpoint_sha256"] = sha256(staging / "checkpoint.npz")
            (staging / "training.json").write_text(
                json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf-8"
            )
            os.replace(staging, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=tuple(TASKS))
    parser.add_argument("train")
    parser.add_argument("validation")
    parser.add_argument("parent")
    parser.add_argument("output")
    parser.add_argument("stage")
    parser.add_argument("rate", type=float)
    parser.add_argument("batch", type=int)
    parser.add_argument("epochs", type=int)
    parser.add_argument("--recipe", default="standard",
                        choices=("standard", "class_balance", "huber", "l2", "l2sp", "sensor_mask_v1",
                                 "b_residual_v1", "c_group_mse_v1"))
    parser.add_argument("--l2sp-strength", type=float)
    parser.add_argument("--residual-mean-weight", type=float)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 2:
        parser.error("Each candidate needs one or two epochs")
    if (args.recipe == "l2sp") != (args.l2sp_strength is not None):
        parser.error("L2-SP requires an explicit --l2sp-strength")
    if (args.recipe == "b_residual_v1") != (args.residual_mean_weight is not None):
        parser.error("The B residual recipe needs --residual-mean-weight")
    for key in ("BASELINE_CPU_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[key] = "1"
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise ValueError("Candidate output must be new")
    for attempt in range(2):
        try:
            train_candidate(args, output)
            return
        except PermissionError as error:
            destination = error.filename2
            if (attempt or getattr(error, "winerror", None) not in (5, 32, 33)
                    or not destination or Path(destination) != output or output.exists() or output.is_symlink()):
                raise
            # The baseline publishes by renaming a staging directory. Retry once within the worker timeout.
            print(f"Retrying candidate publish after Windows error {error.winerror}.", file=sys.stderr, flush=True)
            time.sleep(0.25)


if __name__ == "__main__":
    main()
