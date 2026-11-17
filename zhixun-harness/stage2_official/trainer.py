"""Run the corresponding organizer CPU baseline in a bounded subprocess."""

import argparse
import importlib
import os
from pathlib import Path
import sys
import time

from stage1_local.common import TASKS


def baseline(task, module):
    return importlib.import_module(f"{TASKS[task]}.cpu工业模型.baselines.{module}")


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
    args = parser.parse_args()
    for key in ("BASELINE_CPU_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[key] = "1"
    train_run = baseline(args.task, "train").train_run
    output = Path(args.output)
    for attempt in range(2):
        try:
            train_run(args.task, args.train, args.validation, output, resume=args.parent,
                      epochs=args.epochs, learning_rate=args.rate, batch_size=args.batch, stage_id=args.stage)
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
