"""Predict from a run's verified selected checkpoint, using input-only JSONL."""

import argparse
import os
import re
import subprocess
import sys

from stage1_local.common import ROOT, package, sha256
from stage2_local.batches import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--task", required=True, choices=tuple("ABC"))
    parser.add_argument("--strategy", default="adaptive", choices=("original", "fixed", "adaptive"))
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,47}", args.name):
        parser.error("Use a simple run name")
    run = ROOT / "stage2_local/runs" / args.name
    if run.resolve() != run:
        parser.error("Unexpected run folder")
    summary = read_json(run / "summary.json")["tasks"].get(args.task, {})
    record = summary.get(args.strategy)
    if record is None:
        parser.error("Selection unavailable in this run")
    selected = record if args.strategy == "original" else record["selected"]
    origin = package(args.task)
    path = (ROOT / selected["path"]).resolve()
    if path != origin / "checkpoint.npz" and not path.is_relative_to(run / args.task / args.strategy):
        raise ValueError("Checkpoint is outside this task and strategy")
    if sha256(path) != selected["sha256"]:
        raise ValueError("Selected model checksum differs")
    env = dict(os.environ, BASELINE_CPU_THREADS="1", PYTHONUTF8="1")
    return subprocess.run([sys.executable, "-B", "-m", "baselines.predict", "--checkpoint", str(path)],
                          cwd=origin, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
