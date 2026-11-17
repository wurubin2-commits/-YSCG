"""Copy the files needed for Docker rehearsal to a native Linux filesystem."""

import argparse
from pathlib import Path
from shutil import copyfile

from stage1_local.common import ROOT, TASKS, sha256
from stage2_official.rehearse import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(mode=0o700, parents=True)

    def copy(source, relative):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, target)

    paths = [Path(name) for name in (
        "stage1_local/common.py", "stage2_local/planner.py", "stage2_local/requirements-win-py312.lock.txt",
        "stage2_official/Dockerfile", "stage2_official/contest_submission.py", "stage2_official/trainer.py",
        "stage2_official/policy.py", "stage2_official/continuation.py", "stage2_official/drift.py",
        "stage2_official/rehearse.py", "stage2_official/build_context.py", "stage2_official/verify_linux.sh",
        "stage2_official/sdk/pyproject.toml", "stage2_official/sdk/README.md", "stage2_official/sdk/template/harness.py",
    )]
    paths.extend(path.relative_to(ROOT) for path in (ROOT / "stage2_official/sdk/contest_harness").glob("*.py"))
    for task, name in TASKS.items():
        package = Path(name) / "cpu工业模型"
        paths.extend(package / relative for relative in (
            "checkpoint.npz", "training.json", "public/train.jsonl", "public/validation.jsonl",
            "baselines/model.py", "baselines/data.py", "baselines/train.py", "baselines/predict.py",
            f"baselines/configs/{task}.json",
        ))
    for relative in paths:
        copy(ROOT / relative, relative)

    manifest = load(args.wheelhouse / "verified-wheel-sha256.json")
    for wheel in sorted(args.wheelhouse.glob("*.whl")):
        copy(wheel, Path("wheelhouse") / wheel.name)
    copied = {wheel.name: sha256(wheel).removeprefix("sha256:") for wheel in (destination / "wheelhouse").glob("*.whl")}
    if not copied or copied != manifest:
        raise ValueError("Copied Linux wheels do not match the upstream hash report")
    print(f"{destination}: {len(paths)} rehearsal inputs and {len(copied)} verified wheels", flush=True)


if __name__ == "__main__":
    main()
