"""Create an allowlisted SDK image context without data, checkpoints, logs or credentials."""

import argparse
from pathlib import Path
import re
import shutil

from stage1_local.common import ROOT, TASKS, sha256, write_json
from stage2_official.rehearse import load

IMPORTED = re.compile(r"^\s*from stage2_official(?:\.(\w+)| import ([^\n]+))", re.MULTILINE)


def required_modules(origin):
    """Modules the entry imports, so a renamed file cannot silently leave the image."""
    names = set()
    for match in IMPORTED.finditer(origin.read_text(encoding="utf-8")):
        if match.group(1):
            names.add(match.group(1))
            continue
        for item in match.group(2).split(","):
            name = item.strip().partition(" as ")[0].strip()
            if name and name != "*":
                names.add(name)
    return names


def build(output, wheelhouse=None):
    output = output.resolve()
    output.mkdir(parents=True)
    source = ROOT / "stage2_official"
    files = {}

    def copy(origin, destination):
        target = output / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
        files[target.relative_to(output).as_posix()] = sha256(target)

    copy(source / "Dockerfile", "Dockerfile")
    copy(source / "sdk/template/harness.py", "app/harness.py")
    copy(source / "contest_submission.py", "app/contest_submission.py")
    copy(source / "trainer.py", "app/stage2_official/trainer.py")
    copy(source / "policy.py", "app/stage2_official/policy.py")
    copy(source / "continuation.py", "app/stage2_official/continuation.py")
    copy(source / "operating_conditions.py", "app/stage2_official/operating_conditions.py")
    copy(source / "drift.py", "app/stage2_official/drift.py")
    copy(source / "admission.py", "app/stage2_official/admission.py")
    copy(source / "normalize.py", "app/stage2_official/normalize.py")
    copy(source / "diagnostics.py", "app/stage2_official/diagnostics.py")
    copy(source / "attribution.py", "app/stage2_official/attribution.py")
    copy(source / "tool_layer.py", "app/stage2_official/tool_layer.py")
    copy(ROOT / "stage1_local/common.py", "app/stage1_local/common.py")
    copy(ROOT / "stage2_local/planner.py", "app/stage2_local/planner.py")
    for name in ("pyproject.toml", "README.md"):
        copy(source / "sdk" / name, f"sdk/{name}")
    for path in sorted((source / "sdk/contest_harness").glob("*.py")):
        copy(path, f"sdk/contest_harness/{path.name}")
    for task, dirname in TASKS.items():
        base = Path(dirname) / "cpu工业模型"
        receipt = load(ROOT / base / "training.json")
        for name in ("model.py", "data.py", "train.py", "predict.py", f"configs/{task}.json"):
            path = base / "baselines" / name
            expected = receipt["profile_sha256"] if name.endswith(".json") else receipt["source_sha256"][name]
            if sha256(ROOT / path) != expected:
                raise ValueError(f"Baseline file differs from training receipt: {path}")
            copy(ROOT / path, Path("app") / path)

    # The official entry uses NumPy and Pydantic AI's core types. The cloud logging
    # SDK and external OpenAI clients belong to the local analysis environment.
    names = {"annotated-types", "anyio", "certifi", "genai-prices", "griffelib", "h11", "httpcore2",
             "httpx2", "idna", "logfire-api", "numpy", "opentelemetry-api", "pydantic", "pydantic-ai-slim",
             "pydantic-graph", "pydantic_core", "setuptools", "sniffio", "truststore", "typing-inspection", "typing_extensions"}
    lines = (ROOT / "stage2_local/requirements-win-py312.lock.txt").read_text(encoding="utf-8").splitlines()
    pinned = [line for line in lines if line.partition("==")[0] in names]
    if {line.partition("==")[0] for line in pinned} != names:
        raise ValueError("The local dependency lock is missing a selected runtime pin")
    path = output / "runtime.txt"
    path.write_text("\n".join(pinned) + "\n", encoding="utf-8")
    files[path.name] = sha256(path)
    (output / "wheelhouse").mkdir()
    if wheelhouse is not None:
        wheels = sorted(wheelhouse.glob("*.whl"))
        if not wheels or any(not path.is_file() or path.is_symlink() for path in wheels):
            raise ValueError("Provide a directory of preverified Linux wheels")
        for wheel in wheels:
            copy(wheel, Path("wheelhouse") / wheel.name)
    # A missing module only fails inside the container, where no trace is visible.
    for origin in sorted((output / "app").rglob("*.py")):
        for module in sorted(required_modules(origin)):
            if f"app/stage2_official/{module}.py" not in files:
                raise ValueError(f"Image context omits {module}, imported by {origin.name}")
    write_json(output / "context-manifest.json", files)
    print(f"{output}: {len(files)} inspected files", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--wheelhouse", type=Path, help="Directory containing preverified Linux wheels")
    args = parser.parse_args()
    build(args.output, args.wheelhouse)


if __name__ == "__main__":
    main()
