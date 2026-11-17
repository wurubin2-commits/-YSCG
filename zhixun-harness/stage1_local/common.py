"""Shared paths and evidence helpers for the authorized local smoke runs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "A": "a-气体识别",
    "B": "b-轴承剩余寿命预测",
    "C": "c-发动机寿命预测",
}


def package(code):
    return ROOT / TASKS[code] / "cpu工业模型"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def write_json(path, value):
    # Refuse to overwrite any previous evidence or experiment.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def source_manifest():
    result = {}
    for folder in TASKS.values():
        for path in sorted((ROOT / folder).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.is_symlink():
                raise ValueError(f"Unexpected source symlink: {path}")
            result[path.relative_to(ROOT).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    return result
