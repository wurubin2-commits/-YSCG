"""Release successive slices of the public training data for local testing."""

from collections import Counter
from contextlib import closing, contextmanager
import hashlib
import json
from pathlib import Path

from stage1_local.common import package, sha256


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def group_key(task, meta):
    if task == "A":
        return str(meta["source_batch"])
    if task == "B":
        return meta["bearing"]
    return meta["source_subset"], meta["unit"]


def checked_sources(task):
    origin = package(task)
    receipt = read_json(origin / "training.json")
    hashes = {
        "public/train.jsonl": receipt["train_sha256"],
        "public/validation.jsonl": receipt["validation_sha256"],
        "checkpoint.npz": receipt["checkpoint_sha256"],
        f"baselines/configs/{task}.json": receipt["profile_sha256"],
        **{f"baselines/{name}": digest for name, digest in receipt["source_sha256"].items()},
    }
    for name, digest in hashes.items():
        path = origin / name
        if path.is_symlink() or sha256(path) != digest:
            raise ValueError(f"Source verification failed: {path}")
    return receipt


@contextmanager
def public_batches(task, count, receipt, destination):
    origin = package(task)
    validation_ids, validation_groups = set(), set()
    with (origin / "public/validation.jsonl").open("rb") as validation:
        for line in validation:
            if line.strip():
                row = json.loads(line)
                validation_ids.add(row["id"])
                validation_groups.add(group_key(task, row["meta"]))
    with (origin / "public/train.jsonl").open("rb") as train:
        rows = (line for line in train if line.strip())
        with closing(write_batches(task, count, receipt["train_rows"], rows, destination,
                                   validation_ids, validation_groups)) as batches:
            yield batches


def write_batches(task, count, total, rows, destination, validation_ids, validation_groups):
    if total < count:
        raise ValueError("A batch would be empty")
    seen = set()
    for number in range(1, count + 1):
        size = number * total // count - (number - 1) * total // count
        lines, groups = [], set()
        labels = Counter()
        target_total = 0.0
        for line in rows:
            row = json.loads(line)
            key = group_key(task, row["meta"])
            if row["id"] in seen or row["id"] in validation_ids or key in validation_groups:
                raise ValueError("Duplicate training ID or training/validation overlap")
            seen.add(row["id"])
            groups.add(key)
            if task == "A":
                labels[row["answer"]] += 1
            else:
                target_total += row["answer"]
            lines.append(line)
            if len(lines) == size:
                break
        if len(lines) != size:
            raise ValueError("Training data ended before its recorded row count")
        payload = b"".join(lines)
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        path = destination / f"batch-{number:02d}.jsonl"
        if path.exists():
            if path.is_symlink() or sha256(path) != digest:
                raise ValueError(f"Saved batch differs from its source: {path}")
        else:
            temporary = path.with_suffix(".partial")
            temporary.unlink(missing_ok=True)
            with temporary.open("xb") as stream:
                stream.write(payload)
            temporary.replace(path)
        summary = {"rows": size, "groups": len(groups), "sha256": digest}
        if task == "A":
            summary["class_counts"] = dict(labels)
        else:
            summary["target_mean"] = target_total / size
        yield number, path, summary
    if next(rows, None) is not None:
        raise ValueError("Training data exceeds its recorded row count")
