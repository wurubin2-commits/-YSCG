"""Read local rehearsal receipts and verify the matched round sequence."""

from dataclasses import dataclass
import json
from math import isclose, isfinite

METRICS = {"A": "macro_f1", "B": "rmse", "C": "rmse"}


@dataclass(frozen=True)
class Round:
    task: str
    number: int
    parent_score: float
    candidate_score: float
    selected_score: float
    adopted: bool
    parent_mae: float | None
    candidate_mae: float | None
    rate: float
    rows: int
    updates: int
    elapsed: float
    validation_sha: str
    reason: str

    @property
    def gain(self):
        if self.task == "A":
            return self.candidate_score - self.parent_score
        return self.parent_score - self.candidate_score


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rounds(folder, task):
    source = folder / task / "rounds.json"
    if not source.is_file():
        source = folder / task / "round-10/state/rounds.json"
    records = read_json(source)
    if [row["round"] for row in records] != list(range(1, 11)):
        raise ValueError(f"{source}: expected rounds 1–10")
    return records


def evaluate(record, task, expected_parent):
    local = record["local_train_receipt"]
    parent = record["parent_public_score"]
    candidate = record["candidate_public_score"]
    adopted = record["decision"] == "adopted"
    metric = METRICS[task]
    if record["task"] != task or record["metric"] != metric:
        raise ValueError(f"Unexpected task or metric for {task}")
    if record["decision"] not in {"adopted", "measured_keep"}:
        raise ValueError(f"Unsupported decision for this normal rehearsal: {record['decision']}")
    if not all(isfinite(value) for value in (parent, candidate, record["elapsed_seconds"])):
        raise ValueError(f"Nonfinite score or elapsed time for {task}")
    if expected_parent is not None and record["parent_sha256"] != expected_parent:
        raise ValueError(f"Broken parent chain for {task} round {record['round']}")
    for field in ("train_sha256", "validation_sha256"):
        if record.get(field, local[field]) != local[field]:
            raise ValueError(f"Receipt {field} mismatch for {task}")
    if not (isclose(parent, local["validation_before"][metric], abs_tol=1e-12)
            and isclose(candidate, local["validation_after"][metric], abs_tol=1e-12)):
        raise ValueError(f"Receipt score mismatch for {task}")
    chosen = "candidate" if adopted else "parent"
    if record["selected_artifact"] != record[f"{chosen}_artifact"]:
        raise ValueError(f"Artifact mismatch for {task}")
    if adopted and (candidate <= parent if task == "A" else candidate >= parent):
        raise ValueError(f"Adoption did not improve the metric for {task}")
    train_rows = local.get("train_rows", record.get("batch", {}).get("rows"))
    if type(train_rows) is not int or train_rows < 1:
        raise ValueError(f"Missing training row count for {task}")
    return Round(
        task, record["round"], parent, candidate, candidate if adopted else parent, adopted,
        local["validation_before"].get("mae") if task != "A" else None,
        local["validation_after"].get("mae") if task != "A" else None,
        record["configuration"]["rate"], train_rows,
        local["final_steps"] - local["initial_steps"], record["elapsed_seconds"],
        local["validation_sha256"], record["reason"],
    ), record[f"{chosen}_sha256"]


def read_comparison(reference, current):
    reference_summary, current_summary = (
        read_json(folder / "summary.json")["tasks"] for folder in (reference, current)
    )
    comparison = {}
    for task in METRICS:
        original, compact = (read_rounds(folder, task) for folder in (reference, current))
        selected_sha = None
        rounds = []
        for old, new in zip(original, compact, strict=True):
            old_round, old_selected = evaluate(old, task, selected_sha)
            new_round, new_selected = evaluate(new, task, selected_sha)
            receipt_fields = ("train_sha256", "validation_sha256", "initial_steps", "final_steps")
            same_receipt = all(old["local_train_receipt"][key] == new["local_train_receipt"][key]
                               for key in receipt_fields)
            same_parameters = all(old["configuration"][key] == new["configuration"][key]
                                  for key in ("rate", "batch", "epochs"))
            if not (same_receipt and same_parameters and old["seed"] == new["seed"]
                    and old["parent_sha256"] == new["parent_sha256"]
                    and old["candidate_sha256"] == new["candidate_sha256"]
                    and old_selected == new_selected and old_round.adopted == new_round.adopted
                    and old_round.parent_score == new_round.parent_score
                    and old_round.candidate_score == new_round.candidate_score):
                raise ValueError(f"Rehearsals differ for {task} round {new_round.number}")
            rounds.append(new_round)
            selected_sha = new_selected
        for summary in (reference_summary[task], current_summary[task]):
            if (summary["selected_sha256"] != selected_sha
                    or summary["selected_public"] != rounds[-1].selected_score):
                raise ValueError(f"Final summary mismatch for {task}")
        comparison[task] = rounds
    return comparison
