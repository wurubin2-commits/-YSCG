"""Export locally acknowledged SDK round decisions to SwanLab offline files."""

import argparse
from pathlib import Path

from stage1_local.common import ROOT, write_json
from stage2_local.tracking import record_history
from stage2_official.rehearse import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("Choose an output directory in the project")
    tasks = load(run / "summary.json")["tasks"]
    output.mkdir(parents=True)
    exports = {}
    for task, result in tasks.items():
        metric = result["metric"]
        state = load(run / task / f"round-{result['rounds']:02d}/state/rounds.json")
        rows = []
        for item in state:
            parent, candidate = item["parent_public_score"], item["candidate_public_score"]
            if parent is None or candidate is None:
                raise ValueError("This export requires measured candidate and parent scores")
            accepted = item["decision"] == "adopted"
            evidence = {k: item[k] for k in ("hypothesis", "reason", "configuration", "seed", "model",
                                             "model_call_ids", "history_ids", "usage", "elapsed_seconds",
                                             "parent_sha256", "candidate_sha256", "selected_artifact")}
            rows.append({"round": item["round"], "status": "accepted" if accepted else item["decision"],
                         "parent": {"scores": {metric: parent}},
                         "candidate": {"scores": {metric: candidate}},
                         "selected": {"scores": {metric: candidate if accepted else parent}},
                         "gain": (candidate - parent) * (1 if task == "A" else -1), "plan": evidence,
                         "batch": item["batch"], "epochs_reserved": item["epochs_attempted"], "runtime": None})
        folder = output / task
        folder.mkdir()
        exports[task] = record_history(folder, task, "official-sdk-mock-host", metric, rows)
    write_json(output / "verification.json", exports)
    print(f"{output}: {len(exports)} verified offline experiment files")


if __name__ == "__main__":
    main()
