"""Export locally acknowledged SDK round decisions to SwanLab offline files."""

import argparse
from pathlib import Path

from stage1_local.common import ROOT, write_json
from stage2_local.tracking import record_history, record_history_online
from stage2_official.rehearse import CONTRACTS, load


def cost_summary(rows, metric):
    """Unit-gain cost: the organizer scores gain per compute, token and quota spend."""
    gain = sum(row["gain"] for row in rows if row["gain"] is not None)
    epochs = sum(row["epochs_reserved"] or 0 for row in rows)
    tokens = sum((row["plan"].get("usage") or {}).get("total_tokens", 0) or 0 for row in rows)
    seconds = sum(row["plan"].get("elapsed_seconds") or 0.0 for row in rows)
    accepted = sum(row["status"] == "accepted" for row in rows)
    return {"metric": metric, "rounds": len(rows), "accepted": accepted,
            "cumulative_gain": round(gain, 6), "epochs_spent": epochs,
            "tokens_spent": tokens, "wall_seconds": round(seconds, 4),
            "gain_per_epoch": round(gain / epochs, 6) if epochs else None,
            "gain_per_1k_tokens": round(gain / tokens * 1000, 6) if tokens else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cloud", action="store_true",
                        help="Upload to SwanLab cloud; needs SWANLAB_API_KEY and SWANLAB_PROJECT")
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("Choose an output directory in the project")
    writer = record_history_online if args.cloud else record_history
    tasks = load(run / "summary.json")["tasks"]
    output.mkdir(parents=True)
    exports, costs = {}, {}
    for task, result in tasks.items():
        metric = result["metric"]
        state = load(run / task / f"round-{result['rounds']:02d}/state/rounds.json")
        rows = []
        for item in state:
            parent, candidate = item["parent_public_score"], item["candidate_public_score"]
            if parent is None:
                raise ValueError("This export requires a measured parent score")
            accepted = item["decision"] == "adopted"
            evidence = {k: item[k] for k in ("hypothesis", "reason", "configuration", "seed", "model",
                                             "model_call_ids", "history_ids", "usage", "elapsed_seconds",
                                             "parent_sha256", "candidate_sha256", "selected_artifact")}
            receipt = item["local_train_receipt"] or {}
            evidence.update(stability=receipt.get("stability"), experiment_key=item.get("experiment_key"),
                            batch_shift=item.get("batch_shift"), input_drift=item.get("input_drift"))
            evidence.update({key: item[key] for key in ("prompt_bytes", "output_token_limit", "feedback_rounds",
                                                        "summary_seconds", "model_seconds") if key in item})
            # The metric direction lives in the contract table; a hardcoded task test would
            # silently export an inverted gain for any task added later.
            direction = CONTRACTS[task][2]
            gain = ((candidate - parent) * (1 if direction == "higher" else -1)
                    if candidate is not None else None)
            seconds = item.get("elapsed_seconds")
            rows.append({"round": item["round"], "status": "accepted" if accepted else item["decision"],
                         "parent": {"scores": {metric: parent}},
                         "candidate": {"scores": {metric: candidate}} if candidate is not None else None,
                         "selected": {"scores": {metric: candidate if accepted else parent}},
                         "gain": gain,
                         "gain_per_second": (round(gain / seconds, 6)
                                             if gain is not None and seconds else None),
                         "plan": evidence,
                         "batch": item["batch"], "epochs_reserved": item["epochs_attempted"], "runtime": None})
        folder = output / task
        folder.mkdir()
        exports[task] = writer(folder, task, "official-sdk-mock-host", metric, rows)
        costs[task] = cost_summary(rows, metric)
    write_json(output / "verification.json", exports)
    write_json(output / "cost.json", costs)
    print(f"{output}: {len(exports)} verified offline experiment files")


if __name__ == "__main__":
    main()
