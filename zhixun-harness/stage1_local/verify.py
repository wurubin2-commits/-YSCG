"""Verify finished offline records against local training receipts; never sync."""
import argparse
import json
import math
from pathlib import Path

from google.protobuf.json_format import MessageToDict
from swanlab.sdk.internal.core_python.store import DataStoreReader
from swanlab.proto.swanlab.record.v1.record_pb2 import Record

from common import ROOT, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    folder = args.run.resolve()
    if not folder.is_relative_to(ROOT / "stage1_local/runs"):
        parser.error("Expected a stage1_local/runs directory")
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    initial = json.loads((ROOT / "stage1_local/evidence/initial_source_manifest.json").read_text(encoding="utf-8"))
    before = json.loads((folder / "source_manifest_before.json").read_text(encoding="utf-8"))
    after = json.loads((folder / "source_manifest_after.json").read_text(encoding="utf-8"))
    assert initial == before == after, "Original source manifests differ"
    checks = {"original_file_count": len(initial), "original_bytes": sum(x["bytes"] for x in initial.values()),
              "initial_and_run_manifests_match": True, "tasks": {}}
    for code, outcome in summary["tasks"].items():
        assert outcome["returncode"] == 0, code
        result = outcome["result"]
        assert result["swanlab_mode"] == "offline" and not result["blocked_network_events"]
        candidate = folder / code / "candidate"
        receipt = json.loads((candidate / "training.json").read_text(encoding="utf-8"))
        assert sha256(candidate / "checkpoint.npz") == receipt["checkpoint_sha256"]
        assert receipt["parent_checkpoint_sha256"] == result["parent_checkpoint_sha256"]
        assert sha256(ROOT / result["selected_checkpoint"]) == result["selected_sha256"]
        files = list((folder / code / "swanlab").rglob("*.swanlab"))
        assert len(files) == 1, (code, files)
        reader = DataStoreReader()
        reader.open(files[0])
        scalar = {}
        finished = False
        count = 0
        try:
            while True:
                data = reader.scan()  # Raises on a bad header/checksum; no silent truncation.
                if data is None:
                    break
                record = Record()
                record.ParseFromString(data)
                item = MessageToDict(record, preserving_proto_field_name=True)
                count += 1
                if "scalar" in item:
                    entry = item["scalar"]
                    scalar[(entry["key"], int(entry.get("step", 0)))] = entry["value"].get("number", 0.0)
                if "finish" in item:
                    finished = item["finish"]["state"] == "RUN_STATE_FINISHED"
        finally:
            reader.close()
        assert finished, f"SwanLab run did not finish: {code}"
        for phase, step in (("baseline", 0), ("candidate", result["epochs_added"])):
            for key, expected in result[phase].items():
                actual = scalar[(f"validation/{key}", step)]
                assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (code, key)
        assert ("train/loss", 1) in scalar
        assert scalar[("decision/accepted", result["epochs_added"] + 1)] == int(result["decision"] == "accept_candidate")
        checks["tasks"][code] = {"offline_file": str(files[0].relative_to(ROOT)),
                                  "record_count": count, "finished": True,
                                  "baseline_candidate_and_decision_match": True,
                                  "candidate_checkpoint_hash_matches": True}
    write_json(folder / "verification.json", checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
