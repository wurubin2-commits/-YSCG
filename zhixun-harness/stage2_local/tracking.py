"""Write and verify offline SwanLab records from committed round decisions."""

import math
import os
from uuid import uuid4

from stage1_local.common import ROOT, write_json


def verify(path, metric, history):
    from google.protobuf.json_format import MessageToDict
    from swanlab.proto.swanlab.record.v1.record_pb2 import Record
    from swanlab.sdk.internal.core_python.store import DataStoreReader

    files = list(path.rglob("*.swanlab"))
    if len(files) != 1:
        raise ValueError("Expected one SwanLab offline file")
    reader = DataStoreReader()
    reader.open(files[0])
    finished = False
    values = {}
    try:
        while (data := reader.scan()) is not None:
            record = Record()
            record.ParseFromString(data)
            msg = MessageToDict(record, preserving_proto_field_name=True)
            if "scalar" in msg:
                scalar = msg["scalar"]
                key = scalar["key"], int(scalar.get("step", 0))
                values[key] = scalar["value"].get("number", 0.0)
            if "finish" in msg:
                finished = msg["finish"]["state"] == "RUN_STATE_FINISHED"
    finally:
        reader.close()
    if not finished:
        raise ValueError("SwanLab log is unfinished")
    for item in history:
        step = item["round"]
        if (values["decision/accepted", step] != int(item["status"] == "accepted")
                or not math.isclose(values[f"public/{metric}", step], item["selected"]["scores"][metric],
                                    rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("SwanLab records differ from stored decisions")
    return {"mode": "offline", "file": files[0].relative_to(ROOT).as_posix(), "verified_rounds": len(history)}


def log_history(metric, history):
    """Log one committed round series; shared by the offline and cloud writers."""
    import swanlab

    swanlab.log({f"public/{metric}": history[0]["parent"]["scores"][metric]}, step=0)
    for row in history:
        metrics = {f"public/{metric}": row["selected"]["scores"][metric],
                   "decision/accepted": int(row["status"] == "accepted"),
                   "data/rows": row["batch"]["rows"], "train/reserved_epochs": row["epochs_reserved"]}
        if row["candidate"] is not None:
            metrics.update({f"public/candidate_{metric}": row["candidate"]["scores"][metric],
                            "public/signed_gain": row["gain"]})
        if row["runtime"] and "seconds" in row["runtime"]:
            metrics.update({"resources/seconds": row["runtime"]["seconds"],
                            "resources/peak_worker_mib": row["runtime"]["peak_mib"]})
        swanlab.log(metrics, step=row["round"])


def record_history(root, task, strategy, metric, history):
    destination = root / "swanlab"
    if destination.exists():
        return verify(destination, metric, history)
    staging = root / f".swanlab-{uuid4().hex[:12]}"
    for key in list(os.environ):
        if key.upper().startswith("SWANLAB_"):
            os.environ.pop(key)
    os.environ.update(SWANLAB_MODE="offline", SWANLAB_ROOT=str(root / "swanlab_settings"))
    import swanlab

    settings = swanlab.Settings(mode="offline", interactive=False, api_key=None,
        probe=swanlab.Settings.Probe(hardware=False, runtime=False, requirements=False, git=False,
                                    conda=False, monitor=False, swanlab=True),
        integration=swanlab.Settings.Integration(webhook=swanlab.Settings.Integration.Webhook(url="", value="")))
    swanlab.init(project="zhixun-local", experiment_name=f"{task}-{strategy}", logdir=str(staging),
                 mode="offline", settings=settings, config={"task": task, "strategy": strategy,
                     "decisions": [{"round": row["round"], "status": row["status"], "plan": row["plan"]}
                                   for row in history]})
    try:
        log_history(metric, history)
    finally:
        swanlab.finish()
    verify(staging, metric, history)
    staging.rename(destination)
    return verify(destination, metric, history)


def record_history_online(root, task, strategy, metric, history):
    """Upload the same round series to the SwanLab cloud project.

    The organizer asks for a queryable cloud record, not only a local file.
    Credentials stay in the environment and are never written to disk.
    """
    api_key = os.environ.get("SWANLAB_API_KEY")
    project = os.environ.get("SWANLAB_PROJECT")
    if not api_key or not project:
        raise ValueError("Online recording needs SWANLAB_API_KEY and SWANLAB_PROJECT")
    import swanlab

    experiment = f"{task}-{strategy}"
    run = swanlab.init(project=project, experiment_name=experiment, mode="cloud", api_key=api_key,
                       config={"task": task, "strategy": strategy,
                               "decisions": [{"round": row["round"], "status": row["status"],
                                              "plan": row["plan"]} for row in history]})
    try:
        log_history(metric, history)
    finally:
        swanlab.finish()
    try:
        url = run.url
    except ValueError:
        url = None
    report = {"mode": "cloud", "project": project, "experiment_name": experiment,
              "run_id": str(getattr(run, "id", "") or ""), "url": url, "rounds": len(history)}
    write_json(root / "swanlab-cloud.json", report)
    return report
