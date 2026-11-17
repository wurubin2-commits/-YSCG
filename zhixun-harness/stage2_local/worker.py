"""Run a single trusted evaluation or training job in an isolated subprocess."""

import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

from stage1_local.common import ROOT, package, sha256
from stage2_local.batches import read_json


def atomic_json(path, content):
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(content, file, ensure_ascii=False, allow_nan=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            file.close()
            temporary.unlink()
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def deny_network():
    def guard(event, _args):
        if event in {"socket.connect", "socket.sendto", "socket.getaddrinfo", "socket.gethostbyname"}:
            raise PermissionError("External network is unavailable in a local experiment")

    sys.addaudithook(guard)


def watch_controller(pid, created):
    import psutil

    def poll():
        while True:
            try:
                if psutil.Process(pid).create_time() != created:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(0.2)
        os._exit(75)

    threading.Thread(target=poll, daemon=True).start()


def execute(job_path):
    job = read_json(job_path)
    folder = job_path.parent
    task = job["task"]
    origin = package(task)
    validation = origin / "public/validation.jsonl"
    parent = (ROOT / job["parent"]["path"]).resolve()
    if (parent != origin / "checkpoint.npz"
            and not parent.is_relative_to(folder.parent)):
        raise ValueError("Checkpoint is outside its strategy directory")
    if sha256(parent) != job["parent"]["sha256"] or sha256(validation) != job["validation_sha256"]:
        raise ValueError("Unexpected parent or validation data")

    sys.path.insert(0, str(origin))
    from baselines.data import read_dataset
    from baselines.model import Model
    from baselines.predict import predict
    from baselines.train import metrics, train_run
    import numpy as np

    model = Model.load(parent)
    meta = model.meta
    expected = "classification" if task == "A" else "regression"
    if meta["profile"] != task or meta["task"] != expected or (task == "B" and meta.get("target_unit") != "minutes"):
        raise ValueError("This checkpoint does not match the task")
    valid = read_dataset(validation, meta["task"], meta["classes"], meta["paths"])
    reply = {"job_sha256": sha256(job_path), "before": metrics(model, valid)}
    if job["plan"] is not None:
        # The controller supplies one released path for this round.
        train = (ROOT / job["train"]).resolve()
        if not train.is_relative_to(folder.parent.parent / "batches") or sha256(train) != job["train_sha256"]:
            raise ValueError("Unexpected training batch")
        plan = job["plan"]
        started = time.perf_counter()
        report = train_run(task, train, validation, folder / "candidate", resume=parent,
                           epochs=plan["epochs"], learning_rate=plan["rate"], batch_size=plan["batch"],
                           stage_id=folder.name)
        checkpoint = folder / "candidate/checkpoint.npz"
        candidate = Model.load(checkpoint)
        if (report["parent_checkpoint_sha256"] != job["parent"]["sha256"]
                or report["train_sha256"] != job["train_sha256"]
                or report["validation_sha256"] != job["validation_sha256"]
                or report["initial_steps"] != meta["steps"] or candidate.meta["steps"] <= meta["steps"]
                or candidate.meta["epochs"] != meta["epochs"] + plan["epochs"]
                or not np.array_equal(model.mean, candidate.mean)
                or not np.array_equal(model.scale, candidate.scale)):
            raise ValueError("Continuation provenance, steps or normalization differs")
        keys = ("target_mean", "target_scale", "seed", "hidden", "classes", "paths")
        if any(meta[key] != candidate.meta[key] for key in keys):
            raise ValueError("Continuation changed the model definition")
        after = metrics(candidate, valid)
        if not all(np.isclose(score, report["validation_after"][key], rtol=1e-12, atol=1e-12)
                   for key, score in after.items()):
            raise ValueError("Saved checkpoint does not reproduce the candidate evaluation")
        with validation.open(encoding="utf-8") as stream:
            sample = json.loads(next(stream))
        if sha256(checkpoint) != report["checkpoint_sha256"]:
            raise ValueError("Candidate checksum does not match the training receipt")
        reply.update(candidate={"path": checkpoint.relative_to(ROOT).as_posix(),
                                "sha256": report["checkpoint_sha256"], "scores": after},
                     continuation_seconds=time.perf_counter() - started,
                     initial_steps=meta["steps"], final_steps=candidate.meta["steps"],
                     prediction=predict(candidate, sample["inputs"]))
    atomic_json(folder / "result.json", reply)


if __name__ == "__main__":
    deny_network()
    watch_controller(int(sys.argv[2]), float(sys.argv[3]))
    execute(Path(sys.argv[1]).resolve())
