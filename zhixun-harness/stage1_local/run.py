"""Run existing tests and small CPU continuations, serially, with local-only logs."""
import argparse
from datetime import datetime, timezone, timedelta
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import psutil

from common import ROOT, package, source_manifest, write_json, sha256

HERE = Path(__file__).resolve().parent
MAX_SECONDS = 1800
MAX_TASK_SECONDS = 300
MAX_BYTES = 2 * 1024**3


def process_tree(pid):
    try:
        parent = psutil.Process(pid)
        return [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def kill_tree(pid):
    for child in reversed(process_tree(pid)):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass


def output_bytes():
    return sum(p.stat().st_size for p in (HERE / "runs").rglob("*") if p.is_file())


def execute(command, cwd, log, env, deadline):
    log = Path(log).resolve()
    started = time.monotonic()
    peak_rss = 0
    stopped = None
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            while process.poll() is None:
                now = time.monotonic()
                rss = 0
                # Windows venv python.exe can be a small launcher whose child
                # runs the interpreter; sampling only its PID undercounts RAM.
                for member in process_tree(process.pid):
                    try:
                        rss += member.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak_rss = max(peak_rss, rss)
                if now >= deadline or now - started >= MAX_TASK_SECONDS:
                    stopped = "time_limit"
                elif output_bytes() >= MAX_BYTES:
                    stopped = "output_limit"
                elif psutil.virtual_memory().available < 256 * 1024**2:
                    stopped = "low_system_memory"
                if stopped:
                    kill_tree(process.pid)
                    break
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                kill_tree(process.pid)
            process.wait()
    return {"returncode": process.returncode, "stop_reason": stopped,
            "seconds": time.monotonic() - started, "peak_process_tree_rss_MiB": round(peak_rss / 1024**2, 2),
            "log": str(log.relative_to(ROOT))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("A", "B", "C", "all"), default="all")
    parser.add_argument("--epochs", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / ".venv-zhixun").resolve():
        parser.error("Use the project's .venv-zhixun interpreter")
    if version("numpy") != "1.26.4" or version("swanlab") != "0.10.3":
        parser.error("Install the pinned stage1_local/requirements.txt first")
    if psutil.virtual_memory().available < 1024**3:
        parser.error("Less than 1 GiB memory is available; leave apps untouched and retry after memory is freed")
    (HERE / "runs").mkdir(exist_ok=True)
    if output_bytes() >= MAX_BYTES:
        parser.error("Stage 1 output limit reached; retain evidence and stop")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    run_dir = HERE / "runs" / f"{stamp}-{uuid.uuid4().hex[:6]}"
    run_dir.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("SWANLAB_", "PYTRIO_", "TRIO_"))}
    env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", BASELINE_CPU_THREADS="1",
               SWANLAB_MODE="offline", PIP_DISABLE_PIP_VERSION_CHECK="1")
    print(f"Evidence: {run_dir}", flush=True)
    before = source_manifest()
    write_json(run_dir / "source_manifest_before.json", before)
    print(f"Preserved source inventory: {len(before)} files", flush=True)
    lock = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True,
                          encoding="utf-8", env=env, check=True)
    (run_dir / "requirements.lock.txt").write_text(lock.stdout, encoding="utf-8")
    check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True,
                           encoding="utf-8", env=env)
    write_json(run_dir / "environment.json", {"python": sys.executable, "version": sys.version,
               "numpy": version("numpy"), "swanlab": version("swanlab"), "psutil": version("psutil"),
               "pip_check_returncode": check.returncode, "pip_check": check.stdout + check.stderr,
               "memory_available_GiB": psutil.virtual_memory().available / 1024**3,
               "wrapper_sha256": {p.name: sha256(p) for p in HERE.glob("*.py")}})
    if check.returncode:
        raise RuntimeError("Dependency conflicts; see environment.json")
    summary = {"scope": "local public validation only", "official_scoring_ready": False,
               "run_dir": str(run_dir.relative_to(ROOT)), "tests": {}, "tasks": {}}
    deadline = time.monotonic() + MAX_SECONDS
    for code in ("ABC" if args.task == "all" else args.task):
        test = execute([sys.executable, "-B", "-m", "unittest", "discover", "-s", "baselines/tests", "-v"],
                       package(code), run_dir / f"tests_{code}.log", env, deadline)
        summary["tests"][code] = test
        # Preserve known B upstream packaging errors, but do not proceed on other failures.
        log = (run_dir / f"tests_{code}.log").read_text(encoding="utf-8")
        known_b = (code == "B" and test["returncode"] == 1 and
                   log.count("ModuleNotFoundError: No module named 'scripts'") == 2 and
                   "Ran 12 tests" in log and "FAILED (errors=2)" in log and
                   "test_b_package_delivers_numeric_rul_code_and_weights_without_hidden_data" in log and
                   "test_documented_edition_preserves_trained_code_weights_and_data" in log)
        test["known_upstream_packaging_errors_only"] = known_b
        if test["returncode"] and not known_b:
            summary["tasks"][code] = {"status": "blocked_by_tests"}
            break
        outcome = execute([sys.executable, "-B", str(HERE / "worker.py"), "--task", code,
                           "--output", str(run_dir / code), "--epochs", str(args.epochs)],
                          run_dir, run_dir / f"worker_{code}.log", env, deadline)
        summary["tasks"][code] = outcome
        result_path = run_dir / code / "result.json"
        if result_path.exists():
            outcome["result"] = json.loads(result_path.read_text(encoding="utf-8"))
        print(f"{code}: tests exit={test['returncode']}, worker exit={outcome['returncode']}", flush=True)
        if outcome["returncode"]:
            break
    after = source_manifest()
    write_json(run_dir / "source_manifest_after.json", after)
    summary["original_sources_unchanged"] = before == after
    summary["output_bytes_all_stage1_runs"] = output_bytes()
    write_json(run_dir / "summary.json", summary)
    print(f"Summary: {run_dir / 'summary.json'}", flush=True)
    expected = set("ABC" if args.task == "all" else args.task)
    complete = (set(summary["tasks"]) == expected and before == after and
                all(t.get("returncode") == 0 and "result" in t for t in summary["tasks"].values()))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
