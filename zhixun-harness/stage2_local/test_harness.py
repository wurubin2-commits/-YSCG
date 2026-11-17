"""Check failure branches, durable decisions and the model request contract."""

import asyncio
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import httpx2
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.function import FunctionModel

from stage1_local.common import ROOT, package, sha256
from stage2_local.batches import write_batches
from stage2_local.planner import create_agent, pytrio_connector, suggest
from stage2_local.run import relative, run_round, worker
from stage2_local.worker import atomic_json


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.loop = self.enterContext(asyncio.Runner())
        self.loop.get_loop()
        temporary = self.enterContext(TemporaryDirectory(prefix="test-harness-", dir=ROOT / "stage2_local"))
        self.folder = Path(temporary)
        self.db = self.enterContext(closing(sqlite3.connect(":memory:")))
        self.db.execute("CREATE TABLE rounds (task, strategy, number, intent, decision, PRIMARY KEY(task,strategy,number))")
        baseline = package("A") / "checkpoint.npz"
        self.parent = {"path": relative(baseline), "sha256": sha256(baseline), "scores": {"macro_f1": 0.7}}
        self.batch = {"sha256": "test-digest", "rows": 1, "groups": 1, "class_counts": {"gas": 1}}
        self.source = {"batch_size": 128, "validation_sha256": "test-validation"}

    def round(self, number=1, history=(), *, strategy="fixed", agent=None):
        return run_round(self.db, "A", strategy, number, self.folder / "batch", self.batch, history,
                         self.parent, self.source, self.folder, agent, time.monotonic() + 30)

    def publish_candidate(self, job, score):
        path = job.parent / "candidate/checkpoint.npz"
        path.parent.mkdir()
        path.write_bytes(b"simulated artifact for controller test")
        atomic_json(job.parent / "result.json", {
            "job_sha256": sha256(job), "before": self.parent["scores"],
            "candidate": {"path": relative(path), "sha256": sha256(path), "scores": {"macro_f1": score}},
            "initial_steps": 10, "final_steps": 11, "continuation_seconds": 0.2,
        })

    def test_worker_failures_do_not_replace_current_model(self):
        def corrupt(job, _deadline):
            (job.parent / "result.json").write_text('{"before":', encoding="utf-8")
            return {"status": "ok"}

        with patch("stage2_local.run.worker", side_effect=corrupt):
            first = self.round()
        with patch("stage2_local.run.worker", return_value={"status": "timeout"}) as launch:
            self.assertEqual(self.round(), first)
            launch.assert_not_called()
            second = self.round(2, [first])
        self.assertEqual(first["status"], "invalid_candidate")
        self.assertEqual(second["status"], "timeout")
        self.assertEqual(first["selected"], self.parent)
        self.assertEqual(second["selected"], self.parent)

    def test_process_deadline_terminates_slow_worker(self):
        from subprocess import Popen

        def launch(_command, **kwargs):
            return Popen([sys.executable, "-B", "-c", "import time; time.sleep(30)"], **kwargs)

        with patch("stage2_local.run.subprocess.Popen", side_effect=launch):
            result = worker(self.folder / "job.json", time.monotonic() + 5, timeout=0.3)
        self.assertEqual(result["status"], "timeout")
        self.assertLess(result["seconds"], 5)

    def test_rejected_candidate_keeps_existing_checkpoint(self):
        def worse(job, _deadline):
            self.publish_candidate(job, 0.65)
            return {"status": "ok"}

        with patch("stage2_local.run.worker", side_effect=worse):
            decision = self.round()
        self.assertEqual(decision["status"], "kept")
        self.assertLess(decision["gain"], 0)
        self.assertEqual(decision["selected"], self.parent)

    def test_recovers_complete_result_after_interruption(self):
        def finishes(job, _deadline):
            self.publish_candidate(job, 0.8)
            raise KeyboardInterrupt

        with patch("stage2_local.run.worker", side_effect=finishes):
            with self.assertRaises(KeyboardInterrupt):
                self.round()
        with patch("stage2_local.run.worker") as launch:
            result = self.round()
            self.assertEqual(result["status"], "accepted")
            self.assertEqual(result["epochs_reserved"], 1)
            self.assertEqual(self.round(), result)
            launch.assert_not_called()
            (ROOT / result["selected"]["path"]).write_bytes(b"corrupted artifact")
            with self.assertRaisesRegex(ValueError, "changed"):
                self.round()

    def test_interrupted_launch_without_result_does_not_repeat(self):
        with patch("stage2_local.run.worker", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.round()
        with patch("stage2_local.run.worker") as launch:
            result = self.round()
            self.assertEqual(result["status"], "interrupted")
            self.assertEqual(result["selected"], self.parent)
            self.assertEqual(result["epochs_reserved"], 1)
            launch.assert_not_called()

    def test_invalid_model_plan_and_epoch_budget_prevent_training(self):
        model = FunctionModel(lambda *_: ModelResponse(parts=[TextPart('{"action":"train","epochs":99}')]))
        with patch("stage2_local.run.worker") as launch:
            self.assertEqual(self.round(strategy="adaptive", agent=create_agent(model))["status"], "invalid_plan")
            prior = {"round": 1, "epochs_reserved": 12, "batch": self.batch, "plan": None, "status": "kept", "gain": 0}
            self.assertEqual(self.round(2, [prior])["status"], "epoch_budget")
            launch.assert_not_called()

    def test_future_batch_reads_wait_and_validation_groups_are_excluded(self):
        consumed = []

        def lines():
            for i in (1, 2):
                consumed.append(i)
                row = {"id": str(i), "meta": {"source_batch": i}, "answer": "gas"}
                yield json.dumps(row).encode() + b"\n"

        batches = write_batches("A", 2, 2, lines(), self.folder, set(), {"2"})
        self.assertEqual(next(batches)[0], 1)
        self.assertEqual(consumed, [1])
        with self.assertRaisesRegex(ValueError, "overlap"):
            next(batches)
        self.assertFalse((self.folder / "batch-02.jsonl").exists())

    def test_pytrio_request_uses_plain_chat_and_does_not_retry_rate_limit(self):
        requests = []

        def reply(request):
            requests.append(request)
            body = json.loads(request.content)
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(body["model"], "approved-model")
            self.assertEqual(body["max_tokens"], 512)
            self.assertNotIn("tools", body)
            self.assertNotIn("response_format", body)
            if len(requests) == 2:
                return httpx2.Response(429, json={"error": {"message": "simulated limit"}})
            return httpx2.Response(200, json={"id": "mock", "object": "chat.completion", "created": 0,
                "model": "approved-model", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": '{"action":"keep","reason":"simulated response"}'}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}})

        agent, client = pytrio_connector("https://contest.invalid/v1", "approved-model", "test",
                                         transport=httpx2.MockTransport(reply))
        try:
            self.assertEqual(suggest(agent, {"round": 1}).action, "keep")
            with self.assertRaises(ModelHTTPError):
                suggest(agent, {"round": 2})
            self.assertEqual(len(requests), 2)
        finally:
            self.loop.run(client.close())
