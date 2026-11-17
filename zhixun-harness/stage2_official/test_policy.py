import unittest

from stage2_official.policy import (
    Experiment, already_trained, effective_parameters, experiment_key, history_feedback,
)


class ExperimentHistoryTest(unittest.TestCase):
    configuration = {"action": "train", "reason": "Continue", "rate": 0.001, "batch": 128, "epochs": 1}

    def record(self, index, parent, train, decision):
        return {"round": index, "task": "B", "parent_sha256": parent, "seed": 42,
                "local_train_receipt": {"train_sha256": train}, "train_sha256": train,
                "configuration": self.configuration, "decision": decision,
                "parent_public_score": 570.0, "candidate_public_score": 590.0}

    def test_training_key_ignores_prose_and_validation_and_reads_older_receipts(self):
        old = self.record(1, "start", "batch1", "measured_keep")
        old["validation_sha256"] = "old-validation"
        del old["train_sha256"]
        changed_prose = {**self.configuration, "reason": "Try again", "recipe": "standard"}
        effective = effective_parameters(Experiment.model_validate(changed_prose))

        def seen(parent, train, seed, config=effective):
            return already_trained([old], experiment_key("B", parent, train, seed, config))

        self.assertTrue(seen("start", "batch1", 42))
        self.assertFalse(seen("next", "batch1", 42))
        self.assertFalse(seen("start", "batch2", 42))
        self.assertFalse(seen("start", "batch1", 43))
        self.assertFalse(seen("start", "batch1", 42, {**effective, "batch": 64}))
        old.update(decision="training_failed", local_train_receipt=None)
        self.assertFalse(seen("start", "batch1", 42))

    def test_feedback_keeps_relevant_failure_and_success_without_duplicate_text(self):
        records = [self.record(1, "previous", "batch1", "adopted"),
                   self.record(2, "start", "same", "measured_keep"),
                   self.record(3, "start", "different", "measured_keep"),
                   self.record(4, "start", "same", "duplicate_keep")]
        history = [{"round_index": 2, "hypothesis": "Repeated platform text",
                    "configuration": self.configuration}]
        results = history_feedback(records, history, "start", "same", 5)
        self.assertEqual([item["round"] for item in results], [2, 1, 3])
        self.assertEqual(results[0]["scores"], [570.0, 590.0])
        self.assertEqual(results[0]["scope"], "same_batch")
        self.assertNotIn("hypothesis", results[0])
        self.assertNotIn("reason", results[0]["params"])


if __name__ == "__main__":
    unittest.main()
