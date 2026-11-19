import unittest

from pydantic import ValidationError

from stage2_official.policy import (
    Experiment, already_trained, c_half_rate, effective_parameters, experiment_key, history_feedback,
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

    def test_feedback_carries_the_blocked_slice_of_a_refusal(self):
        row = self.record(2, "start", "same", "stability_keep")
        row["local_train_receipt"]["stability"] = [
            {"slice": "low_rul", "metric": "mae", "before": 308.91, "after": 448.64, "passed": False},
            {"slice": "mid_rul", "metric": "mae", "before": 200.0, "after": 190.0, "passed": True}]
        results = history_feedback([row], [], "start", "same", 3)
        self.assertEqual([item["round"] for item in results], [2])
        self.assertEqual(results[0]["blocked_slices"],
                         [{"slice": "low_rul", "metric": "mae", "before": 308.91, "after": 448.64}])

    def test_l2sp_strength_changes_experiment_identity(self):
        trial = {**self.configuration, "recipe": "l2sp", "l2sp_strength": 1.0}
        record = self.record(1, "start", "batch1", "measured_keep")
        record["configuration"] = trial
        for strength, expected in ((1.0, True), (2.0, False)):
            effective = effective_parameters(Experiment.model_validate({**trial, "l2sp_strength": strength}))
            self.assertEqual(already_trained([record], experiment_key("B", "start", "batch1", 42, effective)), expected)
        with self.assertRaises(ValidationError):
            Experiment.model_validate({**self.configuration, "recipe": "l2sp"})
        with self.assertRaises(ValidationError):
            Experiment.model_validate({**trial, "l2sp_strength": 0.0})

    def test_c_half_rate_preserves_budget_and_records_effective_rate(self):
        plan = Experiment.model_validate(self.configuration)
        candidate = c_half_rate(plan)
        self.assertEqual(candidate.rate, 0.0005)
        self.assertEqual((candidate.batch, candidate.epochs, candidate.reason), (plan.batch, plan.epochs, plan.reason))
        self.assertEqual(effective_parameters(candidate)["rate"], 0.0005)
        low = Experiment.model_validate({**self.configuration, "rate": 0.000075})
        self.assertEqual(c_half_rate(low).rate, 0.00005)
        keep = Experiment(action="keep", reason="No training")
        self.assertIs(c_half_rate(keep), keep)

    def test_b_residual_requires_weight_and_deduplicates_by_value(self):
        trial = {**self.configuration, "recipe": "b_residual_v1", "residual_mean_weight": 0.5}
        record = self.record(1, "start", "batch1", "measured_keep")
        record["configuration"] = trial
        for weight, expected in ((0.5, True), (1.0, False)):
            plan = Experiment.model_validate({**trial, "residual_mean_weight": weight})
            key = experiment_key("B", "start", "batch1", 42, effective_parameters(plan))
            self.assertEqual(already_trained([record], key), expected)
        for weight in (None, 0.0, 1.1):
            with self.assertRaises(ValidationError):
                Experiment.model_validate({**trial, "residual_mean_weight": weight})
        with self.assertRaises(ValidationError):
            Experiment.model_validate({**trial, "recipe": "standard"})


if __name__ == "__main__":
    unittest.main()
