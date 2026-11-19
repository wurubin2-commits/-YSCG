import unittest

from stage2_official.normalize import normalize_proposal, rule_plan


class NormalizeTest(unittest.TestCase):
    def plan(self, body, recipes=("standard",), task="B"):
        return normalize_proposal(body, recipes, task)

    def train(self, **overrides):
        return {"action": "train", "reason": "Continue", "rate": 0.001, "batch": 128, "epochs": 1, **overrides}

    def test_rate_clamped_both_sides_and_original_kept(self):
        plan, report = self.plan(self.train(rate=0.05))
        self.assertEqual(plan.rate, 0.005)
        self.assertEqual(report["clamped"]["rate"], {"from": 0.05, "to": 0.005})
        self.assertEqual(report["source"], "normalized")
        plan, report = self.plan(self.train(rate=0.000001))
        self.assertEqual(plan.rate, 0.00005)

    def test_batch_snaps_to_nearest_option(self):
        plan, report = self.plan(self.train(batch=100))
        self.assertEqual(plan.batch, 128)
        self.assertEqual(report["clamped"]["batch"], {"from": 100, "to": 128})
        plan, _ = self.plan(self.train(batch=33))
        self.assertEqual(plan.batch, 32)
        plan, report = self.plan(self.train(batch="128"))
        self.assertEqual((plan.batch, report["clamped"]["batch"]["to"]), (128, 128))

    def test_epochs_clamped(self):
        plan, report = self.plan(self.train(epochs=9))
        self.assertEqual(plan.epochs, 2)
        plan, report = self.plan(self.train(epochs=0))
        self.assertEqual(plan.epochs, 1)

    def test_unknown_recipe_falls_back_to_standard(self):
        plan, report = self.plan(self.train(recipe="turbo"))
        self.assertEqual(plan.recipe, "standard")
        self.assertEqual(report["clamped"]["recipe"], {"from": "turbo", "to": "standard"})

    def test_keep_with_parameters_is_cleaned(self):
        plan, report = self.plan({"action": "keep", "reason": "Wait",
                                  "rate": 0.001, "batch": 64, "epochs": 1})
        self.assertEqual((plan.rate, plan.batch, plan.epochs), (None, None, None))
        self.assertEqual(report["source"], "normalized")
        self.assertEqual(report["clamped"]["rate"], {"from": 0.001, "to": None})

    def test_unusable_action_and_text_return_none(self):
        plan, report = self.plan({"action": "stop", "reason": "Done"})
        self.assertIsNone(plan)
        self.assertEqual(report["source"], "invalid_action")
        plan, report = self.plan("not json at all")
        self.assertIsNone(plan)
        self.assertEqual(report["source"], "unparseable")

    def test_fenced_reply_is_recovered(self):
        text = 'Here you go:\n```json\n{"action":"train","reason":"Try","rate":0.002,"batch":64,"epochs":1}\n```'
        plan, report = self.plan(text)
        self.assertEqual((plan.rate, plan.batch, plan.epochs), (0.002, 64, 1))
        self.assertEqual(report["source"], "model")

    def test_l2sp_strength_defaults_and_is_dropped_elsewhere(self):
        plan, report = self.plan(self.train(recipe="l2sp"), recipes=("standard", "l2sp"))
        self.assertEqual(plan.l2sp_strength, 1.0)
        self.assertEqual(report["clamped"]["l2sp_strength"]["to"], 1.0)
        plan, report = self.plan(self.train(l2sp_strength=2.0))
        self.assertIsNone(plan.l2sp_strength)
        self.assertEqual(report["clamped"]["l2sp_strength"]["to"], None)

    def test_long_reason_is_trimmed_and_empty_reason_is_filled(self):
        plan, report = self.plan(self.train(reason="x" * 400))
        self.assertEqual(len(plan.reason), 200)
        self.assertEqual(report["clamped"]["reason"], {"from": {"length": 400}, "to": {"length": 200}})
        plan, report = self.plan(self.train(reason="   "))
        self.assertTrue(plan.reason)
        self.assertEqual(report["source"], "normalized")

    def test_clean_proposal_is_reported_as_model(self):
        plan, report = self.plan(self.train())
        self.assertEqual(report["source"], "model")
        self.assertEqual(report["clamped"], {})


class RulePlanTest(unittest.TestCase):
    def record(self, gain, candidate=1.0, rate=0.003, batch=128):
        return {"gain": gain, "candidate_public_score": candidate,
                "configuration": {"action": "train", "rate": rate, "batch": batch, "epochs": 1}}

    def test_default_plan_without_history(self):
        plan, source = rule_plan("B", [], 5)
        self.assertEqual((plan.rate, plan.batch, plan.epochs), (0.003, 128, 1))
        self.assertEqual(source, "rule_default")

    def test_half_rate_after_a_non_positive_gain(self):
        plan, source = rule_plan("A", [self.record(-0.01)], 5)
        self.assertEqual(plan.rate, 0.0015)
        self.assertEqual(source, "rule_half_rate")

    def test_repeat_guard_lowers_rate_again(self):
        prior = [{"gain": None, "candidate_public_score": None,
                  "configuration": {"rate": 0.003, "batch": 128}}]
        plan, source = rule_plan("B", prior, 5)
        self.assertEqual(plan.rate, 0.0015)
        self.assertEqual(source, "rule_half_rate_shift")

    def test_stops_after_full_exploration_without_gains(self):
        prior = [self.record(0.01), self.record(-0.02), self.record(-0.01)]
        self.assertEqual(rule_plan("A", prior, 5), (None, None))

    def test_no_remaining_budget(self):
        self.assertEqual(rule_plan("C", [], 0), (None, None))

    def test_regression_task_uses_its_own_defaults(self):
        plan, _ = rule_plan("C", [], 3)
        self.assertEqual((plan.rate, plan.batch), (0.002, 256))


if __name__ == "__main__":
    unittest.main()
