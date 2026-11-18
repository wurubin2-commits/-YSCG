import unittest

import numpy as np

from stage2_official.diagnostics import collect_features, diagnose, psi_column


class PsiColumnTest(unittest.TestCase):
    def test_same_distribution_scores_near_zero(self):
        rng = np.random.default_rng(7)
        left, right = rng.normal(size=400), rng.normal(size=400)
        self.assertLess(psi_column(left, right), 0.1)

    def test_shifted_distribution_exceeds_cutoff(self):
        rng = np.random.default_rng(7)
        self.assertGreater(psi_column(rng.normal(size=400), rng.normal(1.0, 1.0, 400)), 0.2)

    def test_small_or_constant_columns_are_handled(self):
        self.assertIsNone(psi_column(np.arange(10), np.arange(10)))
        self.assertEqual(psi_column(np.ones(64), np.zeros(64)), 0.0)

    def test_constant_reference_with_variation_is_zero(self):
        rng = np.random.default_rng(3)
        self.assertEqual(psi_column(np.zeros(64), rng.normal(size=64)), 0.0)


class DiagnoseTest(unittest.TestCase):
    def test_reports_structure_and_drift_count(self):
        rng = np.random.default_rng(11)
        train = rng.normal(size=(300, 4))
        validation = train.copy()
        validation[:, 2] += 1.5
        report = diagnose(train, validation, top_k=2)
        self.assertEqual(report["n_features_drifted"], 1)
        self.assertEqual(report["top_features"][0]["index"], 2)
        self.assertEqual(len(report["top_features"]), 2)
        self.assertGreater(report["psi_max"], 0.2)

    def test_mismatched_shapes_return_none(self):
        self.assertIsNone(diagnose(np.zeros((64, 3)), np.zeros((64, 4))))
        self.assertIsNone(diagnose(np.zeros(64), np.zeros(64)))
        self.assertIsNone(diagnose(np.zeros((10, 3)), np.zeros((10, 3))))

    def test_subsampling_keeps_large_projections_cheap(self):
        rng = np.random.default_rng(5)
        report = diagnose(rng.normal(size=(5000, 2)), rng.normal(size=(4000, 2)))
        self.assertEqual(report["psi_mean"].__class__, float)
        self.assertEqual(len(report["top_features"]), 2)


class CollectFeaturesTest(unittest.TestCase):
    def test_bounded_sample_and_full_pass_through(self):
        rows = [{"inputs": {"a": [float(index), 1.0]}} for index in range(600)]
        flatten = lambda inputs: (None, inputs["a"])
        observed, sample = collect_features(rows, flatten, limit=100)
        consumed = list(observed)
        self.assertEqual(len(consumed), 600)
        self.assertEqual(len(sample), 100)

    def test_bad_rows_do_not_abort_the_stream(self):
        rows = [{"inputs": "bad"}, {"inputs": {"a": [1.0]}}, {"inputs": {"a": [2.0]}}]
        flatten = lambda inputs: (None, inputs["a"])
        observed, sample = collect_features(rows, flatten)
        consumed = list(observed)
        self.assertEqual(len(consumed), 3)
        self.assertEqual(len(sample), 2)


if __name__ == "__main__":
    unittest.main()
