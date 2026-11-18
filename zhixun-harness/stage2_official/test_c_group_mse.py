"""Check the condition groups, the mixed objective and the standard-MSE fallback."""

from copy import deepcopy
import json
import unittest

import numpy as np

from stage2_official.continuation import fit
from stage2_official.operating_conditions import (
    CENTERS, C_GROUP_RECIPE, MIN_GROUP_ROWS, OTHER, group_mse_weights, operating_groups,
)
from stage2_official.trainer import baseline


SETTINGS_PATHS = [json.dumps([["key", "settings"], ["index", i]]) for i in range(3)]


class ConditionGroupMSETest(unittest.TestCase):
    def test_assigns_centers_and_uses_every_setting(self):
        base = np.asarray(CENTERS)
        jitter = np.array([0.01, 0.002, 0.0])
        x = np.vstack([base + jitter, (10.01, 0.35, 100.0), (25.0, 0.62, 100.0)])
        expected = list(range(len(CENTERS))) + [OTHER, OTHER]
        np.testing.assert_array_equal(operating_groups(x, SETTINGS_PATHS), expected)

    def test_weights_reproduce_mixed_objective(self):
        groups = np.repeat([0, 1, 2], [128, 256, 512])
        error = np.linspace(-4, 9, len(groups))
        weights, report = group_mse_weights(groups)
        self.assertTrue(report["weighted"])
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        condition_losses = [np.mean(error[groups == i] ** 2) for i in range(3)]
        expected = .75 * np.mean(error ** 2) + .25 * np.mean(condition_losses)
        self.assertAlmostEqual(float(np.mean(weights * error ** 2)), float(expected))

    def test_merges_small_groups_or_uses_standard_mse(self):
        counts = [256, 64, 64]
        _, report = group_mse_weights(np.repeat([0, 1, 2], counts))
        self.assertEqual(report["merged_counts"][OTHER], MIN_GROUP_ROWS)
        self.assertTrue(report["weighted"])
        weights, report = group_mse_weights(np.repeat([0, 1], [256, 64]))
        self.assertIsNone(weights)
        self.assertFalse(report["weighted"])

        x = np.repeat(np.array(CENTERS[:1]), 16, axis=0)
        y = np.arange(len(x), dtype=float)
        parent = baseline("C", "model").Model("regression", SETTINGS_PATHS, [], x, y, batch_size=8)
        parent.meta["profile"] = "C"
        standard, trial = deepcopy(parent), deepcopy(parent)
        self.assertEqual(standard.fit(x, y, 1), fit(trial, x, y, 1, C_GROUP_RECIPE))
        self.assertEqual(standard.meta, trial.meta)
        for field in ("params", "first", "second"):
            for name, values in getattr(standard, field).items():
                np.testing.assert_array_equal(getattr(trial, field)[name], values)
