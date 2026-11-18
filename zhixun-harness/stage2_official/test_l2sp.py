"""Check that L2-SP preserves the organizer B model's continuation semantics."""

from copy import deepcopy
import unittest

import numpy as np

from stage2_official.continuation import fit
from stage2_official.trainer import baseline


class L2SPTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        self.x = rng.normal(size=(96, 3))
        self.y = np.linspace(0.0, 180.0, len(self.x))
        model = baseline("B", "model").Model
        self.parent = model("regression", ["a", "b", "c"], [], self.x, self.y, batch_size=32)
        self.parent.meta["profile"] = "B"
        self.parent.fit(self.x, self.y, 2)

    def assert_same_state(self, first, second):
        self.assertEqual(first.meta, second.meta)
        for name in first.params:
            for field in ("params", "first", "second"):
                np.testing.assert_array_equal(getattr(first, field)[name], getattr(second, field)[name])
        np.testing.assert_array_equal(first.mean, second.mean)
        np.testing.assert_array_equal(first.scale, second.scale)

    def test_zero_strength_matches_standard_continuation(self):
        reference, candidate = deepcopy(self.parent), deepcopy(self.parent)
        self.assertEqual(reference.fit(self.x, self.y, 2),
                         fit(candidate, self.x, self.y, 2, "l2sp", l2sp_strength=0.0))
        self.assert_same_state(reference, candidate)

    def test_anchor_starts_at_parent_and_affects_later_updates(self):
        reference, candidate = deepcopy(self.parent), deepcopy(self.parent)
        reference.fit(self.x[:32], self.y[:32], 1)
        fit(candidate, self.x[:32], self.y[:32], 1, "l2sp", l2sp_strength=1.0)
        self.assert_same_state(reference, candidate)

        reference, candidate = deepcopy(self.parent), deepcopy(self.parent)
        reference.fit(self.x, self.y, 1)
        fit(candidate, self.x, self.y, 1, "l2sp", l2sp_strength=1.0)
        self.assertEqual(reference.meta, candidate.meta)
        self.assertFalse(np.array_equal(reference.params["w1"], candidate.params["w1"]))
