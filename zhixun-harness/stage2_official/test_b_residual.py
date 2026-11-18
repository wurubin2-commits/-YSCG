"""Verify the B residual loss against MSE and the organizer's resumed Adam step."""

from copy import deepcopy
import unittest

import numpy as np

from stage2_official.continuation import B_RESIDUAL_RECIPE, fit, residual_loss_gradient
from stage2_official.trainer import baseline


class BearingResidualTest(unittest.TestCase):
    def test_weight_one_matches_mse_loss_and_gradient(self):
        residual = np.array([-3.0, 0.1, 1.5, 5.0])
        loss, gradient = residual_loss_gradient(residual, 1.0)
        self.assertAlmostEqual(loss, float(np.mean(residual ** 2)), places=12)
        np.testing.assert_allclose(gradient, 2 * residual / len(residual), atol=1e-12, rtol=0)

    def test_half_weight_gradient_matches_finite_differences(self):
        residual = np.array([-3.0, 0.1, 1.5, 5.0])
        _, gradient = residual_loss_gradient(residual, 0.5)
        width = 1e-5
        finite_difference = []
        for displacement in np.eye(len(residual)) * width:
            right = residual_loss_gradient(residual + displacement, 0.5)[0]
            left = residual_loss_gradient(residual - displacement, 0.5)[0]
            finite_difference.append((right - left) / (2 * width))
        np.testing.assert_allclose(gradient, finite_difference, atol=1e-9, rtol=0)

    def test_weight_one_matches_resumed_adam_update(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(32, 3))
        y = np.linspace(0.0, 180.0, len(x))
        create = baseline("B", "model").Model
        parent = create("regression", ["a", "b", "c"], [], x, y, batch_size=32)
        parent.meta["profile"] = "B"
        parent.fit(x, y, 2)
        standard, variant = deepcopy(parent), deepcopy(parent)
        standard_loss = standard.fit(x, y, 1)
        variant_loss = fit(variant, x, y, 1, B_RESIDUAL_RECIPE, residual_mean_weight=1.0)
        np.testing.assert_allclose(standard_loss, variant_loss, atol=1e-12, rtol=0)
        self.assertEqual(standard.meta, variant.meta)
        self.assertEqual(standard.meta["steps"], parent.meta["steps"] + 1)
        for field in ("params", "first", "second"):
            for name, value in getattr(standard, field).items():
                np.testing.assert_allclose(value, getattr(variant, field)[name], atol=1e-12, rtol=0)
