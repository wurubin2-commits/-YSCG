import unittest

import numpy as np

from stage2_official.continuation import mask_sensor_channels


class SensorMaskTest(unittest.TestCase):
    def test_masks_exactly_one_contiguous_sensor_without_touching_input(self):
        features = np.ones((44, 128), dtype=np.float64)
        masked = mask_sensor_channels(features, seed=42, epoch=20)
        np.testing.assert_array_equal(masked, mask_sensor_channels(features, seed=42, epoch=20))
        self.assertFalse(np.array_equal(masked, mask_sensor_channels(features, seed=42, epoch=21)))
        channels = (masked == 0).reshape(44, 16, 8)
        np.testing.assert_array_equal(channels.any(axis=2), channels.all(axis=2))
        self.assertEqual(int(channels.all(axis=2).sum()), 11)
        self.assertLessEqual(int(channels.all(axis=2).sum(axis=1).max()), 1)
        self.assertTrue(np.all(features == 1.0))
