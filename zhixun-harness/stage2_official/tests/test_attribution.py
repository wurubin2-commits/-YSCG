import unittest

from stage2_official.attribution import (
    attribute, class_pressure, dominant_condition, merge_recipes, sensor_channels, target_shift,
)

CLASSES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]


def drift(indices, psi_max=0.6):
    return {"psi_mean": 0.3, "psi_max": psi_max, "n_features_drifted": len(indices),
            "top_features": [{"index": index, "psi": 0.4} for index in indices]}


class SensorChannelTest(unittest.TestCase):
    def test_one_block_owning_the_drift_is_attributed(self):
        # Columns 0-7 belong to sensor 1, so 8, 9, 15 are all channel 1.
        self.assertEqual(sensor_channels(drift([8, 9, 15])["top_features"]), {1: 3})

    def test_scattered_columns_stay_unattributed(self):
        result = attribute("A", {"labels": {name: 100 for name in CLASSES}}, classes=CLASSES,
                           drift=drift([0, 9, 22]))
        self.assertEqual(result["cause"], "covariate_shift")
        self.assertEqual(result["recipes"], [])
        self.assertIn("channel_hits", result["evidence"])

    def test_concentrated_drift_selects_sensor_mask(self):
        result = attribute("A", {"labels": {name: 100 for name in CLASSES}}, classes=CLASSES,
                           drift=drift([40, 41, 47]))
        self.assertEqual(result["cause"], "sensor_channel")
        self.assertEqual(result["channels"], [5])
        self.assertEqual(result["recipes"], ["sensor_mask_v1"])

    def test_out_of_range_column_is_rejected(self):
        self.assertIsNone(sensor_channels([{"index": 128, "psi": 0.9}]))


class ClassPressureTest(unittest.TestCase):
    def test_missing_classes_are_reported(self):
        pressure = class_pressure({"Ethanol": 90, "Ethylene": 10}, CLASSES)
        self.assertEqual(len(pressure["missing"]), 4)
        # A class absent from the batch has a zero share, which is the strongest signal.
        self.assertEqual(pressure["smallest_share"], 0.0)

    def test_empty_labels_return_none(self):
        self.assertIsNone(class_pressure({}, CLASSES))

    def test_missing_class_selects_balanced_loss(self):
        result = attribute("A", {"labels": {"Ethanol": 140, "Ethylene": 8}}, classes=CLASSES, drift=None)
        self.assertEqual(result["cause"], "label_shift")
        self.assertEqual(result["recipes"], ["class_balance"])

    def test_clean_batch_has_no_cause(self):
        result = attribute("A", {"labels": {name: 50 for name in CLASSES}}, classes=CLASSES, drift=None)
        self.assertEqual(result["cause"], "none")
        self.assertEqual(result["recipes"], [])


class TargetShiftTest(unittest.TestCase):
    def test_relative_shift_is_measured_against_the_parent(self):
        self.assertAlmostEqual(target_shift({"target_mean": 900.0}, 600.0), 0.5)
        self.assertIsNone(target_shift({"target_mean": None}, 600.0))
        self.assertIsNone(target_shift({"target_mean": float("nan")}, 600.0))

    def test_large_shift_selects_residual_recipe(self):
        result = attribute("B", {"target_mean": 1000.0}, parent_target_mean=700.0)
        self.assertEqual(result["cause"], "target_shift")
        self.assertEqual(result["recipes"], ["b_residual_v1"])

    def test_small_shift_keeps_standard(self):
        result = attribute("B", {"target_mean": 720.0}, parent_target_mean=700.0)
        self.assertEqual(result["cause"], "none")


class ConditionTest(unittest.TestCase):
    def test_dominant_group_share(self):
        self.assertAlmostEqual(dominant_condition({"0": 700, "1": 300}), 0.7)
        self.assertIsNone(dominant_condition({}))

    def test_dominant_condition_selects_group_mse(self):
        result = attribute("C", {"target_mean": 120.0}, parent_target_mean=110.0,
                           condition_counts={"0": 900, "1": 100})
        self.assertEqual(result["cause"], "operating_condition")
        self.assertEqual(result["recipes"], ["c_group_mse_v1"])

    def test_single_group_does_not_trigger_group_mse(self):
        # One populated group means there is nothing to rebalance.
        result = attribute("C", {"target_mean": 120.0}, parent_target_mean=110.0,
                           condition_counts={"0": 1000})
        self.assertEqual(result["cause"], "none")


class MergeTest(unittest.TestCase):
    def test_existing_order_is_kept_and_duplicates_dropped(self):
        self.assertEqual(merge_recipes(("standard", "l2"), ["sensor_mask_v1"]),
                         ("standard", "l2", "sensor_mask_v1"))
        self.assertEqual(merge_recipes(("standard",), ["standard"]), ("standard",))


if __name__ == "__main__":
    unittest.main()
