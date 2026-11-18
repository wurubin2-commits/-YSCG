"""Frozen C operating-condition bins and per-round weights for group-balanced MSE."""

import json

import numpy as np


C_GROUP_RECIPE = "c_group_mse_v1"
GROUP_FRACTION = 0.25
MIN_GROUP_ROWS = 128
CENTERS = (
    (0, 0.00, 100), (10, 0.25, 100), (20, 0.70, 100),
    (25, 0.62, 60), (35, 0.84, 100), (42, 0.84, 100),
)
TOLERANCE = (0.05, 0.005, 0.5)
OTHER = len(CENTERS)


def operating_groups(x, feature_paths):
    """Map the three raw settings to fixed centers. An unrecognized row is Other."""
    paths = [json.loads(path) for path in feature_paths]
    columns = [paths.index([["key", "settings"], ["index", i]]) for i in range(3)]
    settings = x[:, columns]
    groups = np.full(len(x), OTHER, dtype=np.intp)
    for index, center in enumerate(CENTERS):
        groups[np.all(np.abs(settings - center) <= TOLERANCE, axis=1)] = index
    return groups


def group_mse_weights(groups):
    """Mix ordinary MSE with an equal-weight average over supported groups."""
    raw_counts = np.bincount(groups, minlength=OTHER + 1)
    merged = np.where(raw_counts[groups] < MIN_GROUP_ROWS, OTHER, groups)
    merged_counts = np.bincount(merged, minlength=OTHER + 1)
    present = merged_counts[merged_counts > 0]
    active = len(present) >= 2 and int(present.min()) >= MIN_GROUP_ROWS
    weights = None
    if active:
        weights = (1 - GROUP_FRACTION) + GROUP_FRACTION * len(groups) / (len(present) * merged_counts[merged])
    report = {"group_fraction": GROUP_FRACTION, "min_group_rows": MIN_GROUP_ROWS,
              "recognized_and_other_counts": raw_counts.tolist(),
              "merged_counts": merged_counts.tolist(), "weighted": active}
    return weights, report
