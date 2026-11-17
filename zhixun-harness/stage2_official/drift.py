"""Bounded input-drift diagnostic on the actual released batches."""

from random import Random

import numpy as np

SAMPLE_LIMIT = 512
MIN_ROWS = 32
EDGES = (-np.inf, -2.0, -0.5, 0.5, 2.0, np.inf)


def sample_rows(rows):
    """Yield every row for validation while retaining a uniform, bounded input sample."""
    sample, rng = [], Random(42)

    def observed():
        for index, row in enumerate(rows):
            if index < SAMPLE_LIMIT:
                sample.append(row["inputs"])
            else:
                slot = rng.randrange(index + 1)
                if slot < SAMPLE_LIMIT:
                    sample[slot] = row["inputs"]
            yield row

    return observed(), sample


def histogram(sample, model, flatten):
    if len(sample) < MIN_ROWS:
        return None
    values = []
    try:
        for inputs in sample:
            paths, features = flatten(inputs)
            if paths != model.meta["paths"]:
                return None
            values.append(features)
        normalized = model.normalized(np.asarray(values, dtype=np.float64))
    except (TypeError, ValueError):
        return None
    return [np.histogram(column, bins=EDGES)[0].tolist() for column in normalized.T]


def drift_score(previous, current):
    """Mean total variation of normalized feature histograms, from zero to one."""
    if previous is None or current is None:
        return None
    before, after = np.asarray(previous, dtype=float), np.asarray(current, dtype=float)
    if before.shape != after.shape:
        return None
    before /= before.sum(axis=1, keepdims=True)
    after /= after.sum(axis=1, keepdims=True)
    return round(float(np.abs(after - before).sum(axis=1).mean() / 2), 6)
