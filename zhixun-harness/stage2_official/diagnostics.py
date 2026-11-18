"""Covariate drift between the public training and validation projections.

PSI needs no labels and no model: it compares the two feature distributions that
the platform actually mounts. Bin edges are pooled quantiles and the bin count
follows the smaller projection, which keeps the statistic usable when the two
projections differ in size.
"""

from pathlib import Path
from random import Random

import numpy as np

EPS = 1e-6
MIN_BINS = 5
MAX_BINS = 10
MIN_ROWS = 32
SAMPLE_LIMIT = 512
DRIFT_CUTOFF = 0.2


def psi_column(reference, current):
    """Population stability index between two samples of one feature."""
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if len(reference) < MIN_ROWS or len(current) < MIN_ROWS:
        return None
    if reference.std() < EPS or current.std() < EPS:
        return 0.0
    n_bins = int(np.clip(min(len(reference), len(current)) // 50, MIN_BINS, MAX_BINS))
    edges = np.quantile(np.concatenate([reference, current]), np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    previous = np.histogram(reference, bins=edges)[0].astype(float)
    later = np.histogram(current, bins=edges)[0].astype(float)
    if previous.sum() <= 0 or later.sum() <= 0:
        return None
    previous = previous / previous.sum()
    later = later / later.sum()
    # An empty bin must not dominate: floor it at half a sample instead of 1e-6.
    floor = 0.5 / max(len(reference), len(current))
    previous = np.maximum(previous, floor)
    later = np.maximum(later, floor)
    return float(((previous - later) * np.log(previous / later)).sum())


def _subsample(matrix, limit=SAMPLE_LIMIT):
    """Random, order-independent sample; released rows are often sorted by one column."""
    if len(matrix) <= limit:
        return matrix
    rng = np.random.default_rng(42)
    index = rng.choice(len(matrix), size=limit, replace=False)
    return matrix[np.sort(index)]


def diagnose(train_x, validation_x, top_k=5):
    """Return per-feature PSI, or None when the two projections are incomparable."""
    train_x = np.asarray(train_x, dtype=np.float64)
    validation_x = np.asarray(validation_x, dtype=np.float64)
    if train_x.ndim != 2 or validation_x.ndim != 2 or train_x.shape[1] != validation_x.shape[1]:
        return None
    train_x = _subsample(train_x)
    validation_x = _subsample(validation_x)
    if len(train_x) < MIN_ROWS or len(validation_x) < MIN_ROWS:
        return None
    values = [psi_column(train_x[:, index], validation_x[:, index]) for index in range(train_x.shape[1])]
    if any(value is None for value in values):
        return None
    psi = np.asarray(values, dtype=np.float64)
    order = np.argsort(psi)[::-1][:top_k]
    return {"psi_mean": round(float(psi.mean()), 6), "psi_max": round(float(psi.max()), 6),
            "n_features_drifted": int((psi > DRIFT_CUTOFF).sum()),
            "top_features": [{"index": int(index), "psi": round(float(psi[index]), 6)} for index in order]}


def collect_features(rows, flatten, limit=SAMPLE_LIMIT):
    """Yield every row while keeping a bounded uniform sample of flattened features."""
    sample, rng, seen = [], Random(42), 0

    def observed():
        nonlocal seen
        for row in rows:
            try:
                values = flatten(row["inputs"])[1]
            except (TypeError, ValueError, KeyError):
                values = None
            if values is not None:
                if seen < limit:
                    sample.append(values)
                else:
                    slot = rng.randrange(seen + 1)
                    if slot < limit:
                        sample[slot] = values
                seen += 1
            yield row

    return observed(), sample


_CACHE = {}


def validation_matrix(task, path, model):
    """Parse the validation projection once per file identity and task."""
    path = Path(path)
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, task)
    if key not in _CACHE:
        from stage2_official.trainer import baseline
        data = baseline(task, "data").read_dataset(path, model.meta["task"], model.meta["classes"],
                                                   model.meta["paths"])
        _CACHE.clear()
        _CACHE[key] = data["x"]
    return _CACHE[key]
