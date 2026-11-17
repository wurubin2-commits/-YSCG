"""CPU updates inspired by ZMC's diverse_trainer; use the organizer's checkpoint format."""

import math

import numpy as np


def fit(model, x, y, epochs, recipe):
    # Keep the organizer code path exact for its default optimizer and loss.
    if recipe == "standard":
        return model.fit(x, y, epochs)
    classification = model.meta["task"] == "classification"
    if recipe not in ("class_balance" if classification else "huber", "l2"):
        raise ValueError("Recipe does not apply to this checkpoint")
    x = model.normalized(x)
    if not len(x) or y.shape != (len(x),):
        raise ValueError("Invalid target shape")
    if classification:
        classes = len(model.meta["classes"])
        if np.any(y < 0) or np.any(y >= classes):
            raise ValueError("Unexpected class index")
        counts = np.bincount(y, minlength=classes)
        # Limit amplification of tiny classes, then keep mean weight equal to one.
        weights = np.clip(len(y) / (classes * np.maximum(counts, 1)), 0.25, 4.0)
        weights /= weights[y].mean()
    else:
        y = (y - model.meta["target_mean"]) / model.meta["target_scale"]
    losses = []
    for _ in range(epochs):
        rng = np.random.default_rng(np.random.SeedSequence([model.meta["seed"], model.meta["epochs"]]))
        order = rng.permutation(len(x))
        loss = 0.0
        for start in range(0, len(x), model.meta["batch_size"]):
            rows = order[start:start + model.meta["batch_size"]]
            xb, expected = x[rows], y[rows]
            hidden, logits = model.forward(xb)
            if classification:
                logits -= logits.max(axis=1, keepdims=True)
                delta = np.exp(logits)
                delta /= delta.sum(axis=1, keepdims=True)
                indices = np.arange(len(rows))
                factors = weights[expected] if recipe == "class_balance" else 1.0
                loss -= float(np.sum(factors * np.log(np.maximum(delta[indices, expected], 1e-300))))
                delta[indices, expected] -= 1
                delta *= np.asarray(factors).reshape(-1, 1) / len(rows)
            else:
                error = logits[:, 0] - expected
                if recipe == "huber":
                    clipped = np.clip(error, -1.0, 1.0)
                    loss += float(np.sum(clipped ** 2 + 2 * np.abs(error - clipped)))
                    delta = (2 * clipped / len(rows))[:, None]
                else:
                    loss += float(np.sum(error ** 2))
                    delta = (2 * error / len(rows))[:, None]
            hidden_delta = (delta @ model.params["w2"].T) * (1 - hidden ** 2)
            gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0),
                         "w1": xb.T @ hidden_delta, "b1": hidden_delta.sum(axis=0)}
            if recipe == "l2":
                for name in ("w1", "w2"):
                    gradients[name] += 0.001 * model.params[name]
            norm = math.sqrt(sum(float(np.sum(g ** 2)) for g in gradients.values()))
            if not math.isfinite(norm):
                raise ValueError("Nonfinite gradient")
            model.meta["steps"] += 1
            step = model.meta["steps"]
            for name, grad in gradients.items():
                grad *= min(1.0, 5.0 / max(norm, 1e-12))
                model.first[name] = 0.9 * model.first[name] + 0.1 * grad
                model.second[name] = 0.999 * model.second[name] + 0.001 * grad ** 2
                first = model.first[name] / (1 - 0.9 ** step)
                second = model.second[name] / (1 - 0.999 ** step)
                model.params[name] -= model.meta["learning_rate"] * first / (np.sqrt(second) + 1e-8)
        model.meta["epochs"] += 1
        losses.append(loss / len(x))
    model.validate()
    return losses


def protected_slices(parent, validation):
    """Same-validation comparisons. Anchor membership never feeds training."""
    predicted = parent.predict(validation["x"])
    y = validation["y"]
    if parent.meta["task"] == "classification":
        return [(label, y == i, "recall", i) for i, label in enumerate(parent.meta["classes"])
                if np.count_nonzero(y == i) >= 20], predicted
    low = y <= np.quantile(y, 0.25)
    return ([("low_rul", low, "mae", None)] if np.count_nonzero(low) >= 20 else []), predicted


def stability(parent, candidate, validation):
    slices, original = protected_slices(parent, validation)
    predicted, y = candidate.predict(validation["x"]), validation["y"]
    comparisons = []
    for name, mask, metric, i in slices:
        if metric == "recall":
            before = float(np.mean(original[mask].argmax(axis=1) == i))
            after = float(np.mean(predicted[mask].argmax(axis=1) == i))
            passed = after >= before - 0.12
        else:
            before = float(np.mean(np.abs(original[mask] - y[mask])))
            after = float(np.mean(np.abs(predicted[mask] - y[mask])))
            passed = after <= before * 1.10 + 1.0
        comparisons.append({"slice": name, "rows": int(mask.sum()), "metric": metric,
                            "before": before, "after": after, "passed": passed})
    return comparisons
