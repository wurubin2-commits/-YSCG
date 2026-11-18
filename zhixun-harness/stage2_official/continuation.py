"""CPU updates inspired by ZMC's diverse_trainer; use the organizer's checkpoint format."""

import math
import json

import numpy as np

from stage2_official.operating_conditions import C_GROUP_RECIPE, group_mse_weights, operating_groups


L2SP_RECIPE = "l2sp"
L2SP_MAX_STRENGTH = 10.0
SENSOR_MASK_RECIPE = "sensor_mask_v1"
GAS_SENSOR_FEATURES = 8
GAS_SENSORS = 16
B_RESIDUAL_RECIPE = "b_residual_v1"


def mask_sensor_channels(x, seed, epoch):
    """Replace one complete sensor block with its standardized mean for a quarter of rows."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 1]))
    rows = rng.choice(len(x), size=(len(x) + 2) // 4, replace=False)
    channels = rng.integers(GAS_SENSORS, size=len(rows))
    columns = channels[:, None] * GAS_SENSOR_FEATURES + np.arange(GAS_SENSOR_FEATURES)
    masked = x.copy()
    masked[rows[:, None], columns] = 0.0
    return masked


def residual_loss_gradient(error, mean_weight):
    """Loss and derivative for a mini-batch of normalized regression residuals."""
    bias = error.mean()
    centered = error - bias
    loss = np.mean(centered ** 2) + mean_weight * bias ** 2
    gradient = (2 / len(error)) * (centered + mean_weight * bias)
    return loss, gradient


def fit(model, x, y, epochs, recipe, *, l2sp_strength=None, residual_mean_weight=None, group_mse_report=None):
    if (recipe == L2SP_RECIPE) != (l2sp_strength is not None):
        raise ValueError("L2-SP strength is only defined for the L2-SP recipe")
    if l2sp_strength is not None and not 0.0 <= l2sp_strength <= L2SP_MAX_STRENGTH:
        raise ValueError("L2-SP strength out of range")
    if (recipe == B_RESIDUAL_RECIPE) != (residual_mean_weight is not None):
        raise ValueError("The residual recipe needs an explicit mean weight")
    if residual_mean_weight is not None and (
        not 0.0 < residual_mean_weight <= 1.0 or model.meta.get("profile") != "B"
        or model.meta["task"] != "regression"
    ):
        raise ValueError("Residual mean weight in (0, 1] is only available for B regression")
    # Keep the organizer code path exact for its default optimizer and loss.
    if recipe == "standard":
        return model.fit(x, y, epochs)
    classification = model.meta["task"] == "classification"
    if recipe not in ("class_balance" if classification else "huber", "l2", L2SP_RECIPE, SENSOR_MASK_RECIPE,
                      B_RESIDUAL_RECIPE, C_GROUP_RECIPE):
        raise ValueError("Recipe does not apply to this checkpoint")
    condition_weights = None
    if recipe == C_GROUP_RECIPE:
        if classification or model.meta.get("profile") != "C":
            raise ValueError("Operating-condition MSE is available only for C regression")
        condition_weights, details = group_mse_weights(operating_groups(x, model.meta["paths"]))
        if group_mse_report is not None:
            group_mse_report.update(details)
        if condition_weights is None:
            return model.fit(x, y, epochs)
    if recipe == L2SP_RECIPE and (classification or model.meta["profile"] != "B"):
        raise ValueError("L2-SP is currently available only for B")
    if recipe == SENSOR_MASK_RECIPE:
        expected = [[['key', 'features'], ['index', column]] for column in range(GAS_SENSOR_FEATURES * GAS_SENSORS)]
        if (not classification or model.meta.get("profile") != "A"
                or [json.loads(path) for path in model.meta["paths"]] != expected):
            raise ValueError("Sensor masking requires the verified 128-column A feature order")
    # Starting-point regularization (Li et al., ICML 2018). Implement the penalty
    # on this NumPy model's weight matrices, using the current parent as the anchor.
    parent_weights = {name: model.params[name].copy() for name in ("w1", "w2")} if l2sp_strength else {}
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
        seed, epoch = model.meta["seed"], model.meta["epochs"]
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        order = rng.permutation(len(x))
        epoch_x = mask_sensor_channels(x, seed, epoch) if recipe == SENSOR_MASK_RECIPE else x
        loss = 0.0
        for start in range(0, len(x), model.meta["batch_size"]):
            rows = order[start:start + model.meta["batch_size"]]
            xb, expected = epoch_x[rows], y[rows]
            hidden, logits = model.forward(xb)
            if classification:
                logits -= logits.max(axis=1, keepdims=True)
                delta = np.exp(logits)
                delta /= delta.sum(axis=1, keepdims=True)
                indices = np.arange(len(rows))
                log_likelihood = np.log(np.maximum(delta[indices, expected], 1e-300))
                delta[indices, expected] -= 1
                if recipe == "class_balance":
                    factors = weights[expected]
                    loss -= float(np.sum(factors * log_likelihood))
                    delta *= factors[:, None] / len(rows)
                else:
                    loss -= float(np.sum(log_likelihood))
                    delta /= len(rows)
            else:
                error = logits[:, 0] - expected
                if recipe == "huber":
                    clipped = np.clip(error, -1.0, 1.0)
                    loss += float(np.sum(clipped ** 2 + 2 * np.abs(error - clipped)))
                    delta = (2 * clipped / len(rows))[:, None]
                elif recipe == B_RESIDUAL_RECIPE:
                    batch_loss, gradient = residual_loss_gradient(error, residual_mean_weight)
                    loss += float(batch_loss) * len(rows)
                    delta = gradient[:, None]
                elif condition_weights is not None:
                    weighted_error = condition_weights[rows] * error
                    loss += float(np.dot(error, weighted_error))
                    delta = (2 * weighted_error / len(rows))[:, None]
                else:
                    loss += float(np.sum(error ** 2))
                    delta = (2 * error / len(rows))[:, None]
            hidden_delta = (delta @ model.params["w2"].T) * (1 - hidden ** 2)
            gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0),
                         "w1": xb.T @ hidden_delta, "b1": hidden_delta.sum(axis=0)}
            if recipe == "l2":
                for name in ("w1", "w2"):
                    gradients[name] += 0.001 * model.params[name]
            for name, parent in parent_weights.items():
                offset = model.params[name] - parent
                gradients[name] += l2sp_strength * offset
                loss += len(rows) * 0.5 * l2sp_strength * float(np.sum(offset ** 2))
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
