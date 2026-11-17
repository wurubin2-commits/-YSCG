"""在官方 Model 结构上支持「配置增量」的轻量训练器。

支持的配置键（都作用于一次续训）：
  lr           学习率覆盖
  batch        批大小覆盖
  epochs       训练轮数
  seed         打乱随机种子
  l2           L2 权重衰减
  class_weight "balanced" 时按类别频率反比加权（分类）
  huber        True 时回归用 Huber 损失
  recency      0..1，越晚的样本权重越高（漂移窗口加权）
  bootstrap    整数种子，有放回重采样

训练后模型状态（params/adam/meta）与官方 checkpoint 完全兼容，可直接 model.save。
"""
import math

import numpy as np


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_config(model, x, y, cfg):
    task = model.meta["task"]
    lr = float(cfg.get("lr", model.meta["learning_rate"]))
    batch_size = int(cfg.get("batch", model.meta["batch_size"]))
    epochs = int(cfg.get("epochs", 1))
    seed = int(cfg.get("seed", model.meta["seed"]))
    l2 = float(cfg.get("l2", 0.0))
    huber = bool(cfg.get("huber", False))
    recency = float(cfg.get("recency", 0.0))
    class_weight = cfg.get("class_weight")
    bootstrap = cfg.get("bootstrap")

    xn = model.normalized(x)
    n = len(xn)
    if task == "regression":
        target = (y - model.meta["target_mean"]) / model.meta["target_scale"]
        w = None
    else:
        target = y.astype(np.int64)
        classes = len(model.meta["classes"])
        if class_weight == "balanced":
            counts = np.bincount(target, minlength=classes).astype(float)
            w = counts.sum() / (classes * np.maximum(counts, 1.0))
        else:
            w = np.ones(classes)

    sw = np.ones(n)
    if recency > 0:
        pos = np.linspace(0.0, 1.0, n)
        sw = 1.0 - recency * (1.0 - pos)

    bootstrap_rng = np.random.default_rng(int(bootstrap)) if bootstrap is not None else None
    losses = []
    for _ in range(epochs):
        if bootstrap_rng is not None:
            order = bootstrap_rng.integers(0, n, size=n)
        else:
            rng = np.random.default_rng(np.random.SeedSequence([seed, model.meta["epochs"]]))
            order = rng.permutation(n)
        loss_sum = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            batch, exp = xn[idx], target[idx]
            hidden, out = model.forward(batch)
            if task == "classification":
                probs = _softmax(out)
                cw = w[exp] * sw[idx]
                loss_sum -= float(np.sum(cw * np.log(np.maximum(probs[np.arange(len(idx)), exp], 1e-300))))
                delta = probs
                delta[np.arange(len(idx)), exp] -= 1.0
                delta *= (cw / len(idx))[:, None]
            else:
                err = out[:, 0] - exp
                if huber:
                    d = 1.0
                    abs_err = np.abs(err)
                    mask = abs_err <= d
                    loss = np.where(mask, 0.5 * err ** 2, d * (abs_err - 0.5 * d))
                    grad = np.where(mask, err, d * np.sign(err))
                else:
                    loss = 0.5 * err ** 2
                    grad = err
                loss_sum += float(np.sum(loss * sw[idx]))
                delta = (grad * sw[idx] / len(idx))[:, None]
            grads = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
            hd = (delta @ model.params["w2"].T) * (1.0 - hidden ** 2)
            grads.update(w1=batch.T @ hd, b1=hd.sum(axis=0))
            norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
            if not math.isfinite(norm):
                raise ValueError("nonfinite gradient")
            model.meta["steps"] += 1
            for name, grad in grads.items():
                grad = grad * min(1.0, 5.0 / max(norm, 1e-12))
                if l2 > 0:
                    grad = grad + l2 * model.params[name]
                model.first[name] = 0.9 * model.first[name] + 0.1 * grad
                model.second[name] = 0.999 * model.second[name] + 0.001 * grad ** 2
                m1 = model.first[name] / (1 - 0.9 ** model.meta["steps"])
                m2 = model.second[name] / (1 - 0.999 ** model.meta["steps"])
                model.params[name] -= lr * m1 / (np.sqrt(m2) + 1e-8)
        model.meta["epochs"] += 1
        losses.append(loss_sum / max(n, 1))
    model.validate()
    return losses


def evaluate(model, x, y):
    """返回验证指标字典（分类: accuracy/macro_f1；回归: rmse/mae）。"""
    pred = model.predict(x)
    if model.meta["task"] == "classification":
        labels = pred.argmax(axis=1)
        y = y.astype(np.int64)
        f1s = []
        for c in range(len(model.meta["classes"])):
            tp = int(((labels == c) & (y == c)).sum())
            fp = int(((labels == c) & (y != c)).sum())
            fn = int(((labels != c) & (y == c)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        return {"accuracy": float(np.mean(labels == y)), "macro_f1": float(np.mean(f1s))}
    y = y.astype(np.float64)
    err = pred - y
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err)))}


def ensemble_predict(parent, cand, x):
    """父模型 + 候选模型的集成预测（仅作诊断记录）。"""
    pa, pb = parent.predict(x), cand.predict(x)
    return (pa + pb) / 2.0


def ensemble_evaluate(parent, cand, x, y):
    pred = ensemble_predict(parent, cand, x)
    if cand.meta["task"] == "classification":
        labels = pred.argmax(axis=1)
        y = y.astype(np.int64)
        f1s = []
        for c in range(len(cand.meta["classes"])):
            tp = int(((labels == c) & (y == c)).sum())
            fp = int(((labels == c) & (y != c)).sum())
            fn = int(((labels != c) & (y == c)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        return {"accuracy": float(np.mean(labels == y)), "macro_f1": float(np.mean(f1s))}
    y = y.astype(np.float64)
    err = pred - y
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err)))}
