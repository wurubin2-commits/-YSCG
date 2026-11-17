"""轮间漂移诊断：公开训练投影 vs 验证投影的协变量漂移 + 父模型残差诊断。

复用 M0 漂移监测的 PSI 思想（合并分位数分箱、bin 数自适应），但对象从
"仿真批次流"换成平台每轮挂载的公开 JSONL 投影。所有统计仅用授权公开数据。
"""
import numpy as np

EPS = 1e-6
MIN_BINS = 5
MAX_BINS = 10


def _psi(p, q):
    p = p / (p.sum() + EPS) + EPS
    q = q / (q.sum() + EPS) + EPS
    return float(((p - q) * np.log(p / q)).sum())


def psi_feature(ref_col, cur_col):
    if ref_col.std() < EPS or cur_col.std() < EPS:
        return 0.0
    n_bins = int(np.clip(min(len(ref_col), len(cur_col)) // 50, MIN_BINS, MAX_BINS))
    edges = np.quantile(np.concatenate([ref_col, cur_col]), np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref_col, bins=edges)[0].astype(float)
    q = np.histogram(cur_col, bins=edges)[0].astype(float)
    return _psi(p, q)


def diagnose(model, train, validation, top_k=5):
    """返回结构化诊断：协变量漂移（train→validation）、父模型在两个投影上的表现。

    train / validation 为 baselines.data.read_dataset 的输出。
    """
    x_train, x_val = train["x"], validation["x"]
    psi = np.array([psi_feature(x_train[:, j], x_val[:, j]) for j in range(x_train.shape[1])])
    order = np.argsort(psi)[::-1][:top_k]
    drift = {
        "psi_mean": float(psi.mean()),
        "psi_max": float(psi.max()),
        "n_features_drifted": int((psi > 0.2).sum()),
        "top_features": [{"index": int(j), "psi": round(float(psi[j]), 4)} for j in order],
    }

    def _performance(data):
        pred = model.predict(data["x"])
        y = data["y"]
        if model.meta["task"] == "classification":
            labels = pred.argmax(axis=1)
            observed = sorted(set(y.tolist()) | set(labels.tolist()))
            f1 = [2 * np.sum((labels == k) & (y == k)) / max(np.sum(labels == k) + np.sum(y == k), 1)
                  for k in observed]
            return {"accuracy": float((labels == y).mean()), "macro_f1": float(np.mean(f1)),
                    "confidence": float(pred.max(axis=1).mean())}
        error = pred - y
        return {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
                "bias": float(np.mean(error)),
                "overestimate_frac": float((error > 0).mean())}

    return {"drift": drift, "parent_on_train": _performance(train), "parent_on_validation": _performance(validation)}