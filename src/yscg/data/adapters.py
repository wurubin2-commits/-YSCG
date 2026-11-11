"""真实数据集适配器：CMAPSS（RUL 回归）/ Gas sensor drift（多分类）/ XJTU-SY 轴承。

已按实际文件格式实现（数据集解压于 data/）：
- CMAPSS：`train_FD00x.txt` 空格分隔 26 列 = unit, cycle, setting_1..3, sensor_1..21；
  RUL = 该机组 max_cycle − cycle（可封顶）；`test_FD00x.txt` + `RUL_FD00x.txt`。
- Gas drift：`batchN.dat` 为 libsvm 格式，首列 1–6 气体类别，后 128 维 `idx:value`；
  batch1..batch10 为时间序（含漂移）。
- XJTU-SY 轴承：每个轴承一段 run-to-failure 的 CSV（水平/垂直振动），需特征提取；
  本地未随压缩包提供，接口按官方 README 文档化实现。

统一产出 `LabeledDataStream`（标签总可得，对齐赛题 B 包规则）。
"""
from __future__ import annotations

import glob as _glob
import os

import numpy as np

from .stream import LabeledDataStream
from .task import TaskSpec, TaskType


# ---------------- CMAPSS ----------------

def load_cmapss_fd(data_dir: str, fd: str = "FD001", rul_cap: float = 125.0):
    """读取 CMAPSS 训练集，返回 (X, rul, unit, cycle)。

    X: (n, 24) 特征（3 个操作设定 + 21 个传感器）；rul: (n,) 剩余寿命；unit/cycle: (n,)。
    """
    arr = np.loadtxt(os.path.join(data_dir, f"train_{fd}.txt"))
    unit = arr[:, 0].astype(int)
    cycle = arr[:, 1].astype(int)
    X = arr[:, 2:].astype(np.float32)
    maxc = {int(u): int(cycle[unit == u].max()) for u in np.unique(unit)}
    rul = np.array([float(maxc[int(u)] - int(c)) for u, c in zip(unit, cycle)], dtype=np.float32)
    rul = np.minimum(rul, float(rul_cap))
    return X, rul, unit, cycle


def load_cmapss_test(data_dir: str, fd: str = "FD001"):
    """读取 CMAPSS 测试集，返回 (X, unit)。真实 RUL 见 RUL_FD00x.txt（最后一轮）。"""
    arr = np.loadtxt(os.path.join(data_dir, f"test_{fd}.txt"))
    return arr[:, 2:].astype(np.float32), arr[:, 0].astype(int)


def cmapss_stream(data_dir: str, fd: str = "FD001", batch_size: int = 100,
                  init_frac: float = 0.6, val_frac: float = 0.2):
    """按机组切分构造 RUL 退化流：前 60% 机组训练、20% 验证、20% 作为流（时间序展开）。"""
    X, y, unit, _ = load_cmapss_fd(data_dir, fd)
    units = np.unique(unit)
    n = len(units)
    n_init = max(1, int(n * init_frac))
    n_val = max(1, int(n * val_frac))
    init_u = set(units[:n_init].tolist())
    val_u = set(units[n_init:n_init + n_val].tolist())
    stream_u = set(units[n_init + n_val:].tolist()) or set(units[-1:].tolist())

    m_init = np.isin(unit, list(init_u))
    m_val = np.isin(unit, list(val_u))
    m_stream = np.isin(unit, list(stream_u))
    spec = TaskSpec(name=f"cmapss-{fd}", task_type=TaskType.REGRESSION, metric="mae")
    return LabeledDataStream(
        init_X=X[m_init], init_y=y[m_init], val_X=X[m_val], val_y=y[m_val],
        stream_X=X[m_stream], stream_y=y[m_stream], heldout_X=X[m_val], heldout_y=y[m_val],
        batch_size=batch_size,
    ), spec


# ---------------- Gas sensor drift ----------------

def _parse_libsvm(path: str, n_features: int = 128):
    rows, labels = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            labels.append(int(float(parts[0])))
            feat = np.zeros(n_features, dtype=np.float32)
            for p in parts[1:]:
                if ":" in p:
                    k, v = p.split(":", 1)
                    j = int(k) - 1
                    if 0 <= j < n_features:
                        feat[j] = float(v)
            rows.append(feat)
    X = np.stack(rows) if rows else np.zeros((0, n_features), dtype=np.float32)
    return X, np.array(labels, dtype=int)


def load_gas_drift(gas_dir: str, n_features: int = 128):
    """读取全部 batchN.dat（时间序），返回 (X, y, batch_index)。

    注意：原始类别为 1–6，这里统一映射为 **0–5**（全系统约定 0-indexed，
    与 predict_proba 的 argmax、accuracy、MajorityPredictor 一致）。
    """
    paths = sorted(
        _glob.glob(os.path.join(gas_dir, "batch*.dat")),
        key=lambda p: int("".join(ch for ch in os.path.basename(p) if ch.isdigit()) or 0),
    )
    if not paths:
        raise FileNotFoundError(f"未找到 batch*.dat：{gas_dir}")
    Xs, ys, bidx = [], [], []
    for i, p in enumerate(paths):
        X, y = _parse_libsvm(p, n_features)
        Xs.append(X)
        ys.append(y - 1)  # 1–6 -> 0–5
        bidx.append(np.full(len(y), i + 1, dtype=int))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(bidx)


def gas_drift_stream(gas_dir: str, batch_size: int = 100,
                     n_init_batches: int = 3, n_val_batches: int = 1,
                     n_heldout_batches: int = 1):
    """时间序切分：前 3 批训练、第 4 批验证、**最后 1 批冻结留出**、其余为漂移流。

    留出批不进入流、不参与训练/验证/准入，仅作最终评判（标签总可得）。
    """
    X, y, bidx = load_gas_drift(gas_dir)
    last = int(bidx.max())
    init_n = bidx <= n_init_batches
    val_n = (bidx > n_init_batches) & (bidx <= n_init_batches + n_val_batches)
    held_n = bidx > (last - n_heldout_batches)
    stream_n = (bidx > n_init_batches + n_val_batches) & (bidx <= last - n_heldout_batches)
    spec = TaskSpec(name="gas-drift", task_type=TaskType.MULTICLASS_CLASSIFICATION,
                    n_classes=int(np.unique(y).size), metric="accuracy")
    return LabeledDataStream(
        init_X=X[init_n], init_y=y[init_n], val_X=X[val_n], val_y=y[val_n],
        stream_X=X[stream_n], stream_y=y[stream_n],
        heldout_X=X[held_n], heldout_y=y[held_n],
        batch_size=batch_size,
    ), spec


# ---------------- XJTU-SY 轴承（文档化，数据未随包提供） ----------------
# 官方结构（README）：每个轴承一段 run-to-failure 的 CSV，列含水平/垂直振动；
# 采样 25.6kHz、每次 1.024s（32768 点），按时间快照命名。RUL 由样本序号推得。

def bearing_features(signal: np.ndarray) -> np.ndarray:
    """对一段振动信号提取时域统计特征（均值/标准差/RMS/峰峰值/偏度/峰度）。"""
    s = np.asarray(signal, dtype=np.float64)
    mean = float(s.mean())
    std = float(s.std())
    rms = float(np.sqrt(np.mean(s ** 2)))
    p2p = float(s.max() - s.min())
    if std > 1e-12:
        skew = float(np.mean(((s - mean) / std) ** 3))
        kurt = float(np.mean(((s - mean) / std) ** 4))
    else:
        skew = kurt = 0.0
    return np.array([mean, std, rms, p2p, skew, kurt], dtype=np.float32)


def bearing_stream(csv_dir: str, label: str = "Horizontal_vibration",
                   batch_size: int = 50) -> tuple[LabeledDataStream, TaskSpec]:
    """从某轴承的逐快照 CSV 目录构造 RUL 流（需先用 bearing_features 特征化）。

    目录约定：按文件名排序即时间序；每文件含 `label` 列（振动信号，多行）。
    """
    paths = sorted(_glob.glob(os.path.join(csv_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"未找到轴承 CSV：{csv_dir}")
    feats = []
    for p in paths:
        try:
            arr = np.loadtxt(p, delimiter=",", skiprows=1, usecols=0)
        except Exception:
            arr = np.loadtxt(p, delimiter=",", usecols=0)
        feats.append(bearing_features(arr))
    X = np.stack(feats)
    n = len(X)
    rul = np.arange(n - 1, -1, -1, dtype=np.float32)  # 越靠后越接近失效
    spec = TaskSpec(name="bearing-rul", task_type=TaskType.REGRESSION, metric="mae")
    cut1, cut2 = int(n * 0.6), int(n * 0.8)
    return LabeledDataStream(
        init_X=X[:cut1], init_y=rul[:cut1], val_X=X[cut1:cut2], val_y=rul[cut1:cut2],
        stream_X=X[cut2:], stream_y=rul[cut2:], batch_size=batch_size,
    ), spec
