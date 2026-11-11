"""模型工厂：GBM（LightGBM 优先，缺库时 sklearn HistGradientBoosting 兜底）+ Keras MLP。

CPU-only 对齐：评测环境无 GPU、镜像 ≤2GB，主路线用 GBM；
lightgbm 缺失时自动降级到 sklearn HistGradientBoosting，保证闭环仍可运行。
"""
import joblib
import numpy as np

from .base import BaseModel

try:
    import lightgbm as lgb

    _HAS_LGBM = True
except Exception:  # pragma: no cover - 环境缺 lightgbm 时降级
    _HAS_LGBM = False

try:
    import keras
    from keras import layers

    _HAS_KERAS = True
except Exception:  # pragma: no cover - 环境缺 TF 时降级
    _HAS_KERAS = False


class LGBMModel(BaseModel):
    """表格 GBM 统一封装：分类或回归；lightgbm 缺省时退化为 sklearn GBM。"""

    def __init__(self, seed: int = 42, task: str = "classification"):
        self.seed = seed
        self.task = task
        if _HAS_LGBM:
            cls = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
            self._model = cls(
                n_estimators=100, learning_rate=0.08, num_leaves=31,
                verbosity=-1, n_jobs=1, random_state=seed,
            )
        else:
            # 兜底：RandomForest(n_jobs=1) 纯串行，不创建 multiprocessing 管道，
            # 可运行在受限沙箱 / 无 GPU 的评测环境。
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            cls = RandomForestRegressor if task == "regression" else RandomForestClassifier
            self._model = cls(n_estimators=100, n_jobs=1, random_state=seed)

    def fit(self, X, y, **kwargs):
        params = {}
        for k, v in kwargs.items():
            if hasattr(self._model, k):
                params[k] = v
            elif k == "n_estimators" and hasattr(self._model, "max_iter"):
                params["max_iter"] = v
        if params:
            self._model.set_params(**params)
        self._model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.task == "regression":
            raise RuntimeError("回归模型不支持 predict_proba；请使用 predict()")
        return self._model.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    def save(self, path: str):
        # 连同 task 一起持久化，回滚重载后仍能区分分类/回归
        joblib.dump({"model": self._model, "task": self.task}, path)

    @classmethod
    def load(cls, path: str) -> "LGBMModel":
        obj = cls.__new__(cls)
        payload = joblib.load(path)
        if isinstance(payload, dict) and "model" in payload:
            obj._model = payload["model"]
            obj.task = payload.get("task", "classification")
        else:  # 兼容旧产物
            obj._model = payload
            obj.task = "classification"
        return obj


class MLPModel(BaseModel):
    def __init__(self, seed: int = 42, units: tuple = (64, 32)):
        if not _HAS_KERAS:
            raise RuntimeError("未安装 TensorFlow/Keras，无法使用 MLP")
        self.seed = seed
        self.units = units
        self._model = None

    def _build(self, n_features: int, n_classes: int, lr: float = 0.005):
        keras.utils.set_random_seed(self.seed)
        model = keras.Sequential(
            [keras.Input(shape=(n_features,))]
            + [layers.Dense(u, activation="relu") for u in self.units]
            + [layers.Dense(n_classes, activation="softmax")]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X, y, **kwargs):
        n_classes = int(np.max(y)) + 1
        lr = kwargs.get("lr", 0.005)
        self._model = self._build(X.shape[1], n_classes, lr)
        epochs = kwargs.get("epochs", 15)
        self._model.fit(
            X, y, epochs=epochs, batch_size=64, verbose=0,
            validation_split=0.15,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict(X, verbose=0)

    def save(self, path: str):
        self._model.save(path)

    @classmethod
    def load(cls, path: str) -> "MLPModel":
        obj = cls.__new__(cls)
        obj._model = keras.models.load_model(path)
        return obj


def create_model(model_type: str, seed: int = 42, **params) -> BaseModel:
    if model_type == "lgbm":
        return LGBMModel(seed=seed, task=params.get("task", "classification"))
    if model_type == "mlp":
        return MLPModel(seed=seed, units=params.get("units", (64, 32)))
    raise ValueError(f"未知模型类型: {model_type}")


MODEL_EXT = {"lgbm": ".joblib", "mlp": ".keras"}
