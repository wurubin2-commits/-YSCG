"""模型工厂：LightGBM（表格基线，快而省）与 Keras MLP（深度学习路线）。"""
import joblib
import lightgbm as lgb
import numpy as np

from .base import BaseModel

try:
    import keras
    from keras import layers

    _HAS_KERAS = True
except Exception:  # pragma: no cover - 环境缺 TF 时降级
    _HAS_KERAS = False


class LGBMModel(BaseModel):
    def __init__(self, seed: int = 42):
        self._model = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.08, num_leaves=31,
            verbosity=-1, n_jobs=1, random_state=seed,
        )

    def fit(self, X, y, **kwargs):
        self._model.set_params(**{k: v for k, v in kwargs.items() if hasattr(self._model, k)})
        self._model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(X)

    def save(self, path: str):
        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path: str) -> "LGBMModel":
        obj = cls.__new__(cls)
        obj._model = joblib.load(path)
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
        return LGBMModel(seed=seed)
    if model_type == "mlp":
        return MLPModel(seed=seed, units=params.get("units", (64, 32)))
    raise ValueError(f"未知模型类型: {model_type}")


MODEL_EXT = {"lgbm": ".joblib", "mlp": ".keras"}
