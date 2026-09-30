"""Native-NaN LightGBM with training-only nominal category support."""
from collections.abc import Sequence
from typing import Self
import numpy as np
import pandas as pd
from src.models.base import BaseModel

class LightGBMModel(BaseModel):
    """Deterministic single-thread booster; nominal codes are not ordinal."""
    def __init__(self, *, seed: int = 42, categorical_features: Sequence[int] | None = None,
                 n_estimators: int = 150, learning_rate: float = .05,
                 num_leaves: int = 15, min_child_samples: int = 20,
                 reg_lambda: float = 1.0) -> None:
        from lightgbm import LGBMClassifier
        self.categorical = tuple(categorical_features or ())
        self.classifier = LGBMClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
            min_child_samples=min_child_samples, reg_lambda=reg_lambda,
            random_state=seed, n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1)

    def _frame(self, x: np.ndarray, *, fit: bool) -> pd.DataFrame:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or len(x) == 0 or np.isinf(x).any():
            raise ValueError("Expected a matrix with finite values or NaN.")
        if fit:
            self.n_features_in_ = x.shape[1]
            self.categories_ = {i: np.unique(x[~np.isnan(x[:, i]), i]) for i in self.categorical}
        elif x.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from training.")
        frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
        for i, categories in self.categories_.items():
            frame[f"x{i}"] = pd.Categorical(x[:, i], categories=categories)
        return frame

    def fit(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> Self:
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("Training labels must contain both classes.")
        self.classifier.fit(self._frame(x, fit=True), y)
        return self

    def predict_proba(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        return self.classifier.predict_proba(self._frame(x, fit=False))
