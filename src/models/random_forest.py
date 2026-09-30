"""Random Forest baseline using the same train-only encoding as the MLP."""

from collections.abc import Sequence
from typing import Self

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.data.preprocess import build_preprocessor
from src.models.base import BaseModel


class RandomForestModel(BaseModel):
    """Deterministic small-data baseline with explicit imputation."""

    def __init__(
        self, *, seed: int = 42, categorical_features: Sequence[int] | None = None,
        n_estimators: int = 150, min_samples_leaf: int = 3,
        max_depth: int | None = None,
    ) -> None:
        self.pipeline = Pipeline([
            ("preprocess", build_preprocessor(categorical_features)),
            ("classifier", RandomForestClassifier(
                n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                max_depth=max_depth, random_state=seed, n_jobs=1,
            )),
        ])

    def fit(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> Self:
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("Training labels must contain both classes 0 and 1.")
        self.pipeline.fit(x, y)
        return self

    def predict_proba(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        return self.pipeline.predict_proba(x)
