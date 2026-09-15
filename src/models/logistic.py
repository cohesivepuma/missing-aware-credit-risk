"""Logistic Regression baseline with train-only imputation and scaling."""

from typing import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.preprocess import build_preprocessor
from src.models.base import BaseModel


class LogisticRegressionModel(BaseModel):
    """Ignore mask intentionally; this is the feature-value-only baseline."""

    def __init__(self, *, seed: int = 42, C: float = 1.0, max_iter: int = 1000) -> None:
        self.pipeline = Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(C=C, max_iter=max_iter, random_state=seed, solver="lbfgs"),
                ),
            ]
        )

    def fit(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> Self:
        """Fit preprocessing and classifier together using only training rows."""
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("Training labels must contain both classes 0 and 1.")
        self.pipeline.fit(x, y)
        return self

    def predict_proba(
        self, x: np.ndarray, mask: np.ndarray | None = None
    ) -> NDArray[np.float64]:
        """Return [P(non-default), P(default)]; toy labels are synthetic."""
        return self.pipeline.predict_proba(x)
