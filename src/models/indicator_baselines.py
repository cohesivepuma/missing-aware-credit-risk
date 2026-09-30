"""Dimension-matched value/mask controls for linear and forest classifiers."""
from collections.abc import Sequence
from typing import Self
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.data.preprocess import build_preprocessor
from src.models.base import BaseModel

class IndicatorBaseline(BaseModel):
    """Append ones or the original observed mask after train-only encoding."""
    def __init__(self, *, family: str, use_mask: bool, seed: int = 42,
                 categorical_features: Sequence[int] | None = None, **parameters) -> None:
        self.use_mask = use_mask
        self.preprocessor = build_preprocessor(categorical_features)
        if family == "logistic":
            self.classifier = LogisticRegression(random_state=seed, solver="lbfgs", **parameters)
        elif family == "random_forest":
            self.classifier = RandomForestClassifier(random_state=seed, n_jobs=1, **parameters)
        else:
            raise ValueError("family must be logistic or random_forest.")

    def _inputs(self, x: np.ndarray, mask: np.ndarray | None, *, fit: bool) -> np.ndarray:
        if self.use_mask and (mask is None or not np.array_equal(mask, ~np.isnan(x))):
            raise ValueError("Mask must match original NaNs: 1=observed.")
        encoded = self.preprocessor.fit_transform(x) if fit else self.preprocessor.transform(x)
        extra = np.asarray(mask, dtype=float) if self.use_mask else np.ones_like(x)
        return np.column_stack([encoded, extra])

    def fit(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> Self:
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("Training labels must contain both classes.")
        self.classifier.fit(self._inputs(x, mask, fit=True), y)
        return self

    def predict_proba(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        return self.classifier.predict_proba(self._inputs(x, mask, fit=False))
