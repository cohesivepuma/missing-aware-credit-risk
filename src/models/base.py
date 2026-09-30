"""Small shared model interface; probability columns are [P(y=0), P(y=1)]."""

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from numpy.typing import NDArray


class BaseModel(ABC):
    """External API shared by sklearn models and future PyTorch adapters."""

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> Self:
        """Fit training values; mask uses 1=observed, 0=missing."""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(
        self, x: np.ndarray, mask: np.ndarray | None = None
    ) -> NDArray[np.float64]:
        """Return an (n_samples, 2) probability array ordered by classes [0, 1]."""
        raise NotImplementedError


class PlannedModel(BaseModel):
    """Explicit placeholder shared by models outside the phase 1 scope."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> Self:
        raise NotImplementedError(f"{type(self).__name__} training is not implemented yet.")

    def predict_proba(
        self, x: np.ndarray, mask: np.ndarray | None = None
    ) -> NDArray[np.float64]:
        raise NotImplementedError(f"{type(self).__name__} prediction is not implemented yet.")
