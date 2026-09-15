"""Calibration interface; fit on validation data, never on the test set."""

from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray


class ProbabilityCalibrator:
    """Planned positive-class probability adapter.

    TODO: implement Platt and isotonic calibration of P(y=1), with clipping
    before a logit transform. Temperature scaling requires explicit logits
    and will get a separate adapter. Freeze the base model before fitting.
    """

    def __init__(self, method: Literal["platt", "isotonic"] = "platt") -> None:
        if method not in {"platt", "isotonic"}:
            raise ValueError("Supported planned methods: 'platt', 'isotonic'.")
        self.method = method

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> Self:
        """Fit using 1D validation P(y=1) and validation labels."""
        raise NotImplementedError("Probability calibration is planned for a later phase.")

    def predict_proba(self, probabilities: np.ndarray) -> NDArray[np.float64]:
        """Return calibrated 1D P(y=1), matching the input row order."""
        raise NotImplementedError("Probability calibration is planned for a later phase.")
