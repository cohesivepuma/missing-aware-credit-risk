"""Positive-class probability calibration fitted on validation rows only.

The base model is trained and frozen first; a calibrator then maps its 1D
P(y=1) output to a calibrated 1D P(y=1) using the independent
validation_calibration rows. The test set is only used for the final paired
comparison: a method, threshold or bin count must never be chosen on it.
"""

from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.metrics.metrics import validate_binary_inputs, validate_probabilities

CalibrationMethod = Literal["platt", "isotonic"]
SUPPORTED_METHODS: tuple[CalibrationMethod, ...] = ("platt", "isotonic")

DEFAULT_CLIP_EPSILON = 1e-6
"""Default clip bound applied before the logit transform, giving |logit| <= 13.9."""

PLATT_REGULARIZATION_C = 1e6
"""Effectively unregularized logistic fit, matching the standard sigmoid calibration."""


def _require_clip_epsilon(clip_epsilon: float) -> float:
    """Reject booleans, non-numbers and values outside (0, 0.5)."""
    if isinstance(clip_epsilon, bool) or not isinstance(clip_epsilon, (int, float)):
        raise ValueError("clip_epsilon must be a number in (0, 0.5).")
    value = float(clip_epsilon)
    if not np.isfinite(value) or not 0.0 < value < 0.5 or 1.0 - value == 1.0:
        raise ValueError(
            "clip_epsilon must be in (0, 0.5) and large enough that "
            "1 - clip_epsilon is below 1 in float64."
        )
    return value


def _validate_logits(logits: np.ndarray) -> NDArray[np.float64]:
    """Return validated 1D logits; reject empty input, NaN and infinity."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Expected a nonempty 1D logits array.")
    if not np.isfinite(values).all():
        raise ValueError("Logits must be finite.")
    return values


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable sigmoid that never overflows for large magnitudes."""
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative = np.exp(values[~positive])
    result[~positive] = negative / (1.0 + negative)
    return result


class LogitsAdapter:
    """Logits contract for models that expose raw scores instead of probabilities.

    Temperature scaling is a planned extension, so only the contract and the
    transforms that do not depend on a fitted temperature live here: 1D logits
    in the same row order as P(y=1), a stable sigmoid, the clipped logit used by
    Platt scaling, and the division by a frozen temperature.
    """

    def __init__(self, *, clip_epsilon: float = DEFAULT_CLIP_EPSILON) -> None:
        self.clip_epsilon = _require_clip_epsilon(clip_epsilon)

    def to_logits(self, probabilities: np.ndarray) -> NDArray[np.float64]:
        """Clip to [eps, 1-eps] and apply logit(p); the result is never infinite."""
        p = validate_probabilities(probabilities)
        clipped = np.clip(p, self.clip_epsilon, 1.0 - self.clip_epsilon)
        return np.log(clipped / (1.0 - clipped))

    def to_probabilities(self, logits: np.ndarray) -> NDArray[np.float64]:
        """Return the stable sigmoid of 1D logits."""
        return _sigmoid(_validate_logits(logits))

    def apply_temperature(self, logits: np.ndarray, temperature: float) -> NDArray[np.float64]:
        """Return sigmoid(logits / temperature) for a frozen positive temperature."""
        if isinstance(temperature, bool) or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        return self.to_probabilities(_validate_logits(logits) / float(temperature))

    @staticmethod
    def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
        """Planned: fit one temperature from validation logits; not part of this change."""
        raise NotImplementedError("Temperature scaling is a planned extension.")


class ProbabilityCalibrator:
    """Map 1D P(y=1) to calibrated 1D P(y=1), fitted on validation rows only.

    Platt scaling fits an effectively unregularized logistic regression on the
    clipped logit of the base probabilities. Its slope may be positive, zero or
    negative; clipping and numerical saturation can introduce ties. Isotonic
    regression fits a nondecreasing map with linear interpolation between
    fitted thresholds, bounded to [0, 1]. Both preserve the input row order and
    neither may be fitted or selected on the test set.
    """

    def __init__(
        self,
        method: CalibrationMethod = "platt",
        *,
        clip_epsilon: float = DEFAULT_CLIP_EPSILON,
    ) -> None:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Supported methods: {', '.join(SUPPORTED_METHODS)}.")
        self.method = method
        self.clip_epsilon = _require_clip_epsilon(clip_epsilon)
        self._model: LogisticRegression | IsotonicRegression | None = None

    @property
    def is_fitted(self) -> bool:
        """True once fit has succeeded; predict_proba requires a fitted calibrator."""
        return self._model is not None

    @property
    def fitted_parameters(self) -> dict[str, float | int]:
        """Compact description of the fitted map, safe to serialize into a report."""
        if self._model is None:
            raise RuntimeError("fit must be called before reading fitted_parameters.")
        if self.method == "platt":
            return {
                "a": float(self._model.coef_[0][0]),
                "b": float(self._model.intercept_[0]),
            }
        return {"n_thresholds": int(len(self._model.X_thresholds_))}

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> Self:
        """Fit on 1D validation P(y=1) plus labels; never pass test rows here."""
        labels, p = validate_binary_inputs(y, probabilities)
        if len(np.unique(labels)) != 2:
            raise ValueError("Calibration requires both classes in the validation labels.")
        if self.method == "platt":
            logits = LogitsAdapter(clip_epsilon=self.clip_epsilon).to_logits(p).reshape(-1, 1)
            model = LogisticRegression(
                C=PLATT_REGULARIZATION_C, solver="lbfgs", max_iter=1000
            )
            model.fit(logits, labels)
            self._model = model
        else:
            model = IsotonicRegression(
                y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
            )
            # Passing a copy keeps the caller's probability array untouched.
            model.fit(p.copy(), labels)
            self._model = model
        return self

    def predict_proba(self, probabilities: np.ndarray) -> NDArray[np.float64]:
        """Return calibrated 1D P(y=1) in the input row order, always within [0, 1]."""
        if self._model is None:
            raise RuntimeError("fit must be called before predict_proba.")
        p = validate_probabilities(probabilities)
        if self.method == "platt":
            logits = LogitsAdapter(clip_epsilon=self.clip_epsilon).to_logits(p).reshape(-1, 1)
            calibrated = self._model.predict_proba(logits)[:, 1]
        else:
            calibrated = self._model.predict(p)
        return np.clip(np.asarray(calibrated, dtype=np.float64), 0.0, 1.0)
