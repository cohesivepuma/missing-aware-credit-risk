"""Binary metrics and the shared probability binning protocol for P(y=1)."""

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score

DEFAULT_N_BINS = 10
"""Default number of equally spaced probability bins for ECE and reliability plots."""


def validate_probabilities(probabilities: np.ndarray) -> NDArray[np.float64]:
    """Return validated 1D P(y=1); reject empty input, NaN, infinity and out-of-range."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("Expected a nonempty 1D probability array.")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and between 0 and 1.")
    return p


def validate_binary_inputs(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return aligned 1D labels and probabilities after one shared protocol check.

    Metrics, calibration and reliability diagrams all validate through this
    function so the 0/1 label encoding and the [0, 1] probability range are
    enforced identically everywhere.
    """
    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or len(y) == 0:
        raise ValueError("Expected aligned, nonempty 1D labels and probabilities.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1.")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and between 0 and 1.")
    return y.astype(np.int64), p


def assign_probability_bins(
    probabilities: np.ndarray, *, n_bins: int = DEFAULT_N_BINS
) -> NDArray[np.int64]:
    """Map each P(y=1) to an equal-width bin index in [0, n_bins).

    Bins are [lower, upper) except the last one, which also contains 1.0 so a
    probability of exactly 1.0 never falls outside the protocol. ECE and the
    reliability diagram both call this helper, which keeps one binning
    definition instead of two drifting copies.
    """
    p = validate_probabilities(probabilities)
    _require_n_bins(n_bins)
    return np.minimum((p * n_bins).astype(int), n_bins - 1).astype(np.int64)


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, *, n_bins: int = DEFAULT_N_BINS
) -> float:
    """Positive-class ECE using equally spaced bins weighted by sample count.

    ECE = sum_b (n_b / n) * abs(mean(P(y=1))_b - mean(y)_b).
    Bins are [lower, upper), except the last includes probability 1.
    This definition is not top-label confidence ECE; compare consistent bins.
    """
    y, p = validate_binary_inputs(y_true, probabilities)
    _require_n_bins(n_bins)
    bins = assign_probability_bins(p, n_bins=n_bins)
    ece = 0.0
    for bin_index in range(n_bins):
        in_bin = bins == bin_index
        if in_bin.any():
            ece += float(in_bin.mean() * abs(p[in_bin].mean() - y[in_bin].mean()))
    return ece


def evaluate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
    n_bins: int = DEFAULT_N_BINS,
) -> dict[str, float]:
    """Return auc, f1, brier and ece; AUC requires both binary classes.

    Predict label 1 when P(y=1) >= threshold. AUC/F1 are higher-is-better;
    Brier/ECE are lower-is-better. Threshold selection belongs on validation.
    """
    y, p = validate_binary_inputs(y_true, probabilities)
    if len(np.unique(y)) != 2:
        raise ValueError("AUC requires both classes in y_true.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between 0 and 1.")
    return {
        "auc": float(roc_auc_score(y, p)),
        "f1": float(f1_score(y, p >= threshold, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p, n_bins=n_bins),
    }


def _require_n_bins(n_bins: int) -> int:
    """Reject non-integer, boolean and non-positive bin counts."""
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")
    return n_bins
