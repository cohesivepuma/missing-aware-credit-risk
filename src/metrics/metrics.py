"""Binary metrics for 1D positive-class probabilities P(y=1)."""

import numpy as np
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score


def _validate_inputs(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or len(y) == 0:
        raise ValueError("Expected aligned, nonempty 1D labels and probabilities.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1.")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and between 0 and 1.")
    return y, p


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, *, n_bins: int = 10
) -> float:
    """Positive-class ECE using equally spaced bins weighted by sample count.

    ECE = sum_b (n_b / n) * abs(mean(P(y=1))_b - mean(y)_b).
    Bins are [lower, upper), except the last includes probability 1.
    This definition is not top-label confidence ECE; compare consistent bins.
    """
    y, p = _validate_inputs(y_true, probabilities)
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
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
    n_bins: int = 10,
) -> dict[str, float]:
    """Return auc, f1, brier and ece; AUC requires both binary classes.

    Predict label 1 when P(y=1) >= threshold. AUC/F1 are higher-is-better;
    Brier/ECE are lower-is-better. Threshold selection belongs on validation.
    """
    y, p = _validate_inputs(y_true, probabilities)
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
