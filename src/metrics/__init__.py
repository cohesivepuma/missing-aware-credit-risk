"""Shared binary evaluation metrics and probability binning protocol."""

from src.metrics.metrics import (
    DEFAULT_N_BINS,
    assign_probability_bins,
    evaluate_metrics,
    expected_calibration_error,
    validate_binary_inputs,
    validate_probabilities,
)

__all__ = [
    "DEFAULT_N_BINS",
    "assign_probability_bins",
    "evaluate_metrics",
    "expected_calibration_error",
    "validate_binary_inputs",
    "validate_probabilities",
]
