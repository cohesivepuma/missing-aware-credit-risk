"""Check metric values against small hand-computed examples and edge cases."""

import numpy as np
import pytest

from src.metrics.metrics import evaluate_metrics, expected_calibration_error


def test_metrics_known_values() -> None:
    result = evaluate_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
    assert set(result) == {"auc", "f1", "brier", "ece"}
    assert result["auc"] == pytest.approx(0.75)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["brier"] == pytest.approx(0.158125)
    assert result["ece"] == pytest.approx(0.3375)
    assert all(0 <= value <= 1 for value in result.values())


def test_perfect_predictions_include_zero_and_one() -> None:
    y = np.array([0, 1])
    assert evaluate_metrics(y, y) == {"auc": 1.0, "f1": 1.0, "brier": 0.0, "ece": 0.0}


def test_ece_weights_bins_by_sample_count() -> None:
    assert expected_calibration_error(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.1, 0.1, 0.9]), n_bins=2
    ) == pytest.approx(0.2)


@pytest.mark.parametrize("probabilities", [[-0.1, 0.8], [0.2, 1.1], [np.nan, 0.8], [0.2, np.inf]])
def test_reject_invalid_probabilities(probabilities: list[float]) -> None:
    with pytest.raises(ValueError, match="Probabilities"):
        evaluate_metrics(np.array([0, 1]), np.array(probabilities))


def test_auc_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="both classes"):
        evaluate_metrics(np.array([0, 0]), np.array([0.1, 0.2]))


def test_reject_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="aligned"):
        evaluate_metrics(np.array([0, 1]), np.array([0.1]))


@pytest.mark.parametrize("n_bins", [0, -1, 1.5, True])
def test_reject_invalid_bins(n_bins: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        expected_calibration_error(np.array([0, 1]), np.array([0.1, 0.9]), n_bins=n_bins)
