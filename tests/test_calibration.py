"""Calibration tests: probability boundaries, input validation and fit/predict separation."""

import numpy as np
import pytest

from src.calibration.calibrators import (
    DEFAULT_CLIP_EPSILON,
    LogitsAdapter,
    ProbabilityCalibrator,
)
from src.metrics.metrics import evaluate_metrics


def _overconfident_scores(seed: int = 0, n: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Return labels from a known logistic model plus deliberately overconfident scores.

    The true positive probability is sigmoid(z) while the reported score is
    sigmoid(3 * z), so a correct calibration map has to shrink the log-odds by
    roughly a factor of three.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(0.0, 2.0, size=n)
    labels = rng.binomial(1, 1.0 / (1.0 + np.exp(-latent)))
    overconfident = 1.0 / (1.0 + np.exp(-3.0 * latent))
    return labels.astype(np.int64), overconfident


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibration_reduces_ece_on_overconfident_scores(method: str) -> None:
    """Both methods must actually fix a known miscalibration instead of only running."""
    labels, scores = _overconfident_scores()
    calibrator = ProbabilityCalibrator(method).fit(scores[:2000], labels[:2000])
    calibrated = calibrator.predict_proba(scores[2000:])

    before = evaluate_metrics(labels[2000:], scores[2000:])["ece"]
    after = evaluate_metrics(labels[2000:], calibrated)["ece"]
    assert after < before


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibrated_probabilities_stay_valid_and_monotone(method: str) -> None:
    labels, scores = _overconfident_scores(n=2000)
    calibrator = ProbabilityCalibrator(method).fit(scores[:1000], labels[:1000])
    boundaries = np.array([0.0, 5e-9, 0.5, 1.0 - 5e-9, 1.0])
    calibrated = calibrator.predict_proba(boundaries)

    assert calibrated.shape == boundaries.shape
    assert np.isfinite(calibrated).all()
    assert ((calibrated >= 0.0) & (calibrated <= 1.0)).all()
    assert np.all(np.diff(calibrated) >= 0.0)


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_predict_before_fit_is_rejected(method: str) -> None:
    """Fitting and predicting stay separate; an unfitted calibrator refuses to predict."""
    calibrator = ProbabilityCalibrator(method)
    assert calibrator.is_fitted is False
    with pytest.raises(RuntimeError, match="fit must be called"):
        calibrator.predict_proba(np.array([0.2, 0.8]))
    with pytest.raises(RuntimeError, match="fit must be called"):
        _ = calibrator.fitted_parameters


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_fit_does_not_modify_caller_inputs(method: str) -> None:
    labels, scores = _overconfident_scores(n=2000)
    fit_scores, fit_labels = scores[:1000].copy(), labels[:1000].copy()
    ProbabilityCalibrator(method).fit(fit_scores, fit_labels)

    assert np.array_equal(fit_scores, scores[:1000])
    assert np.array_equal(fit_labels, labels[:1000])


def test_fitted_parameters_describe_the_map() -> None:
    labels, scores = _overconfident_scores(n=2000)
    platt = ProbabilityCalibrator("platt").fit(scores[:1000], labels[:1000])
    isotonic = ProbabilityCalibrator("isotonic").fit(scores[:1000], labels[:1000])

    assert platt.fitted_parameters["a"] == pytest.approx(1 / 3, abs=0.1)
    assert isotonic.fitted_parameters["n_thresholds"] >= 1


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_fit_requires_both_classes(method: str) -> None:
    with pytest.raises(ValueError, match="both classes"):
        ProbabilityCalibrator(method).fit(np.array([0.2, 0.4, 0.6]), np.array([1, 1, 1]))


def test_reject_unsupported_method() -> None:
    with pytest.raises(ValueError, match="Supported methods"):
        ProbabilityCalibrator("temperature")


@pytest.mark.parametrize("clip_epsilon", [0.0, -1e-6, 0.5, 1.0, np.nan, np.inf, True, "1e-6"])
def test_reject_invalid_clip_epsilon(clip_epsilon: object) -> None:
    with pytest.raises(ValueError, match="clip_epsilon"):
        ProbabilityCalibrator("platt", clip_epsilon=clip_epsilon)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("probabilities", "y"),
    [
        ([[0.2, 0.8]], [1]),
        ([0.2, 0.8], [1]),
        ([0.2, 0.8, 0.5], [1, 0]),
        ([-0.1, 0.8], [1, 0]),
        ([0.2, np.nan], [1, 0]),
    ],
)
def test_fit_rejects_misaligned_or_invalid_inputs(
    probabilities: list[float], y: list[int]
) -> None:
    with pytest.raises(ValueError):
        ProbabilityCalibrator("platt").fit(
            np.array(probabilities, dtype=np.float64), np.array(y)
        )


@pytest.mark.parametrize("probabilities", [[0.2, 1.2], [-0.5, 0.5], [np.nan, 0.5], [np.inf, 0.5], []])
def test_predict_rejects_invalid_probabilities(probabilities: list[float]) -> None:
    labels, scores = _overconfident_scores(n=1000)
    calibrator = ProbabilityCalibrator("platt").fit(scores, labels)
    with pytest.raises(ValueError):
        calibrator.predict_proba(np.array(probabilities, dtype=np.float64))


def test_logits_adapter_clips_probability_boundaries() -> None:
    """A probability of exactly 0 or 1 must not turn into an infinite logit."""
    adapter = LogitsAdapter()
    logits = adapter.to_logits(np.array([0.0, 1.0]))

    assert np.isfinite(logits).all()
    assert logits[0] == pytest.approx(np.log(DEFAULT_CLIP_EPSILON / (1.0 - DEFAULT_CLIP_EPSILON)))
    assert adapter.to_probabilities(logits) == pytest.approx([0.0, 1.0], abs=1e-5)


def test_logits_adapter_round_trips_interior_probabilities() -> None:
    adapter = LogitsAdapter()
    probabilities = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    assert adapter.to_probabilities(adapter.to_logits(probabilities)) == pytest.approx(
        probabilities
    )


@pytest.mark.parametrize("logits", [[np.nan, 0.0], [np.inf], [], [[0.0, 1.0]]])
def test_logits_adapter_rejects_invalid_logits(logits: list[float]) -> None:
    with pytest.raises(ValueError):
        LogitsAdapter().to_probabilities(np.array(logits, dtype=np.float64))


@pytest.mark.parametrize("temperature", [0.0, -1.0, np.nan, np.inf, True])
def test_apply_temperature_rejects_nonpositive_values(temperature: object) -> None:
    with pytest.raises(ValueError, match="temperature"):
        LogitsAdapter().apply_temperature(np.array([0.0, 1.0]), temperature)  # type: ignore[arg-type]


def test_temperature_fitting_is_explicitly_not_implemented() -> None:
    """The planned Temperature Scaling extension must not look implemented."""
    with pytest.raises(NotImplementedError, match="planned extension"):
        LogitsAdapter.fit_temperature(np.array([0.0, 1.0]), np.array([0, 1]))
