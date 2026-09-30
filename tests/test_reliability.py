"""Reliability binning tests: shared ECE protocol and diagram output."""

from pathlib import Path

import numpy as np
import pytest

from src.calibration.reliability import plot_reliability_diagram, reliability_bins
from src.metrics.metrics import expected_calibration_error

# Matplotlib 3.10 still calls pyparsing APIs that newer pyparsing deprecates.
# matplotlib is imported inside the plotting tests so a strict ``pytest -W error``
# run does not fail during collection.


def test_bins_follow_the_same_protocol_as_ece() -> None:
    """The diagram and the ECE number must describe one binning definition."""
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.05, 0.15, 0.35, 0.55, 0.85, 1.0])
    stats = reliability_bins(y, p, n_bins=5)

    assert stats["counts"].sum() == len(y)
    assert np.array_equal(stats["counts"], np.array([2, 1, 1, 0, 2]))
    assert stats["edges"] == pytest.approx(np.linspace(0.0, 1.0, 6))
    assert np.isnan(stats["mean_probability"][3])
    assert np.isnan(stats["positive_frequency"][3])

    weights = stats["counts"] / len(y)
    manual = float(
        np.nansum(weights * np.abs(stats["mean_probability"] - stats["positive_frequency"]))
    )
    assert manual == pytest.approx(expected_calibration_error(y, p, n_bins=5))


def test_probability_one_stays_inside_the_last_bin() -> None:
    stats = reliability_bins(np.array([1, 0]), np.array([1.0, 1.0]), n_bins=4)

    assert stats["counts"][-1] == 2
    assert stats["counts"][:-1].sum() == 0


def test_statistics_match_hand_computed_bin_means() -> None:
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.2, 0.9, 0.8])
    stats = reliability_bins(y, p, n_bins=2)

    assert stats["counts"] == pytest.approx([3, 2])
    assert stats["mean_probability"] == pytest.approx([0.2, 0.85])
    assert stats["positive_frequency"] == pytest.approx([1 / 3, 1.0])


@pytest.mark.filterwarnings("ignore::pyparsing.PyparsingDeprecationWarning")
def test_plot_writes_a_diagram_for_several_variants(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # Render to the file only; never open a window.
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    p = np.array([0.1, 0.2, 0.4, 0.6, 0.35, 0.8, 0.9, 0.25])
    path = tmp_path / "nested" / "reliability.png"
    written = plot_reliability_diagram(
        {"uncalibrated": (y, p), "platt": (y, np.clip(p * 0.7 + 0.15, 0.0, 1.0))},
        path=path,
        n_bins=4,
        title="test diagram",
    )

    assert written == path
    assert path.is_file()
    assert path.stat().st_size > 0


@pytest.mark.filterwarnings("ignore::pyparsing.PyparsingDeprecationWarning")
def test_plot_rejects_empty_curves(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        plot_reliability_diagram({}, path=tmp_path / "empty.png")


@pytest.mark.parametrize("n_bins", [0, -2, True, 1.5])
def test_reject_invalid_bin_counts(n_bins: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        reliability_bins(np.array([0, 1]), np.array([0.2, 0.8]), n_bins=n_bins)


@pytest.mark.filterwarnings("ignore::pyparsing.PyparsingDeprecationWarning")
@pytest.mark.parametrize("other_labels", [[0, 1], [1, 0, 0]])
def test_plot_rejects_unpaired_evaluation_labels(
    tmp_path: Path, other_labels: list[int]
) -> None:
    y = np.array([0, 1, 0])
    other_y = np.array(other_labels)
    path = tmp_path / "unpaired.png"
    with pytest.raises(ValueError, match="same evaluation.*row order"):
        plot_reliability_diagram(
            {
                "uncalibrated": (y, np.array([0.1, 0.9, 0.2])),
                "platt": (other_y, np.full(len(other_y), 0.5)),
            },
            path=path,
        )
    assert not path.exists()


@pytest.mark.parametrize(
    ("y", "p"), [([0, 1], [0.2]), ([0, 2], [0.2, 0.8]), ([0, 1], [0.2, 1.5])]
)
def test_reject_misaligned_or_invalid_inputs(y: list[int], p: list[float]) -> None:
    with pytest.raises(ValueError):
        reliability_bins(np.array(y), np.array(p, dtype=np.float64))
