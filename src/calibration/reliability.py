"""Reliability diagram data and plotting for 1D positive-class probabilities.

The diagram always uses the same equal-width binning as
``expected_calibration_error``, so a figure and the ECE number in the same
report describe one protocol instead of two.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.metrics.metrics import (
    DEFAULT_N_BINS,
    assign_probability_bins,
    expected_calibration_error,
    validate_binary_inputs,
)

Curves = Mapping[str, tuple[NDArray[np.int64], NDArray[np.float64]]]


def reliability_bins(
    y_true: np.ndarray, probabilities: np.ndarray, *, n_bins: int = DEFAULT_N_BINS
) -> dict[str, NDArray]:
    """Return per-bin reliability statistics for one probability vector.

    ``counts`` has one entry per bin. ``mean_probability`` and
    ``positive_frequency`` hold NaN for empty bins so plotting can skip them
    while the sample count stays visible. Bin edges are equally spaced and the
    last bin also contains a probability of exactly 1.0.
    """
    y, p = validate_binary_inputs(y_true, probabilities)
    bins = assign_probability_bins(p, n_bins=n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)
    mean_probability = np.full(n_bins, np.nan, dtype=np.float64)
    positive_frequency = np.full(n_bins, np.nan, dtype=np.float64)
    for index in range(n_bins):
        in_bin = bins == index
        counts[index] = int(in_bin.sum())
        if counts[index]:
            mean_probability[index] = float(p[in_bin].mean())
            positive_frequency[index] = float(y[in_bin].mean())
    return {
        "edges": np.linspace(0.0, 1.0, n_bins + 1),
        "counts": counts,
        "mean_probability": mean_probability,
        "positive_frequency": positive_frequency,
    }


def plot_reliability_diagram(
    curves: Curves,
    *,
    path: str | Path,
    n_bins: int = DEFAULT_N_BINS,
    title: str | None = None,
    dpi: int = 150,
) -> Path:
    """Write a reliability diagram and return the saved path.

    ``curves`` maps a variant name to ``(y_true, probabilities)``. Every variant
    must be scored on the same evaluation rows, which keeps an uncalibrated and
    a calibrated curve paired instead of comparing different samples. The upper
    panel shows each curve against the ideal diagonal, the lower panel shows the
    sample count of every bin, and the legend repeats the ECE of each curve.
    """
    if not curves:
        raise ValueError("curves must contain at least one (y_true, probabilities) pair.")
    reference_y = next(iter(curves.values()))[0]
    if any(not np.array_equal(y, reference_y) for y, _ in curves.values()):
        raise ValueError(
            "All reliability curves must use the same evaluation labels in the same row order."
        )

    labels = list(curves)
    statistics = {
        label: reliability_bins(*curves[label], n_bins=n_bins) for label in labels
    }
    errors = {
        label: expected_calibration_error(*curves[label], n_bins=n_bins) for label in labels
    }
    drawn = int(len(next(iter(curves.values()))[0]))

    from matplotlib import pyplot as plt

    figure, (curve_axis, count_axis) = plt.subplots(
        2,
        1,
        figsize=(9.5, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    curve_axis.plot(
        [0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="0.45", label="ideal"
    )
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_width = 1.0 / n_bins
    # Grouped bars must stay inside one bin, otherwise neighbouring bins overlap.
    bar_width = 0.8 * bin_width / len(labels)
    offset_base = -(len(labels) - 1) / 2.0

    for position, label in enumerate(labels):
        stats = statistics[label]
        curve_axis.plot(
            stats["mean_probability"],
            stats["positive_frequency"],
            marker="o",
            linewidth=1.4,
            label=f"{label} (ECE={errors[label]:.3f})",
        )
        bars = count_axis.bar(
            centers + (offset_base + position) * bar_width,
            stats["counts"],
            width=bar_width,
            label=label,
        )
        count_axis.bar_label(bars, fontsize=6, padding=1, fmt="%d")

    curve_axis.set_ylabel("Observed positive frequency")
    curve_axis.set_xlim(0.0, 1.0)
    curve_axis.set_ylim(0.0, 1.0)
    curve_axis.grid(alpha=0.3)
    curve_axis.legend(fontsize=8, loc="best")

    count_axis.set_xlabel("Mean predicted P(y=1) per bin")
    count_axis.set_ylabel("Samples")
    count_axis.set_ylim(bottom=0.0)
    count_axis.grid(alpha=0.3, axis="y")

    figure.suptitle(
        title
        if title is not None
        else f"Reliability diagram | {n_bins} equal-width bins | n={drawn} evaluation rows"
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return destination
