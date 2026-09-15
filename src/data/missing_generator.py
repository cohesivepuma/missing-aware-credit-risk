"""Reproducible MCAR, MAR and MNAR injection for the shared dataset contract."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np

from src.data.loader import Dataset

MissingMechanism = Literal["mcar", "mar", "mnar"]
MissingnessDirection = Literal["higher", "lower"]

_QUANTILES = (0.2, 0.4, 0.6, 0.8)


@dataclass(frozen=True)
class MissingnessPlan:
    """Train-fitted parameters that can be reused across data splits."""

    mechanism: MissingMechanism
    rate: float
    seed: int
    n_features: int
    injectable_features: tuple[int, ...]
    driver_features: tuple[int, ...]
    direction: MissingnessDirection
    strength: float
    driver_thresholds: tuple[tuple[float, ...], ...]
    feature_thresholds: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class MissingnessReport:
    """Rates and counts for one plan application."""

    mechanism: MissingMechanism
    requested_rate: float
    injected_rate: float
    overall_missing_rate: float
    candidate_cells: int
    injected_cells: int
    injectable_features: tuple[int, ...]
    driver_features: tuple[int, ...]
    seed: int


@dataclass(frozen=True)
class MissingnessResult:
    """Dataset plus metadata when callers need to persist mechanism details."""

    data: Dataset
    report: MissingnessReport


def _validate_rate(rate: float) -> float:
    if isinstance(rate, bool) or not np.isscalar(rate) or not np.isfinite(rate):
        raise ValueError("rate must be a finite number between 0 and 1.")
    value = float(rate)
    if not 0 <= value <= 1:
        raise ValueError("rate must be between 0 and 1.")
    return value


def _validate_strength(strength: float) -> float:
    if (
        isinstance(strength, bool)
        or not np.isscalar(strength)
        or not np.isfinite(strength)
        or float(strength) <= 0
    ):
        raise ValueError("strength must be a finite positive number.")
    return float(strength)


def _features(
    indices: Sequence[int] | None,
    *,
    n_features: int,
    name: str,
) -> tuple[int, ...]:
    if indices is None:
        return tuple(range(n_features))
    if isinstance(indices, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty sequence of feature indices.")
    try:
        values = tuple(indices)
    except TypeError as error:
        raise ValueError(f"{name} must be a nonempty sequence of feature indices.") from error
    if not values:
        raise ValueError(f"{name} must be a nonempty sequence of feature indices.")
    if any(
        not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_))
        for index in values
    ):
        raise ValueError(f"{name} must contain integer feature indices.")
    result = tuple(sorted(int(index) for index in values))
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate feature indices.")
    if any(index < 0 or index >= n_features for index in result):
        raise ValueError(f"{name} contains an out-of-range feature index.")
    return result


def _validate_dataset(data: Dataset) -> None:
    x, mask, y = data["x"], data["mask"], data["y"]
    if (
        x.ndim != 2
        or x.shape[0] == 0
        or x.shape[1] == 0
        or mask.shape != x.shape
        or y.ndim != 1
        or len(y) != len(x)
    ):
        raise ValueError("Expected aligned nonempty x, mask and y arrays.")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("mask must contain only 0=missing and 1=observed.")
    if not np.array_equal(mask, (~np.isnan(x)).astype(np.uint8)):
        raise ValueError("mask must align exactly with original NaN values in x.")


def _thresholds(values: np.ndarray) -> tuple[float, ...]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return ()
    if np.all(finite == finite[0]):
        return (float(finite[0]),)
    return tuple(float(value) for value in np.quantile(finite, _QUANTILES))


def _scores(
    values: np.ndarray,
    thresholds: tuple[float, ...],
    direction: MissingnessDirection,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scores = np.full(values.shape, 0.5, dtype=np.float64)
    finite = np.isfinite(values)
    if not thresholds or not finite.any():
        return scores
    scores[finite] = np.searchsorted(thresholds, values[finite], side="left") / len(
        thresholds
    )
    if direction == "lower":
        scores[finite] = 1 - scores[finite]
    return np.clip(scores, 0, 1)


def fit_missingness_plan(
    data: Dataset,
    *,
    mechanism: MissingMechanism,
    rate: float,
    seed: int = 42,
    eligible_features: Sequence[int] | None = None,
    driver_features: Sequence[int] | None = None,
    direction: MissingnessDirection = "higher",
    strength: float = 1.0,
) -> MissingnessPlan:
    """Fit mechanism thresholds on training data and freeze them for reuse.

    ``rate`` is the share of currently observed eligible cells converted to
    missing. The selected count is rounded to the nearest integer. For MAR,
    driver columns are removed from the injectable set so they always remain
    observable. MNAR should be restricted to numeric or explicitly ordered
    fields through ``eligible_features``; arbitrary category codes are not a
    meaningful numerical ordering.
    """
    if mechanism not in {"mcar", "mar", "mnar"}:
        raise ValueError("mechanism must be 'mcar', 'mar' or 'mnar'.")
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'.")
    _validate_dataset(data)
    target_rate = _validate_rate(rate)
    target_strength = _validate_strength(strength)
    n_features = data["x"].shape[1]
    eligible = _features(
        eligible_features, n_features=n_features, name="eligible_features"
    )

    if mechanism == "mar":
        if driver_features is None:
            if len(eligible) < 2:
                raise ValueError("MAR requires a driver feature separate from targets.")
            drivers = (eligible[0],)
        else:
            drivers = _features(
                driver_features, n_features=n_features, name="driver_features"
            )
        if not set(drivers).issubset(eligible):
            raise ValueError("driver_features must be a subset of eligible_features.")
        injectable = tuple(feature for feature in eligible if feature not in drivers)
        if not injectable:
            raise ValueError("MAR must leave at least one injectable feature.")
    else:
        if driver_features is not None:
            raise ValueError("driver_features is only valid for MAR.")
        drivers = ()
        injectable = eligible

    driver_thresholds: tuple[tuple[float, ...], ...] = ()
    feature_thresholds: tuple[tuple[float, ...], ...] = ()
    if mechanism == "mar":
        driver_thresholds = tuple(
            _thresholds(data["x"][:, feature]) for feature in drivers
        )
    elif mechanism == "mnar":
        feature_thresholds = tuple(
            _thresholds(data["x"][:, feature]) for feature in injectable
        )

    return MissingnessPlan(
        mechanism=mechanism,
        rate=target_rate,
        seed=int(seed),
        n_features=n_features,
        injectable_features=injectable,
        driver_features=drivers,
        direction=direction,
        strength=target_strength,
        driver_thresholds=driver_thresholds,
        feature_thresholds=feature_thresholds,
    )


def _select_cells(
    data: Dataset,
    plan: MissingnessPlan,
    candidates: np.ndarray,
) -> np.ndarray:
    n_inject = int(np.floor(plan.rate * len(candidates) + 0.5))
    if n_inject == 0:
        return np.array([], dtype=np.int64)
    if n_inject == len(candidates):
        return candidates

    rows, columns = np.unravel_index(candidates, data["x"].shape)
    if plan.mechanism == "mcar":
        weights = np.ones(len(candidates), dtype=np.float64)
    elif plan.mechanism == "mar":
        row_scores = np.zeros(len(data["y"]), dtype=np.float64)
        for feature, thresholds in zip(
            plan.driver_features, plan.driver_thresholds, strict=True
        ):
            row_scores += _scores(data["x"][:, feature], thresholds, plan.direction)
        row_scores /= len(plan.driver_features)
        weights = 1 + plan.strength * row_scores[rows]
    else:
        weights = np.ones(len(candidates), dtype=np.float64)
        for feature, thresholds in zip(
            plan.injectable_features, plan.feature_thresholds, strict=True
        ):
            cells = columns == feature
            if cells.any():
                feature_scores = _scores(
                    data["x"][:, feature], thresholds, plan.direction
                )
                weights[cells] = 1 + plan.strength * feature_scores[rows[cells]]

    probabilities = weights / weights.sum()
    rng = np.random.default_rng(plan.seed)
    return np.sort(
        rng.choice(candidates, size=n_inject, replace=False, p=probabilities)
    ).astype(np.int64, copy=False)


@overload
def apply_missingness(
    data: Dataset, plan: MissingnessPlan, *, return_report: Literal[False] = False
) -> Dataset: ...


@overload
def apply_missingness(
    data: Dataset, plan: MissingnessPlan, *, return_report: Literal[True]
) -> MissingnessResult: ...


def apply_missingness(
    data: Dataset,
    plan: MissingnessPlan,
    *,
    return_report: bool = False,
) -> Dataset | MissingnessResult:
    """Apply a frozen plan without mutating ``data``."""
    _validate_dataset(data)
    if data["x"].shape[1] != plan.n_features:
        raise ValueError("Missingness plan and dataset feature counts do not match.")

    result: Dataset = {
        "x": data["x"].copy(),
        "mask": data["mask"].copy(),
        "y": data["y"].copy(),
    }
    candidate_matrix = np.zeros_like(result["mask"], dtype=bool)
    candidate_matrix[:, plan.injectable_features] = (
        result["mask"][:, plan.injectable_features] == 1
    )
    candidates = np.flatnonzero(candidate_matrix.ravel()).astype(np.int64, copy=False)
    selected = _select_cells(data, plan, candidates)
    rows, columns = np.unravel_index(selected, result["x"].shape)
    result["x"][rows, columns] = np.nan
    result["mask"][rows, columns] = 0

    report = MissingnessReport(
        mechanism=plan.mechanism,
        requested_rate=plan.rate,
        injected_rate=(len(selected) / len(candidates) if len(candidates) else 0.0),
        overall_missing_rate=float(np.mean(result["mask"] == 0)),
        candidate_cells=len(candidates),
        injected_cells=len(selected),
        injectable_features=plan.injectable_features,
        driver_features=plan.driver_features,
        seed=plan.seed,
    )
    if return_report:
        return MissingnessResult(data=result, report=report)
    return result


def generate_missingness(
    data: Dataset,
    *,
    mechanism: MissingMechanism,
    rate: float,
    seed: int = 42,
    eligible_features: Sequence[int] | None = None,
    driver_features: Sequence[int] | None = None,
    direction: MissingnessDirection = "higher",
    strength: float = 1.0,
    return_report: bool = False,
) -> Dataset | MissingnessResult:
    """Fit on ``data`` and immediately inject missingness into a copy.

    For train/validation/test experiments, fit one plan on train and call
    ``apply_missingness`` with that plan on every split. This keeps MAR/MNAR
    threshold parameters train-only while giving all splits the same protocol.
    """
    plan = fit_missingness_plan(
        data,
        mechanism=mechanism,
        rate=rate,
        seed=seed,
        eligible_features=eligible_features,
        driver_features=driver_features,
        direction=direction,
        strength=strength,
    )
    return apply_missingness(data, plan, return_report=return_report)
