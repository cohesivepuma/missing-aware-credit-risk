"""Check mask conventions, future-generator boundaries and seed reproducibility."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.loader import load_dataset
from src.data.missing_generator import (
    MissingnessResult,
    apply_missingness,
    fit_missingness_plan,
    generate_missingness,
)
from src.data.preprocess import split_dataset
from src.models.logistic import LogisticRegressionModel
from src.utils.seed import set_seed


def test_toy_data_is_observed_and_reproducible() -> None:
    first, second = load_dataset(seed=42), load_dataset(seed=42)
    assert first["mask"].shape == first["x"].shape
    assert np.all(first["mask"] == 1)  # 1=observed, 0=missing
    for key in ("x", "mask", "y"):
        np.testing.assert_array_equal(first[key], second[key])


def test_csv_mask_marks_nan(tmp_path: Path) -> None:
    path = tmp_path / "tiny.csv"
    path.write_text("income,age,default\n1,20,0\n,30,1\n", encoding="utf-8")
    data = load_dataset("csv", path=path, target_column="default")
    np.testing.assert_array_equal(data["mask"], [[1, 1], [0, 1]])
    assert np.isnan(data["x"][1, 0])


def test_set_seed_repeats_python_numpy_and_torch() -> None:
    set_seed(42)
    first = (random.random(), np.random.random(3), torch.rand(3))
    set_seed(42)
    second = (random.random(), np.random.random(3), torch.rand(3))
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)


def test_split_and_logistic_predictions_are_reproducible() -> None:
    data = load_dataset(seed=42)
    first, second = split_dataset(data, seed=42), split_dataset(data, seed=42)
    assert {name: len(part["y"]) for name, part in first.items()} == {
        "train": 600, "validation": 200, "test": 200
    }
    for name in first:
        for key in ("x", "mask", "y"):
            np.testing.assert_array_equal(first[name][key], second[name][key])

    predictions = []
    for parts in (first, second):
        set_seed(42)
        model = LogisticRegressionModel(seed=42)
        model.fit(parts["train"]["x"], parts["train"]["y"], parts["train"]["mask"])
        predictions.append(model.predict_proba(parts["test"]["x"], parts["test"]["mask"]))
    assert predictions[0].shape == (200, 2)
    np.testing.assert_allclose(predictions[0].sum(axis=1), 1)
    np.testing.assert_array_equal(predictions[0], predictions[1])


def test_splits_are_disjoint_and_keep_masks_aligned() -> None:
    data = load_dataset(seed=42)
    data["x"][:, 0] = np.arange(1000)  # unique row identifiers
    data["x"][::3, 1] = np.nan
    data["mask"] = (~np.isnan(data["x"])).astype(np.uint8)
    parts = split_dataset(data, seed=42)
    row_sets = [set(part["x"][:, 0]) for part in parts.values()]
    assert len(set.union(*row_sets)) == 1000
    assert sum(len(rows) for rows in row_sets) == 1000
    for part in parts.values():
        np.testing.assert_array_equal(part["mask"], ~np.isnan(part["x"]))


def test_preprocessing_fits_only_training_rows_and_keeps_empty_columns() -> None:
    model = LogisticRegressionModel()
    train = np.array([[1, np.nan], [2, np.nan], [3, np.nan], [4, np.nan]])
    model.fit(train, np.array([0, 0, 1, 1]))
    imputer = model.pipeline.named_steps["preprocess"].named_steps["imputer"]
    original_statistics = imputer.statistics_.copy()
    probabilities = model.predict_proba(np.array([[1000, np.nan], [np.nan, np.nan]]))
    np.testing.assert_array_equal(imputer.statistics_, original_statistics)
    assert original_statistics[0] == 2.5
    assert probabilities.shape == (2, 2)
    assert np.isfinite(probabilities).all()


@pytest.mark.parametrize("mechanism", ["mcar", "mar", "mnar"])
@pytest.mark.parametrize("rate", [0.1, 0.3, 0.5])
def test_missing_generator_exactly_controls_rate_without_mutating_input(
    mechanism: str, rate: float
) -> None:
    data = load_dataset(seed=42)
    original_x = data["x"].copy()
    original_mask = data["mask"].copy()
    original_y = data["y"].copy()
    options = {"driver_features": [0]} if mechanism == "mar" else {}

    result = generate_missingness(
        data,
        mechanism=mechanism,
        rate=rate,
        seed=42,
        return_report=True,
        **options,
    )

    assert isinstance(result, MissingnessResult)
    assert result.report.mechanism == mechanism
    assert result.report.requested_rate == rate
    assert result.report.injected_rate == pytest.approx(rate)
    assert result.report.injected_cells == round(rate * result.report.candidate_cells)
    np.testing.assert_array_equal(data["x"], original_x)
    np.testing.assert_array_equal(data["mask"], original_mask)
    np.testing.assert_array_equal(data["y"], original_y)
    np.testing.assert_array_equal(result.data["y"], original_y)
    np.testing.assert_array_equal(
        result.data["mask"], (~np.isnan(result.data["x"])).astype(np.uint8)
    )


def test_existing_missing_cells_are_preserved() -> None:
    x = np.arange(60, dtype=np.float64).reshape(20, 3)
    x[0, 1] = np.nan
    x[5, 2] = np.nan
    data = {
        "x": x,
        "mask": (~np.isnan(x)).astype(np.uint8),
        "y": np.arange(20) % 2,
    }

    result = generate_missingness(data, mechanism="mcar", rate=0.5, seed=7)

    assert np.isnan(result["x"][0, 1]) and result["mask"][0, 1] == 0
    assert np.isnan(result["x"][5, 2]) and result["mask"][5, 2] == 0
    np.testing.assert_array_equal(result["mask"], (~np.isnan(result["x"])).astype(np.uint8))


def test_frozen_plan_is_reusable_and_reproducible_across_splits() -> None:
    rows = 100
    train_x = np.column_stack((np.arange(rows), np.linspace(0, 1, rows)))
    test_x = np.column_stack((np.arange(rows) + 1000, np.linspace(2, 3, rows)))
    y = np.arange(rows) % 2
    train = {"x": train_x, "mask": np.ones_like(train_x, dtype=np.uint8), "y": y}
    test = {"x": test_x, "mask": np.ones_like(test_x, dtype=np.uint8), "y": y}

    plan = fit_missingness_plan(
        train,
        mechanism="mar",
        rate=0.3,
        seed=13,
        eligible_features=[0, 1],
        driver_features=[0],
    )
    expected_thresholds = tuple(np.quantile(train_x[:, 0], [0.2, 0.4, 0.6, 0.8]))
    assert plan.driver_thresholds[0] == pytest.approx(expected_thresholds)

    first = apply_missingness(test, plan)
    second = apply_missingness(test, plan)
    np.testing.assert_array_equal(first["x"], second["x"])
    np.testing.assert_array_equal(first["mask"], second["mask"])
    assert np.all(first["mask"][:, 0] == 1)


def test_mar_driver_stays_observed_and_high_driver_has_more_missingness() -> None:
    rows = 500
    x = np.column_stack(
        (np.arange(rows, dtype=np.float64), np.ones(rows), np.ones(rows))
    )
    data = {"x": x, "mask": np.ones_like(x, dtype=np.uint8), "y": np.arange(rows) % 2}

    result = generate_missingness(
        data,
        mechanism="mar",
        rate=0.5,
        seed=42,
        eligible_features=[0, 1, 2],
        driver_features=[0],
    )

    assert np.all(result["mask"][:, 0] == 1)
    low_missing = np.sum(result["mask"][: rows // 2, 1:] == 0)
    high_missing = np.sum(result["mask"][rows // 2 :, 1:] == 0)
    assert high_missing > low_missing


@pytest.mark.parametrize(
    ("direction", "comparison"),
    [("higher", "higher"), ("lower", "lower")],
)
def test_mnar_depends_on_the_injected_feature_value(
    direction: str, comparison: str
) -> None:
    rows = 1000
    x = np.column_stack((np.arange(rows, dtype=np.float64), np.ones(rows)))
    data = {"x": x, "mask": np.ones_like(x, dtype=np.uint8), "y": np.arange(rows) % 2}

    result = generate_missingness(
        data,
        mechanism="mnar",
        rate=0.5,
        seed=42,
        eligible_features=[0],
        direction=direction,
    )

    missing_mean = x[result["mask"][:, 0] == 0, 0].mean()
    observed_mean = x[result["mask"][:, 0] == 1, 0].mean()
    if comparison == "higher":
        assert missing_mean > observed_mean
    else:
        assert missing_mean < observed_mean


@pytest.mark.parametrize("mechanism", ["mcar", "mar", "mnar"])
def test_missingness_does_not_depend_on_labels(mechanism: str) -> None:
    x = np.column_stack((np.arange(200, dtype=np.float64), np.arange(200) % 7))
    mask = np.ones_like(x, dtype=np.uint8)
    first = {"x": x.copy(), "mask": mask.copy(), "y": np.arange(200) % 2}
    second = {"x": x.copy(), "mask": mask.copy(), "y": 1 - first["y"]}
    options = {"driver_features": [0]} if mechanism == "mar" else {}

    first_result = generate_missingness(
        first, mechanism=mechanism, rate=0.3, seed=42, **options
    )
    second_result = generate_missingness(
        second, mechanism=mechanism, rate=0.3, seed=42, **options
    )

    np.testing.assert_array_equal(first_result["mask"], second_result["mask"])
    np.testing.assert_array_equal(first_result["x"], second_result["x"])


@pytest.mark.parametrize("mechanism", ["mcar", "mar", "mnar"])
def test_rate_boundaries_are_exact(mechanism: str) -> None:
    data = load_dataset(seed=42, n_samples=50, n_features=4, n_informative=2)
    options = {"driver_features": [0]} if mechanism == "mar" else {}

    unchanged = generate_missingness(
        data, mechanism=mechanism, rate=0.0, seed=42, **options
    )
    fully_missing = generate_missingness(
        data, mechanism=mechanism, rate=1.0, seed=42, **options
    )

    np.testing.assert_array_equal(unchanged["mask"], np.ones_like(data["mask"]))
    if mechanism == "mar":
        assert np.all(fully_missing["mask"][:, 0] == 1)
        assert np.all(fully_missing["mask"][:, 1:] == 0)
    else:
        assert np.all(fully_missing["mask"] == 0)


@pytest.mark.parametrize("rate", [-0.1, 1.1, float("nan"), True])
def test_missing_generator_rejects_invalid_rates(rate: float) -> None:
    with pytest.raises(ValueError, match="rate"):
        generate_missingness(load_dataset(), mechanism="mcar", rate=rate)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mechanism": "invalid", "rate": 0.3}, "mechanism"),
        ({"mechanism": "mcar", "rate": 0.3, "direction": "sideways"}, "direction"),
        ({"mechanism": "mcar", "rate": 0.3, "strength": 0}, "strength"),
        (
            {"mechanism": "mcar", "rate": 0.3, "driver_features": [0]},
            "driver_features",
        ),
        (
            {
                "mechanism": "mar",
                "rate": 0.3,
                "eligible_features": [0, 1],
                "driver_features": [0, 1],
            },
            "injectable",
        ),
        (
            {"mechanism": "mnar", "rate": 0.3, "eligible_features": [0, 0]},
            "duplicate",
        ),
    ],
)
def test_missing_generator_rejects_invalid_configuration(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        generate_missingness(load_dataset(), **kwargs)
