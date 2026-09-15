"""Check mask conventions, future-generator boundaries and seed reproducibility."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.loader import load_dataset
from src.data.missing_generator import generate_missingness
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
def test_missing_generator_is_explicitly_pending(mechanism: str) -> None:
    data = load_dataset(seed=42)
    with pytest.raises(NotImplementedError, match="phase 2"):
        generate_missingness(data, mechanism=mechanism, rate=0.3)
    assert np.all(data["mask"] == 1)


@pytest.mark.parametrize("rate", [-0.1, 1.1, float("nan")])
def test_missing_generator_rejects_invalid_rates(rate: float) -> None:
    with pytest.raises(ValueError, match="rate"):
        generate_missingness(load_dataset(), mechanism="mcar", rate=rate)
