"""Deterministic splits and preprocessing fitted on training data only."""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.loader import Dataset, dataset_fingerprint


def make_split_indices(
    y: np.ndarray,
    *,
    validation_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
    calibration_fraction: float | None = None,
) -> dict[str, np.ndarray]:
    """Return zero-based original row indices, with optional validation children.

    calibration_fraction is relative to validation, not the complete dataset.
    The parent 'validation' view contains both children and is not an additional
    disjoint partition. Fit/tune/calibration/test must remain mutually exclusive.
    """
    if not (0 < validation_size < 1 and 0 < test_size < 1):
        raise ValueError("validation_size and test_size must be between 0 and 1.")
    if validation_size + test_size >= 1:
        raise ValueError("Validation and test fractions must leave training data.")
    if calibration_fraction is not None and not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1.")
    indices = np.arange(len(y))
    train_validation, test = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=y
    )
    train, validation = train_test_split(
        train_validation,
        test_size=validation_size / (1 - test_size),
        random_state=seed,
        stratify=y[train_validation],
    )
    result = {"train": train, "validation": validation, "test": test}
    if calibration_fraction is not None:
        tune, calibration = train_test_split(
            validation, test_size=calibration_fraction, random_state=seed,
            stratify=y[validation],
        )
        result.update(validation_tune=tune, validation_calibration=calibration)
    return result


def _validate_indices(indices: dict[str, np.ndarray], n_rows: int, has_children: bool) -> None:
    """Reject corrupted manifests with duplicate, missing or out-of-range rows."""
    expected = {"train", "validation", "test"}
    if has_children:
        expected |= {"validation_tune", "validation_calibration"}
    if set(indices) != expected:
        raise ValueError("Split manifest has unexpected split names.")
    for rows in indices.values():
        if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or len(rows) == 0:
            raise ValueError("Split indices must be nonempty integer arrays.")
        if np.any((rows < 0) | (rows >= n_rows)) or len(np.unique(rows)) != len(rows):
            raise ValueError("Split manifest contains duplicate or out-of-range rows.")
    partition = np.concatenate([indices[name] for name in ("train", "validation", "test")])
    if not np.array_equal(np.sort(partition), np.arange(n_rows)):
        raise ValueError("Train/validation/test must partition all original rows.")
    if has_children:
        children = np.concatenate([indices["validation_tune"], indices["validation_calibration"]])
        if not np.array_equal(np.sort(children), np.sort(indices["validation"])):
            raise ValueError("Validation children must partition the parent validation rows.")


def split_dataset(
    data: Dataset,
    *,
    validation_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
    calibration_fraction: float | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Dataset]:
    """Create or reuse fingerprint-checked row indices; keep x/mask/y aligned.

    Relative manifest paths resolve from the repository root. With no manifest
    and no calibration_fraction, the original three-split toy API is unchanged.
    The mask always uses 1=observed, 0=missing, including after imputation.
    """
    parameters = {"validation_size": validation_size, "test_size": test_size,
                  "seed": seed, "calibration_fraction": calibration_fraction}
    path = None if manifest_path is None else Path(__file__).resolve().parents[2] / manifest_path
    fingerprint = dataset_fingerprint(data)
    if path is not None and path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("dataset_sha256") != fingerprint or manifest.get("parameters") != parameters:
            raise ValueError("Split manifest does not match the data or requested split configuration.")
        indices = {name: np.asarray(rows) for name, rows in manifest["indices"].items()}
    else:
        indices = make_split_indices(data["y"], **parameters)
    _validate_indices(indices, len(data["y"]), calibration_fraction is not None)
    if path is not None and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1, "dataset_sha256": fingerprint,
            "row_index_convention": "zero-based original file order",
            "parameters": parameters,
            "indices": {name: rows.tolist() for name, rows in indices.items()},
            "class_counts": {name: {str(label): int(np.sum(data["y"][rows] == label)) for label in (0, 1)} for name, rows in indices.items()},
        }
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def subset(rows: np.ndarray) -> Dataset:
        return {"x": data["x"][rows], "mask": data["mask"][rows], "y": data["y"][rows]}

    return {name: subset(rows) for name, rows in indices.items()}


def build_preprocessor(categorical_features: Sequence[int] | None = None) -> Pipeline | ColumnTransformer:
    """Train-fitted numeric imputation/scaling and optional categorical one-hot.

    Keep all-missing columns so dimensions remain compatible with future masks.
    Imputation does not change the original mask kept in the data dictionary.
    """
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    if categorical_features is None or len(categorical_features) == 0:
        return numeric
    columns = list(categorical_features)
    if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in columns) or len(set(columns)) != len(columns):
        raise ValueError("categorical_features must contain unique nonnegative integer indices.")
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("categorical", categorical, columns)], remainder=numeric, sparse_threshold=0,
    )
