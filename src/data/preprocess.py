"""Deterministic splits and preprocessing fitted on training data only."""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loader import Dataset


def split_dataset(
    data: Dataset,
    *,
    validation_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, Dataset]:
    """Stratify into train/validation/test; sizes are fractions of the full data.

    Keep x, mask and y aligned. The mask always describes the original values,
    including after future imputation: 1=observed, 0=missing.
    """
    if not (0 < validation_size < 1 and 0 < test_size < 1):
        raise ValueError("validation_size and test_size must be between 0 and 1.")
    if validation_size + test_size >= 1:
        raise ValueError("Validation and test fractions must leave training data.")
    indices = np.arange(len(data["y"]))
    train_validation, test = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=data["y"]
    )
    train, validation = train_test_split(
        train_validation,
        test_size=validation_size / (1 - test_size),
        random_state=seed,
        stratify=data["y"][train_validation],
    )

    def subset(rows: np.ndarray) -> Dataset:
        return {"x": data["x"][rows], "mask": data["mask"][rows], "y": data["y"][rows]}

    return {"train": subset(train), "validation": subset(validation), "test": subset(test)}


def build_preprocessor() -> Pipeline:
    """Median imputation then scaling; call fit only on the training split.

    Keep all-missing columns so dimensions remain compatible with future masks.
    Imputation does not change the original mask kept in the data dictionary.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
