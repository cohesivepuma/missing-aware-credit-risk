"""Load offline toy data or a user-supplied numeric credit dataset."""

from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import make_classification


class Dataset(TypedDict):
    """Raw x may contain NaN; mask is 1=observed, 0=missing; y is binary."""

    x: NDArray[np.float64]
    mask: NDArray[np.uint8]
    y: NDArray[np.int64]


def load_dataset(
    name: str = "toy",
    *,
    seed: int = 42,
    n_samples: int = 1000,
    n_features: int = 12,
    n_informative: int = 6,
    path: str | Path | None = None,
    target_column: str | None = None,
) -> Dataset:
    """Return {x, mask, y}. CSV targets must already use 1=default, 0=non-default.

    Toy data is synthetic and only verifies the pipeline, not credit performance.
    CSV features must be numeric; categorical encoding is a future dataset task.
    """
    if name == "toy":
        x, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=0,
            random_state=seed,
        )
    elif name == "csv":
        if path is None or target_column is None:
            raise ValueError("CSV loading requires path and target_column.")
        frame = pd.read_csv(path)
        if target_column not in frame.columns:
            raise ValueError(f"Target column {target_column!r} not found.")
        y = frame[target_column].to_numpy()
        features = frame.drop(columns=[target_column])
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
            raise ValueError("CSV features must be numeric; encode categories first.")
        x = features.to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"Unsupported dataset: {name!r}. Choose 'toy' or 'csv'.")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    if x.ndim != 2 or x.shape[1] == 0 or x.shape[0] != len(y):
        raise ValueError("Expected a nonempty feature matrix aligned with labels.")
    if np.isinf(x).any():
        raise ValueError("Features may contain NaN, but not infinity.")
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("Target must contain both classes, encoded as 0 and 1.")
    return {
        "x": x,
        "mask": (~np.isnan(x)).astype(np.uint8),  # 1=observed, 0=missing
        "y": y.astype(np.int64),
    }
