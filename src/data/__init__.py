"""Data contract: x contains values, mask uses 1=observed / 0=missing, y is 0/1."""

from src.data.loader import Dataset, dataset_fingerprint, load_dataset
from src.data.missing_generator import (
    MissingnessPlan,
    MissingnessReport,
    MissingnessResult,
    apply_missingness,
    fit_missingness_plan,
    generate_missingness,
)
from src.data.preprocess import make_split_indices, split_dataset

__all__ = [
    "Dataset",
    "MissingnessPlan",
    "MissingnessReport",
    "MissingnessResult",
    "apply_missingness",
    "dataset_fingerprint",
    "fit_missingness_plan",
    "generate_missingness",
    "load_dataset",
    "make_split_indices",
    "split_dataset",
]
