"""Data contract: x contains values, mask uses 1=observed / 0=missing, y is 0/1."""

from src.data.loader import Dataset, load_dataset
from src.data.preprocess import split_dataset

__all__ = ["Dataset", "load_dataset", "split_dataset"]
