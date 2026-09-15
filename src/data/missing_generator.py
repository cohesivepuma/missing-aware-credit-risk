"""Planned MCAR/MAR/MNAR injection interface; not implemented in phase 1."""

from typing import Literal

from src.data.loader import Dataset

MissingMechanism = Literal["mcar", "mar", "mnar"]


def generate_missingness(
    data: Dataset,
    *,
    mechanism: MissingMechanism,
    rate: float,
    seed: int = 42,
) -> Dataset:
    """Plan: copy data, inject NaN into x, and set mask to 0 for missing values.

    Observed mask entries remain 1; y must never be masked or used to decide
    feature missingness. TODO: define mechanisms, rate semantics, eligible
    features and train-only mechanism fitting before implementing MAR/MNAR.
    """
    if mechanism not in {"mcar", "mar", "mnar"}:
        raise ValueError("mechanism must be 'mcar', 'mar' or 'mnar'.")
    if not 0 <= rate <= 1:
        raise ValueError("rate must be between 0 and 1.")
    raise NotImplementedError("MCAR/MAR/MNAR generation is planned for phase 2.")
