"""Official German Credit field schema and deterministic raw-file conversion.

Categorical numbers are transport codes, not ordinal measurements. Missingness
is injected into these 20 original fields before train-fitted one-hot encoding.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/144/statlog%2Bgerman%2Bcredit%2Bdata.zip"
RAW_SHA256 = "b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871"
TARGET_COLUMN = "bad_credit"
EXPECTED_ROWS = 1000


@dataclass(frozen=True)
class Feature:
    """One original UCI field, in the order used by x and its missing mask."""

    name: str
    kind: Literal["numeric", "categorical"]
    codes: tuple[str, ...] = ()
    unit: str | None = None
    maximum: int | None = None


FEATURES = (
    Feature("checking_status", "categorical", ("A11", "A12", "A13", "A14")),
    Feature("duration", "numeric", unit="months"),
    Feature("credit_history", "categorical", ("A30", "A31", "A32", "A33", "A34")),
    Feature("purpose", "categorical", tuple(f"A4{i}" for i in range(11))),
    Feature("credit_amount", "numeric", unit="DM"),
    Feature("savings_status", "categorical", ("A61", "A62", "A63", "A64", "A65")),
    Feature("employment", "categorical", ("A71", "A72", "A73", "A74", "A75")),
    Feature("installment_rate", "numeric", maximum=4),
    Feature("personal_status_sex", "categorical", ("A91", "A92", "A93", "A94", "A95")),
    Feature("other_debtors", "categorical", ("A101", "A102", "A103")),
    Feature("residence_since", "numeric", maximum=4),
    Feature("property", "categorical", ("A121", "A122", "A123", "A124")),
    Feature("age", "numeric", unit="years", maximum=150),
    Feature("other_installment_plans", "categorical", ("A141", "A142", "A143")),
    Feature("housing", "categorical", ("A151", "A152", "A153")),
    Feature("existing_credits", "numeric", maximum=4),
    Feature("job", "categorical", ("A171", "A172", "A173", "A174")),
    Feature("dependents", "numeric", maximum=2),
    Feature("telephone", "categorical", ("A191", "A192")),
    Feature("foreign_worker", "categorical", ("A201", "A202")),
)


def categorical_feature_indices() -> list[int]:
    """Return zero-based categorical column positions in the original x."""
    return [index for index, feature in enumerate(FEATURES) if feature.kind == "categorical"]


def feature_schema() -> dict[str, Any]:
    """Return portable metadata for models and the future missingness module."""
    return {
        "schema_version": 1,
        "source_page": SOURCE_PAGE,
        "raw_sha256": RAW_SHA256,
        "target": TARGET_COLUMN,
        "target_mapping": {"1 (Good)": 0, "2 (Bad)": 1},
        "mask_convention": "1 = observed, 0 = missing",
        "missingness_unit": "original field before one-hot encoding",
        "row_index_convention": "zero-based original file order; never a model feature",
        "categorical_feature_indices": categorical_feature_indices(),
        "features": [
            {"index": index, **asdict(feature),
             "transport_mapping": {code: value for value, code in enumerate(feature.codes)}}
            for index, feature in enumerate(FEATURES)
        ],
    }


def convert_raw_file(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Check and convert original fields without learning dataset statistics.

    Labels map 1=Good to 0 and 2=Bad to 1. Explicit 'unknown/no account'
    categories remain observed categories; they are not converted to NaN.
    Row order is preserved and duplicates are reported, never silently dropped.
    """
    raw = pd.read_csv(path, sep=r"\s+", header=None, dtype="string")
    if raw.shape[1] != len(FEATURES) + 1 or len(raw) == 0:
        raise ValueError("German Credit requires 20 original fields and one target.")
    if raw.isna().any().any():
        raise ValueError("The official raw German Credit file must not contain missing cells.")
    raw.columns = [feature.name for feature in FEATURES] + ["credit_class"]
    converted = pd.DataFrame(index=raw.index)
    for feature in FEATURES:
        values = raw[feature.name]
        if feature.kind == "categorical":
            mapping = {code: index for index, code in enumerate(feature.codes)}
            if not values.isin(mapping).all():
                raise ValueError(f"Unknown categorical code in {feature.name}.")
            converted[feature.name] = values.map(mapping).astype(np.float64)
        else:
            numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
            invalid = ~np.isfinite(numeric) | (numeric < 1) | (numeric != np.floor(numeric))
            if feature.maximum is not None:
                invalid |= numeric > feature.maximum
            if invalid.any():
                raise ValueError(f"Invalid numeric values in {feature.name}.")
            converted[feature.name] = numeric
    if not raw["credit_class"].isin(["1", "2"]).all():
        raise ValueError("Raw targets must be 1=Good or 2=Bad.")
    converted[TARGET_COLUMN] = raw["credit_class"].map({"1": 0, "2": 1}).astype(np.int64)
    quality = {
        "rows": len(raw),
        "features": len(FEATURES),
        "numeric_features": sum(feature.kind == "numeric" for feature in FEATURES),
        "categorical_features": len(categorical_feature_indices()),
        "missing_feature_cells": int(raw.drop(columns="credit_class").isna().sum().sum()),
        "missing_targets": int(raw["credit_class"].isna().sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_feature_rows": int(raw.drop(columns="credit_class").duplicated().sum()),
        "class_counts": {str(label): int(count) for label, count in converted[TARGET_COLUMN].value_counts().sort_index().items()},
        "numeric_ranges": {
            feature.name: {"min": float(converted[feature.name].min()), "max": float(converted[feature.name].max())}
            for feature in FEATURES if feature.kind == "numeric"
        },
        "explicit_unknown_categories_are_observed": True,
        "identifiers_or_target_in_features": False,
        "row_order_preserved": True,
    }
    return converted, quality
