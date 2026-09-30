"""Pinned UCI snapshots and schemas; category carrier codes are not learned."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.german_credit import categorical_feature_indices
from src.data.loader import Dataset, load_dataset

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Source:
    url: str
    filename: str
    sha256: str
    rows: int
    positive_label: str
    categorical: tuple[int, ...]

SOURCES = {
    "credit_approval": Source(
        "https://archive.ics.uci.edu/static/public/27/credit+approval.zip", "crx.data",
        "fff49bc186cbddb3ace7371d40d9fbbb3af4f126019c13ff3f562249b1454f4d",
        690, "approved (+)", (0, 3, 4, 5, 6, 8, 9, 11, 12)),
    "taiwan_default": Source(
        "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
        "default of credit card clients.xls",
        "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933",
        30000, "default payment next month", (1, 2, 3)),
}
APPROVAL_CATEGORIES = {
    0: ("b", "a"), 3: ("u", "y", "l", "t"), 4: ("g", "p", "gg"),
    5: ("c", "d", "cc", "i", "j", "k", "m", "r", "q", "w", "x", "e", "aa", "ff"),
    6: ("v", "h", "bb", "j", "n", "z", "dd", "ff", "o"),
    8: ("t", "f"), 9: ("t", "f"), 11: ("t", "f"), 12: ("g", "p", "s"),
}
TAIWAN_FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0"] + [
    f"PAY_{i}" for i in range(2, 7)] + [f"BILL_AMT{i}" for i in range(1, 7)] + [
    f"PAY_AMT{i}" for i in range(1, 7)]

def load_benchmark(name: str, root: Path = ROOT) -> tuple[Dataset, list[int], dict]:
    """Check raw snapshot; retain every row, original NaNs and outcome meaning."""
    if name == "german_credit":
        data = load_dataset(name, path=root / "data/processed/german_credit/german_credit.csv")
        from src.data.german_credit import RAW_SHA256
        metadata = {"raw_sha256": RAW_SHA256, "positive_label": "Bad", "source": "UCI 144"}
        categorical = categorical_feature_indices()
    else:
        if name not in SOURCES:
            raise ValueError(f"Unknown benchmark: {name}")
        spec = SOURCES[name]
        path = root / "data/raw" / name / spec.filename
        if not path.exists():
            raise FileNotFoundError("Run python scripts/prepare_credit_benchmarks.py first.")
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum != spec.sha256:
            raise ValueError(f"Raw source checksum mismatch: {name}")
        if name == "credit_approval":
            frame = pd.read_csv(path, header=None, dtype="string", na_values="?", keep_default_na=False)
            if frame.shape != (spec.rows, 16) or not frame[15].isin(["+", "-"]).all():
                raise ValueError("Credit Approval schema/labels differ from the pinned source.")
            converted = {}
            for column in range(15):
                values = frame[column]
                if column in APPROVAL_CATEGORIES:
                    mapping = {code: i for i, code in enumerate(APPROVAL_CATEGORIES[column])}
                    if not values.dropna().isin(mapping).all():
                        raise ValueError(f"Unknown category in A{column + 1}")
                    converted[column] = values.map(mapping).to_numpy(dtype=float, na_value=np.nan)
                else:
                    converted[column] = pd.to_numeric(values, errors="raise").to_numpy(dtype=float, na_value=np.nan)
            x = np.column_stack(list(converted.values()))
            y = (frame[15] == "+").to_numpy(dtype=np.int64)
        else:
            frame = pd.read_excel(path, header=1, engine="xlrd")
            expected = ["ID", *TAIWAN_FEATURES, "default payment next month"]
            if list(frame.columns) != expected or len(frame) != spec.rows or frame.isna().any().any():
                raise ValueError("Taiwan source schema differs from the pinned snapshot.")
            if not np.array_equal(frame.ID, np.arange(1, spec.rows + 1)):
                raise ValueError("Unexpected Taiwan IDs.")
            for column, allowed in {"SEX": [1, 2], "EDUCATION": range(7), "MARRIAGE": range(4)}.items():
                if not frame[column].isin(allowed).all():
                    raise ValueError(f"Unknown Taiwan category in {column}")
            x = frame[TAIWAN_FEATURES].to_numpy(dtype=np.float64)
            y = frame.iloc[:, -1].to_numpy(dtype=np.int64)
        if np.isinf(x).any() or not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("Invalid features or labels.")
        data = {"x": x, "mask": (~np.isnan(x)).astype(np.uint8), "y": y}
        categorical = list(spec.categorical)
        metadata = {"raw_sha256": checksum, "positive_label": spec.positive_label, "source": spec.url}
    metadata.update(rows=len(data["y"]), features=data["x"].shape[1],
                    natural_missing_cells=int(np.isnan(data["x"]).sum()),
                    duplicate_feature_rows=int(pd.DataFrame(data["x"]).duplicated().sum()),
                    class_counts=np.bincount(data["y"]).tolist(), categorical=categorical)
    return data, categorical, metadata
