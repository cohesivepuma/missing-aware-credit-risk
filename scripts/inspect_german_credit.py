"""Preview at most five raw German Credit rows before preparing the dataset."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    """Print file size, inferred sample dtypes and at most five sample rows."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=root / "data/raw/german_credit/german.data")
    args = parser.parse_args()
    preview = pd.read_csv(args.path, sep=r"\s+", header=None, nrows=5)
    preview.columns = [f"attribute_{i + 1}" for i in range(preview.shape[1] - 1)] + ["credit_class"]
    print(json.dumps({
        "file": str(args.path),
        "bytes": args.path.stat().st_size,
        "sample_rows": len(preview),
        "columns": list(preview.columns),
        "sample_dtypes": {column: str(dtype) for column, dtype in preview.dtypes.items()},
        "preview": preview.to_dict(orient="records"),
    }, indent=2))


if __name__ == "__main__":
    main()
