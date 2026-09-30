"""Get verified official credit data and reusable fixed split indices."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prepare import prepare_german_credit


def main() -> None:
    """Download to local ignored folders, prepare data and print an audit summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw/german_credit")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/processed/german_credit")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare_german_credit(raw_dir=args.raw_dir, output_dir=args.output_dir, seed=args.seed), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
