"""Export within-seed paired mask and calibration effects from a benchmark."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.research import METRICS


def paired_ablation(report: dict) -> pd.DataFrame:
    """Positive deltas increase a metric; never turn pooled rows into fake seeds."""
    rows = []
    for run in report["runs"]:
        records = {(r["model"], r["calibration"]): r for r in run["records"]}
        comparisons = []
        for variant in ("uncalibrated", *report["config"]["calibration"]["methods"]):
            if ("mlp", variant) in records and ("mask_aware_mlp", variant) in records:
                comparisons.append(("mask", "mask_aware_mlp", variant, records[("mlp", variant)], records[("mask_aware_mlp", variant)]))
        for model in report["config"]["models"]:
            for method in report["config"]["calibration"]["methods"]:
                comparisons.append(("calibration", model, method, records[(model, "uncalibrated")], records[(model, method)]))
        for kind, model, variant, reference, treatment in comparisons:
            rows.append({
                "seed": run["seed"], "mechanism": run["mechanism"], "rate": run["rate"],
                "effect": kind, "model": model, "calibration": variant,
                **{f"delta_{metric}": treatment[metric] - reference[metric] for metric in METRICS},
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "results/benchmark.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/ablation.csv")
    args = parser.parse_args()
    frame = paired_ablation(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Saved {len(frame)} paired effects to {args.output}")


if __name__ == "__main__":
    main()
