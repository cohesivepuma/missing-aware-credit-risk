"""Run the complete-data control and the mechanisms x rates x seeds matrix."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.research import environment, fit_scenario, load_config, summarize_records, validate_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiment.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seeds:
        config["seeds"] = args.seeds
    validate_config(config)
    settings = [("complete", 0.0)] + [
        (mechanism, rate) for mechanism in config["missingness"]["mechanisms"]
        for rate in config["missingness"]["rates"]
    ]
    runs = []
    total = len(settings) * len(config["seeds"])
    for seed in config["seeds"]:
        for mechanism, rate in settings:
            scenario = fit_scenario(config, seed, mechanism, rate)
            runs.append(scenario.report)
            print(f"[{len(runs):02d}/{total}] seed={seed} {mechanism} target_rate={rate:.0%}", flush=True)
    records = [row for run in runs for row in run["records"]]
    config_json = json.dumps(config, sort_keys=True, ensure_ascii=False)
    report = {
        "schema_version": 1, "config": config,
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "protocol": {"rate_denominator": "observed cells in the six shared target fields",
                     "std_ddof": 1, "threshold_selected_on_test": False},
        "runs": runs, "summary": summarize_records(records), "environment": environment(),
    }
    output = PROJECT_ROOT / (args.output or config["output"]["path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(records).to_csv(output.with_suffix(".csv"), index=False)
    pd.DataFrame(report["summary"]).to_csv(output.with_name(output.stem + "_summary.csv"), index=False)
    print(f"Saved {len(records)} measured result rows to {output}")


if __name__ == "__main__":
    main()
