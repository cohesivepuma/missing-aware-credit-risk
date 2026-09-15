"""Run the offline reproducible Logistic Regression baseline from a YAML file."""

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

# Allow the requested direct script invocation without an editable installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.data.loader import dataset_fingerprint, load_dataset
from src.data.preprocess import split_dataset
from src.metrics.metrics import evaluate_metrics
from src.models.logistic import LogisticRegressionModel
from src.utils.seed import set_seed


def run_baseline(config: dict[str, Any]) -> dict[str, Any]:
    """Fit on train, report test metrics, and reserve validation for calibration."""
    seed = config["seed"]
    set_seed(seed)
    data_options = dict(config["data"])
    if data_options.get("path") is not None:
        data_options["path"] = str(PROJECT_ROOT / data_options["path"])
    data = load_dataset(seed=seed, **data_options)
    splits = split_dataset(data, seed=seed, **config["split"])
    model_options = dict(config["model"])
    if model_options.pop("name") != "logistic":
        raise ValueError("Only model.name='logistic' is implemented in phase 1.")
    model = LogisticRegressionModel(seed=seed, **model_options)
    train, test = splits["train"], splits["test"]
    model.fit(train["x"], train["y"], mask=train["mask"])
    probabilities = model.predict_proba(test["x"], mask=test["mask"])[:, 1]
    metrics = evaluate_metrics(test["y"], probabilities, **config["evaluation"])

    return {
        "config": config,
        "dataset_sha256": dataset_fingerprint(data),
        "split_sizes": {name: len(part["y"]) for name, part in splits.items()},
        "evaluation_split": "test",
        "metrics": metrics,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "torch", "PyYAML")
            },
        },
    }


def main() -> None:
    """Read config, print real metrics and write an ignored local JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/baseline.yaml")
    parser.add_argument("--output", type=Path, help="Override output path (relative to repo root).")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    report = run_baseline(config)
    output = args.output or Path(config["output"]["path"])
    output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Dataset: {config['data']['name']} | Model: logistic | Seed: {config['seed']}")
    print(f"Split sizes: {report['split_sizes']}")
    print("Test metrics:")
    print(json.dumps(report["metrics"], indent=2))
    print(f"Report saved to: {output}")


if __name__ == "__main__":
    main()
