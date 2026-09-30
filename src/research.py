"""Shared experiment and demo service; only training and calibration splits fit."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
from pathlib import Path
import platform
from typing import Any

import numpy as np
import yaml

from src.calibration.calibrators import ProbabilityCalibrator, SUPPORTED_METHODS
from src.data.german_credit import categorical_feature_indices
from src.data.loader import Dataset, dataset_fingerprint, load_dataset
from src.data.missing_generator import apply_missingness, fit_missingness_plan
from src.data.preprocess import split_dataset
from src.metrics.metrics import evaluate_metrics
from src.models.base import BaseModel
from src.models.logistic import LogisticRegressionModel
from src.models.mask_aware_mlp import MaskAwareMLP
from src.models.mlp import MLPModel
from src.models.random_forest import RandomForestModel
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_TYPES = {
    "logistic": LogisticRegressionModel,
    "random_forest": RandomForestModel,
    "mlp": MLPModel,
    "mask_aware_mlp": MaskAwareMLP,
}
MODEL_LABELS = {
    "logistic": "Logistic Regression", "random_forest": "Random Forest",
    "mlp": "MLP · 无掩码", "mask_aware_mlp": "MLP · 缺失感知",
}
METRICS = ("auc", "f1", "brier", "ece")
PARTITIONS = ("train", "validation_tune", "validation_calibration", "test")


def load_config(path: str | Path = "configs/experiment.yaml") -> dict[str, Any]:
    """Load the portable research configuration relative to the repository."""
    return yaml.safe_load((PROJECT_ROOT / path).read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    """Fail before starting a matrix if the protocol is incomplete or ambiguous."""
    seeds = config["seeds"]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be nonempty and unique.")
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("Invalid seed.")
    if not config["split"].get("calibration_fraction"):
        raise ValueError("An independent validation_calibration split is required.")
    manifest = config["split"].get("manifest_path")
    if len(seeds) > 1 and manifest and "{seed}" not in manifest:
        raise ValueError("Multiple seeds require a {seed} manifest template.")
    models = config["models"]
    if not models or len(models) != len(set(models)) or any(m not in MODEL_TYPES for m in models):
        raise ValueError("models must list unique implemented models.")
    methods = config["calibration"]["methods"]
    if len(methods) != len(set(methods)) or any(m not in SUPPORTED_METHODS for m in methods):
        raise ValueError("Unsupported or duplicate calibration method.")
    missing = config["missingness"]
    mechanisms = missing["mechanisms"]
    if not mechanisms or len(mechanisms) != len(set(mechanisms)) or any(m not in ("mcar", "mar", "mnar") for m in mechanisms):
        raise ValueError("Unsupported or duplicate missingness mechanism.")
    rates = missing["rates"]
    if not rates or len(rates) != len(set(rates)) or any(not 0 < r <= 1 for r in rates):
        raise ValueError("Missingness rates must be unique and in (0, 1].")
    if set(missing["target_features"]) & set(missing["mar_driver_features"]):
        raise ValueError("MAR drivers must be separate from the shared target fields.")
    if config["data"]["name"] == "german_credit" and set(missing["target_features"]) & set(categorical_feature_indices()):
        raise ValueError("This MNAR protocol must target numerical/ordered fields.")


def load_study_data(config: dict[str, Any], seed: int) -> Dataset:
    options = dict(config["data"])
    if options.get("path"):
        options["path"] = PROJECT_ROOT / options["path"]
    return load_dataset(seed=seed, **options)


@dataclass
class Scenario:
    """In-memory fitted objects for the demo, plus a serializable audit report."""

    models: dict[str, BaseModel]
    calibrators: dict[str, dict[str, ProbabilityCalibrator]]
    clean_splits: dict[str, Dataset]
    splits: dict[str, Dataset]
    probabilities: dict[str, dict[str, np.ndarray]]
    report: dict[str, Any]


def fit_scenario(config: dict[str, Any], seed: int, mechanism: str, rate: float) -> Scenario:
    """Use identical rows and masks for every model, then calibrate frozen scores."""
    validate_config(config)
    if seed not in config["seeds"]:
        raise ValueError("Scenario seed must be listed in the study configuration.")
    if mechanism not in ("complete", "mcar", "mar", "mnar"):
        raise ValueError("Unknown missingness mechanism.")
    if (mechanism == "complete") != (rate == 0):
        raise ValueError("Only the complete-data scenario uses rate=0.")
    set_seed(seed)
    data = load_study_data(config, seed)
    options = dict(config["split"])
    if options.get("manifest_path"):
        options["manifest_path"] = options["manifest_path"].format(seed=seed)
    all_splits = split_dataset(data, seed=seed, **options)
    clean = {name: all_splits[name] for name in PARTITIONS}
    splits = clean
    plan_report = None
    missing_reports: dict[str, Any] = {}
    if mechanism != "complete":
        settings = config["missingness"]
        targets = settings["target_features"]
        drivers = settings["mar_driver_features"] if mechanism == "mar" else None
        eligible = sorted(set(targets) | set(drivers or []))
        plan = fit_missingness_plan(
            clean["train"], mechanism=mechanism, rate=rate, seed=seed,
            eligible_features=eligible, driver_features=drivers,
            direction=settings["direction"], strength=settings["strength"],
        )
        plan_report = asdict(plan)
        splits = {}
        for offset, name in enumerate(PARTITIONS):
            # Independent split draws; train-derived thresholds remain frozen.
            split_plan = replace(plan, seed=(seed + 104729 * offset) % 2**32)
            result = apply_missingness(clean[name], split_plan, return_report=True)
            splits[name] = result.data
            missing_reports[name] = asdict(result.report)
    train, calibration, test = (splits[n] for n in ("train", "validation_calibration", "test"))
    models, calibrators, probabilities, records = {}, {}, {}, []
    model_details = {}
    categorical = categorical_feature_indices() if config["data"]["name"] == "german_credit" else config.get("categorical_features")
    for name in config["models"]:
        params = deepcopy(config.get("model_parameters", {}).get(name, {}))
        model = MODEL_TYPES[name](seed=seed, categorical_features=categorical, **params)
        model.fit(train["x"], train["y"], mask=train["mask"])
        models[name] = model
        calibration_scores = model.predict_proba(calibration["x"], mask=calibration["mask"])[:, 1]
        scores = model.predict_proba(test["x"], mask=test["mask"])[:, 1]
        variants = {"uncalibrated": scores}
        calibrators[name] = {}
        for method in config["calibration"]["methods"]:
            calibrator = ProbabilityCalibrator(method)
            calibrator.fit(calibration_scores, calibration["y"])
            calibrators[name][method] = calibrator
            variants[method] = calibrator.predict_proba(scores)
        probabilities[name] = variants
        model_details[name] = {
            "parameters": params,
            "parameter_count": getattr(model, "parameter_count_", None),
            "calibrators": {method: c.fitted_parameters for method, c in calibrators[name].items()},
        }
        for variant, values in variants.items():
            records.append({
                "seed": seed, "mechanism": mechanism, "rate": rate,
                "overall_test_missing_rate": float(np.mean(test["mask"] == 0)),
                "model": name, "calibration": variant,
                **evaluate_metrics(test["y"], values, **config["evaluation"]),
            })
    report = {
        "seed": seed, "mechanism": mechanism, "rate": rate,
        "dataset_sha256": dataset_fingerprint(data),
        "split_sizes": {name: len(part["y"]) for name, part in splits.items()},
        "split_sha256": {name: dataset_fingerprint(part) for name, part in clean.items()},
        "mask_sha256": {name: hashlib.sha256(part["mask"].tobytes()).hexdigest() for name, part in splits.items()},
        "missingness_plan": plan_report, "missingness": missing_reports,
        "fit_split": "train", "calibration_fit_split": "validation_calibration",
        "evaluation_split": "test", "validation_tune_used": False,
        "models": model_details, "records": records,
        "predictions": {name: {v: p.tolist() for v, p in variants.items()} for name, variants in probabilities.items()},
        "test_labels": test["y"].tolist(),
    }
    return Scenario(models, calibrators, clean, splits, probabilities, report)


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate each setting over seeds; sample SD (ddof=1), no significance claim."""
    groups = defaultdict(list)
    for row in records:
        groups[(row["mechanism"], row["rate"], row["model"], row["calibration"])].append(row)
    summary = []
    for (mechanism, rate, model, calibration), rows in sorted(groups.items()):
        item = dict(mechanism=mechanism, rate=rate, model=model, calibration=calibration, n_seeds=len(rows))
        for metric in METRICS:
            values = [row[metric] for row in rows]
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(rows) > 1 else 0.0
        summary.append(item)
    return summary


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "pandas", "scikit-learn", "torch", "streamlit", "PyYAML")},
    }
