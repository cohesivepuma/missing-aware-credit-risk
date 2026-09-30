"""Compare uncalibrated and validation-calibrated probabilities on one test split."""

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

import numpy as np
import yaml

from src.calibration.calibrators import SUPPORTED_METHODS, ProbabilityCalibrator
from src.calibration.reliability import plot_reliability_diagram
from src.data.loader import dataset_fingerprint, load_dataset
from src.data.preprocess import split_dataset
from src.metrics.metrics import DEFAULT_N_BINS, evaluate_metrics
from src.models.logistic import LogisticRegressionModel
from src.utils.seed import set_seed

METRIC_NAMES = ("auc", "f1", "brier", "ece")
DEFAULT_CALIBRATION: dict[str, Any] = {
    "methods": list(SUPPORTED_METHODS),
    "clip_epsilon": 1e-6,
}
CALIBRATION_FIT_SPLIT = "validation_calibration"


def _resolve(path: str | Path) -> Path:
    """Resolve a config or CLI path against the repository root unless absolute."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _display_path(path: Path | None) -> str | None:
    """Return a repository-relative path when possible so reports stay portable."""
    if path is None:
        return None
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _configured_output(config: dict[str, Any]) -> str | Path:
    """Prefer output.calibration_path so a shared config never overwrites its baseline report."""
    output = config["output"]
    if "calibration_path" in output:
        return output["calibration_path"]
    if "path" in output:
        baseline_path = Path(output["path"])
        return baseline_path.with_name(
            f"{baseline_path.stem}_calibration{baseline_path.suffix}"
        )
    raise ValueError("Config output needs 'calibration_path' or 'path'.")


def _split_options(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Return seed-specific split options; a manifest template may contain {seed}."""
    options = dict(config["split"])
    options["seed"] = seed
    manifest = options.get("manifest_path")
    if manifest is not None:
        template = str(manifest)
        options["manifest_path"] = template.format(seed=seed) if "{seed}" in template else template
    return options


def run_seed(
    config: dict[str, Any], seed: int, *, diagram_path: str | Path | None = None
) -> dict[str, Any]:
    """Train the frozen base model, calibrate on validation, and score the same test rows.

    Every variant is evaluated on one identical test split, so the reported
    before/after difference is paired instead of comparing different samples.
    """
    set_seed(seed)
    data_options = dict(config["data"])
    if data_options.get("path") is not None:
        data_options["path"] = str(_resolve(data_options["path"]))
    data = load_dataset(seed=seed, **data_options)
    splits = split_dataset(data, **_split_options(config, seed))
    if CALIBRATION_FIT_SPLIT not in splits:
        raise ValueError(
            "Calibration needs split.calibration_fraction so that "
            f"{CALIBRATION_FIT_SPLIT} exists as an independent validation child split."
        )

    model_options = dict(config["model"])
    if model_options.pop("name") != "logistic":
        raise ValueError("Only model.name='logistic' is implemented.")
    model = LogisticRegressionModel(seed=seed, **model_options)

    train = splits["train"]
    calibration_rows = splits[CALIBRATION_FIT_SPLIT]
    test = splits["test"]
    model.fit(train["x"], train["y"], mask=train["mask"])
    # The base model is frozen here: nothing below refits, retunes or replaces it.
    calibration_scores = model.predict_proba(
        calibration_rows["x"], mask=calibration_rows["mask"]
    )[:, 1]
    test_scores = model.predict_proba(test["x"], mask=test["mask"])[:, 1]

    calibration_options = {**DEFAULT_CALIBRATION, **config.get("calibration", {})}
    # 'none' is the uncalibrated reference that every run already reports.
    methods = [method for method in calibration_options["methods"] if method != "none"]
    clip_epsilon = float(calibration_options["clip_epsilon"])
    if not methods:
        raise ValueError("calibration.methods must list at least one method besides 'none'.")
    unknown = [method for method in methods if method not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(f"Unsupported calibration methods: {', '.join(map(str, unknown))}.")

    evaluation_options = config.get("evaluation", {})
    threshold = evaluation_options.get("threshold", 0.5)
    n_bins = evaluation_options.get("n_bins", DEFAULT_N_BINS)

    variants: dict[str, np.ndarray] = {"uncalibrated": test_scores}
    fitted_parameters: dict[str, dict[str, float | int]] = {}
    for method in methods:
        calibrator = ProbabilityCalibrator(method, clip_epsilon=clip_epsilon)
        calibrator.fit(calibration_scores, calibration_rows["y"])
        variants[method] = calibrator.predict_proba(test_scores)
        fitted_parameters[method] = dict(calibrator.fitted_parameters)

    metrics = {
        name: evaluate_metrics(test["y"], values, threshold=threshold, n_bins=n_bins)
        for name, values in variants.items()
    }
    uncalibrated = metrics["uncalibrated"]
    deltas = {
        method: {name: metrics[method][name] - uncalibrated[name] for name in METRIC_NAMES}
        for method in methods
    }

    diagram_written: Path | None = None
    if diagram_path is not None:
        curves = {name: (test["y"], values) for name, values in variants.items()}
        diagram_written = plot_reliability_diagram(
            curves,
            path=diagram_path,
            n_bins=n_bins,
            title=(
                f"{config['data']['name']} | seed {seed} | {n_bins} bins | "
                f"n={len(test['y'])} test rows"
            ),
        )

    return {
        "seed": seed,
        "dataset_sha256": dataset_fingerprint(data),
        "split_sizes": {name: len(part["y"]) for name, part in splits.items()},
        "calibration": {
            "fit_split": CALIBRATION_FIT_SPLIT,
            "fit_samples": len(calibration_rows["y"]),
            "evaluation_split": "test",
            "evaluation_samples": len(test["y"]),
            "methods": methods,
            "clip_epsilon": clip_epsilon,
            "threshold": threshold,
            "n_bins": n_bins,
            "fitted_parameters": fitted_parameters,
        },
        "metrics": metrics,
        "delta_vs_uncalibrated": deltas,
        "reliability_diagram": _display_path(diagram_written),
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """Return per-variant mean and standard deviation of every metric across seeds."""
    variants = list(runs[0]["metrics"])
    return {
        variant: {
            name: {
                "mean": float(np.mean([run["metrics"][variant][name] for run in runs])),
                "std": float(np.std([run["metrics"][variant][name] for run in runs])),
            }
            for name in METRIC_NAMES
        }
        for variant in variants
    }


def main() -> None:
    """Read a config, run one or more seeds, and write the paired comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/calibration.yaml"
    )
    parser.add_argument("--output", type=Path, help="Override report path (relative to repo root).")
    parser.add_argument(
        "--diagram-dir", type=Path, help="Override diagram directory (relative to repo root)."
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", help="Override the config seed for a multi-seed summary."
    )
    parser.add_argument("--no-diagram", action="store_true", help="Skip reliability diagrams.")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seeds = list(args.seeds) if args.seeds else [int(config["seed"])]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds are not allowed.")
    manifest = config["split"].get("manifest_path")
    if len(seeds) > 1 and manifest is not None and "{seed}" not in str(manifest):
        raise ValueError(
            "A multi-seed run needs split.manifest_path to be a template containing '{seed}'."
        )

    output = _resolve(args.output or _configured_output(config))
    diagram_dir = None if args.no_diagram else _resolve(args.diagram_dir or output.parent)
    if diagram_dir is not None:
        import matplotlib

        matplotlib.use("Agg")  # Batch CLI renders to files without opening a window.

    runs = []
    for seed in seeds:
        diagram_path = None if diagram_dir is None else diagram_dir / (
            f"reliability_{config['data']['name']}_seed{seed}.png"
        )
        runs.append(run_seed(config, seed, diagram_path=diagram_path))

    report: dict[str, Any] = {
        "config": config,
        "seeds": seeds,
        "runs": runs,
        "summary": summarize(runs) if len(runs) > 1 else None,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "PyYAML")
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    displayed = report["summary"] or runs[0]["metrics"]
    print(
        f"Dataset: {config['data']['name']} | Base model: {config['model']['name']} "
        f"| Seeds: {seeds}"
    )
    print(
        f"Calibrated on {CALIBRATION_FIT_SPLIT} ({runs[0]['calibration']['fit_samples']} rows); "
        f"all variants scored on the same {runs[0]['calibration']['evaluation_samples']} test rows."
    )
    print(f"Reliability bins: {runs[0]['calibration']['n_bins']}")
    print("Metrics by variant (AUC/F1 higher is better, Brier/ECE lower is better):")
    print(json.dumps(displayed, indent=2))
    print(f"Report saved to: {output}")
    if diagram_dir is not None:
        print(f"Reliability diagrams: {diagram_dir}")


if __name__ == "__main__":
    main()
