"""Check calibration split isolation and safe report output through the runner."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from experiments import run_calibration
from src.calibration.calibrators import ProbabilityCalibrator
from src.data.loader import load_dataset
from src.data.preprocess import split_dataset
from src.models.logistic import LogisticRegressionModel


@pytest.fixture
def calibration_config() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "configs/calibration.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["data"]["n_samples"] = 300
    return config


def test_shared_config_preserves_the_existing_baseline_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calibration_config: dict[str, Any]
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"baseline": "keep this report"}\n', encoding="utf-8")
    original = baseline.read_bytes()
    calibration_config["output"] = {"path": str(baseline)}
    config_path = tmp_path / "shared.yaml"
    config_path.write_text(yaml.safe_dump(calibration_config), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["run_calibration.py", "--config", str(config_path), "--no-diagram"]
    )

    run_calibration.main()

    assert baseline.read_bytes() == original
    report_path = Path(run_calibration._configured_output(calibration_config))
    assert report_path != baseline
    assert report_path.is_file()


def test_calibration_only_fits_independent_validation_rows(
    monkeypatch: pytest.MonkeyPatch, calibration_config: dict[str, Any]
) -> None:
    seed = calibration_config["seed"]
    data = load_dataset(seed=seed, **calibration_config["data"])
    splits = split_dataset(data, seed=seed, **calibration_config["split"])
    base_fits, predictions, calibration_fits = [], [], []
    original_fit = LogisticRegressionModel.fit
    original_predict = LogisticRegressionModel.predict_proba
    original_calibration_fit = ProbabilityCalibrator.fit

    def record_fit(
        self: LogisticRegressionModel, x: np.ndarray, y: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> LogisticRegressionModel:
        base_fits.append((x.copy(), y.copy()))
        return original_fit(self, x, y, mask=mask)

    def record_predict(
        self: LogisticRegressionModel, x: np.ndarray, mask: np.ndarray | None = None,
    ) -> np.ndarray:
        predictions.append(x.copy())
        return original_predict(self, x, mask=mask)

    def record_calibration_fit(
        self: ProbabilityCalibrator, probabilities: np.ndarray, y: np.ndarray,
    ) -> ProbabilityCalibrator:
        calibration_fits.append(y.copy())
        return original_calibration_fit(self, probabilities, y)

    monkeypatch.setattr(LogisticRegressionModel, "fit", record_fit)
    monkeypatch.setattr(LogisticRegressionModel, "predict_proba", record_predict)
    monkeypatch.setattr(ProbabilityCalibrator, "fit", record_calibration_fit)
    report = run_calibration.run_seed(calibration_config, seed)

    assert len(base_fits) == 1
    np.testing.assert_array_equal(base_fits[0][0], splits["train"]["x"])
    np.testing.assert_array_equal(base_fits[0][1], splits["train"]["y"])
    assert len(predictions) == 2
    np.testing.assert_array_equal(predictions[0], splits["validation_calibration"]["x"])
    np.testing.assert_array_equal(predictions[1], splits["test"]["x"])
    assert len(calibration_fits) == 2
    for labels in calibration_fits:
        np.testing.assert_array_equal(labels, splits["validation_calibration"]["y"])
    assert report["calibration"]["fit_split"] == "validation_calibration"
    assert report["calibration"]["evaluation_split"] == "test"


def test_runner_rejects_a_shared_validation_split(
    calibration_config: dict[str, Any]
) -> None:
    del calibration_config["split"]["calibration_fraction"]
    with pytest.raises(ValueError, match="independent validation child split"):
        run_calibration.run_seed(calibration_config, calibration_config["seed"])
