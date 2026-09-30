"""Behaviour checks for matched neural controls and the research service."""
from copy import deepcopy

import numpy as np
import pytest

from src.data.loader import load_dataset
from src.models.mlp import MLPModel
from src.models.mask_aware_mlp import MaskAwareMLP
from src.models.random_forest import RandomForestModel
from src.research import fit_scenario, summarize_records, validate_config


def small_config():
    return {
        "seeds": [42], "data": {"name": "toy", "n_samples": 160, "n_features": 6, "n_informative": 3},
        "split": {"validation_size": .25, "test_size": .25, "calibration_fraction": .5},
        "missingness": {"mechanisms": ["mcar", "mar", "mnar"], "rates": [.3],
                        "target_features": [1, 2], "mar_driver_features": [0], "direction": "higher", "strength": 1},
        "models": ["logistic", "random_forest", "mlp", "mask_aware_mlp"],
        "model_parameters": {"random_forest": {"n_estimators": 10},
                             "mlp": {"epochs": 2}, "mask_aware_mlp": {"epochs": 2}},
        "calibration": {"methods": ["platt", "isotonic"]},
        "evaluation": {"threshold": .5, "n_bins": 10},
    }


def test_complete_data_controls_are_identical():
    data = load_dataset(n_samples=80, n_features=6, n_informative=3)
    plain = MLPModel(epochs=3).fit(data["x"], data["y"])
    aware = MaskAwareMLP(epochs=3).fit(data["x"], data["y"], data["mask"])
    assert plain.parameter_count_ == aware.parameter_count_
    np.testing.assert_array_equal(plain.predict_proba(data["x"]), aware.predict_proba(data["x"], data["mask"]))


@pytest.mark.parametrize("model_type", [MLPModel, MaskAwareMLP, RandomForestModel])
def test_missing_predictions_are_reproducible_and_keep_inputs(model_type):
    data = load_dataset(n_samples=80, n_features=6, n_informative=3)
    data["x"][:40, 0] = np.nan
    data["mask"][:40, 0] = 0
    original = deepcopy(data)
    options = {"n_estimators": 10} if model_type is RandomForestModel else {"epochs": 2}
    a = model_type(**options).fit(data["x"], data["y"], data["mask"])
    b = model_type(**options).fit(data["x"], data["y"], data["mask"])
    first = a.predict_proba(data["x"], data["mask"])
    np.testing.assert_array_equal(first, b.predict_proba(data["x"], data["mask"]))
    np.testing.assert_allclose(first.sum(axis=1), 1)
    assert np.isfinite(first).all() and np.all((first >= 0) & (first <= 1))
    for key in data:
        np.testing.assert_array_equal(data[key], original[key])


def test_mask_contract_and_prediction_before_fit():
    data = load_dataset(n_samples=80)
    with pytest.raises(RuntimeError):
        MLPModel().predict_proba(data["x"])
    with pytest.raises(ValueError, match="aligned mask"):
        MaskAwareMLP(epochs=1).fit(data["x"], data["y"])
    with pytest.raises(ValueError, match="original NaN"):
        MaskAwareMLP(epochs=1).fit(data["x"], data["y"], np.zeros_like(data["mask"]))


@pytest.mark.parametrize("mechanism", ["mcar", "mar", "mnar"])
def test_scenario_shared_fields_and_calibration_isolation(mechanism, monkeypatch):
    import src.research as research
    from src.calibration.calibrators import ProbabilityCalibrator
    seen = []
    original = ProbabilityCalibrator.fit
    def spy(self, p, y):
        seen.append((p.copy(), y.copy()))
        return original(self, p, y)
    monkeypatch.setattr(ProbabilityCalibrator, "fit", spy)
    result = fit_scenario(small_config(), 42, mechanism, .3)
    assert result.report["missingness_plan"]["injectable_features"] == (1, 2)
    assert sum(result.report["split_sizes"].values()) == 160
    assert len(result.report["records"]) == 12
    for name, part in result.splits.items():
        assert (part["mask"][:, [0, 3, 4, 5]] == 1).all()
        np.testing.assert_array_equal(part["mask"], ~np.isnan(part["x"]))
    for p, y in seen:
        np.testing.assert_array_equal(y, result.splits["validation_calibration"]["y"])
        assert len(y) == 20
    for name, model in result.models.items():
        test = result.splits["test"]
        np.testing.assert_array_equal(model.predict_proba(test["x"], test["mask"])[:, 1], result.probabilities[name]["uncalibrated"])


def test_runner_rejects_shared_validation_and_duplicate_seeds():
    config = small_config()
    config["split"].pop("calibration_fraction")
    with pytest.raises(ValueError, match="independent"):
        validate_config(config)
    config = small_config()
    config["seeds"] = [42, 42]
    with pytest.raises(ValueError, match="unique"):
        validate_config(config)


def test_summary_uses_sample_standard_deviation():
    records = [dict(mechanism="mcar", rate=.3, model="mlp", calibration="platt", auc=v, f1=v, brier=v, ece=v) for v in (.1, .3)]
    summary = summarize_records(records)[0]
    assert summary["auc_mean"] == pytest.approx(.2)
    assert summary["auc_std"] == pytest.approx(np.std([.1, .3], ddof=1))
