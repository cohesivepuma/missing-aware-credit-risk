"""Behavioral checks for uploaded-data isolation and frozen batch scoring."""

import io
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from service.engine import (
    demo_frame, profile_frame, read_csv_bytes, score_batch, train_analysis,
    validate_training,
)


def _config(frame, **updates):
    config = {
        "target_column": "outcome", "positive_value": "1", "positive_label": "模拟事件",
        "id_column": None,
        "feature_columns": [name for name in frame if name != "outcome"],
        "categorical_columns": [column["name"] for column in profile_frame(frame)["columns"]
                                if column["name"] != "outcome" and column["kind"] == "categorical"],
        "models": ["logistic"], "seed": 42,
    }
    config.update(updates)
    if "categorical_columns" not in updates:
        config["categorical_columns"] = [name for name in config["categorical_columns"]
                                          if name in config["feature_columns"]]
    return config


@pytest.fixture(scope="module")
def trained():
    frame = demo_frame().iloc[:240].reset_index(drop=True)
    result, bundle = train_analysis(frame, _config(frame))
    return frame, result, bundle


def test_csv_preserves_literal_categories_and_missing_definition():
    frame = read_csv_bytes(b'\xef\xbb\xbfid,category,value\r\n001,NA,1\r\n002,None,?\r\n003,NULL,\r\n004," quoted, value ",4\r\n')
    assert frame["id"].tolist() == ["001", "002", "003", "004"]
    assert frame["category"].tolist() == ["NA", "None", "NULL", "quoted, value"]
    assert frame["value"].iloc[1:3].isna().all()
    assert isinstance(frame["value"].iloc[0], str)


@pytest.mark.parametrize("content", [b"", b"\n", b"x,y\n", b"x,x\n1,2\n", b"x, x\n1,2\n",
                                      b"x,\n1,2\n", b"x,y\n1\n", b"x,y\n1,2,3\n",
                                      b'x,y\n"unterminated,1', b"x\n\xff", b"x\n\x00"])
def test_csv_rejects_empty_ambiguous_and_malformed_files(content):
    with pytest.raises(ValueError):
        read_csv_bytes(content)


def test_csv_enforces_declared_resource_limits():
    for content in (b"a\n" + b"x" * (20 * 1024 * 1024),
                    b"a\n" + b"x\n" * 50_001,
                    (",".join(f"x{i}" for i in range(101)) + "\n" + ",".join("1" for _ in range(101))).encode()):
        with pytest.raises(ValueError):
            read_csv_bytes(content)


def test_profile_is_finite_json_and_samples_are_strings():
    frame = read_csv_bytes(b"number,category,empty\n1,NA,\n2,None,?\n3,blue,\n")
    profile = profile_frame(frame)
    assert profile["columns"][0]["kind"] == "numeric"
    assert profile["columns"][1]["kind"] == "categorical"
    assert profile["columns"][2]["missing_count"] == 3
    assert profile["preview"][0]["empty"] is None
    assert all(isinstance(value, str) for column in profile["columns"] for value in column["sample_values"])
    json.dumps(profile, allow_nan=False)


def test_synthetic_data_is_reproducible_and_has_usable_missingness():
    first = demo_frame()
    pd.testing.assert_frame_equal(first, demo_frame())
    assert len(first) >= 500
    assert first.attrs["source"] == "synthetic"
    assert set(first.outcome) == {"0", "1"}
    assert first.drop(columns="outcome").isna().any().all()
    assert not first.outcome.isna().any()


def test_identical_numeric_equivalent_features_never_cross_partitions():
    rows = []
    for group in range(120):
        # Conflicting labels and alternate numeric spellings must not break grouping.
        rows.extend([{"amount": str(group), "kind": "A", "outcome": str(group % 2)},
                     {"amount": f"{group}.0", "kind": "A", "outcome": str(1 - group % 2)}])
    frame = pd.DataFrame(rows)
    result, _ = train_analysis(frame, _config(frame))
    groups = []
    indices = result["provenance"]["split_indices"]
    for name in ("train", "calibration", "test"):
        groups.append(set(frame.iloc[indices[name]].amount.astype(float)))
        assert min(result["class_counts"][name]) > 0
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert sorted(sum(indices.values(), [])) == list(range(len(frame)))
    assert result["provenance"]["independent_feature_groups"] == 120
    assert result["split_sizes"] == {"train": 144, "calibration": 48, "test": 48}


def test_categories_and_imputation_are_learned_only_on_training_rows():
    frame = pd.DataFrame({"amount": [str(i) for i in range(200)],
                          "category": [f"rare-{i}" if i < 90 else "common" for i in range(200)],
                          "outcome": [str(i % 2) for i in range(200)]})
    frame.loc[::7, "amount"] = np.nan
    result, bundle = train_analysis(frame, _config(frame))
    train_rows = result["provenance"]["split_indices"]["train"]
    expected = set(frame.iloc[train_rows].category)
    assert set(bundle["category_maps"]["category"]) == expected
    assert expected < set(frame.category)
    preprocessor = bundle["models"]["logistic"]["estimator"].pipeline.named_steps["preprocess"]
    fitted_median = preprocessor.named_transformers_["remainder"].named_steps["imputer"].statistics_[0]
    assert fitted_median == frame.iloc[train_rows].amount.astype(float).median()
    assert bundle["provenance"]["calibrator_fit_partition"] == "calibration"


def test_repeat_training_is_deterministic_and_every_variant_is_reported(trained):
    frame, first, bundle = trained
    second, again = train_analysis(frame, _config(frame))
    assert first == second
    assert [(m["model"], m["calibration"]) for m in first["metrics"]] == [
        ("logistic", "raw"), ("logistic", "platt"), ("logistic", "isotonic")]
    for metric in first["metrics"]:
        assert sum(metric["reliability"]["counts"]) == first["split_sizes"]["test"]
        assert len(metric["reliability"]["counts"]) == 10
        assert 0 <= metric["auc"] <= 1 and 0 <= metric["brier"] <= 1
    first_scores = score_batch(bundle, frame.iloc[:12], "logistic", "platt")[1]
    second_scores = score_batch(again, frame.iloc[:12], "logistic", "platt")[1]
    pd.testing.assert_frame_equal(first_scores, second_scores)
    json.dumps(first, allow_nan=False)


class _MaskCheckingEstimator:
    def __init__(self, category_index):
        self.category_index = category_index

    def predict_proba(self, values, mask=None):
        assert values[0, self.category_index] == -1
        assert mask[0, self.category_index]
        assert np.isnan(values[1, self.category_index])
        assert not mask[1, self.category_index]
        assert np.array_equal(mask, ~np.isnan(values))
        return np.tile([0.7, 0.3], (len(values), 1))


def test_unseen_category_is_distinct_from_missing_and_never_refits(trained):
    frame, _, original = trained
    bundle = dict(original, models={"logistic": {"estimator": _MaskCheckingEstimator(
        original["feature_columns"].index("home_ownership")), "calibrators": {}}})
    batch = frame.iloc[:2].copy()
    batch.loc[batch.index[0], "home_ownership"] = "new_category"
    batch.loc[batch.index[1], "home_ownership"] = np.nan
    before = dict(original["category_maps"]["home_ownership"])
    summary, scored = score_batch(bundle, batch, "logistic", "raw")
    quality = next(item for item in summary["quality"] if item["name"] == "home_ownership")
    assert quality["unseen_rate"] == 0.5
    assert quality["batch_missing_rate"] == 0.5
    assert original["category_maps"]["home_ownership"] == before
    assert scored.probability.tolist() == [0.3, 0.3]
    assert any("未见" in text for text in summary["warnings"])


def test_scoring_requires_frozen_numeric_schema_and_feature_headers(trained):
    frame, _, bundle = trained
    with pytest.raises(ValueError, match="缺少必需字段"):
        score_batch(bundle, frame.drop(columns="annual_income"), "logistic", "raw")
    for invalid in ("not-a-number", "inf", "1e400", "10000000000001"):
        batch = frame.iloc[:3].copy()
        batch.loc[batch.index[0], "annual_income"] = invalid
        with pytest.raises(ValueError, match="annual_income"):
            score_batch(bundle, batch, "logistic", "raw")


def test_scoring_preserves_order_and_ignores_target_and_extra_fields(trained):
    frame, _, bundle = trained
    batch = frame.iloc[[12, 2, 29]].drop(columns="outcome")
    _, plain = score_batch(bundle, batch, "logistic", "raw")
    expanded = batch.assign(outcome=["completely", "unrelated", "labels"], notes="private text")
    summary, extra = score_batch(bundle, expanded, "logistic", "raw")
    pd.testing.assert_frame_equal(plain, extra)
    assert extra.record_id.tolist() == ["1", "2", "3"]
    assert set(extra.columns) == {"record_id", "probability", "missing_count"}
    assert any("额外字段" in notice for notice in summary["warnings"])
    reversed_scores = score_batch(bundle, batch.iloc[::-1], "logistic", "raw")[1]
    assert reversed_scores.record_id.tolist() == ["1", "2", "3"]
    np.testing.assert_array_equal(reversed_scores.missing_count, plain.missing_count.iloc[::-1])
    # BLAS may round a permuted batch differently at the last floating-point bit.
    np.testing.assert_allclose(reversed_scores.probability, plain.probability.iloc[::-1],
                               rtol=1e-13, atol=1e-15)


def test_explicit_ids_are_required_unique_preserved_and_excluded_from_features():
    frame = demo_frame().iloc[:160].copy()
    frame.insert(0, "case_id", [f"case-{i:04}" for i in range(len(frame))])
    config = _config(frame, id_column="case_id", feature_columns=[name for name in frame if name not in ("case_id", "outcome")])
    _, bundle = train_analysis(frame, config)
    batch = frame.iloc[[7, 3, 19]].copy()
    _, scored = score_batch(bundle, batch, "logistic", "raw")
    assert scored.record_id.tolist() == batch.case_id.tolist()
    assert "case_id" not in bundle["feature_columns"]
    with pytest.raises(ValueError, match="case_id"):
        score_batch(bundle, batch.drop(columns="case_id"), "logistic", "raw")
    for bad in (["same", "same", "other"], ["one", "?", "three"]):
        with pytest.raises(ValueError, match="非空且唯一"):
            score_batch(bundle, batch.assign(case_id=bad), "logistic", "raw")
    frame.loc[0, "outcome"] = np.nan
    frame.loc[0, "case_id"] = frame.loc[1, "case_id"]
    with pytest.raises(ValueError, match="非空且唯一"):
        validate_training(frame, config)


def test_missing_target_rows_are_disclosed_and_never_partitioned(trained):
    frame = trained[0].copy()
    frame.loc[[0, 7, 13], "outcome"] = np.nan
    result, _ = train_analysis(frame, _config(frame))
    assert result["rows"] == len(frame) - 3
    assert result["provenance"]["omitted_row_indices"] == [0, 7, 13]
    rows = sum(result["provenance"]["split_indices"].values(), [])
    assert not set(rows) & {0, 7, 13}
    assert sum(result["split_sizes"].values()) == result["rows"]


def test_all_missing_batch_produces_finite_predictions_and_quality_warnings(trained):
    _, _, bundle = trained
    batch = pd.DataFrame(np.nan, index=range(3), columns=bundle["feature_columns"])
    summary, scored = score_batch(bundle, batch, "logistic", "isotonic")
    assert summary["missing_rate"] == 1
    assert scored.missing_count.tolist() == [len(bundle["feature_columns"])] * 3
    assert np.isfinite(scored.probability).all()
    assert any("全部特征缺失" in text for text in summary["warnings"])
    assert all(item["unseen_rate"] == 0 for item in summary["quality"])
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("mutation", [
    {"feature_columns": ["age", "outcome"]}, {"positive_value": "absent"},
    {"positive_label": ""}, {"models": []}, {"models": ["unknown"]}, {"seed": True},
    {"categorical_columns": ["outcome"]}, {"feature_columns": ["age", "age"]},
])
def test_configuration_errors_fail_before_training(trained, mutation):
    frame = trained[0]
    with pytest.raises(ValueError):
        validate_training(frame, _config(frame, **mutation))


def test_binary_target_and_independent_minority_group_checks():
    frame = demo_frame().iloc[:120].copy()
    with pytest.raises(ValueError, match="至少需要 100"):
        validate_training(frame.iloc[:99], _config(frame))
    with pytest.raises(ValueError, match="恰好包含两个"):
        validate_training(frame.assign(outcome="one"), _config(frame))
    frame["outcome"] = "0"
    frame.loc[:3, "outcome"] = "1"
    with pytest.raises(ValueError, match="5 组独立"):
        validate_training(frame, _config(frame))


def test_numeric_coded_categories_can_be_explicitly_overridden_and_cardinality_is_bounded():
    frame = pd.DataFrame({"code": [str(i % 3) for i in range(180)],
                          "amount": [str(i) for i in range(180)],
                          "outcome": [str(i % 2) for i in range(180)]})
    _, bundle = train_analysis(frame, _config(frame, categorical_columns=["code"]))
    assert bundle["category_maps"]["code"] == {"0": 0, "1": 1, "2": 2}
    with pytest.raises(ValueError, match="超过 100"):
        validate_training(frame, _config(frame, categorical_columns=["amount"]))


def test_bundle_can_be_serialized_without_changing_predictions(trained):
    frame, _, bundle = trained
    buffer = io.BytesIO()
    joblib.dump(bundle, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    before = score_batch(bundle, frame.iloc[:10], "logistic", "platt")[1]
    after = score_batch(restored, frame.iloc[:10], "logistic", "platt")[1]
    pd.testing.assert_frame_equal(before, after)


def test_all_model_adapters_train_calibrate_and_score_with_real_missingness():
    frame = demo_frame().iloc[:160].copy()
    events = []
    models = ["logistic", "random_forest", "lightgbm", "mlp", "mask_aware_mlp"]
    result, bundle = train_analysis(frame, _config(frame, models=models), lambda pct, msg: events.append((pct, msg)))
    assert len(result["metrics"]) == 15
    assert events[0][0] > 0 and events[-1][0] == 100
    assert [event[0] for event in events] == sorted(event[0] for event in events)
    batch = frame.iloc[:3].copy()
    batch.loc[0, "home_ownership"] = "previously_unseen"
    for name in models:
        for method in ("raw", "platt", "isotonic"):
            summary, scored = score_batch(bundle, batch, name, method)
            assert len(scored) == 3 and scored.probability.between(0, 1).all()
            json.dumps(summary, allow_nan=False)
    json.dumps(result, allow_nan=False)
