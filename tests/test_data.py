"""Offline checks for credit conversion, source integrity and shared data splits."""

import hashlib
import io
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request
from zipfile import ZipFile

import numpy as np
import pytest
import yaml

from src.data import german_credit, prepare
from src.data.german_credit import FEATURES, TARGET_COLUMN, categorical_feature_indices, convert_raw_file
from src.data.loader import dataset_fingerprint, load_dataset
from src.data.preprocess import make_split_indices, split_dataset
from src.models.logistic import LogisticRegressionModel


@pytest.fixture
def raw_file(tmp_path: Path) -> Path:
    """Two synthetic rows in the official format, including an explicit unknown."""
    path = tmp_path / "german.data"
    path.write_text(
        "A11 12 A30 A40 1000 A65 A71 1 A91 A101 1 A121 25 A141 A151 1 A171 1 A191 A201 1\n"
        "A14 24 A32 A43 2000 A61 A75 2 A93 A103 2 A122 35 A143 A152 2 A173 2 A192 A202 2\n",
        encoding="utf-8",
    )
    return path


def test_raw_conversion_maps_labels_and_preserves_original_fields(raw_file: Path, tmp_path: Path) -> None:
    frame, quality = convert_raw_file(raw_file)
    assert list(frame.columns) == [feature.name for feature in FEATURES] + [TARGET_COLUMN]
    np.testing.assert_array_equal(frame[TARGET_COLUMN], [0, 1])
    assert frame.loc[0, "savings_status"] == 4  # A65 remains a category, not NaN.
    assert quality["missing_feature_cells"] == 0
    assert quality["duplicate_rows"] == 0
    assert quality["class_counts"] == {"0": 1, "1": 1}
    assert quality["numeric_features"] == 7
    assert quality["categorical_features"] == 13
    path = tmp_path / "transport.csv"
    frame.to_csv(path, index=False)
    data = load_dataset("csv", path=path, target_column=TARGET_COLUMN)
    assert data["x"].shape == data["mask"].shape == (2, 20)
    assert np.all(data["mask"] == 1)


@pytest.mark.parametrize("column,value,message", [(0, "A99", "categorical"), (1, "-1", "numeric"), (20, "0", "targets")])
def test_raw_conversion_rejects_invalid_values(raw_file: Path, column: int, value: str, message: str) -> None:
    rows = raw_file.read_text(encoding="utf-8").splitlines()
    first = rows[0].split()
    first[column] = value
    rows[0] = " ".join(first)
    raw_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        convert_raw_file(raw_file)


def test_duplicate_rows_are_reported_without_silent_removal(raw_file: Path) -> None:
    first = raw_file.read_text(encoding="utf-8").splitlines()[0]
    with raw_file.open("a", encoding="utf-8") as handle:
        handle.write(first + "\n")
    frame, quality = convert_raw_file(raw_file)
    assert len(frame) == 3
    assert quality["duplicate_rows"] == quality["duplicate_feature_rows"] == 1


def _archive_bytes(raw: bytes) -> bytes:
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("german.data", raw)
    return stream.getvalue()


def test_download_verifies_bytes_retries_and_reuses_cache(raw_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = raw_file.read_bytes()
    monkeypatch.setattr(prepare, "RAW_SHA256", hashlib.sha256(raw).hexdigest())
    calls = []

    def fake_urlopen(request: Request, timeout: int) -> io.BytesIO:
        calls.append(request.full_url)
        if len(calls) == 1:
            raise URLError("temporary connection failure")
        return io.BytesIO(_archive_bytes(raw))

    monkeypatch.setattr(prepare, "urlopen", fake_urlopen)
    directory = tmp_path / "download"
    metadata = prepare.download_german_credit(directory)
    assert (directory / "german.data").read_bytes() == raw
    assert metadata["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert metadata["source_url"] == german_credit.SOURCE_URL
    assert prepare.download_german_credit(directory) == metadata
    assert len(calls) == 2  # no network when a verified cache exists
    (directory / "german.data").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Existing"):
        prepare.download_german_credit(directory)
    assert len(calls) == 2


def test_wrong_download_checksum_is_rejected_before_writing(raw_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prepare, "urlopen", lambda request, timeout: io.BytesIO(_archive_bytes(raw_file.read_bytes())))
    directory = tmp_path / "download"
    with pytest.raises(ValueError, match="SHA256"):
        prepare.download_german_credit(directory)
    assert not directory.exists()


@pytest.fixture
def prepared_csv(raw_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Synthetic processed fixture; source-download verification is tested separately."""
    frame, _ = convert_raw_file(raw_file)
    path = tmp_path / "german_credit.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setattr(german_credit, "EXPECTED_ROWS", 2)
    metadata = {"schema_version": 1, "raw_sha256": german_credit.RAW_SHA256,
                "processed_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_name("metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_german_loader_checks_metadata_and_returns_contract(prepared_csv: Path) -> None:
    data = load_dataset("german_credit", path=prepared_csv)
    assert data["x"].dtype == np.float64
    assert data["mask"].dtype == np.uint8
    assert data["y"].dtype == np.int64
    np.testing.assert_array_equal(data["y"], [0, 1])
    assert np.all(data["mask"] == 1)
    prepared_csv.write_text(prepared_csv.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset("german_credit", path=prepared_csv)


def test_german_loader_rejects_reordered_feature_schema(prepared_csv: Path) -> None:
    rows = prepared_csv.read_text(encoding="utf-8").splitlines()
    columns = rows[0].split(",")
    columns[0], columns[1] = columns[1], columns[0]
    rows[0] = ",".join(columns)
    prepared_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    metadata_path = prepared_csv.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["processed_sha256"] = hashlib.sha256(prepared_csv.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_dataset("german_credit", path=prepared_csv)


def test_manifest_roundtrip_and_validation_children_are_disjoint(tmp_path: Path) -> None:
    data = load_dataset(seed=42)
    path = tmp_path / "indices.json"
    first = split_dataset(data, seed=42, calibration_fraction=0.5, manifest_path=path)
    second = split_dataset(data, seed=42, calibration_fraction=0.5, manifest_path=path)
    assert {name: len(part["y"]) for name, part in first.items()} == {
        "train": 600, "validation": 200, "test": 200,
        "validation_tune": 100, "validation_calibration": 100,
    }
    for name in first:
        for key in ("x", "mask", "y"):
            np.testing.assert_array_equal(first[name][key], second[name][key])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["dataset_sha256"] == dataset_fingerprint(data)
    indices = manifest["indices"]
    partition = [indices[name] for name in ("train", "validation_tune", "validation_calibration", "test")]
    flattened = np.concatenate(partition)
    np.testing.assert_array_equal(np.sort(flattened), np.arange(1000))
    np.testing.assert_array_equal(np.sort(indices["validation"]), np.sort(np.concatenate(partition[1:3])))


@pytest.mark.parametrize("change", ["seed", "data", "indices"])
def test_manifest_rejects_mismatches_and_corrupted_indices(tmp_path: Path, change: str) -> None:
    data = load_dataset(seed=42)
    path = tmp_path / "indices.json"
    split_dataset(data, seed=42, calibration_fraction=0.5, manifest_path=path)
    seed = 42
    if change == "seed":
        seed = 43
    elif change == "data":
        data["x"][0, 0] += 1
    else:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["indices"]["train"][0] = manifest["indices"]["train"][1]
        path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        split_dataset(data, seed=seed, calibration_fraction=0.5, manifest_path=path)


@pytest.mark.parametrize("fraction", [0.0, 1.0, float("nan")])
def test_reject_invalid_calibration_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="calibration_fraction"):
        make_split_indices(load_dataset()["y"], calibration_fraction=fraction)


def test_categorical_encoder_learns_only_training_categories() -> None:
    train = np.array([[10.0, 0], [20.0, 1], [30.0, 0], [40.0, 1]])
    model = LogisticRegressionModel(categorical_features=[1])
    model.fit(train, np.array([0, 1, 0, 1]))
    transformer = model.pipeline.named_steps["preprocess"]
    encoder = transformer.named_transformers_["categorical"].named_steps["onehot"]
    numeric = transformer.named_transformers_["remainder"]
    np.testing.assert_array_equal(encoder.categories_[0], [0, 1])
    assert numeric.named_steps["imputer"].statistics_[0] == 25
    test = np.array([[1000000.0, 2], [np.nan, np.nan]])
    original_test = test.copy()
    probabilities = model.predict_proba(test)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_array_equal(test, original_test)
    np.testing.assert_array_equal(encoder.categories_[0], [0, 1])
    assert numeric.named_steps["imputer"].statistics_[0] == 25


def test_german_config_matches_canonical_categorical_fields() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/german_credit.yaml").read_text(encoding="utf-8"))
    assert config["model"]["categorical_features"] == categorical_feature_indices()
