"""Download, verify and prepare the small official German Credit dataset."""

import hashlib
import io
import json
from datetime import datetime, timezone
from http.client import IncompleteRead
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

from src.data.german_credit import EXPECTED_ROWS, RAW_SHA256, SOURCE_PAGE, SOURCE_URL, convert_raw_file, feature_schema
from src.data.loader import load_dataset
from src.data.preprocess import split_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def download_german_credit(raw_dir: str | Path) -> dict[str, Any]:
    """Reuse verified local data, or download official bytes with bounded retries.

    Only extract the explicitly named raw member; do not extract arbitrary paths.
    A changed existing raw file fails validation instead of being overwritten.
    """
    directory = Path(raw_dir)
    raw_path = directory / "german.data"
    metadata_path = directory / "download_metadata.json"
    if raw_path.exists():
        if file_sha256(raw_path) != RAW_SHA256:
            raise ValueError("Existing german.data does not match the pinned official SHA256.")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["raw_sha256"] != RAW_SHA256 or metadata["source_url"] != SOURCE_URL:
                raise ValueError("Raw download metadata does not match the official source.")
            return metadata

    archive_bytes = b""
    last_error: Exception | None = None
    for _ in range(3):
        try:
            request = Request(SOURCE_URL, headers={"User-Agent": "missing-aware-credit-risk/1.0"})
            with urlopen(request, timeout=30) as response:
                archive_bytes = response.read(5 * 1024 * 1024 + 1)
            if len(archive_bytes) > 5 * 1024 * 1024:
                raise ValueError("Unexpectedly large German Credit archive.")
            with ZipFile(io.BytesIO(archive_bytes)) as archive:
                if archive.getinfo("german.data").file_size > 1024 * 1024:
                    raise ValueError("Unexpectedly large german.data member.")
                raw_bytes = archive.read("german.data")
            if hashlib.sha256(raw_bytes).hexdigest() != RAW_SHA256:
                raise ValueError("Downloaded german.data fails the pinned official SHA256 check.")
            break
        except (URLError, IncompleteRead, TimeoutError, BadZipFile) as error:
            last_error = error
    else:
        raise RuntimeError(f"Unable to download {SOURCE_URL}; check network and retry.") from last_error

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "uci-german-credit.zip").write_bytes(archive_bytes)
    raw_path.write_bytes(raw_bytes)
    metadata = {
        "source_page": SOURCE_PAGE,
        "source_url": SOURCE_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "raw_sha256": RAW_SHA256,
        "raw_bytes": len(raw_bytes),
        "citation": "Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77",
        "license": "CC BY 4.0",
    }
    _write_json(metadata_path, metadata)
    return metadata


def prepare_german_credit(
    *,
    raw_dir: str | Path = PROJECT_ROOT / "data/raw/german_credit",
    output_dir: str | Path = PROJECT_ROOT / "data/processed/german_credit",
    seed: int = 42,
) -> dict[str, Any]:
    """Prepare fixed transport codes and save train/tune/calibration/test indices.

    Conversion never fits a learned encoder, imputer or scaler on the full data.
    The raw data and generated artifacts remain ignored by Git.
    """
    source = download_german_credit(raw_dir)
    frame, quality = convert_raw_file(Path(raw_dir) / "german.data")
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} German Credit rows.")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "german_credit.csv"
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    _write_json(directory / "schema.json", feature_schema())
    _write_json(directory / "quality_report.json", quality)
    _write_json(directory / "metadata.json", {
        "schema_version": 1, **source,
        "processed_sha256": file_sha256(csv_path),
        "quality": quality,
    })
    data = load_dataset("german_credit", path=csv_path)
    manifest_path = directory / f"splits_seed{seed}.json"
    splits = split_dataset(data, seed=seed, calibration_fraction=0.5, manifest_path=manifest_path)
    return {
        "csv_path": str(csv_path),
        "split_manifest": str(manifest_path),
        "split_sizes": {name: len(part["y"]) for name, part in splits.items()},
        "quality": quality,
        "raw_sha256": source["raw_sha256"],
    }
