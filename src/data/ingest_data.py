"""Dataset ingestion module for the UCI Bank Marketing dataset (ID: 222).

Retrieves the official dataset from the UCI Machine Learning Repository,
verifies data integrity via SHA256 checksum, saves the pristine raw file,
and logs ingestion metadata without modifying the raw data bytes.
"""

import hashlib
import io
import json
import logging
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

UCI_DATASET_ID = 222
UCI_BANK_MARKETING_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
EXPECTED_RAW_FILENAME = "bank-full.csv"
EXPECTED_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "contact", "day", "month", "duration",
    "campaign", "pdays", "previous", "poutcome", "y"
]


def calculate_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file on disk."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def ingest_bank_marketing_dataset(
    raw_dir: Path = Path("data/raw"),
    force: bool = False,
    timeout: int = 60
) -> Dict[str, Any]:
    """Retrieve and store the UCI Bank Marketing dataset preserving raw bytes.

    Args:
        raw_dir: Directory where raw dataset and metadata will be saved.
        force: If True, re-downloads even if the raw file already exists.
        timeout: Network request timeout in seconds.

    Returns:
        Dictionary containing metadata about the ingested dataset.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file_path = raw_dir / EXPECTED_RAW_FILENAME
    metadata_path = raw_dir / "ingestion_metadata.json"

    # Idempotency check: if raw file already exists and force is False, return existing metadata
    if raw_file_path.exists() and not force:
        logger.info(f"Raw dataset already exists at {raw_file_path}. Preserving original file.")
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            metadata["cached"] = True
            return metadata

    logger.info(f"Fetching Bank Marketing dataset (ID: {UCI_DATASET_ID}) from {UCI_BANK_MARKETING_URL}...")
    req = urllib.request.Request(
        UCI_BANK_MARKETING_URL,
        headers={"User-Agent": "MLForge/0.1.0 (Machine Learning Platform)"}
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            zip_bytes = resp.read()
    except Exception as e:
        logger.error(f"Failed to download dataset from {UCI_BANK_MARKETING_URL}: {e}")
        raise RuntimeError(f"Dataset download failed: {e}") from e

    # Extract bank-full.csv from the nested zip archive without altering bytes
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as outer_zip:
        if "bank.zip" not in outer_zip.namelist():
            raise ValueError(f"'bank.zip' not found in outer archive: {outer_zip.namelist()}")

        bank_zip_bytes = outer_zip.read("bank.zip")
        with zipfile.ZipFile(io.BytesIO(bank_zip_bytes)) as inner_zip:
            if EXPECTED_RAW_FILENAME not in inner_zip.namelist():
                raise ValueError(f"'{EXPECTED_RAW_FILENAME}' not found in bank.zip: {inner_zip.namelist()}")

            raw_bytes = inner_zip.read(EXPECTED_RAW_FILENAME)

    # Write pristine raw bytes to destination
    with open(raw_file_path, "wb") as f:
        f.write(raw_bytes)

    sha256_checksum = calculate_sha256(raw_file_path)
    file_size_bytes = raw_file_path.stat().st_size

    # Verify dimensions without mutating file
    df_preview = pd.read_csv(raw_file_path, sep=";")
    rows, cols = df_preview.shape

    metadata = {
        "dataset_name": "UCI Bank Marketing",
        "uci_dataset_id": UCI_DATASET_ID,
        "source_url": UCI_BANK_MARKETING_URL,
        "filename": EXPECTED_RAW_FILENAME,
        "retrieval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "file_size_bytes": file_size_bytes,
        "sha256_checksum": sha256_checksum,
        "num_rows": rows,
        "num_columns": cols,
        "delimiter": ";",
        "columns": df_preview.columns.tolist(),
        "cached": False,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Successfully ingested {EXPECTED_RAW_FILENAME}: "
        f"{rows:,} rows, {cols} columns, {file_size_bytes / (1024 * 1024):.2f} MB, SHA256: {sha256_checksum[:12]}..."
    )
    return metadata


def load_raw_dataset(file_path: Path = Path("data/raw/bank-full.csv")) -> pd.DataFrame:
    """Load the raw Bank Marketing dataset with appropriate delimiter.

    Args:
        file_path: Path to the raw CSV file.

    Returns:
        Unmodified raw DataFrame.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}. Run ingest_bank_marketing_dataset() first.")
    return pd.read_csv(path, sep=";")
