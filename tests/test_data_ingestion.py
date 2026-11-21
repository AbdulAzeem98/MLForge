"""Unit and integration tests for data ingestion, validation, and EDA modules."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.ingest_data import (
    calculate_sha256,
    ingest_bank_marketing_dataset,
    load_raw_dataset,
    EXPECTED_COLUMNS,
)
from src.data.validate_data import (
    validate_bank_marketing_dataset,
    ValidationResult,
    EXPECTED_SCHEMA,
)


@pytest.fixture
def valid_synthetic_df() -> pd.DataFrame:
    """Fixture providing a valid synthetic Bank Marketing DataFrame."""
    return pd.DataFrame({
        "age": [30, 45, 52],
        "job": ["management", "technician", "unknown"],
        "marital": ["single", "married", "divorced"],
        "education": ["tertiary", "secondary", "unknown"],
        "default": ["no", "no", "no"],
        "balance": [1000, 250, -50],
        "housing": ["yes", "no", "yes"],
        "loan": ["no", "yes", "no"],
        "contact": ["cellular", "telephone", "unknown"],
        "day": [15, 5, 28],
        "month": ["may", "jul", "nov"],
        "duration": [120, 300, 45],
        "campaign": [1, 2, 1],
        "pdays": [-1, 100, -1],
        "previous": [0, 1, 0],
        "poutcome": ["unknown", "success", "unknown"],
        "y": ["no", "yes", "no"],
    })


# =====================================================================
# Unit Tests (Synthetic Data)
# =====================================================================

def test_validation_passes_on_valid_df(valid_synthetic_df):
    """Test that valid synthetic DataFrame passes validation completely."""
    result = validate_bank_marketing_dataset(valid_synthetic_df)
    assert result.is_valid is True
    assert result.total_rows == 3
    assert result.total_cols == 17
    assert len(result.missing_columns) == 0
    assert len(result.domain_anomalies) == 0


def test_validation_fails_missing_target_column(valid_synthetic_df):
    """Test that missing target column 'y' is detected and marks dataset invalid."""
    df_missing_target = valid_synthetic_df.drop(columns=["y"])
    result = validate_bank_marketing_dataset(df_missing_target)
    assert result.is_valid is False
    assert "y" in result.missing_columns


def test_validation_fails_missing_required_column(valid_synthetic_df):
    """Test that missing required feature column is flagged."""
    df_missing_age = valid_synthetic_df.drop(columns=["age"])
    result = validate_bank_marketing_dataset(df_missing_age)
    assert result.is_valid is False
    assert "age" in result.missing_columns


def test_validation_fails_on_invalid_target_values(valid_synthetic_df):
    """Test that invalid values in target column (e.g. not in {'yes', 'no'}) are caught."""
    df_invalid_target = valid_synthetic_df.copy()
    df_invalid_target.loc[0, "y"] = "maybe"
    result = validate_bank_marketing_dataset(df_invalid_target)
    assert result.is_valid is False
    assert any("unexpected values" in a for a in result.domain_anomalies)


def test_unknown_treated_as_categorical_placeholder_not_null(valid_synthetic_df):
    """Test that 'unknown' is tracked as a categorical level without generating nulls."""
    result = validate_bank_marketing_dataset(valid_synthetic_df)
    # Standard missing values should be 0
    assert sum(result.standard_missing_values.values()) == 0
    # Categorical placeholders should count 'unknown' in job, education, contact, poutcome
    assert "job" in result.categorical_placeholders
    assert result.categorical_placeholders["job"]["count"] == 1
    assert "poutcome" in result.categorical_placeholders
    assert result.categorical_placeholders["poutcome"]["count"] == 2


def test_pdays_sentinel_analysis(valid_synthetic_df):
    """Test that pdays = -1 is recognized as the 'no prior contact' sentinel."""
    result = validate_bank_marketing_dataset(valid_synthetic_df)
    assert "pdays_-1" in result.sentinel_values
    assert result.sentinel_values["pdays_-1"]["count"] == 2
    assert result.sentinel_values["pdays_-1"]["percentage"] == round((2 / 3) * 100, 2)


def test_domain_range_anomalies(valid_synthetic_df):
    """Test detection of domain constraint violations (age, day, month)."""
    df_anomalous = valid_synthetic_df.copy()
    df_anomalous.loc[0, "age"] = 150  # unrealistic human age
    df_anomalous.loc[1, "day"] = 35   # day outside 1-31
    df_anomalous.loc[2, "month"] = "invalid_month"

    result = validate_bank_marketing_dataset(df_anomalous)
    assert result.is_valid is False
    assert any("age outside [18, 120]" in a for a in result.domain_anomalies)
    assert any("day outside [1, 31]" in a for a in result.domain_anomalies)
    assert any("invalid month names" in a for a in result.domain_anomalies)


def test_duplicate_row_detection(valid_synthetic_df):
    """Test that duplicate rows are detected and counted."""
    df_dup = pd.concat([valid_synthetic_df, valid_synthetic_df.iloc[[0]]], ignore_index=True)
    result = validate_bank_marketing_dataset(df_dup)
    assert result.duplicate_rows_count == 1


def test_duration_leakage_warning_emitted(valid_synthetic_df):
    """Test that the prospective pre-call data leakage warning is logged for 'duration'."""
    result = validate_bank_marketing_dataset(valid_synthetic_df)
    assert len(result.leakage_warnings) > 0
    assert any("duration" in w.lower() for w in result.leakage_warnings)


from tests.conftest import create_synthetic_nested_zip


class MockHttpResponse:
    """Mock HTTP response wrapping in-memory bytes for urllib.request.urlopen."""

    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# =====================================================================
# Integration Tests (Hermetic Ingestion Pipeline)
# =====================================================================

@pytest.mark.integration
def test_actual_dataset_ingestion_and_integrity(tmp_path, monkeypatch, synthetic_bank_df):
    """Test full ingestion workflow using a hermetic nested ZIP fixture in tmp_path."""
    raw_dir = tmp_path / "raw"
    raw_file = raw_dir / "bank-full.csv"
    meta_file = raw_dir / "ingestion_metadata.json"

    # Mock urllib.request.urlopen to avoid external network calls during CI
    zip_bytes = create_synthetic_nested_zip(synthetic_bank_df)
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=60: MockHttpResponse(zip_bytes))

    # Ingest using mock zip bytes
    metadata = ingest_bank_marketing_dataset(raw_dir=raw_dir, force=True)
    assert raw_file.exists(), "Raw dataset file must exist"
    assert meta_file.exists(), "Metadata file must exist"

    # Verify metadata fields
    n_expected_rows = len(synthetic_bank_df)
    assert metadata["uci_dataset_id"] == 222
    assert metadata["num_rows"] == n_expected_rows
    assert metadata["num_columns"] == 17
    assert metadata["filename"] == "bank-full.csv"

    # Verify SHA256 matches actual file on disk
    computed_hash = calculate_sha256(raw_file)
    assert metadata["sha256_checksum"] == computed_hash

    # Verify non-destructive idempotency: re-running does not alter file
    original_mtime = raw_file.stat().st_mtime
    re_metadata = ingest_bank_marketing_dataset(raw_dir=raw_dir, force=False)
    assert re_metadata["cached"] is True
    assert raw_file.stat().st_mtime == original_mtime
    assert calculate_sha256(raw_file) == computed_hash

    # Load and validate actual DataFrame
    df = load_raw_dataset(raw_file)
    assert df.shape == (n_expected_rows, 17)
    assert df.columns.tolist() == EXPECTED_COLUMNS

    validation = validate_bank_marketing_dataset(df)
    assert validation.is_valid is True
    assert sum(validation.standard_missing_values.values()) == 0
    assert validation.duplicate_rows_count == 0

    # Check target distribution
    target_summary = validation.target_summary
    assert target_summary["counts"]["yes"] == int(round(n_expected_rows * 0.12))
    assert round(target_summary["percentages"]["yes"], 1) == 12.0
