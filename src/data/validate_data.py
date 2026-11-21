"""Data validation module for the Bank Marketing dataset.

Performs schema verification, missing-value and placeholder audits,
duplicate checks, domain anomaly checks, target column validation,
and target-leakage flagging.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

EXPECTED_SCHEMA: Dict[str, str] = {
    "age": "int64",
    "job": "object",
    "marital": "object",
    "education": "object",
    "default": "object",
    "balance": "int64",
    "housing": "object",
    "loan": "object",
    "contact": "object",
    "day": "int64",
    "month": "object",
    "duration": "int64",
    "campaign": "int64",
    "pdays": "int64",
    "previous": "int64",
    "poutcome": "object",
    "y": "object",
}

NUMERICAL_COLUMNS: List[str] = [
    "age", "balance", "day", "duration", "campaign", "pdays", "previous"
]

CATEGORICAL_COLUMNS: List[str] = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome"
]

TARGET_COLUMN: str = "y"
VALID_MONTHS: set = {
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec"
}


@dataclass
class ValidationResult:
    """Encapsulates the complete dataset validation diagnostics."""
    is_valid: bool
    total_rows: int
    total_cols: int
    missing_columns: List[str] = field(default_factory=list)
    unexpected_columns: List[str] = field(default_factory=list)
    type_mismatches: Dict[str, Dict[str, str]] = field(default_factory=dict)
    standard_missing_values: Dict[str, int] = field(default_factory=dict)
    categorical_placeholders: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sentinel_values: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    duplicate_rows_count: int = 0
    duplicate_rows_pct: float = 0.0
    domain_anomalies: List[str] = field(default_factory=list)
    target_summary: Dict[str, Any] = field(default_factory=dict)
    leakage_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to JSON-serializable dictionary."""
        return asdict(self)


def validate_bank_marketing_dataset(df: pd.DataFrame) -> ValidationResult:
    """Validate a Bank Marketing DataFrame against project rules and schema constraints.

    Args:
        df: Input DataFrame to validate.

    Returns:
        ValidationResult object containing comprehensive diagnostics.
    """
    total_rows, total_cols = df.shape
    missing_cols = [c for c in EXPECTED_SCHEMA if c not in df.columns]
    unexpected_cols = [c for c in df.columns if c not in EXPECTED_SCHEMA]
    type_mismatches: Dict[str, Dict[str, str]] = {}
    domain_anomalies: List[str] = []
    leakage_warnings: List[str] = []

    # 1. Column and Type Verification
    for col in df.columns:
        if col in EXPECTED_SCHEMA:
            actual_dtype = str(df[col].dtype)
            expected_dtype = EXPECTED_SCHEMA[col]
            # Allow compatible integer types (e.g. int32 or int64)
            if "int" in expected_dtype and "int" not in actual_dtype:
                type_mismatches[col] = {"expected": expected_dtype, "actual": actual_dtype}
            elif expected_dtype == "object" and actual_dtype != "object":
                type_mismatches[col] = {"expected": expected_dtype, "actual": actual_dtype}

    # 2. Standard Missing Values Check (null / NaN)
    standard_missing = df.isna().sum().to_dict()

    # 3. Categorical Placeholder Audit ('unknown' values)
    # Note: 'unknown' is treated as a valid categorical level requiring analysis, not automatically null
    categorical_placeholders: Dict[str, Dict[str, Any]] = {}
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            unknown_mask = df[col].astype(str).str.lower() == "unknown"
            count = int(unknown_mask.sum())
            if count > 0:
                pct = round((count / total_rows) * 100, 2)
                categorical_placeholders[col] = {
                    "count": count,
                    "percentage": pct,
                    "note": f"Categorical placeholder 'unknown' present in {pct}% of records."
                }

    # 4. Sentinel Value Audit (pdays = -1)
    # Note: pdays = -1 is a meaningful domain sentinel indicating the client was not previously contacted
    sentinel_values: Dict[str, Dict[str, Any]] = {}
    if "pdays" in df.columns:
        pdays_sentinel_mask = df["pdays"] == -1
        pdays_sentinel_count = int(pdays_sentinel_mask.sum())
        pdays_sentinel_pct = round((pdays_sentinel_count / total_rows) * 100, 2)
        sentinel_values["pdays_-1"] = {
            "count": pdays_sentinel_count,
            "percentage": pdays_sentinel_pct,
            "meaning": "Client was not previously contacted during any prior campaign."
        }

    # 5. Duplicate Row Audit
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_count / total_rows) * 100, 4) if total_rows > 0 else 0.0

    # 6. Domain Range and Constraint Validation
    if "age" in df.columns:
        invalid_ages = df[~df["age"].between(18, 120)]
        if len(invalid_ages) > 0:
            domain_anomalies.append(f"Found {len(invalid_ages)} records with age outside [18, 120].")

    if "day" in df.columns:
        invalid_days = df[~df["day"].between(1, 31)]
        if len(invalid_days) > 0:
            domain_anomalies.append(f"Found {len(invalid_days)} records with day outside [1, 31].")

    if "month" in df.columns:
        invalid_months = df[~df["month"].str.lower().isin(VALID_MONTHS)]
        if len(invalid_months) > 0:
            domain_anomalies.append(f"Found {len(invalid_months)} records with invalid month names.")

    if "duration" in df.columns:
        negative_duration = (df["duration"] < 0).sum()
        if negative_duration > 0:
            domain_anomalies.append(f"Found {negative_duration} records with negative duration.")

    # 7. Target Column Validation
    target_summary: Dict[str, Any] = {}
    if TARGET_COLUMN in df.columns:
        target_series = df[TARGET_COLUMN].astype(str).str.lower()
        unique_targets = set(target_series.unique())
        expected_targets = {"yes", "no"}
        if not unique_targets.issubset(expected_targets):
            domain_anomalies.append(f"Target '{TARGET_COLUMN}' contains unexpected values: {unique_targets - expected_targets}")

        val_counts = target_series.value_counts().to_dict()
        no_count = val_counts.get("no", 0)
        yes_count = val_counts.get("yes", 0)
        imbalance_ratio = round(no_count / yes_count, 2) if yes_count > 0 else float("inf")

        target_summary = {
            "target_column": TARGET_COLUMN,
            "classes": list(unique_targets),
            "counts": val_counts,
            "percentages": {k: round((v / total_rows) * 100, 2) for k, v in val_counts.items()},
            "imbalance_ratio": f"{imbalance_ratio}:1 (no:yes)",
            "minority_class_percentage": round((yes_count / total_rows) * 100, 2),
        }
    else:
        domain_anomalies.append(f"Critical: Target column '{TARGET_COLUMN}' is missing from DataFrame.")

    # 8. Data Leakage Warning for Pre-call Prediction
    if "duration" in df.columns:
        leakage_warnings.append(
            "LEAKAGE WARNING: Feature 'duration' represents call duration in seconds. "
            "This attribute is unknown before a call is performed (the prospective prediction scenario) "
            "and strongly correlates with the target 'y'. As per UCI guidelines and best ML practices, "
            "'duration' must be strictly excluded from prospective pre-call feature sets."
        )

    # Determine overall validity
    is_valid = (
        len(missing_cols) == 0
        and len(type_mismatches) == 0
        and TARGET_COLUMN in df.columns
        and len(domain_anomalies) == 0
    )

    return ValidationResult(
        is_valid=is_valid,
        total_rows=total_rows,
        total_cols=total_cols,
        missing_columns=missing_cols,
        unexpected_columns=unexpected_cols,
        type_mismatches=type_mismatches,
        standard_missing_values=standard_missing,
        categorical_placeholders=categorical_placeholders,
        sentinel_values=sentinel_values,
        duplicate_rows_count=duplicate_count,
        duplicate_rows_pct=duplicate_pct,
        domain_anomalies=domain_anomalies,
        target_summary=target_summary,
        leakage_warnings=leakage_warnings,
    )
