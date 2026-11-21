"""Shared pytest fixtures for hermetic test execution across MLForge test suites.

Provides deterministic synthetic data, mock archives, and in-memory test services
to ensure zero access to sealed holdout test datasets or production model weights during CI.
"""

import io
import json
from pathlib import Path
from typing import Any, Dict
import zipfile

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import torch

from src.api.service import PredictionService
from src.features.preprocessing import build_preprocessor_pipeline, fit_preprocessor
from src.models.pytorch_model import MLForgeClassifier


# All 12 standard Bank Marketing jobs
ALL_JOBS = [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown"
]
# All 12 months
ALL_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
# All education categories
ALL_EDUCATION = ["primary", "secondary", "tertiary", "unknown"]
# All contact categories
ALL_CONTACT = ["cellular", "telephone", "unknown"]
# All poutcome categories
ALL_POUTCOME = ["failure", "other", "success", "unknown"]
# Marital categories
ALL_MARITAL = ["divorced", "married", "single"]


def generate_synthetic_bank_marketing_df(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic synthetic Bank Marketing DataFrame satisfying all schema rules.

    Guarantees:
    - Exactly 17 columns matching EXPECTED_SCHEMA.
    - All domain constraints satisfied (age in [18, 120], day in [1, 31], valid months).
    - All 12 jobs, 12 months, 4 educations, 3 contacts, and 4 poutcomes represented for 51-feature encoding.
    - Exactly 12% positive class ("yes") for stratification testing.
    - Deterministic and non-duplicated rows.
    """
    rng = np.random.RandomState(seed)

    records = []
    # Base positive rate: exactly 24 / 200 = 12%
    n_positive = int(round(n_rows * 0.12))

    for i in range(n_rows):
        job = ALL_JOBS[i % len(ALL_JOBS)]
        month = ALL_MONTHS[i % len(ALL_MONTHS)]
        edu = ALL_EDUCATION[i % len(ALL_EDUCATION)]
        contact = ALL_CONTACT[i % len(ALL_CONTACT)]
        poutcome = ALL_POUTCOME[i % len(ALL_POUTCOME)]
        marital = ALL_MARITAL[i % len(ALL_MARITAL)]
        is_contacted = (poutcome != "unknown")

        record = {
            "age": int(20 + (i * 3) % 65),
            "job": job,
            "marital": marital,
            "education": edu,
            "default": "yes" if i == 50 else "no",
            "balance": int(-200 + (i * 123) % 8000),
            "housing": "yes" if (i % 2 == 0) else "no",
            "loan": "yes" if (i % 7 == 0) else "no",
            "contact": contact,
            "day": int(1 + (i * 5) % 28),
            "month": month,
            "duration": int(60 + (i * 17) % 500),
            "campaign": int(1 + (i % 5)),
            "pdays": int(10 + (i * 11) % 250) if is_contacted else -1,
            "previous": int(1 + (i % 4)) if is_contacted else 0,
            "poutcome": poutcome,
            "y": "yes" if i < n_positive else "no",
        }
        records.append(record)

    df = pd.DataFrame(records)
    # Shuffle deterministically so classes are randomly dispersed
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def create_synthetic_nested_zip(df: pd.DataFrame) -> bytes:
    """Create a nested ZIP archive in memory matching the UCI Bank Marketing format.

    Format: outer.zip -> bank.zip -> bank-full.csv (semicolon-separated).
    """
    # 1. Inner bank-full.csv bytes
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, sep=";", index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    # 2. Inner bank.zip
    inner_zip_buffer = io.BytesIO()
    with zipfile.ZipFile(inner_zip_buffer, "w", zipfile.ZIP_DEFLATED) as inner_zip:
        inner_zip.writestr("bank-full.csv", csv_bytes)
    inner_zip_bytes = inner_zip_buffer.getvalue()

    # 3. Outer zip containing bank.zip
    outer_zip_buffer = io.BytesIO()
    with zipfile.ZipFile(outer_zip_buffer, "w", zipfile.ZIP_DEFLATED) as outer_zip:
        outer_zip.writestr("bank.zip", inner_zip_bytes)
    return outer_zip_buffer.getvalue()


@pytest.fixture(scope="module")
def synthetic_bank_df() -> pd.DataFrame:
    """Fixture providing a deterministic 200-row synthetic Bank Marketing DataFrame."""
    return generate_synthetic_bank_marketing_df(n_rows=200, seed=42)


@pytest.fixture(scope="module")
def dummy_prediction_service(synthetic_bank_df) -> PredictionService:
    """Fixture providing an in-memory PredictionService with lightweight models and 51-feature preprocessor.

    Permits complete API contract, routing, feature engineering, and inference latency testing
    without loading or needing production model checkpoints on disk.
    """
    # Extract features matching the 16 preprocessor columns
    df = synthetic_bank_df.copy()
    pdays_val = df["pdays"]
    df["is_previously_contacted"] = (pdays_val != -1).astype(int)
    df["pdays_recency"] = np.where(pdays_val == -1, 0.0, pdays_val.astype(float))
    X_raw = df.drop(columns=["duration", "pdays", "y"])

    num_cols = ["age", "balance", "day", "campaign", "previous", "is_previously_contacted", "pdays_recency"]
    cat_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]

    # Fit preprocessor to get exact 51 features
    prep = build_preprocessor_pipeline(num_cols, cat_cols)
    prep = fit_preprocessor(prep, X_raw)

    transformed_matrix = prep.transform(X_raw)
    n_features = transformed_matrix.shape[1]
    assert n_features == 51, f"Expected 51 transformed features, got {n_features}"

    feature_names = [str(col) for col in prep.get_feature_names_out()]
    df_transformed = pd.DataFrame(transformed_matrix, columns=feature_names)

    # Fit lightweight dummy classifiers
    y_dummy = (synthetic_bank_df["y"] == "yes").astype(int).values
    lr = LogisticRegression(max_iter=50, random_state=42)
    lr.fit(df_transformed, y_dummy)

    rf = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
    rf.fit(df_transformed, y_dummy)

    pt_model = MLForgeClassifier(input_dim=51, hidden_dim1=16, hidden_dim2=8)
    pt_model.eval()

    thresholds = {
        "logistic_regression": 0.660,
        "random_forest": 0.635,
        "pytorch": 0.685,
    }

    feature_names = [str(col) for col in prep.get_feature_names_out()]

    metadata = {
        "project": "MLForge",
        "dataset": "UCI Bank Marketing",
        "target_column": "y",
        "target_mapping": {"no": 0, "yes": 1},
        "excluded_features": ["duration"],
        "raw_input_feature_count": 15,
        "preprocessor_input_feature_count": 16,
        "transformed_feature_count": 51,
        "feature_names": feature_names,
    }

    return PredictionService(
        preprocessor=prep,
        lr_model=lr,
        rf_model=rf,
        pytorch_model=pt_model,
        thresholds=thresholds,
        metadata=metadata,
    )
