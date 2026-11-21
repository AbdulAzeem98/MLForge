"""Automated test suite for data preprocessing and feature engineering."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.data.split_data import stratified_split_dataset, SplitResult
from src.features.build_features import (
    extract_features_and_target,
    execute_feature_pipeline,
    LEAKAGE_FEATURE_TO_EXCLUDE,
)
from src.features.preprocessing import (
    build_preprocessor_pipeline,
    fit_preprocessor,
    transform_features,
    save_fitted_preprocessor,
    load_fitted_preprocessor,
)


@pytest.fixture
def synthetic_raw_df() -> pd.DataFrame:
    """Fixture providing a synthetic raw Bank Marketing DataFrame with known values."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "age": np.random.randint(20, 65, size=n),
        "job": np.random.choice(["admin.", "technician", "management", "unknown"], size=n),
        "marital": np.random.choice(["married", "single", "divorced"], size=n),
        "education": np.random.choice(["primary", "secondary", "tertiary", "unknown"], size=n),
        "default": np.random.choice(["no", "yes"], size=n, p=[0.9, 0.1]),
        "balance": np.random.randint(-500, 5000, size=n),
        "housing": np.random.choice(["yes", "no"], size=n),
        "loan": np.random.choice(["yes", "no"], size=n),
        "contact": np.random.choice(["cellular", "telephone", "unknown"], size=n),
        "day": np.random.randint(1, 29, size=n),
        "month": np.random.choice(["may", "jun", "jul", "aug"], size=n),
        "duration": np.random.randint(10, 1000, size=n),  # Leakage feature
        "campaign": np.random.randint(1, 5, size=n),
        "pdays": np.random.choice([-1, 10, 50, 150], size=n, p=[0.7, 0.1, 0.1, 0.1]),
        "previous": np.random.choice([0, 1, 2], size=n, p=[0.7, 0.2, 0.1]),
        "poutcome": np.random.choice(["unknown", "failure", "success"], size=n),
        "y": np.random.choice(["no", "yes"], size=n, p=[0.88, 0.12]),
    })


# =====================================================================
# Unit Tests (Synthetic Data)
# =====================================================================

def test_target_encoding(synthetic_raw_df):
    """Test 1: Target variable 'y' is cleanly encoded as binary integers (no=0, yes=1)."""
    X, y, row_ids, _, _ = extract_features_and_target(synthetic_raw_df)
    assert set(y.unique()).issubset({0, 1})
    assert y.dtype == int

    # Verify direct alignment with raw column
    expected_y = (synthetic_raw_df["y"] == "yes").astype(int)
    assert (y == expected_y).all()


def test_duration_exclusion(synthetic_raw_df):
    """Test 2: 'duration' feature is completely excluded from feature matrix X."""
    X, y, row_ids, num_cols, cat_cols = extract_features_and_target(synthetic_raw_df)
    assert LEAKAGE_FEATURE_TO_EXCLUDE not in X.columns
    assert LEAKAGE_FEATURE_TO_EXCLUDE not in num_cols
    assert LEAKAGE_FEATURE_TO_EXCLUDE not in cat_cols


def test_pdays_sentinel_handling(synthetic_raw_df):
    """Test 3: pdays=-1 sentinel is split into is_previously_contacted and pdays_recency."""
    X, y, row_ids, num_cols, _ = extract_features_and_target(synthetic_raw_df)

    assert "is_previously_contacted" in X.columns
    assert "pdays_recency" in X.columns
    assert "pdays" not in X.columns  # Replaced by engineered features

    # Check mapping logic
    for i, raw_pday in enumerate(synthetic_raw_df["pdays"]):
        if raw_pday == -1:
            assert X.loc[i, "is_previously_contacted"] == 0
            assert X.loc[i, "pdays_recency"] == 0.0
        else:
            assert X.loc[i, "is_previously_contacted"] == 1
            assert X.loc[i, "pdays_recency"] == float(raw_pday)


def test_stratification_and_ratios(synthetic_raw_df):
    """Test 4: Stratified split preserves target ratio and exact 70/15/15 partitions."""
    X, y, row_ids, _, _ = extract_features_and_target(synthetic_raw_df)
    split_res = stratified_split_dataset(
        X, y, row_ids, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )

    n = len(synthetic_raw_df)
    # Check exact counts
    assert len(split_res.X_train) == int(np.round(n * 0.70))
    assert len(split_res.X_val) == int(np.round(n * 0.15))
    assert len(split_res.X_test) == int(np.round(n * 0.15))

    # Check stratification: target positive rate preserved across splits
    overall_pos_rate = y.mean()
    train_pos_rate = split_res.y_train.mean()
    val_pos_rate = split_res.y_val.mean()
    test_pos_rate = split_res.y_test.mean()

    assert np.isclose(train_pos_rate, overall_pos_rate, atol=0.03)
    assert np.isclose(val_pos_rate, overall_pos_rate, atol=0.06)
    assert np.isclose(test_pos_rate, overall_pos_rate, atol=0.06)


def test_split_disjointness_no_overlap(synthetic_raw_df):
    """Test 5: Zero overlap between train, validation, and test row identifiers."""
    X, y, row_ids, _, _ = extract_features_and_target(synthetic_raw_df)
    split_res = stratified_split_dataset(X, y, row_ids, random_seed=42)

    assert split_res.verify_disjointness() is True
    set_train = set(split_res.train_indices)
    set_val = set(split_res.val_indices)
    set_test = set(split_res.test_indices)

    assert len(set_train & set_val) == 0
    assert len(set_train & set_test) == 0
    assert len(set_val & set_test) == 0
    assert len(set_train | set_val | set_test) == len(synthetic_raw_df)


def test_preprocessor_fit_on_train_only(synthetic_raw_df):
    """Test 6: Preprocessor learns statistics from X_train only and ignores X_val/X_test."""
    X, y, row_ids, num_cols, cat_cols = extract_features_and_target(synthetic_raw_df)
    split_res = stratified_split_dataset(X, y, row_ids, random_seed=42)

    preprocessor = build_preprocessor_pipeline(num_cols, cat_cols)
    preprocessor = fit_preprocessor(preprocessor, split_res.X_train)

    # Verify numerical scaler learned train mean
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    train_age_mean = split_res.X_train["age"].mean()
    idx_age = num_cols.index("age")
    assert np.isclose(scaler.mean_[idx_age], train_age_mean, atol=1e-5)

    # Validation data mean should be different from fitted scaler mean
    val_age_mean = split_res.X_val["age"].mean()
    assert not np.isclose(scaler.mean_[idx_age], val_age_mean, atol=1e-4)


def test_handle_unknown_categorical_levels():
    """Test 7: OneHotEncoder with handle_unknown='ignore' handles novel categories with zeros."""
    train_df = pd.DataFrame({
        "age": [25, 30, 35],
        "job": ["management", "technician", "management"],
    })
    # Val df introduces completely unseen job "astronaut"
    val_df = pd.DataFrame({
        "age": [40],
        "job": ["astronaut"],
    })

    preprocessor = build_preprocessor_pipeline(
        numerical_features=["age"],
        categorical_features=["job"],
    )
    preprocessor.fit(train_df)

    val_trans, feat_names = transform_features(preprocessor, val_df)
    assert len(val_trans) == 1
    # Unknown job should result in 0 for all job one-hot columns
    job_cols = [c for c in feat_names if "job" in c]
    for col in job_cols:
        assert val_trans.loc[0, col] == 0.0


def test_transformed_output_dimensions_and_types(synthetic_raw_df):
    """Test 8: Output feature names generated dynamically and data types are float64."""
    X, y, row_ids, num_cols, cat_cols = extract_features_and_target(synthetic_raw_df)
    split_res = stratified_split_dataset(X, y, row_ids, random_seed=42)

    preprocessor = build_preprocessor_pipeline(num_cols, cat_cols)
    preprocessor.fit(split_res.X_train)

    X_train_trans, train_features = transform_features(preprocessor, split_res.X_train)
    X_val_trans, val_features = transform_features(preprocessor, split_res.X_val)

    # Feature counts must be dynamically determined and match across splits
    assert len(train_features) == X_train_trans.shape[1]
    assert train_features == val_features
    assert X_train_trans.dtypes.apply(lambda dt: np.issubdtype(dt, np.floating)).all()
    assert not X_train_trans.isna().any().any()
    assert not X_val_trans.isna().any().any()


# =====================================================================
# Integration Test (Real UCI Dataset Pipeline)
# =====================================================================

@pytest.mark.integration
def test_full_pipeline_execution(tmp_path, synthetic_bank_df):
    """Integration test verifying end-to-end preprocessing pipeline on synthetic data in tmp_path."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    raw_csv = raw_dir / "bank-synthetic.csv"
    synthetic_bank_df.to_csv(raw_csv, sep=";", index=False)

    processed_dir = tmp_path / "processed"
    model_dir = tmp_path / "models"

    metadata = execute_feature_pipeline(
        raw_data_path=raw_csv,
        processed_dir=processed_dir,
        model_dir=model_dir,
        random_seed=42,
    )

    # Verify all expected artifacts exist in temporary directory
    assert (processed_dir / "X_train.parquet").exists()
    assert (processed_dir / "X_val.parquet").exists()
    assert (processed_dir / "X_test.parquet").exists()
    assert (processed_dir / "y_train.parquet").exists()
    assert (processed_dir / "y_val.parquet").exists()
    assert (processed_dir / "y_test.parquet").exists()
    assert (processed_dir / "split_indices.json").exists()
    assert (processed_dir / "preprocessing_metadata.json").exists()
    assert (model_dir / "preprocessor.joblib").exists()

    # Load temporary Parquet datasets and verify dimensions
    X_train = pd.read_parquet(processed_dir / "X_train.parquet")
    X_val = pd.read_parquet(processed_dir / "X_val.parquet")
    X_test_part = pd.read_parquet(processed_dir / "X_test.parquet")
    y_train = pd.read_parquet(processed_dir / "y_train.parquet")["y"]
    y_val = pd.read_parquet(processed_dir / "y_val.parquet")["y"]
    y_test_part = pd.read_parquet(processed_dir / "y_test.parquet")["y"]

    # Verify row counts: 70% (140), 15% (30), 15% (30) = 200
    n_total = len(synthetic_bank_df)
    assert len(X_train) == int(round(n_total * 0.70))
    assert len(X_val) == int(round(n_total * 0.15))
    assert len(X_test_part) == int(round(n_total * 0.15))
    assert len(X_train) + len(X_val) + len(X_test_part) == n_total

    # Verify columns dynamically generated and consistent
    dynamic_feature_count = metadata["transformed_feature_count"]
    assert X_train.shape[1] == dynamic_feature_count
    assert X_val.shape[1] == dynamic_feature_count
    assert X_test_part.shape[1] == dynamic_feature_count
    assert list(X_train.columns) == list(X_val.columns) == list(X_test_part.columns)

    # Verify duration is not in transformed features
    assert not any("duration" in col.lower() for col in X_train.columns)

    # Verify target distributions match within tolerance across splits
    overall_pos_pct = (synthetic_bank_df["y"] == "yes").mean() * 100
    train_pos_pct = (y_train == 1).mean() * 100
    val_pos_pct = (y_val == 1).mean() * 100
    test_pos_pct = (y_test_part == 1).mean() * 100

    assert np.isclose(train_pos_pct, overall_pos_pct, atol=2.0)
    assert np.isclose(val_pos_pct, overall_pos_pct, atol=4.0)
    assert np.isclose(test_pos_pct, overall_pos_pct, atol=4.0)

    # Verify split indices disjointness
    with open(processed_dir / "split_indices.json", "r") as f:
        indices = json.load(f)
    s_train = set(indices["train_indices"])
    s_val = set(indices["val_indices"])
    s_test = set(indices["test_indices"])
    assert len(s_train & s_val) == 0
    assert len(s_train & s_test) == 0
    assert len(s_val & s_test) == 0
    assert len(s_train | s_val | s_test) == n_total

    # Verify preprocessor serialization can be reloaded and transform raw inputs seamlessly
    raw_df = pd.read_csv(raw_csv, sep=";")
    X_raw, _, _, _, _ = extract_features_and_target(raw_df)
    loaded_preprocessor = load_fitted_preprocessor(model_dir / "preprocessor.joblib")
    re_trans, re_features = transform_features(loaded_preprocessor, X_raw.iloc[:5])
    assert re_features == list(X_train.columns)
    assert re_trans.shape == (5, dynamic_feature_count)
