"""Feature engineering and preprocessing execution pipeline for MLForge.

Transforms the UCI Bank Marketing dataset into leakage-safe, stratified
training, validation, and test datasets. Excludes call duration, engineers
pdays sentinel features, tracks row identifiers separately, and dynamically
records transformed feature names and dimensions.
"""

import sys
from pathlib import Path

# Add project root to sys.path to enable src.* imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.data.ingest_data import calculate_sha256, load_raw_dataset
from src.data.split_data import stratified_split_dataset, save_split_indices, SplitResult
from src.features.preprocessing import (
    build_preprocessor_pipeline,
    fit_preprocessor,
    transform_features,
    save_fitted_preprocessor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Feature configuration
LEAKAGE_FEATURE_TO_EXCLUDE = "duration"
TARGET_COLUMN = "y"
TARGET_MAPPING = {"no": 0, "yes": 1}

BASE_NUMERICAL_COLS = ["age", "balance", "day", "campaign", "previous"]
BASE_CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome"
]


def extract_features_and_target(
    raw_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, List[str], List[str]]:
    """Transform raw DataFrame into predictive feature matrix X and encoded target y.

    Applies feature engineering:
    - Encodes target: no -> 0, yes -> 1.
    - Excludes duration (pre-call data leakage).
    - Preserves 'unknown' as distinct categorical category.
    - Creates is_previously_contacted (0 for pdays == -1, 1 otherwise).
    - Creates pdays_recency (0.0 for pdays == -1, pdays otherwise).
    - Preserves row identifiers separately for auditability.

    Args:
        raw_df: Unmodified raw DataFrame.

    Returns:
        Tuple of (X, y, row_ids, numerical_cols, categorical_cols).
    """
    df = raw_df.copy()

    # 1. Separate row identifiers (never included in feature set X)
    row_ids = np.arange(len(df), dtype=int)

    # 2. Encode Target
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in raw data.")
    y = df[TARGET_COLUMN].map(TARGET_MAPPING).astype(int)

    # 3. Exclude Target Leakage Feature
    if LEAKAGE_FEATURE_TO_EXCLUDE in df.columns:
        df = df.drop(columns=[LEAKAGE_FEATURE_TO_EXCLUDE])
        logger.info(f"Excluded target leakage feature '{LEAKAGE_FEATURE_TO_EXCLUDE}' from predictive set.")

    # 4. Feature Engineering for pdays Sentinel Value (-1)
    if "pdays" in df.columns:
        # Binary indicator of prior contact
        df["is_previously_contacted"] = (df["pdays"] != -1).astype(int)
        # Numeric recency: 0.0 when uncontacted, exact days elapsed when previously contacted
        df["pdays_recency"] = np.where(df["pdays"] == -1, 0.0, df["pdays"].astype(float))
        # Drop original raw pdays to avoid redundant collinearity
        df = df.drop(columns=["pdays"])
        logger.info("Engineered 'is_previously_contacted' and 'pdays_recency' from 'pdays' sentinel.")

    # Drop target from feature set
    X = df.drop(columns=[TARGET_COLUMN])

    # Dynamic identification of feature subsets
    numerical_cols = [c for c in X.columns if c in BASE_NUMERICAL_COLS or c in ["is_previously_contacted", "pdays_recency"]]
    categorical_cols = [c for c in X.columns if c in BASE_CATEGORICAL_COLS]

    logger.info(
        f"Feature extraction complete: {X.shape[1]} raw features "
        f"({len(numerical_cols)} numerical, {len(categorical_cols)} categorical)."
    )
    return X, y, row_ids, numerical_cols, categorical_cols


def execute_feature_pipeline(
    raw_data_path: Path = Path("data/raw/bank-full.csv"),
    processed_dir: Path = Path("data/processed"),
    model_dir: Path = Path("models"),
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete feature engineering, splitting, fitting, and transformation pipeline."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load pristine raw data and verify checksum
    raw_data_path = Path(raw_data_path)
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_data_path}")
    raw_checksum = calculate_sha256(raw_data_path)
    raw_df = load_raw_dataset(raw_data_path)

    # 2. Extract features and target
    X, y, row_ids, num_cols, cat_cols = extract_features_and_target(raw_df)

    # 3. Stratified split (70% train, 15% val, 15% test)
    split_res = stratified_split_dataset(
        X=X,
        y=y,
        row_ids=row_ids,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=random_seed,
    )

    # Save row identifiers for auditability
    save_split_indices(split_res, processed_dir / "split_indices.json")

    # 4. Build and fit preprocessor EXCLUSIVELY on training features
    preprocessor = build_preprocessor_pipeline(
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )
    preprocessor = fit_preprocessor(preprocessor, split_res.X_train)

    # 5. Transform Train, Validation, and Test datasets using training preprocessor
    X_train_trans, feature_names = transform_features(preprocessor, split_res.X_train)
    X_val_trans, _ = transform_features(preprocessor, split_res.X_val)
    X_test_trans, _ = transform_features(preprocessor, split_res.X_test)

    # 6. Save fitted preprocessor pipeline
    preprocessor_path = save_fitted_preprocessor(preprocessor, model_dir / "preprocessor.joblib")

    # 7. Save processed datasets in Parquet format
    X_train_trans.to_parquet(processed_dir / "X_train.parquet", index=False)
    X_val_trans.to_parquet(processed_dir / "X_val.parquet", index=False)
    X_test_trans.to_parquet(processed_dir / "X_test.parquet", index=False)

    split_res.y_train.to_frame(name="y").to_parquet(processed_dir / "y_train.parquet", index=False)
    split_res.y_val.to_frame(name="y").to_parquet(processed_dir / "y_val.parquet", index=False)
    split_res.y_test.to_frame(name="y").to_parquet(processed_dir / "y_test.parquet", index=False)

    # 8. Compute target distributions across splits
    def get_distribution(y_series: pd.Series) -> Dict[str, Any]:
        counts = y_series.value_counts().to_dict()
        n = len(y_series)
        return {
            "count_0": int(counts.get(0, 0)),
            "count_1": int(counts.get(1, 0)),
            "pct_0": round((counts.get(0, 0) / n) * 100, 2),
            "pct_1": round((counts.get(1, 0) / n) * 100, 2),
            "total": n,
        }

    target_dists = {
        "train": get_distribution(split_res.y_train),
        "validation": get_distribution(split_res.y_val),
        "test": get_distribution(split_res.y_test),
        "overall": get_distribution(y),
    }

    # 9. Record full metadata manifest
    metadata = {
        "random_seed": random_seed,
        "raw_dataset": {
            "path": str(raw_data_path),
            "sha256": raw_checksum,
            "total_rows": len(raw_df),
        },
        "target_column": TARGET_COLUMN,
        "target_mapping": TARGET_MAPPING,
        "excluded_features": [LEAKAGE_FEATURE_TO_EXCLUDE],
        "engineered_features": {
            "is_previously_contacted": "Binary flag (0 for pdays == -1, 1 otherwise).",
            "pdays_recency": "Numeric days since last contact (0.0 for pdays == -1, actual days for pdays > -1).",
        },
        "numerical_features_raw": num_cols,
        "categorical_features_raw": cat_cols,
        "transformed_feature_count": len(feature_names),
        "transformed_feature_names": feature_names,
        "split_dimensions": {
            "X_train": list(X_train_trans.shape),
            "X_val": list(X_val_trans.shape),
            "X_test": list(X_test_trans.shape),
            "y_train": [len(split_res.y_train)],
            "y_val": [len(split_res.y_val)],
            "y_test": [len(split_res.y_test)],
        },
        "target_distributions": target_dists,
        "fitted_preprocessor_path": str(preprocessor_path),
    }

    metadata_path = processed_dir / "preprocessing_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Preprocessing pipeline completed successfully!\n"
        f"  Transformed feature count: {len(feature_names)}\n"
        f"  X_train: {X_train_trans.shape}\n"
        f"  X_val:   {X_val_trans.shape}\n"
        f"  X_test:  {X_test_trans.shape}\n"
        f"  Metadata saved to: {metadata_path}"
    )
    return metadata


if __name__ == "__main__":
    execute_feature_pipeline()
