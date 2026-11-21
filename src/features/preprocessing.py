"""Preprocessing pipeline module for MLForge.

Implements Scikit-learn ColumnTransformer combining median-imputed, standard-scaled
numerical pipelines with constant-imputed, one-hot encoded categorical pipelines.
Supports dynamic feature name generation via get_feature_names_out().
Strictly fits on training data only to prevent data leakage.
"""

import logging
from pathlib import Path
from typing import List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_preprocessor_pipeline(
    numerical_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Construct an unfitted Scikit-learn ColumnTransformer.

    Args:
        numerical_features: List of continuous and numeric feature column names.
        categorical_features: List of categorical feature column names.

    Returns:
        Unfitted ColumnTransformer instance.
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """Fit the preprocessor strictly on training data.

    Args:
        preprocessor: Unfitted ColumnTransformer.
        X_train: Training feature DataFrame.

    Returns:
        Fitted ColumnTransformer.
    """
    logger.info(f"Fitting preprocessor exclusively on training data ({len(X_train):,} rows)...")
    preprocessor.fit(X_train)
    return preprocessor


def transform_features(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """Transform a feature DataFrame using the fitted preprocessor.

    Derives feature names dynamically using get_feature_names_out().

    Args:
        preprocessor: Fitted ColumnTransformer.
        X: Feature DataFrame to transform.

    Returns:
        Tuple of (Transformed DataFrame with dynamic feature names, list of feature names).
    """
    transformed_array = preprocessor.transform(X)
    feature_names = [str(col) for col in preprocessor.get_feature_names_out()]

    transformed_df = pd.DataFrame(
        transformed_array,
        columns=feature_names,
        index=X.index,
    )
    return transformed_df, feature_names


def save_fitted_preprocessor(
    preprocessor: ColumnTransformer,
    output_path: Path = Path("models/preprocessor.joblib"),
) -> Path:
    """Serialize the fitted preprocessor pipeline to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output_path)
    logger.info(f"Saved fitted preprocessor pipeline to {output_path}")
    return output_path


def load_fitted_preprocessor(
    model_path: Path = Path("models/preprocessor.joblib"),
) -> ColumnTransformer:
    """Load a serialized fitted preprocessor pipeline from disk."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {model_path}")
    return joblib.load(model_path)
