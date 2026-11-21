"""Baseline model training module for MLForge — Model 2: Random Forest.

Trains an ensemble Random Forest with balanced class weighting on CPU.
Explicitly documents hyperparameters as baseline starting points rather than tuned optima.
Benchmarks latency, evaluates on validation data, and serializes the model.
Strictly adheres to validation-only governance (never accesses test data).
"""

import sys
from pathlib import Path

# Add project root to sys.path to enable src.* imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import time
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.metrics import (
    benchmark_inference_latency,
    compute_classification_metrics,
    compute_threshold_independent_metrics,
    find_optimal_f1_threshold,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Baseline hyperparameter configuration (established as baseline settings, not claimed as optimal)
BASELINE_RF_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,  # Utilize all CPU cores for parallel tree induction
}


def train_random_forest(
    processed_dir: Path = Path("data/processed"),
    model_output_path: Path = Path("models/random_forest.joblib"),
    rf_params: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Train, benchmark, and evaluate Random Forest on validation data.

    Args:
        processed_dir: Directory containing processed train and validation Parquet files.
        model_output_path: Destination path for serialized model artifact.
        rf_params: Dictionary of hyperparameters (defaults to BASELINE_RF_PARAMS).

    Returns:
        Dictionary containing trained model, validation predictions, and metrics.
    """
    processed_dir = Path(processed_dir)
    model_output_path = Path(model_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    params = rf_params or BASELINE_RF_PARAMS

    # 1. Load Training and Validation Sets (Strict validation-only governance: never load test)
    logger.info("Loading training and validation datasets...")
    X_train = pd.read_parquet(processed_dir / "X_train.parquet")
    y_train = pd.read_parquet(processed_dir / "y_train.parquet")["y"].values
    X_val = pd.read_parquet(processed_dir / "X_val.parquet")
    y_val = pd.read_parquet(processed_dir / "y_val.parquet")["y"].values

    # 2. Configure Model
    model = RandomForestClassifier(**params)

    # 3. Fit on Training Data and Measure Wall-Clock Training Duration
    logger.info(f"Fitting Random Forest on {len(X_train):,} training records with parameters: {params}...")
    start_train = time.perf_counter()
    model.fit(X_train.values, y_train)
    train_duration_sec = time.perf_counter() - start_train
    logger.info(f"Random Forest training completed in {train_duration_sec:.3f} seconds.")

    # 4. Measure Validation Inference Latency
    latency_stats = benchmark_inference_latency(
        predict_fn=lambda x: model.predict_proba(x)[:, 1],
        X_sample=X_val.values,
        num_warmups=3,
        num_repeats=10,
    )

    # 5. Extract Validation Predictions
    y_val_prob = model.predict_proba(X_val.values)[:, 1]

    # 6. Compute Threshold-Independent and Threshold-Dependent Metrics
    thresh_indep = compute_threshold_independent_metrics(y_val, y_val_prob)
    default_metrics = compute_classification_metrics(y_val, y_val_prob, threshold=0.50)
    best_thresh, best_f1 = find_optimal_f1_threshold(y_val, y_val_prob)
    tuned_metrics = compute_classification_metrics(y_val, y_val_prob, threshold=best_thresh)

    # 7. Serialize Fitted Model
    joblib.dump(model, model_output_path)
    logger.info(f"Saved Random Forest model artifact to {model_output_path}")

    results = {
        "model_name": "Random Forest",
        "model_path": str(model_output_path),
        "hyperparameters": params,
        "note": "Parameters are baseline settings designed for CPU efficiency, not asserted to be optimal.",
        "train_time_sec": round(train_duration_sec, 4),
        "latency": latency_stats,
        "threshold_independent_metrics": thresh_indep,
        "default_threshold_metrics": default_metrics,
        "tuned_threshold_metrics": tuned_metrics,
        "validation_probabilities": y_val_prob.tolist(),
        "y_val": y_val.tolist(),
    }
    return results


if __name__ == "__main__":
    train_random_forest()
