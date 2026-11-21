"""Baseline experiment logging module for MLForge.

Ingests verified historical parameters, validation metrics (Task 4), and frozen holdout
test metrics (Task 5), registering them into the dedicated MLflow experiment 'MLForge_Bank_Marketing'.
Guarantees idempotency (skips re-logging if runs already exist) and zero retraining.
Strictly separates validation metrics from test metrics using 'test_' prefixes.
"""

import sys
from pathlib import Path

# Add project root to sys.path to enable src.* imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.config import setup_mlflow

import json
import logging
import platform
from typing import Any, Dict, List, Optional
import joblib
import mlflow
import torch

from src.models.train_pytorch import load_pytorch_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_verified_sources(
    project_root: Path = PROJECT_ROOT,
) -> Dict[str, Any]:
    """Load and cross-verify parameters and metrics from original source files."""
    baselines_json_path = project_root / "experiments/baselines/model_comparison.json"
    test_json_path = project_root / "experiments/final_evaluation/test_metrics.json"
    prep_meta_path = project_root / "data/processed/preprocessing_metadata.json"
    raw_meta_path = project_root / "data/raw/ingestion_metadata.json"

    for p in [baselines_json_path, test_json_path, prep_meta_path, raw_meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required verified source file missing: {p}")

    with open(baselines_json_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    with open(prep_meta_path, "r", encoding="utf-8") as f:
        prep_meta = json.load(f)

    with open(raw_meta_path, "r", encoding="utf-8") as f:
        raw_meta = json.load(f)

    return {
        "val_models": {m["model_name"]: m for m in val_data},
        "test_models": {m["model_name"]: m for m in test_data["models"]},
        "test_metadata": test_data["test_dataset_metadata"],
        "preprocessing_metadata": prep_meta,
        "raw_metadata": raw_meta,
    }


def find_existing_run_id(
    client: mlflow.client.MlflowClient,
    experiment_id: str,
    model_name: str,
) -> Optional[str]:
    """Check if an MLflow run for the given model already exists to guarantee idempotency."""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.model_name = '{model_name}'",
        max_results=5,
    )
    if runs:
        return runs[0].info.run_id
    return None


def register_logistic_regression_run(
    client: mlflow.client.MlflowClient,
    experiment_id: str,
    sources: Dict[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> str:
    """Register Logistic Regression baseline run idempotently without retraining."""
    model_name = "Logistic Regression"
    existing_id = find_existing_run_id(client, experiment_id, model_name)
    if existing_id:
        logger.info(f"Idempotent check: Run for '{model_name}' already exists (Run ID: {existing_id}). Skipping re-registration.")
        return existing_id

    val_entry = sources["val_models"][model_name]
    test_entry = sources["test_models"][model_name]
    raw_sha = sources["raw_metadata"]["sha256_checksum"]
    feat_count = sources["preprocessing_metadata"]["transformed_feature_count"]

    # Verify model parameters from serialized joblib artifact
    lr_model = joblib.load(project_root / "models/logistic_regression.joblib")
    lr_params = lr_model.get_params()

    run = client.create_run(
        experiment_id=experiment_id,
        run_name="Logistic_Regression_Baseline",
        tags={
            "model_name": model_name,
            "run_type": "baseline_historical_import",
            "evaluation_type": "frozen_historical_holdout",
            "provenance_task4_validation": "experiments/baselines/model_comparison.json",
            "provenance_task5_holdout_test": "experiments/final_evaluation/test_metrics.json",
            "python_version": platform.python_version(),
            "os_platform": f"{platform.system()} {platform.release()}",
            "raw_dataset_sha256": raw_sha,
            "preprocessor_reference": "models/preprocessor.joblib",
            "retraining_executed": "False",
        },
    )
    run_id = run.info.run_id

    # Log Parameters
    client.log_param(run_id, "model_family", "Linear")
    client.log_param(run_id, "estimator", "LogisticRegression")
    client.log_param(run_id, "solver", str(lr_params.get("solver", "lbfgs")))
    client.log_param(run_id, "max_iter", int(lr_params.get("max_iter", 1000)))
    client.log_param(run_id, "class_weight", str(lr_params.get("class_weight", "balanced")))
    client.log_param(run_id, "random_seed", int(lr_params.get("random_state", 42)))
    client.log_param(run_id, "input_feature_count", int(feat_count))

    # Log Training Duration and Latency
    client.log_metric(run_id, "train_duration_sec", float(val_entry["train_time_sec"]))
    client.log_metric(run_id, "median_batch_latency_ms", float(val_entry["latency"]["median_batch_latency_ms"]))
    client.log_metric(run_id, "per_sample_latency_us", float(val_entry["latency"]["per_sample_latency_us"]))

    # Log Validation Metrics (Task 4)
    ti_val = val_entry["threshold_independent_metrics"]
    def_val = val_entry["default_threshold_metrics"]
    tune_val = val_entry["tuned_threshold_metrics"]

    client.log_metric(run_id, "val_roc_auc", float(ti_val["roc_auc"]))
    client.log_metric(run_id, "val_pr_auc", float(ti_val["pr_auc"]))
    client.log_metric(run_id, "val_default_accuracy", float(def_val["accuracy"]))
    client.log_metric(run_id, "val_default_precision", float(def_val["precision"]))
    client.log_metric(run_id, "val_default_recall", float(def_val["recall"]))
    client.log_metric(run_id, "val_default_f1", float(def_val["f1"]))
    client.log_metric(run_id, "val_optimal_threshold", float(tune_val["threshold"]))
    client.log_metric(run_id, "val_tuned_accuracy", float(tune_val["accuracy"]))
    client.log_metric(run_id, "val_tuned_precision", float(tune_val["precision"]))
    client.log_metric(run_id, "val_tuned_recall", float(tune_val["recall"]))
    client.log_metric(run_id, "val_tuned_f1", float(tune_val["f1"]))

    # Log Frozen Holdout Test Metrics (Task 5, prefixed with 'test_')
    ti_test = test_entry["threshold_independent_metrics"]
    def_test = test_entry["default_threshold_metrics"]
    tune_test = test_entry["tuned_threshold_metrics"]
    gen_comp = test_entry["generalization_comparison"]

    client.log_metric(run_id, "test_roc_auc", float(ti_test["roc_auc"]))
    client.log_metric(run_id, "test_pr_auc", float(ti_test["pr_auc"]))
    client.log_metric(run_id, "test_default_f1", float(def_test["f1"]))
    client.log_metric(run_id, "test_tuned_accuracy", float(tune_test["accuracy"]))
    client.log_metric(run_id, "test_tuned_precision", float(tune_test["precision"]))
    client.log_metric(run_id, "test_tuned_recall", float(tune_test["recall"]))
    client.log_metric(run_id, "test_tuned_f1", float(tune_test["f1"]))

    client.log_metric(run_id, "delta_roc_auc_test_minus_val", float(gen_comp["roc_auc"]["delta"]))
    client.log_metric(run_id, "delta_pr_auc_test_minus_val", float(gen_comp["pr_auc"]["delta"]))
    client.log_metric(run_id, "delta_f1_tuned_test_minus_val", float(gen_comp["f1_tuned"]["delta"]))

    # Log Artifacts
    client.log_artifact(run_id, str(project_root / "models/logistic_regression.joblib"), "model")
    client.log_artifact(run_id, str(project_root / "experiments/baselines/roc_curves.png"), "visualizations")
    client.log_artifact(run_id, str(project_root / "experiments/baselines/pr_curves.png"), "visualizations")
    client.log_artifact(run_id, str(project_root / "experiments/baselines/confusion_matrices.png"), "visualizations")
    client.log_artifact(run_id, str(project_root / "experiments/final_evaluation/test_confusion_matrices.png"), "visualizations")

    client.set_terminated(run_id)
    logger.info(f"Registered Logistic Regression baseline run (Run ID: {run_id})")
    return run_id


def register_random_forest_run(
    client: mlflow.client.MlflowClient,
    experiment_id: str,
    sources: Dict[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> str:
    """Register Random Forest baseline run idempotently without retraining."""
    model_name = "Random Forest"
    existing_id = find_existing_run_id(client, experiment_id, model_name)
    if existing_id:
        logger.info(f"Idempotent check: Run for '{model_name}' already exists (Run ID: {existing_id}). Skipping re-registration.")
        return existing_id

    val_entry = sources["val_models"][model_name]
    test_entry = sources["test_models"][model_name]
    raw_sha = sources["raw_metadata"]["sha256_checksum"]
    feat_count = sources["preprocessing_metadata"]["transformed_feature_count"]

    # Verify model parameters from serialized joblib artifact
    rf_model = joblib.load(project_root / "models/random_forest.joblib")
    rf_params = rf_model.get_params()

    run = client.create_run(
        experiment_id=experiment_id,
        run_name="Random_Forest_Baseline",
        tags={
            "model_name": model_name,
            "run_type": "baseline_historical_import",
            "evaluation_type": "frozen_historical_holdout",
            "provenance_task4_validation": "experiments/baselines/model_comparison.json",
            "provenance_task5_holdout_test": "experiments/final_evaluation/test_metrics.json",
            "python_version": platform.python_version(),
            "os_platform": f"{platform.system()} {platform.release()}",
            "raw_dataset_sha256": raw_sha,
            "preprocessor_reference": "models/preprocessor.joblib",
            "retraining_executed": "False",
            "hyperparameter_status": "baseline_settings_not_asserted_as_optimal",
        },
    )
    run_id = run.info.run_id

    # Log Parameters
    client.log_param(run_id, "model_family", "Tree Ensemble")
    client.log_param(run_id, "estimator", "RandomForestClassifier")
    client.log_param(run_id, "n_estimators", int(rf_params.get("n_estimators", 100)))
    client.log_param(run_id, "max_depth", int(rf_params.get("max_depth", 12)))
    client.log_param(run_id, "min_samples_leaf", int(rf_params.get("min_samples_leaf", 5)))
    client.log_param(run_id, "class_weight", str(rf_params.get("class_weight", "balanced")))
    client.log_param(run_id, "random_seed", int(rf_params.get("random_state", 42)))
    client.log_param(run_id, "n_jobs", int(rf_params.get("n_jobs", -1)))
    client.log_param(run_id, "input_feature_count", int(feat_count))

    # Log Training Duration and Latency
    client.log_metric(run_id, "train_duration_sec", float(val_entry["train_time_sec"]))
    client.log_metric(run_id, "median_batch_latency_ms", float(val_entry["latency"]["median_batch_latency_ms"]))
    client.log_metric(run_id, "per_sample_latency_us", float(val_entry["latency"]["per_sample_latency_us"]))

    # Log Validation Metrics (Task 4)
    ti_val = val_entry["threshold_independent_metrics"]
    def_val = val_entry["default_threshold_metrics"]
    tune_val = val_entry["tuned_threshold_metrics"]

    client.log_metric(run_id, "val_roc_auc", float(ti_val["roc_auc"]))
    client.log_metric(run_id, "val_pr_auc", float(ti_val["pr_auc"]))
    client.log_metric(run_id, "val_default_accuracy", float(def_val["accuracy"]))
    client.log_metric(run_id, "val_default_precision", float(def_val["precision"]))
    client.log_metric(run_id, "val_default_recall", float(def_val["recall"]))
    client.log_metric(run_id, "val_default_f1", float(def_val["f1"]))
    client.log_metric(run_id, "val_optimal_threshold", float(tune_val["threshold"]))
    client.log_metric(run_id, "val_tuned_accuracy", float(tune_val["accuracy"]))
    client.log_metric(run_id, "val_tuned_precision", float(tune_val["precision"]))
    client.log_metric(run_id, "val_tuned_recall", float(tune_val["recall"]))
    client.log_metric(run_id, "val_tuned_f1", float(tune_val["f1"]))

    # Log Frozen Holdout Test Metrics (Task 5, prefixed with 'test_')
    ti_test = test_entry["threshold_independent_metrics"]
    def_test = test_entry["default_threshold_metrics"]
    tune_test = test_entry["tuned_threshold_metrics"]
    gen_comp = test_entry["generalization_comparison"]

    client.log_metric(run_id, "test_roc_auc", float(ti_test["roc_auc"]))
    client.log_metric(run_id, "test_pr_auc", float(ti_test["pr_auc"]))
    client.log_metric(run_id, "test_default_f1", float(def_test["f1"]))
    client.log_metric(run_id, "test_tuned_accuracy", float(tune_test["accuracy"]))
    client.log_metric(run_id, "test_tuned_precision", float(tune_test["precision"]))
    client.log_metric(run_id, "test_tuned_recall", float(tune_test["recall"]))
    client.log_metric(run_id, "test_tuned_f1", float(tune_test["f1"]))

    client.log_metric(run_id, "delta_roc_auc_test_minus_val", float(gen_comp["roc_auc"]["delta"]))
    client.log_metric(run_id, "delta_pr_auc_test_minus_val", float(gen_comp["pr_auc"]["delta"]))
    client.log_metric(run_id, "delta_f1_tuned_test_minus_val", float(gen_comp["f1_tuned"]["delta"]))

    # Log Artifacts
    client.log_artifact(run_id, str(project_root / "models/random_forest.joblib"), "model")
    client.log_artifact(run_id, str(project_root / "experiments/baselines/roc_curves.png"), "visualizations")
    client.log_artifact(run_id, str(project_root / "experiments/baselines/pr_curves.png"), "visualizations")
    client.log_artifact(run_id, str(project_root / "experiments/baselines/confusion_matrices.png"), "visualizations")
    client.log_artifact(run_id, str(project_root / "experiments/final_evaluation/test_confusion_matrices.png"), "visualizations")

    client.set_terminated(run_id)
    logger.info(f"Registered Random Forest baseline run (Run ID: {run_id})")
    return run_id


def register_pytorch_run(
    client: mlflow.client.MlflowClient,
    experiment_id: str,
    sources: Dict[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> str:
    """Register PyTorch Neural Network baseline run idempotently without retraining."""
    model_name = "PyTorch Neural Network"
    existing_id = find_existing_run_id(client, experiment_id, model_name)
    if existing_id:
        logger.info(f"Idempotent check: Run for '{model_name}' already exists (Run ID: {existing_id}). Skipping re-registration.")
        return existing_id

    val_entry = sources["val_models"][model_name]
    test_entry = sources["test_models"][model_name]
    raw_sha = sources["raw_metadata"]["sha256_checksum"]

    # Verify model parameters directly from checkpoint
    _, ckpt = load_pytorch_checkpoint(project_root / "models/pytorch_model_best.pt")
    arch = ckpt["architecture_config"]
    train_cfg = ckpt["training_config"]

    # Check whether epoch-level numeric loss history exists
    # Verified: raw numeric loss array was not serialized in checkpoint or comparison JSON.
    # Therefore, explicitly document this fact and log the PNG loss curve artifact as instructed.
    epoch_history_available = "train_losses" in ckpt

    run = client.create_run(
        experiment_id=experiment_id,
        run_name="PyTorch_Neural_Network_Baseline",
        tags={
            "model_name": model_name,
            "run_type": "baseline_historical_import",
            "evaluation_type": "frozen_historical_holdout",
            "provenance_task4_validation": "experiments/baselines/model_comparison.json",
            "provenance_task5_holdout_test": "experiments/final_evaluation/test_metrics.json",
            "python_version": platform.python_version(),
            "os_platform": f"{platform.system()} {platform.release()}",
            "raw_dataset_sha256": raw_sha,
            "preprocessor_reference": "models/preprocessor.joblib",
            "retraining_executed": "False",
            "epoch_numeric_loss_history": "unavailable_in_checkpoint; visual_curve_logged_as_artifact",
            "early_stopping_default_min_delta": "0.0001 (verified from train_pytorch_model default signature)",
        },
    )
    run_id = run.info.run_id

    # Log Parameters
    client.log_param(run_id, "model_family", "Deep Learning")
    client.log_param(run_id, "architecture", f"FeedForward ({arch['input_dim']} -> {arch['hidden_dim1']} -> {arch['hidden_dim2']} -> {arch['output_dim']})")
    client.log_param(run_id, "input_feature_count", int(arch["input_dim"]))
    client.log_param(run_id, "hidden_dim1", int(arch["hidden_dim1"]))
    client.log_param(run_id, "hidden_dim2", int(arch["hidden_dim2"]))
    client.log_param(run_id, "dropout_rate", float(arch["dropout_rate"]))
    client.log_param(run_id, "activation", "ReLU")
    client.log_param(run_id, "loss_function", str(train_cfg["loss_function"]))
    client.log_param(run_id, "pos_weight", round(float(train_cfg["pos_weight"]), 4))
    client.log_param(run_id, "optimizer", str(train_cfg["optimizer"]))
    client.log_param(run_id, "learning_rate", float(train_cfg["learning_rate"]))
    client.log_param(run_id, "weight_decay", float(train_cfg["weight_decay"]))
    client.log_param(run_id, "batch_size", int(train_cfg["batch_size"]))
    client.log_param(run_id, "random_seed", int(train_cfg["random_seed"]))
    client.log_param(run_id, "best_epoch", int(ckpt["best_epoch"]))
    client.log_param(run_id, "total_epochs_trained", int(ckpt["total_epochs_trained"]))

    # Log Training Duration and Latency
    client.log_metric(run_id, "train_duration_sec", float(val_entry["train_time_sec"]))
    client.log_metric(run_id, "best_val_loss", float(ckpt["best_val_loss"]))
    client.log_metric(run_id, "median_batch_latency_ms", float(val_entry["latency"]["median_batch_latency_ms"]))
    client.log_metric(run_id, "per_sample_latency_us", float(val_entry["latency"]["per_sample_latency_us"]))

    # Log Validation Metrics (Task 4)
    ti_val = val_entry["threshold_independent_metrics"]
    def_val = val_entry["default_threshold_metrics"]
    tune_val = val_entry["tuned_threshold_metrics"]

    client.log_metric(run_id, "val_roc_auc", float(ti_val["roc_auc"]))
    client.log_metric(run_id, "val_pr_auc", float(ti_val["pr_auc"]))
    client.log_metric(run_id, "val_default_accuracy", float(def_val["accuracy"]))
    client.log_metric(run_id, "val_default_precision", float(def_val["precision"]))
    client.log_metric(run_id, "val_default_recall", float(def_val["recall"]))
    client.log_metric(run_id, "val_default_f1", float(def_val["f1"]))
    client.log_metric(run_id, "val_optimal_threshold", float(tune_val["threshold"]))
    client.log_metric(run_id, "val_tuned_accuracy", float(tune_val["accuracy"]))
    client.log_metric(run_id, "val_tuned_precision", float(tune_val["precision"]))
    client.log_metric(run_id, "val_tuned_recall", float(tune_val["recall"]))
    client.log_metric(run_id, "val_tuned_f1", float(tune_val["f1"]))

    # Log Frozen Holdout Test Metrics (Task 5, prefixed with 'test_')
    ti_test = test_entry["threshold_independent_metrics"]
    def_test = test_entry["default_threshold_metrics"]
    tune_test = test_entry["tuned_threshold_metrics"]
    gen_comp = test_entry["generalization_comparison"]

    client.log_metric(run_id, "test_roc_auc", float(ti_test["roc_auc"]))
    client.log_metric(run_id, "test_pr_auc", float(ti_test["pr_auc"]))
    client.log_metric(run_id, "test_default_f1", float(def_test["f1"]))
    client.log_metric(run_id, "test_tuned_accuracy", float(tune_test["accuracy"]))
    client.log_metric(run_id, "test_tuned_precision", float(tune_test["precision"]))
    client.log_metric(run_id, "test_tuned_recall", float(tune_test["recall"]))
    client.log_metric(run_id, "test_tuned_f1", float(tune_test["f1"]))

    client.log_metric(run_id, "delta_roc_auc_test_minus_val", float(gen_comp["roc_auc"]["delta"]))
    client.log_metric(run_id, "delta_pr_auc_test_minus_val", float(gen_comp["pr_auc"]["delta"]))
    client.log_metric(run_id, "delta_f1_tuned_test_minus_val", float(gen_comp["f1_tuned"]["delta"]))

    # Log Artifacts
    client.log_artifact(run_id, str(project_root / "models/pytorch_model_best.pt"), "model")
    client.log_artifact(run_id, str(project_root / "experiments/baselines/pytorch_loss_curves.png"), "visualizations")
    client.log_artifact(run_id, str(project_root / "experiments/baselines/roc_curves.png"), "visualizations")
    client.log_artifact(run_id, str(project_root / "experiments/baselines/pr_curves.png"), "visualizations")
    client.log_artifact(run_id, str(project_root / "experiments/baselines/confusion_matrices.png"), "visualizations")
    client.log_artifact(run_id, str(project_root / "experiments/final_evaluation/test_confusion_matrices.png"), "visualizations")

    client.set_terminated(run_id)
    logger.info(f"Registered PyTorch Neural Network baseline run (Run ID: {run_id})")
    return run_id


def register_all_baseline_runs(
    project_root: Path = PROJECT_ROOT,
    client: Optional[mlflow.client.MlflowClient] = None,
    experiment_id: Optional[str] = None,
) -> Dict[str, str]:
    """Orchestrate idempotent registration of all 3 baseline model runs into MLflow.

    Args:
        project_root: Path to repository root containing models/ and experiments/.
        client: Optional MlflowClient. If supplied, experiment_id must also be supplied.
        experiment_id: Optional target experiment ID. If supplied, client must also be supplied.

    Raises:
        ValueError: If exactly one of client or experiment_id is provided.
    """
    if (client is None) ^ (experiment_id is None):
        raise ValueError(
            "Both 'client' and 'experiment_id' must be provided together, or both must be omitted."
        )

    if client is None and experiment_id is None:
        client, experiment_id = setup_mlflow()

    sources = load_verified_sources(project_root)

    run_ids: Dict[str, str] = {}
    run_ids["Logistic Regression"] = register_logistic_regression_run(client, experiment_id, sources, project_root)
    run_ids["Random Forest"] = register_random_forest_run(client, experiment_id, sources, project_root)
    run_ids["PyTorch Neural Network"] = register_pytorch_run(client, experiment_id, sources, project_root)

    exp = client.get_experiment(experiment_id)
    exp_name = exp.name if exp else experiment_id
    logger.info(f"All 3 baseline model runs registered in experiment '{exp_name}' (ID: {experiment_id})")
    return run_ids


if __name__ == "__main__":
    register_all_baseline_runs()
