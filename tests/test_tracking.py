"""Automated test suite for MLflow tracking, parameter/metric integrity, artifact download, and governance."""

import ast
import json
from pathlib import Path
import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch
from mlflow.tracking import MlflowClient

from src.models.pytorch_model import MLForgeClassifier

from src.tracking.config import (
    DEFAULT_EXPERIMENT_NAME,
    get_artifact_location,
    get_tracking_uri,
    setup_mlflow,
)
from src.tracking.log_baselines import register_all_baseline_runs


@pytest.fixture(scope="module")
def mlflow_setup(tmp_path_factory):
    """Module-level fixture configuring an isolated temporary tracking backend."""
    tmp_dir = tmp_path_factory.mktemp("mlflow_tracking_isolated")
    db_path = tmp_dir / "test_mlruns.db"
    artifact_dir = tmp_dir / "test_mlartifacts"

    # Set up directories in tmp_dir for source files and artifacts
    (tmp_dir / "models").mkdir(parents=True)
    (tmp_dir / "experiments/baselines").mkdir(parents=True)
    (tmp_dir / "experiments/final_evaluation").mkdir(parents=True)
    (tmp_dir / "data/processed").mkdir(parents=True)
    (tmp_dir / "data/raw").mkdir(parents=True)

    # Copy lightweight tracked verified source JSONs
    for src_rel in [
        "experiments/baselines/model_comparison.json",
        "experiments/final_evaluation/test_metrics.json",
    ]:
        orig = Path(src_rel)
        if orig.exists():
            (tmp_dir / src_rel).write_bytes(orig.read_bytes())

    # Create hermetic synthetic metadata directly inside the temporary test project
    prep_meta = {
        "transformed_feature_count": 51,
        "target_column": "y",
        "excluded_features": ["duration"],
    }
    raw_meta = {
        "dataset_name": "UCI Bank Marketing",
        "sha256_checksum": "d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d",
    }
    (tmp_dir / "data/processed/preprocessing_metadata.json").write_text(
        json.dumps(prep_meta, indent=2), encoding="utf-8"
    )
    (tmp_dir / "data/raw/ingestion_metadata.json").write_text(
        json.dumps(raw_meta, indent=2), encoding="utf-8"
    )

    # Populate lightweight valid model artifacts matching expected parameters
    lr = LogisticRegression(solver="lbfgs", class_weight="balanced", random_state=42)
    joblib.dump(lr, tmp_dir / "models/logistic_regression.joblib")

    rf = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
    joblib.dump(rf, tmp_dir / "models/random_forest.joblib")

    pt_model = MLForgeClassifier(input_dim=51, hidden_dim1=32, hidden_dim2=16)
    ckpt = {
        "model_state_dict": pt_model.state_dict(),
        "architecture_config": {
            "input_dim": 51,
            "hidden_dim1": 32,
            "hidden_dim2": 16,
            "dropout_rate": 0.3,
            "output_dim": 1,
        },
        "input_feature_count": 51,
        "best_epoch": 11,
        "total_epochs_trained": 21,
        "best_val_loss": 0.35,
        "training_config": {
            "loss_function": "BCEWithLogitsLoss",
            "pos_weight": 5.98,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 256,
            "weight_decay": 1e-4,
            "random_seed": 42,
            "patience": 5,
            "min_delta": 0.0001,
        },
    }
    torch.save(ckpt, tmp_dir / "models/pytorch_model_best.pt")

    dummy_payload = b"MLForge dummy artifact content for hermetic testing " * 30
    (tmp_dir / "models/preprocessor.joblib").write_bytes(dummy_payload)
    for art_rel in [
        "experiments/baselines/pytorch_loss_curves.png",
        "experiments/baselines/baseline_roc_pr_curves.png",
        "experiments/baselines/roc_curves.png",
        "experiments/baselines/pr_curves.png",
        "experiments/baselines/confusion_matrices.png",
        "experiments/final_evaluation/test_confusion_matrices.png",
    ]:
        (tmp_dir / art_rel).write_bytes(dummy_payload)

    client, exp_id = setup_mlflow(
        experiment_name=DEFAULT_EXPERIMENT_NAME,
        db_path=db_path,
        artifact_dir=artifact_dir,
    )
    run_ids = register_all_baseline_runs(
        project_root=tmp_dir,
        client=client,
        experiment_id=exp_id,
    )
    return client, exp_id, run_ids, tmp_dir


# =====================================================================
# Tests: Backend Connection & Experiment Retrieval
# =====================================================================

def test_sqlite_backend_initialization(tmp_path):
    """Test 1: Verify SQLite URI structure and database initialization in tmp_path."""
    test_db = tmp_path / "custom_mlruns.db"
    uri = get_tracking_uri(test_db)
    assert uri.startswith("sqlite:///")
    assert "custom_mlruns.db" in uri

    test_art = tmp_path / "custom_mlartifacts"
    artifact_uri = get_artifact_location(test_art)
    assert artifact_uri.startswith("file:")
    assert test_art.exists()


def test_experiment_creation_and_retrieval(mlflow_setup):
    """Test 2: Verify experiment 'MLForge_Bank_Marketing' is active and retrievable."""
    client, exp_id, _, _ = mlflow_setup
    exp = client.get_experiment(exp_id)

    assert exp is not None
    assert exp.name == DEFAULT_EXPERIMENT_NAME
    assert exp.lifecycle_stage == "active"


# =====================================================================
# Tests: Run Distinguishability & Count
# =====================================================================

def test_exactly_three_distinguishable_runs(mlflow_setup):
    """Test 3: Exactly three historical runs exist and are distinguishable by model_name."""
    client, exp_id, run_ids, _ = mlflow_setup
    runs = client.search_runs(experiment_ids=[exp_id])

    assert len(runs) == 3, f"Expected exactly 3 runs, found {len(runs)}"

    model_names = {r.data.tags.get("model_name") for r in runs}
    expected_names = {"Logistic Regression", "Random Forest", "PyTorch Neural Network"}
    assert model_names == expected_names


# =====================================================================
# Tests: Parameter and Metric Integrity
# =====================================================================

def test_parameter_and_metric_integrity(mlflow_setup):
    """Test 4: Verified parameters and metrics match Task 4 validation artifacts."""
    client, exp_id, _, _ = mlflow_setup
    runs = client.search_runs(experiment_ids=[exp_id])
    run_map = {r.data.tags["model_name"]: r for r in runs}

    # 1. Logistic Regression
    lr_data = run_map["Logistic Regression"].data
    assert lr_data.params["solver"] == "lbfgs"
    assert lr_data.params["class_weight"] == "balanced"
    assert lr_data.metrics["val_roc_auc"] == pytest.approx(0.7720, abs=1e-4)
    assert lr_data.metrics["val_optimal_threshold"] == pytest.approx(0.660, abs=1e-3)
    assert lr_data.metrics["val_tuned_f1"] == pytest.approx(0.4490, abs=1e-4)

    # 2. Random Forest
    rf_data = run_map["Random Forest"].data
    assert int(rf_data.params["n_estimators"]) == 100
    assert int(rf_data.params["max_depth"]) == 12
    assert rf_data.metrics["val_roc_auc"] == pytest.approx(0.7977, abs=1e-4)
    assert rf_data.metrics["val_optimal_threshold"] == pytest.approx(0.635, abs=1e-3)
    assert rf_data.metrics["val_tuned_f1"] == pytest.approx(0.4849, abs=1e-4)

    # 3. PyTorch Neural Network
    pt_data = run_map["PyTorch Neural Network"].data
    assert int(pt_data.params["input_feature_count"]) == 51
    assert float(pt_data.params["learning_rate"]) == 0.001
    assert int(pt_data.params["best_epoch"]) == 11
    assert pt_data.metrics["val_roc_auc"] == pytest.approx(0.7996, abs=1e-4)
    assert pt_data.metrics["val_optimal_threshold"] == pytest.approx(0.685, abs=1e-3)
    assert pt_data.metrics["val_tuned_f1"] == pytest.approx(0.4887, abs=1e-4)


# =====================================================================
# Tests: Historical Test Metric Separation
# =====================================================================

def test_historical_test_metric_separation(mlflow_setup):
    """Test 5: Holdout test metrics carry 'test_' prefix and distinct provenance tags."""
    client, exp_id, _, _ = mlflow_setup
    runs = client.search_runs(experiment_ids=[exp_id])

    for run in runs:
        metrics = run.data.metrics
        tags = run.data.tags

        # Provenance verification
        assert tags["evaluation_type"] == "frozen_historical_holdout"
        assert tags["run_type"] == "baseline_historical_import"
        assert "provenance_task5_holdout_test" in tags

        # Separation verification
        assert "test_roc_auc" in metrics
        assert "test_pr_auc" in metrics
        assert "test_tuned_f1" in metrics
        assert "delta_roc_auc_test_minus_val" in metrics


# =====================================================================
# Tests: Artifact Download and Integrity
# =====================================================================

def test_artifact_download_and_integrity(mlflow_setup, tmp_path):
    """Test 6: Model artifacts and visualization charts can be downloaded and verified."""
    client, exp_id, run_ids, _ = mlflow_setup
    runs = client.search_runs(experiment_ids=[exp_id])
    pt_run = next(r for r in runs if r.data.tags["model_name"] == "PyTorch Neural Network")

    # Download model artifact
    downloaded_model = client.download_artifacts(pt_run.info.run_id, "model/pytorch_model_best.pt", dst_path=str(tmp_path))
    assert Path(downloaded_model).exists()
    assert Path(downloaded_model).stat().st_size > 1000

    # Download visualization artifact
    downloaded_viz = client.download_artifacts(pt_run.info.run_id, "visualizations/pytorch_loss_curves.png", dst_path=str(tmp_path))
    assert Path(downloaded_viz).exists()
    assert Path(downloaded_viz).stat().st_size > 1000


# =====================================================================
# Tests: Idempotent Registration
# =====================================================================

def test_idempotent_registration(mlflow_setup):
    """Test 7: Repeated execution does not duplicate baseline runs."""
    client, exp_id, initial_ids, tmp_dir = mlflow_setup

    # Execute second registration
    repeated_ids = register_all_baseline_runs(
        project_root=tmp_dir,
        client=client,
        experiment_id=exp_id,
    )

    # Verify run IDs are identical
    assert initial_ids == repeated_ids

    # Verify total count in experiment remains exactly 3
    runs = client.search_runs(experiment_ids=[exp_id])
    assert len(runs) == 3


# =====================================================================
# Tests: Argument Contract Validation
# =====================================================================

def test_argument_contract_client_without_experiment_id(mlflow_setup):
    """Verify passing client without experiment_id raises clear ValueError."""
    client, _, _, tmp_dir = mlflow_setup
    with pytest.raises(ValueError, match="Both 'client' and 'experiment_id' must be provided together"):
        register_all_baseline_runs(project_root=tmp_dir, client=client, experiment_id=None)


def test_argument_contract_experiment_id_without_client(mlflow_setup):
    """Verify passing experiment_id without client raises clear ValueError."""
    _, exp_id, _, tmp_dir = mlflow_setup
    with pytest.raises(ValueError, match="Both 'client' and 'experiment_id' must be provided together"):
        register_all_baseline_runs(project_root=tmp_dir, client=None, experiment_id=exp_id)


def test_argument_contract_both_omitted_calls_setup_mlflow(mlflow_setup, monkeypatch):
    """Verify omitting both client and experiment_id triggers setup_mlflow fallback."""
    client, exp_id, _, tmp_dir = mlflow_setup
    setup_called = False

    def mock_setup():
        nonlocal setup_called
        setup_called = True
        return client, exp_id

    monkeypatch.setattr("src.tracking.log_baselines.setup_mlflow", mock_setup)
    res = register_all_baseline_runs(project_root=tmp_dir, client=None, experiment_id=None)
    assert setup_called is True
    assert len(res) == 3


# =====================================================================
# Tests: Governance (No Retraining or Test Data Access)
# =====================================================================

def test_governance_no_retraining_or_test_access():
    """Test 8: AST check ensuring tracking modules do not call fit/train or load test data."""
    target_files = [
        Path("src/tracking/config.py"),
        Path("src/tracking/log_baselines.py"),
    ]

    forbidden_calls = {"fit", "fit_transform", "train_test_split", "train"}
    forbidden_tokens = ["X_test.parquet", "y_test.parquet", "predict_proba"]

    for file_path in target_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check method call (e.g. model.fit())
                if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_calls:
                    pytest.fail(f"Governance violation: '{node.func.attr}()' called in {file_path}")
                # Check function call (e.g. train())
                elif isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
                    pytest.fail(f"Governance violation: '{node.func.id}()' called in {file_path}")
            # Check string literals for test data references
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                for token in forbidden_tokens:
                    if token in node.value:
                        pytest.fail(f"Governance violation: '{token}' found in {file_path}")
