"""Automated test suite for model training, architectures, evaluation metrics, and governance."""

import ast
from pathlib import Path
import numpy as np
import pytest
import torch

from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_threshold_independent_metrics,
    find_optimal_f1_threshold,
    benchmark_inference_latency,
)
from src.models.pytorch_model import MLForgeClassifier, TabularDataset
from src.models.train_pytorch import load_pytorch_checkpoint, set_seed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib


@pytest.fixture
def synthetic_batch() -> tuple:
    """Fixture returning synthetic X and y arrays for fast model unit tests."""
    np.random.seed(42)
    X = np.random.randn(50, 51).astype(np.float32)
    y = np.random.choice([0, 1], size=50, p=[0.8, 0.2]).astype(np.int64)
    return X, y


# =====================================================================
# Unit Tests: Output Shapes and Probability Ranges
# =====================================================================

def test_model_output_shapes_and_probability_ranges(synthetic_batch):
    """Test 1 & 2: Output shapes match sample count and probabilities are strictly in [0, 1]."""
    X, y = synthetic_batch
    n_samples = len(X)

    # 1. Logistic Regression
    lr = LogisticRegression(class_weight="balanced", random_state=42)
    lr.fit(X, y)
    lr_probs = lr.predict_proba(X)[:, 1]
    assert lr_probs.shape == (n_samples,)
    assert (lr_probs >= 0.0).all() and (lr_probs <= 1.0).all()

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X, y)
    rf_probs = rf.predict_proba(X)[:, 1]
    assert rf_probs.shape == (n_samples,)
    assert (rf_probs >= 0.0).all() and (rf_probs <= 1.0).all()

    # 3. PyTorch Model
    model = MLForgeClassifier(input_dim=51, hidden_dim1=32, hidden_dim2=16)
    model.eval()
    tensor_x = torch.tensor(X, dtype=torch.float32)
    logits = model(tensor_x)
    assert logits.shape == (n_samples, 1)

    pt_probs = model.predict_proba(X)
    assert pt_probs.shape == (n_samples,)
    assert (pt_probs >= 0.0).all() and (pt_probs <= 1.0).all()


# =====================================================================
# Unit Tests: Training Execution & Forward/Backward Pass
# =====================================================================

def test_pytorch_training_step(synthetic_batch):
    """Test 3: PyTorch forward pass, BCEWithLogitsLoss, backward pass, and optimizer step."""
    X, y = synthetic_batch
    dataset = TabularDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)

    model = MLForgeClassifier(input_dim=51, hidden_dim1=16, hidden_dim2=8)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    batch_x, batch_y = next(iter(loader))

    initial_param = next(model.parameters()).clone()

    optimizer.zero_grad()
    logits = model(batch_x)
    loss = criterion(logits, batch_y)
    assert loss.item() > 0.0
    loss.backward()
    optimizer.step()

    updated_param = next(model.parameters())
    # Weights should have updated after backward and step
    assert not torch.equal(initial_param, updated_param)


# =====================================================================
# Unit Tests: Reproducibility
# =====================================================================

def test_model_reproducibility(synthetic_batch):
    """Test 4: Training with identical random seeds produces identical outputs."""
    X, y = synthetic_batch

    # Test Scikit-learn LR
    lr1 = LogisticRegression(class_weight="balanced", random_state=42).fit(X, y)
    lr2 = LogisticRegression(class_weight="balanced", random_state=42).fit(X, y)
    np.testing.assert_allclose(lr1.predict_proba(X), lr2.predict_proba(X))

    # Test PyTorch initialization reproducibility
    set_seed(42)
    pt1 = MLForgeClassifier(input_dim=51, hidden_dim1=32, hidden_dim2=16)
    set_seed(42)
    pt2 = MLForgeClassifier(input_dim=51, hidden_dim1=32, hidden_dim2=16)

    for p1, p2 in zip(pt1.parameters(), pt2.parameters()):
        assert torch.equal(p1, p2)


# =====================================================================
# Unit Tests: Evaluation Metrics Correctness
# =====================================================================

def test_evaluation_metric_correctness():
    """Test 5: Validates exact calculation of classification and ranking metrics."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])

    # Threshold 0.5 -> preds = [0, 0, 1, 1] (perfect classification)
    metrics_50 = compute_classification_metrics(y_true, y_prob, threshold=0.50)
    assert metrics_50["accuracy"] == 1.0
    assert metrics_50["precision"] == 1.0
    assert metrics_50["recall"] == 1.0
    assert metrics_50["f1"] == 1.0
    assert metrics_50["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}

    # Threshold-independent metrics
    rank_metrics = compute_threshold_independent_metrics(y_true, y_prob)
    assert rank_metrics["roc_auc"] == 1.0
    assert rank_metrics["pr_auc"] == 1.0

    # Optimal threshold search
    opt_t, opt_f1 = find_optimal_f1_threshold(y_true, y_prob)
    assert np.isclose(opt_f1, 1.0)
    assert 0.4 < opt_t <= 0.6


# =====================================================================
# Unit Tests: Checkpoint Saving and Loading
# =====================================================================

def test_checkpoint_saving_and_loading(tmp_path, synthetic_batch):
    """Test 6: Verify serialization and reconstruction using map_location='cpu'."""
    X, _ = synthetic_batch
    model = MLForgeClassifier(input_dim=51, hidden_dim1=64, hidden_dim2=32)
    model.eval()

    probs_original = model.predict_proba(X)

    ckpt_path = tmp_path / "test_ckpt.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture_config": {
            "input_dim": 51,
            "hidden_dim1": 64,
            "hidden_dim2": 32,
            "dropout_rate": 0.3,
            "output_dim": 1,
        },
        "input_feature_count": 51,
        "preprocessing_artifact_reference": "models/preprocessor.joblib",
        "best_epoch": 5,
        "best_val_loss": 0.35,
        "training_config": {"learning_rate": 1e-3},
    }
    torch.save(checkpoint, ckpt_path)

    loaded_model, loaded_meta = load_pytorch_checkpoint(ckpt_path)
    probs_loaded = loaded_model.predict_proba(X)

    np.testing.assert_allclose(probs_original, probs_loaded, atol=1e-6)
    assert loaded_meta["best_epoch"] == 5
    assert loaded_meta["input_feature_count"] == 51


# =====================================================================
# Unit Tests: Validation-Only Governance Enforcement
# =====================================================================

def test_validation_only_governance_enforcement():
    """Test 7: Verify runtime data flow never loads or references X_test or y_test."""
    files_to_check = [
        Path("src/models/train_baselines.py"),
        Path("src/models/random_forest.py"),
        Path("src/models/train_pytorch.py"),
        Path("src/evaluation/compare_models.py"),
    ]

    forbidden_patterns = ["X_test.parquet", "y_test.parquet", "X_test", "y_test"]

    for file_path in files_to_check:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content, filename=str(file_path))

            # Inspect all string literals in the AST
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    for forbidden in forbidden_patterns:
                        assert forbidden not in node.value, (
                            f"Governance violation: '{forbidden}' found in {file_path}"
                        )
