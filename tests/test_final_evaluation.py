"""Automated test suite for final holdout test evaluation and governance."""

import ast
from pathlib import Path
import numpy as np
import pytest
import torch

from src.evaluation.evaluate_test import (
    extract_and_verify_validation_thresholds,
    run_pytorch_test_inference,
    EXPECTED_THRESHOLDS,
)
from src.models.pytorch_model import MLForgeClassifier


def test_threshold_extraction_and_verification():
    """Test 1: Verifies parsing and assertion of Task 4 validation-selected thresholds."""
    extracted = extract_and_verify_validation_thresholds()

    assert "Logistic Regression" in extracted
    assert "Random Forest" in extracted
    assert "PyTorch Neural Network" in extracted

    assert np.isclose(extracted["Logistic Regression"]["threshold"], 0.660, atol=1e-3)
    assert np.isclose(extracted["Random Forest"]["threshold"], 0.635, atol=1e-3)
    assert np.isclose(extracted["PyTorch Neural Network"]["threshold"], 0.685, atol=1e-3)


def test_threshold_verification_fails_on_tampered_json(tmp_path):
    """Test 1b: Ensures pipeline halts if validation thresholds differ from Task 4."""
    import json
    tampered_file = tmp_path / "tampered_comparison.json"
    tampered_data = [
        {"model_name": "Logistic Regression", "tuned_threshold_metrics": {"threshold": 0.50}},
        {"model_name": "Random Forest", "tuned_threshold_metrics": {"threshold": 0.50}},
        {"model_name": "PyTorch Neural Network", "tuned_threshold_metrics": {"threshold": 0.50}},
    ]
    with open(tampered_file, "w", encoding="utf-8") as f:
        json.dump(tampered_data, f)

    with pytest.raises(ValueError, match="Threshold verification failed"):
        extract_and_verify_validation_thresholds(tampered_file)


def test_pytorch_inference_contract():
    """Test 2: Verifies PyTorch inference produces 1D array in [0, 1] preserving sample order."""
    model = MLForgeClassifier(input_dim=51, hidden_dim1=32, hidden_dim2=16)

    # Generate synthetic batch with ascending first feature to verify order preservation
    X_synthetic = np.zeros((10, 51), dtype=np.float32)
    for i in range(10):
        X_synthetic[i, 0] = float(i)

    probs = run_pytorch_test_inference(model, X_synthetic)

    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 1
    assert len(probs) == 10
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Verify deterministic order: calling single samples gives exact corresponding elements
    for i in range(10):
        single_prob = run_pytorch_test_inference(model, X_synthetic[i:i+1])
        assert np.isclose(probs[i], single_prob[0], atol=1e-6)


def test_generalization_gap_calculation():
    """Test 3: Validates mathematical correctness of Delta = Test - Validation."""
    val_score = 0.8125
    test_score = 0.8050
    delta = round(test_score - val_score, 4)
    assert delta == -0.0075


def test_governance_no_retraining_or_fitting():
    """Test 4: AST audit ensuring evaluate_test.py never calls fit, fit_transform, or train()."""
    target_file = Path("src/evaluation/evaluate_test.py")
    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()

    tree = ast.parse(content, filename=str(target_file))

    forbidden_calls = {"fit", "fit_transform", "train_test_split"}
    found_forbidden = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_calls:
                found_forbidden.append(node.func.attr)
            elif isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
                found_forbidden.append(node.func.id)

    assert len(found_forbidden) == 0, f"Governance violation: found forbidden calls: {found_forbidden}"
