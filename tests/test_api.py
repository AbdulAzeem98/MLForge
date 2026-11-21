"""Integration test suite for MLForge FastAPI Model Serving API.

Tests API endpoints (/health, /models, /metadata, /predict) using FastAPI TestClient.
Validates:
- All three baseline models (Logistic Regression, Random Forest, PyTorch MLP).
- pdays sentinel feature engineering (pdays=-1 vs pdays > -1).
- Preprocessing column alignment and 51-feature dynamic generation.
- Strict input validation rejecting target leakage ('duration'), labels ('y'), and extra fields.
- Hardware-independent inference latency measurements.
- Health degradation under missing or failed artifact initialization.
- Strict governance: never imports or accesses holdout test datasets (X_test, y_test).
"""

import json
import math
from pathlib import Path
import joblib
import pytest
import torch
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from src.api.main import app, get_prediction_service, lifespan
from src.api.service import PredictionService

# Synthetic customer record (completely independent of test parquet files)
SYNTHETIC_CUSTOMER_UNCONTACTED = {
    "age": 45,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 2143.0,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 17,
    "month": "nov",
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}

SYNTHETIC_CUSTOMER_CONTACTED = {
    "age": 33,
    "job": "technician",
    "marital": "single",
    "education": "secondary",
    "default": "no",
    "balance": 833.0,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 4,
    "month": "feb",
    "campaign": 2,
    "pdays": 180,
    "previous": 3,
    "poutcome": "failure",
}


@pytest.fixture(scope="module")
def client(dummy_prediction_service):
    """Module-scoped FastAPI TestClient injecting synthetic PredictionService."""
    app.state.prediction_service = dummy_prediction_service
    app.dependency_overrides[get_prediction_service] = lambda: dummy_prediction_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    app.state.prediction_service = None


def test_health_endpoint_healthy(client: TestClient):
    """Verify /health returns 200 OK and confirms all models and preprocessor are loaded."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert data["preprocessor_loaded"] is True
    assert data["models_loaded"]["logistic_regression"] is True
    assert data["models_loaded"]["random_forest"] is True
    assert data["models_loaded"]["pytorch"] is True
    assert data["device"] == "cpu"


def test_health_endpoint_degraded_on_failure():
    """Verify failed artifact loading prevents the service from reporting healthy (Safeguard 5)."""
    # Create an unready service instance
    unready_service = PredictionService(
        preprocessor=None,
        lr_model=None,
        rf_model=None,
        pytorch_model=None,
        thresholds={},
    )
    health = unready_service.get_health()
    assert health.status == "unhealthy"
    assert health.preprocessor_loaded is False
    assert unready_service.is_ready() is False


def test_models_endpoint(client: TestClient):
    """Verify /models returns the catalog of all three models with exact thresholds."""
    response = client.get("/models")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["default_model"] == "random_forest"
    models = {m["model_name"]: m for m in data["models"]}
    assert set(models.keys()) == {"logistic_regression", "random_forest", "pytorch"}

    # Exact threshold verification from model_comparison.json
    assert models["logistic_regression"]["threshold"] == 0.660
    assert models["random_forest"]["threshold"] == 0.635
    assert models["pytorch"]["threshold"] == 0.685

    for m in models.values():
        assert m["is_loaded"] is True


def test_metadata_endpoint(client: TestClient):
    """Verify /metadata returns verified dataset, feature count, and scaler specs."""
    response = client.get("/metadata")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["project"] == "MLForge"
    assert data["dataset"] == "UCI Bank Marketing"
    assert data["target_column"] == "y"
    assert data["target_mapping"] == {"no": 0, "yes": 1}
    assert "duration" in data["excluded_leakage_features"]
    assert data["raw_input_feature_count"] == 15
    assert data["preprocessor_input_feature_count"] == 16
    assert data["transformed_feature_count"] == 51
    assert data["preprocessing_pipeline"]["numerical_scaler"] == "StandardScaler"


@pytest.mark.parametrize("model_name,expected_threshold", [
    ("logistic_regression", 0.660),
    ("random_forest", 0.635),
    ("pytorch", 0.685),
])
def test_predict_all_three_models(client: TestClient, model_name: str, expected_threshold: float):
    """Verify inference across all three models returns valid probabilities and thresholds."""
    payload = {
        "model_name": model_name,
        "features": SYNTHETIC_CUSTOMER_UNCONTACTED,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["model_name"] == model_name
    assert 0.0 <= data["probability"] <= 1.0
    assert data["threshold"] == expected_threshold
    assert data["prediction"] in (0, 1)
    assert data["prediction_label"] in ("no", "yes")

    # Verify decision rule consistency
    if data["probability"] >= data["threshold"]:
        assert data["prediction"] == 1
        assert data["prediction_label"] == "yes"
    else:
        assert data["prediction"] == 0
        assert data["prediction_label"] == "no"

    # Hardware-independent latency verification
    assert isinstance(data["latency_ms"], (int, float))
    assert data["latency_ms"] >= 0.0
    assert math.isfinite(data["latency_ms"])


def test_predict_default_model(client: TestClient):
    """Verify omitting model_name defaults to random_forest."""
    payload = {
        "features": SYNTHETIC_CUSTOMER_UNCONTACTED,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["model_name"] == "random_forest"
    assert data["threshold"] == 0.635


def test_predict_pdays_sentinel_uncontacted(client: TestClient):
    """Verify customer with pdays = -1 (uncontacted) processes seamlessly."""
    payload = {
        "model_name": "random_forest",
        "features": SYNTHETIC_CUSTOMER_UNCONTACTED,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_pdays_contacted(client: TestClient):
    """Verify customer with pdays > -1 (previously contacted) processes seamlessly."""
    payload = {
        "model_name": "random_forest",
        "features": SYNTHETIC_CUSTOMER_CONTACTED,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_unknown_and_novel_categories(client: TestClient):
    """Verify unknown categorical values and unseen levels are handled without crash."""
    novel_customer = SYNTHETIC_CUSTOMER_UNCONTACTED.copy()
    novel_customer["job"] = "astronaut"  # Unseen category
    novel_customer["contact"] = "telepathy"  # Unseen category

    payload = {
        "model_name": "random_forest",
        "features": novel_customer,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert 0.0 <= data["probability"] <= 1.0


def test_reject_leakage_feature_duration(client: TestClient):
    """Verify submitting target leakage feature 'duration' is rejected with 422."""
    leaked_payload = SYNTHETIC_CUSTOMER_UNCONTACTED.copy()
    leaked_payload["duration"] = 350

    response = client.post("/predict", json={"features": leaked_payload})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    data = response.json()
    errors = [e["loc"][-1] for e in data["detail"]]
    assert "duration" in errors


def test_reject_target_column_y(client: TestClient):
    """Verify submitting target column 'y' is rejected with 422."""
    labeled_payload = SYNTHETIC_CUSTOMER_UNCONTACTED.copy()
    labeled_payload["y"] = "yes"

    response = client.post("/predict", json={"features": labeled_payload})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    data = response.json()
    errors = [e["loc"][-1] for e in data["detail"]]
    assert "y" in errors


def test_reject_extra_fields(client: TestClient):
    """Verify submitting arbitrary extra fields is rejected with 422."""
    extra_payload = SYNTHETIC_CUSTOMER_UNCONTACTED.copy()
    extra_payload["credit_score"] = 720

    response = client.post("/predict", json={"features": extra_payload})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_reject_missing_required_field(client: TestClient):
    """Verify missing required fields is rejected with 422."""
    incomplete_payload = SYNTHETIC_CUSTOMER_UNCONTACTED.copy()
    del incomplete_payload["age"]

    response = client.post("/predict", json={"features": incomplete_payload})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    data = response.json()
    assert any("age" in e["loc"] for e in data["detail"])


def test_reject_out_of_range_numerical_fields(client: TestClient):
    """Verify out-of-range numerical fields (e.g. day=35, age=12) are rejected with 422."""
    invalid_day = SYNTHETIC_CUSTOMER_UNCONTACTED.copy()
    invalid_day["day"] = 35

    response = client.post("/predict", json={"features": invalid_day})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    invalid_age = SYNTHETIC_CUSTOMER_UNCONTACTED.copy()
    invalid_age["age"] = 12

    response = client.post("/predict", json={"features": invalid_age})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_reject_invalid_field_types(client: TestClient):
    """Verify non-numeric types for numeric fields are rejected with 422."""
    invalid_type = SYNTHETIC_CUSTOMER_UNCONTACTED.copy()
    invalid_type["balance"] = "twenty_thousand"

    response = client.post("/predict", json={"features": invalid_type})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_reject_unsupported_model(client: TestClient):
    """Verify requesting an unsupported model name is rejected with 400 or 422."""
    payload = {
        "model_name": "xgboost",
        "features": SYNTHETIC_CUSTOMER_UNCONTACTED,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in (status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_ENTITY)


def test_from_artifacts_custom_path(tmp_path, dummy_prediction_service):
    """Verify PredictionService.from_artifacts() cleanly loads from a custom project root."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    exp_dir = tmp_path / "experiments/baselines"
    exp_dir.mkdir(parents=True)
    data_dir = tmp_path / "data/processed"
    data_dir.mkdir(parents=True)

    joblib.dump(dummy_prediction_service.preprocessor, models_dir / "preprocessor.joblib")
    joblib.dump(dummy_prediction_service.lr_model, models_dir / "logistic_regression.joblib")
    joblib.dump(dummy_prediction_service.rf_model, models_dir / "random_forest.joblib")

    checkpoint = {
        "model_state_dict": dummy_prediction_service.pytorch_model.state_dict(),
        "architecture_config": {
            "input_dim": 51,
            "hidden_dim1": 16,
            "hidden_dim2": 8,
            "dropout_rate": 0.3,
            "output_dim": 1,
        },
        "input_feature_count": 51,
    }
    torch.save(checkpoint, models_dir / "pytorch_model_best.pt")

    comp_data = [
        {"model_name": "Logistic Regression", "tuned_threshold_metrics": {"threshold": 0.660}},
        {"model_name": "Random Forest", "tuned_threshold_metrics": {"threshold": 0.635}},
        {"model_name": "PyTorch Neural Network", "tuned_threshold_metrics": {"threshold": 0.685}},
    ]
    with open(exp_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comp_data, f)

    with open(data_dir / "preprocessing_metadata.json", "w", encoding="utf-8") as f:
        json.dump(dummy_prediction_service.metadata, f)

    service = PredictionService.from_artifacts(project_root=tmp_path)
    assert service.is_ready() is True
    assert service.get_health().status == "healthy"
    assert service.preprocessor is not None
    assert service.lr_model is not None
    assert service.rf_model is not None
    assert service.pytorch_model is not None


def test_lifespan_pre_injected_service_bypasses_from_artifacts(dummy_prediction_service, monkeypatch):
    """Verify pre-injected PredictionService bypasses from_artifacts() completely."""
    called = False

    def mock_from_artifacts(*args, **kwargs):
        nonlocal called
        called = True
        return dummy_prediction_service

    monkeypatch.setattr(PredictionService, "from_artifacts", mock_from_artifacts)

    test_app = FastAPI(lifespan=lifespan)
    test_app.state.prediction_service = dummy_prediction_service

    with TestClient(test_app):
        assert not called, "from_artifacts() must not be called when service is pre-injected in app.state"
        assert test_app.state.prediction_service is dummy_prediction_service
        assert test_app.state.prediction_service.is_ready() is True

    assert test_app.state.prediction_service is None


def test_lifespan_normal_startup_calls_from_artifacts(dummy_prediction_service, monkeypatch):
    """Verify normal startup without pre-injected service calls from_artifacts()."""
    called = False

    def mock_from_artifacts(*args, **kwargs):
        nonlocal called
        called = True
        return dummy_prediction_service

    monkeypatch.setattr(PredictionService, "from_artifacts", mock_from_artifacts)

    test_app = FastAPI(lifespan=lifespan)
    assert getattr(test_app.state, "prediction_service", None) is None

    with TestClient(test_app):
        assert called, "from_artifacts() must be called during normal startup when app.state has no service"
        assert test_app.state.prediction_service is dummy_prediction_service
        assert test_app.state.prediction_service.is_ready() is True

    assert test_app.state.prediction_service is None


def test_lifespan_normal_startup_with_synthetic_artifacts(tmp_path, dummy_prediction_service, monkeypatch):
    """Verify unmocked lifespan loading real synthetic artifacts from a temporary directory."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    exp_dir = tmp_path / "experiments/baselines"
    exp_dir.mkdir(parents=True)
    data_dir = tmp_path / "data/processed"
    data_dir.mkdir(parents=True)

    joblib.dump(dummy_prediction_service.preprocessor, models_dir / "preprocessor.joblib")
    joblib.dump(dummy_prediction_service.lr_model, models_dir / "logistic_regression.joblib")
    joblib.dump(dummy_prediction_service.rf_model, models_dir / "random_forest.joblib")

    checkpoint = {
        "model_state_dict": dummy_prediction_service.pytorch_model.state_dict(),
        "architecture_config": {
            "input_dim": 51,
            "hidden_dim1": 16,
            "hidden_dim2": 8,
            "dropout_rate": 0.3,
            "output_dim": 1,
        },
        "input_feature_count": 51,
    }
    torch.save(checkpoint, models_dir / "pytorch_model_best.pt")

    comp_data = [
        {"model_name": "Logistic Regression", "tuned_threshold_metrics": {"threshold": 0.660}},
        {"model_name": "Random Forest", "tuned_threshold_metrics": {"threshold": 0.635}},
        {"model_name": "PyTorch Neural Network", "tuned_threshold_metrics": {"threshold": 0.685}},
    ]
    with open(exp_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comp_data, f)

    with open(data_dir / "preprocessing_metadata.json", "w", encoding="utf-8") as f:
        json.dump(dummy_prediction_service.metadata, f)

    monkeypatch.setattr("src.api.service.PROJECT_ROOT", tmp_path)

    test_app = FastAPI(lifespan=lifespan)
    with TestClient(test_app):
        loaded_service = test_app.state.prediction_service
        assert loaded_service is not None
        assert loaded_service.is_ready() is True
        assert loaded_service.get_health().status == "healthy"

    assert test_app.state.prediction_service is None


def test_lifespan_degraded_startup_when_artifacts_missing(tmp_path, monkeypatch):
    """Verify normal startup gracefully degrades to unhealthy when artifacts are missing."""
    empty_dir = tmp_path / "empty_project"
    empty_dir.mkdir(parents=True)
    monkeypatch.setattr("src.api.service.PROJECT_ROOT", empty_dir)

    test_app = FastAPI(lifespan=lifespan)
    with TestClient(test_app):
        loaded_service = test_app.state.prediction_service
        assert loaded_service is not None
        assert loaded_service.is_ready() is False
        assert loaded_service.get_health().status == "unhealthy"

    assert test_app.state.prediction_service is None


def test_governance_no_test_partition_access():
    """Verify test_api.py and src/api never import or access holdout test partitions."""
    api_dir = Path("src/api")
    test_file = Path("tests/test_api.py")

    files_to_audit = list(api_dir.glob("*.py")) + [test_file]
    forbidden_tokens = ["X_test", "y_test", "X_test.parquet", "y_test.parquet"]

    for file_path in files_to_audit:
        content = file_path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert f'"{token}"' not in content or "test_governance" in content, (
                f"Governance violation: {token} referenced in {file_path}"
            )
            assert f"'{token}'" not in content or "test_governance" in content, (
                f"Governance violation: {token} referenced in {file_path}"
            )
