# MLForge

An end-to-end, production-grade Machine Learning engineering and inference platform built for CPU environments.

[![CI](https://github.com/AbdulAzeem98/MLForge/actions/workflows/ci.yml/badge.svg)](https://github.com/AbdulAzeem98/MLForge/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch CPU](https://img.shields.io/badge/PyTorch-2.15.0%2Bcpu-red.svg)](https://download.pytorch.org/whl/cpu)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.165.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-78%20passed-brightgreen.svg)](#automated-testing--cicd-pipeline)
[![Docker](https://img.shields.io/badge/Docker-Hardened%20Multi--Stage-2496ED.svg)](#production-docker-containerization)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Executive Overview

**MLForge** is an enterprise-grade Machine Learning system solving financial term-deposit conversion prediction on the UCI Bank Marketing dataset (45,211 customers). Rather than treating machine learning as an isolated modeling experiment, MLForge demonstrates complete ML engineering lifecycle rigor:

- **Strict Data Governance & Leakage Prevention:** Explicit elimination of prospective target leakage (`duration`), automated data quality gates, and stratified splitting with a sealed holdout test partition.
- **Cost-Sensitive Decision Threshold Optimization:** Rejection of synthetic oversampling distortion (e.g. SMOTE) in favor of validation-tuned F1 decision thresholds under severe class imbalance (11.7% conversion).
- **Multi-Model Architectural Comparison:** Standardized training, tuning, and evaluation across Scikit-Learn Logistic Regression, Random Forest, and a custom PyTorch Deep Neural Network (MLP).
- **Auditable Experiment Tracking:** Local SQLite-backed MLflow tracking server capturing model hyper-parameters, diagnostic curves (ROC, PR, training loss), and checkpoint artifacts.
- **Production FastAPI Serving Engine:** Asynchronous ASGI inference service with application-scoped lifecycle management, strict Pydantic v2 schemas (`extra="forbid"`), and graceful HTTP 503 degraded-mode resilience.
- **Hardened Multi-Stage Containerization:** Security-hardened Docker deployment featuring dependency closure isolation via `uv`, non-root execution (`UID:GID 10001:10001`), read-only root filesystem, decoupled model artifact mounting, and cryptographic SHA-256 integrity verification.

---

## End-to-End System Architecture

```text
                                         MLFORGE END-TO-END PIPELINE
                                         
   +---------------------------------------------------------------------------------------------------+
   |  1. INGESTION & DATA GOVERNANCE                                                                   |
   |     UCI Bank Marketing (45,211 rows) ---> SHA-256 Checksum Validation ---> Data Quality Suite     |
   |     LEAKAGE GATE: 'duration' permanently stripped from prospective feature space                  |
   +---------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
   +---------------------------------------------------------------------------------------------------+
   |  2. STRATIFIED PARTITIONING & FEATURE PIPELINE                                                    |
   |     Stratified 70/15/15 Split (y=11.7%) -> Train (31,647) | Val (6,782) | Holdout Test (6,782)    |
   |     Domain Transforms: pdays sentinel flag (-1 vs contacted) + median imputation                  |
   |     ColumnTransformer (fitted strictly on X_train): RobustScaler + OneHotEncoder -> 51 Features  |
   +---------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
   +---------------------------------------------------------------------------------------------------+
   |  3. MULTI-MODEL TRAINING & VALIDATION THRESHOLD TUNING                                            |
   |     * Logistic Regression (L-BFGS, balanced weights)           -> Tuned Threshold: 0.660          |
   |     * Random Forest (100 trees, balanced subsampling)          -> Tuned Threshold: 0.635          |
   |     * PyTorch Neural Network (51->64->32->1, BatchNorm, Adam)  -> Tuned Threshold: 0.685          |
   |     MLflow SQLite Tracking (mlruns.db) -> Parameters, Metrics, ROC/PR Curves, Checkpoints         |
   +---------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
   +---------------------------------------------------------------------------------------------------+
   |  4. FROZEN HOLDOUT EVALUATION & AUDIT                                                             |
   |     Unsealed holdout test set evaluated strictly with frozen models and validation thresholds     |
   |     Generalization gap: |Test - Val| <= 0.014 across all models, confirming zero data leakage    |
   |     Cryptographic Manifest: SHA-256 hashes generated for all 4 model artifacts (~7.1 MB)          |
   +---------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
   +---------------------------------------------------------------------------------------------------+
   |  5. PRODUCTION INFERENCE & HARDENED CONTAINERIZATION                                              |
   |     FastAPI ASGI Server (src.api.main:app) with Lifespan DI & Pydantic v2 schemas                 |
   |     Degraded Mode Fallback (HTTP 503) if artifacts absent; 11-60ms inference latency              |
   |     Multi-Stage Dockerfile (uv 0.13.1, non-root UID 10001, --read-only, tmpfs, dropped caps)     |
   +---------------------------------------------------------------------------------------------------+
```

---

## Problem Formulation & Governance Decisions

### 1. The Business Problem
The objective is predicting whether a banking client will subscribe to a long-term deposit (`y = "yes" / "no"`). In direct marketing campaigns, outbound call capacity is strictly constrained. False positives squander outbound sales agent time on uninterested prospects, while false negatives miss valuable deposit capital.

### 2. The Target Leakage Trap (`duration`)
During initial exploratory analysis, the feature `duration` (last call duration in seconds) displayed overwhelming correlation with deposit conversion (ROC-AUC > 0.85 alone). However, **call duration is intrinsically unknown prior to placing the call**. Including `duration` introduces fatal prospective target leakage:
- A model trained with `duration` learns that long calls indicate success, but real-time pre-call scoring systems cannot know call duration beforehand.
- **Architectural Enforcement:** `duration` was permanently excluded during preprocessing, was omitted from the fitted feature schema, and is explicitly rejected with `HTTP 422 Unprocessable Content` by Pydantic API schemas at the serving boundary.

### 3. Sentinel Feature Engineering (`pdays`)
In the raw data, 81.74% of records had `pdays = -1` (client was not previously contacted). Treating `-1` as a linear numerical feature corrupts distance-based scalers and linear weights:
- Engineered a binary indicator: `is_previously_contacted = (pdays != -1)`.
- Engineered an imputed recency feature: `pdays_imputed`, setting sentinels to the median of previously contacted clients.
- Applied logarithmic transforms (`log_campaign`, `log_previous`) to mitigate extreme right-skewed call frequency distributions.

---

## Dataset & Preprocessing Pipeline

- **Source Dataset:** UCI Bank Marketing (`data/raw/bank-full.csv`, 45,211 rows, 16 raw input features).
- **Splitting Strategy:** 70/15/15 stratified train/val/test split on `y` (preserving the 11.69%–11.71% positive ratio):
  - **Training Set:** 31,647 records (3,702 positive)
  - **Validation Set:** 6,782 records (793 positive)
  - **Holdout Test Set:** 6,782 records (793 positive) — strictly sealed until final evaluation.
- **Transformation Pipeline:** Scikit-Learn `ColumnTransformer` fitted **strictly** on the training partition:
  - **Numerical Features (7 columns):** Imputed with median, scaled using `RobustScaler` (mitigating outlier influence).
  - **Categorical Features (9 columns):** Imputed with `"missing"`, encoded with `OneHotEncoder(handle_unknown='ignore')`.
  - **Output Representation:** 51 continuous/one-hot dimensions exported to optimized Parquet format (`X_train.parquet`, `X_val.parquet`, `X_test.parquet`).
  - **Serialization:** Preprocessor pipeline persisted to `models/preprocessor.joblib` (5,919 bytes).

---

## Model Architectures & Validation Threshold Tuning

### Why Tune Decision Thresholds?
In imbalanced classification (11.7% positive class), the standard decision threshold of `0.500` is arbitrary and suboptimal. Lower thresholds produce excessive false alarms, while default thresholds fail to capture minority conversions. Synthetic oversampling methods like SMOTE introduce synthetic artifacts into feature distributions.

Instead, MLForge trains models using balanced loss functions, then performs a post-training grid search over $\tau \in [0.05, 0.95]$ on the validation set to find the **F1-maximizing decision threshold**.

### Model Implementations
1. **Logistic Regression:** L-BFGS solver, `class_weight='balanced'`, `max_iter=1000`.
2. **Random Forest Classifier:** 100 estimators, `class_weight='balanced_subsample'`, `min_samples_leaf=5`, constrained tree depth.
3. **PyTorch Deep Neural Network (MLP):**
   - Architecture: `51 -> 64 -> 32 -> 1` fully connected layers with `BatchNorm1d`, `ReLU`, and `Dropout` (0.3 / 0.2).
   - Loss Function: `BCEWithLogitsLoss` with positive class weighting (`pos_weight = 7.55`).
   - Optimization: Adam optimizer (`lr=0.001`), `ReduceLROnPlateau` scheduler, and `EarlyStopping` (patience=10, min_delta=0.0001, best epoch 11 of 21).

### Validation Optimization Results

| Model Architecture | Default Thresh (0.50) F1 | Tuned Threshold ($\tau$) | Tuned Val F1 | Val Accuracy (Default $\to$ Tuned) | Val ROC-AUC | Val PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.3778 | **0.660** | **0.4490** | 75.72% $\to$ **87.81%** | 0.7720 | 0.4162 |
| **Random Forest** | 0.4435 | **0.635** | **0.4849** | 81.27% $\to$ **87.94%** | 0.7977 | 0.4621 |
| **PyTorch Neural Network** | 0.4199 | **0.685** | **0.4887** | 79.02% $\to$ **87.63%** | 0.7996 | 0.4455 |

*Impact:* Post-training threshold tuning boosted F1-score by **+0.041 to +0.071** and overall accuracy from ~79% to nearly **88%**, suppressing hundreds of costly false positives while preserving recall.

---

## Sealed Holdout Evaluation & Generalization Analysis

Following validation tuning, models and thresholds were frozen. The holdout test set (6,782 records, 793 positives) was unsealed and evaluated in a single pass without retraining:

### Holdout Test Performance (At Tuned Thresholds)

| Metric | Logistic Regression ($\tau=0.660$) | Random Forest ($\tau=0.635$) | PyTorch Neural Net ($\tau=0.685$) |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | 87.61% | 87.70% | 87.56% |
| **Test Precision** | 0.4661 | 0.4741 | 0.4692 |
| **Test Recall** | 0.4073 | 0.4741 | 0.4905 |
| **Test F1-Score** | **0.4347** | **0.4741** | **0.4797** |
| **Test ROC-AUC** | **0.7682** | **0.7914** | **0.7856** |
| **Test PR-AUC** | **0.4147** | **0.4512** | **0.4311** |
| **Generalization $\Delta$ (ROC-AUC)** | **-0.0038** | **-0.0063** | **-0.0140** |
| **Generalization $\Delta$ (F1 Tuned)** | **-0.0143** | **-0.0108** | **-0.0090** |

$$\text{Generalization Gap } \Delta = \text{Metric}_{\text{Test}} - \text{Metric}_{\text{Validation}}$$

Across all three architectures, the generalization gap remained strictly bounded ($|\Delta| \le 0.014$). This confirms absence of data leakage, absence of threshold over-fitting, and stability under real-world holdout distributions.

---

## MLflow Experiment Tracking & Lineage

MLForge uses an embedded, local SQLite tracking database and file artifact repository to ensure 100% auditable experiment provenance without cloud subscriptions:

- **Tracking URI:** `sqlite:///D:/MLForge/mlruns.db`
- **Artifact Store:** `file:///D:/MLForge/mlartifacts/1`
- **Experiment Name:** `MLForge_Bank_Marketing` (Experiment ID: `1`)
- **Compatibility Pinning:** Pinned `sqlalchemy>=2.0,<2.1` (`2.0.58`) to resolve upstream SQLite dialect compatibility with `mlflow==2.22.5`.

### Registered Baseline Model Runs

| Architecture | MLflow Run ID | Status | Tracked Artifacts |
| :--- | :--- | :---: | :--- |
| **Logistic Regression** | `14b87d2fb2cc4f2a9a737ff7eae0360b` | Finished | Hyperparameters, confusion matrices, ROC/PR curves, model binary |
| **Random Forest** | `54ca60bc9ecd4bbd812da94483c02caf` | Finished | Hyperparameters, confusion matrices, ROC/PR curves, model binary |
| **PyTorch MLP** | `d00c8c6552b04a7aa9a7e5e4fb74bf24` | Finished | Checkpoint dictionary, loss curves, ROC/PR curves, state dict |

---

## Production FastAPI Inference Serving

The inference engine (`src/api/main.py`) exposes high-throughput, low-latency RESTful inference over HTTP:

### Architectural Features
- **Application-Scoped Lifecycle:** `PredictionService` is initialized during FastAPI `lifespan` startup, pre-loading artifacts into memory and injecting instances via `Depends(get_prediction_service)`. Eliminates global state and prevents per-request disk reads.
- **Multi-Model Routing:** Route inference dynamically to `logistic_regression`, `random_forest` (default), or `pytorch`.
- **Dynamic Preprocessing:** Real-time alignment of 15 raw features into 51 transformed features matching the fitted preprocessor pipeline.
- **Strict Pydantic v2 Contracts:** Rejects data leakage attributes (`duration`), labels (`y`), and arbitrary fields using `ConfigDict(extra="forbid")`.
- **Latency Profiling:** Dynamic per-request latency calculation (`latency_ms`), typically 11–60 ms on commodity CPU.
- **Degraded-Mode Resilience:** If model artifacts are missing or corrupted, the service does not crash; it logs internal exceptions and returns controlled `HTTP 503 Service Unavailable` with sanitized client responses across all endpoints.

### API Routes Specification

| Method | Endpoint | Description | Healthy Status | Degraded Status |
| :---: | :--- | :--- | :---: | :---: |
| `GET` | `/health` | Health & model readiness probe | `HTTP 200 OK` | `HTTP 503 Service Unavailable` |
| `GET` | `/models` | Catalog of registered models and decision thresholds | `HTTP 200 OK` | `HTTP 503 Service Unavailable` |
| `GET` | `/metadata` | Pipeline metadata, target mappings, excluded features | `HTTP 200 OK` | `HTTP 503 Service Unavailable` |
| `POST` | `/predict` | Executes model inference against customer attributes | `HTTP 200 OK` | `HTTP 503 Service Unavailable` |
| `GET` | `/docs` | Interactive Swagger OpenAPI documentation | `HTTP 200 OK` | `HTTP 200 OK` |

---

## Model Artifact Integrity Manifest

To prevent model poisoning, accidental weight corruption, or unauthorized checkpoint modification, MLForge maintains a deterministic cryptographic SHA-256 manifest:

- **Manifest File:** [`experiments/baselines/model_artifact_checksums.json`](experiments/baselines/model_artifact_checksums.json)
- **Hash Algorithm:** `SHA-256`
- **Schema Version:** `1.0.0`

### Artifact Checksum Table

| Relative Path | Exact Size | SHA-256 Digest |
| :--- | :---: | :--- |
| `models/preprocessor.joblib` | 5,919 bytes | `acd4e8fcab320ecd28112c3096185e08d6c5d5c1bafa7f14fed8014e9baef0a2` |
| `models/logistic_regression.joblib` | 2,255 bytes | `fe5a5f1d69d9a257209eab9fd4fa2e7a05b61da66cc38a55f575f698f9f9f31e` |
| `models/random_forest.joblib` | 7,022,057 bytes | `0b816250a9e64abe696b8176b61ad1690a1038e6886e5026fa490c28b2681766` |
| `models/pytorch_model_best.pt` | 63,769 bytes | `1b301287931468b162f6d072d60709ee77ad911603ff2d1699b25e5c91354692` |

---

## Production Docker Containerization

MLForge provides a hardened, multi-stage production container designed according to defense-in-depth principles:

### 1. Multi-Stage Dockerfile Architecture
- **Stage 1 (Builder):** Uses `python:3.11-slim-bookworm` and copies the official `uv:0.13.1` binary. Compiles locked serving dependencies strictly from `requirements-serve.lock` into an isolated virtual environment (`/opt/venv`). Offline training libraries (MLflow, Optuna, Matplotlib, Seaborn) are stripped, slashing image size.
- **Stage 2 (Runtime):** Clean Debian Bookworm runtime image copying only `/opt/venv`, application source (`src/`), and validation thresholds (`experiments/baselines/model_comparison.json`).
- **Non-Root Execution:** Runs under dedicated system user and group `mlforge` (`UID:GID 10001:10001`).
- **Filesystem Immutability:** Runs with read-only root (`--read-only`), read-only model artifact mount (`/app/models:ro`), and memory-backed temporary scratch space (`--tmpfs /tmp`).
- **Internal Probe:** Native Python `urllib` healthcheck targeting `http://127.0.0.1:8000/health`.

### 2. Build Context vs. Final Image Boundaries

- **Control Files Transmitted to Builder Daemon:** `Dockerfile` and `.dockerignore` are read as engine control files; they are **not** copied into the runtime image.
- **Admitted into Build Context by `.dockerignore`:** `requirements-serve.*`, `requirements.*`, `src/`, `experiments/baselines/model_comparison.json`.
- **Copied into Runtime Image by `Dockerfile`:** `/opt/venv` (from builder), `/app/src/`, `/app/experiments/baselines/model_comparison.json`.
- **Excluded Entirely from Build Context:** `data/` (all raw and holdout datasets), `models/` (all model weights), `.git/`, `.venv/`, `mlruns/`, `mlartifacts/`, `tests/`, and secrets.

---

## Project Structure

```text
MLForge/
├── .github/
│   └── workflows/
│       └── ci.yml                      # Automated GitHub Actions CI workflow
├── configs/                            # Configuration files and placeholders
├── data/                               # Data directory (excluded from git/docker)
│   ├── raw/
│   │   ├── bank-full.csv               # UCI Bank Marketing dataset (45,211 rows)
│   │   └── ingestion_metadata.json     # Ingestion shape and SHA-256 hash
│   └── processed/
│       ├── split_indices.json          # Stratified split row indices
│       ├── preprocessing_metadata.json # 51 feature names and scaler metadata
│       ├── X_train.parquet             # Training features (31,647 x 51)
│       ├── y_train.parquet             # Training labels (31,647)
│       ├── X_val.parquet               # Validation features (6,782 x 51)
│       ├── y_val.parquet               # Validation labels (6,782)
│       ├── X_test.parquet              # Holdout test features (6,782 x 51)
│       └── y_test.parquet              # Holdout test labels (6,782)
├── experiments/
│   ├── eda/                            # EDA reports, distributions, correlation plots
│   ├── baselines/                      # Model comparison JSON, PR/ROC curves, loss plots
│   │   ├── model_comparison.json       # Validation metrics and tuned decision thresholds
│   │   └── model_artifact_checksums.json # Cryptographic SHA-256 artifact manifest
│   └── final_evaluation/               # Holdout evaluation metrics and generalization report
├── mlartifacts/                        # Local MLflow artifact store
├── models/                             # Trained model binaries (git-ignored, mounted at runtime)
│   ├── preprocessor.joblib             # Fitted Scikit-Learn ColumnTransformer pipeline
│   ├── logistic_regression.joblib      # Trained Logistic Regression baseline
│   ├── random_forest.joblib            # Trained Random Forest baseline
│   └── pytorch_model_best.pt           # PyTorch deep neural network checkpoint
├── src/                                # Core application source code
│   ├── api/                            # FastAPI inference service
│   │   ├── main.py                     # ASGI application, routes, and lifespan manager
│   │   ├── schemas.py                  # Pydantic v2 validation contracts (extra="forbid")
│   │   └── service.py                  # PredictionService lifecycle and model routing
│   ├── data/                           # Ingestion, validation, and stratified splitting
│   ├── features/                       # Feature engineering and preprocessing pipelines
│   ├── models/                         # Model architectures and training pipelines
│   ├── evaluation/                     # Metric calculations and holdout evaluation
│   ├── tracking/                       # MLflow tracking configuration and logger
│   └── utils/                          # Shared utilities and logging
├── tests/                              # Comprehensive test suite (78 tests, all passing)
│   ├── test_api.py                     # API integration tests & degraded-mode tests (25 tests)
│   ├── test_data_ingestion.py          # Data ingestion and schema validation (10 tests)
│   ├── test_environment.py             # Environment and CPU PyTorch verification (12 tests)
│   ├── test_final_evaluation.py        # Holdout evaluation contracts (5 tests)
│   ├── test_models.py                  # Model shapes and checkpoint integrity (6 tests)
│   ├── test_preprocessing.py           # Split disjointness and feature transforms (9 tests)
│   └── test_tracking.py                # MLflow tracking server and artifact tests (11 tests)
├── .dockerignore                       # Strict default-deny Docker build context allow-list
├── Dockerfile                          # Multi-stage production Dockerfile
├── check_environment.py                # Standalone environment diagnostic script
├── mlruns.db                           # SQLite database backing MLflow tracking
├── pytest.ini                          # Pytest configuration
├── requirements.txt                    # Project dependencies (CPU PyTorch pin)
├── requirements.lock                   # Fully resolved uv lockfile for full dependencies
├── requirements-serve.txt              # Serving-only dependency specification
├── requirements-serve.lock             # Frozen serving-only dependency lockfile
├── PROJECT_STATUS.md                   # Detailed project status and handover document
└── README.md                           # This documentation
```

---

## Quickstart & Verification Guide

### 1. Prerequisites
- **Operating System:** Windows 10/11, Linux, or macOS.
- **Python:** Version 3.11 (64-bit).
- **Package Manager:** `uv` recommended (or standard `pip`).
- **Hardware:** Commodity CPU (No NVIDIA GPU or CUDA required).

### 2. Environment Setup

```powershell
# Clone the repository
git clone https://github.com/AbdulAzeem98/MLForge.git
cd MLForge

# Create Python 3.11 virtual environment via uv
uv venv .venv --python 3.11

# Activate the virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# Install locked dependencies (using CPU PyTorch wheel)
uv pip install -r requirements.lock
```

### 3. Run Automated Tests

Execute the full automated test suite (78 tests):

```powershell
python -m pytest -v
```

Execute API integration tests only (25 tests):

```powershell
python -m pytest tests/test_api.py -v
```

### 4. Verify Model Artifact Checksums

Verify the cryptographic SHA-256 digests and file sizes of all four model artifacts against the integrity manifest:

```powershell
python -c '
import hashlib, json, pathlib
manifest = json.loads(pathlib.Path("experiments/baselines/model_artifact_checksums.json").read_text(encoding="utf-8"))
for item in manifest["artifacts"]:
    p = pathlib.Path(item["path"])
    raw = p.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    size = len(raw)
    expected_size = item["size_bytes"]
    expected_digest = item["sha256"]
    assert size == expected_size, f"Size mismatch on {p}: {size} != {expected_size}"
    assert digest == expected_digest, f"Digest mismatch on {p}: {digest} != {expected_digest}"
    print(f"Verified {p}: {size} bytes (SHA-256: {digest[:16]}...)")
print("All model artifacts verified successfully (size and SHA-256 match).")
'
```

### 5. Launch the Local Serving API

Start the FastAPI application via Uvicorn:

```powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

- **Swagger OpenAPI Docs:** Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser.
- **Health Check:** `curl http://127.0.0.1:8000/health`
- **Model Catalog:** `curl http://127.0.0.1:8000/models`

### 6. Submit a Prediction Request

Submit a prospective customer record using PowerShell:

```powershell
$headers = @{ "Content-Type" = "application/json" }
$body = @'
{
  "model_name": "random_forest",
  "features": {
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
    "poutcome": "unknown"
  }
}
'@

Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -Headers $headers -Body $body
```

**Expected Response (`HTTP 200 OK`):**
```json
{
  "model_name": "random_forest",
  "probability": 0.3812,
  "threshold": 0.635,
  "prediction": 0,
  "prediction_label": "no",
  "latency_ms": 40.72
}
```

*(Note: Target leakage feature `duration` and ground truth `y` are strictly forbidden by the schema; submitting them will trigger `HTTP 422 Unprocessable Content`).*

### 7. Launch the MLflow Tracking Dashboard

Explore tracked runs, parameters, curves, and artifacts in the MLflow UI:

```powershell
mlflow ui --backend-store-uri sqlite:///D:/MLForge/mlruns.db --port 5000 --host 127.0.0.1
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Production Docker Deployment

### 1. Build the Docker Image

Build the container image from the repository root:

```bash
docker build -t mlforge-inference:latest .
```

### 2. Run the Hardened Container

Run the container using Windows PowerShell with security hardening flags and explicit `--mount`:

```powershell
docker run -d `
  --name mlforge-api `
  -p 8000:8000 `
  --mount "type=bind,source=$($PWD.Path)\models,target=/app/models,readonly" `
  --read-only `
  --tmpfs /tmp:rw,noexec,nosuid,size=128m `
  --cap-drop ALL `
  --security-opt no-new-privileges:true `
  --restart on-failure:5 `
  mlforge-inference:latest
```

*(Single-line equivalent for Linux/bash: `docker run -d --name mlforge-api -p 8000:8000 --mount "type=bind,source=$(pwd)/models,target=/app/models,readonly" --read-only --tmpfs /tmp:rw,noexec,nosuid,size=128m --cap-drop ALL --security-opt no-new-privileges:true --restart on-failure:5 mlforge-inference:latest`)*

#### Security Flags Breakdown:
- `--mount "type=bind,source=...,target=/app/models,readonly"`: Mounts host `models/` read-only, preventing in-container model tampering or poisoning.
- `--read-only`: Mounts the container root filesystem as strictly read-only.
- `--tmpfs /tmp:rw,noexec,nosuid,size=128m`: Mounts a 128 MB RAM-backed writable scratch space for temporary runtime buffers, with binary execution blocked (`noexec`).
- `--cap-drop ALL`: Strips all Linux kernel capabilities from the container.
- `--security-opt no-new-privileges:true`: Prevents privilege escalation via setuid/setgid binaries.
- `--restart on-failure:5`: Automatically restarts on process crashes up to 5 times.

### 3. Container Management & Troubleshooting

```powershell
# Inspect container health probe status
docker inspect --format='{{json .State.Health}}' mlforge-api

# Stream real-time container logs
docker logs --tail 100 -f mlforge-api

# Stop and remove the container
docker stop mlforge-api
docker rm mlforge-api
```

- **HTTP 503 / Degraded Status:** If the container starts but returns HTTP 503, verify that the volume mount points to the correct directory containing all four verified artifact files.
- **Port Conflict (8000 already bound):** Remap the host port: `-p 8080:8000`.
- **Operational Disclaimer:** Standalone functional serving verification has been validated via in-process ASGI testing on Windows. Live container execution remains pending testing on a Docker-capable host.

---

## Automated Testing & CI/CD Pipeline

MLForge incorporates a comprehensive automated test suite of **78 tests** passing with 0 failures:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.17, pytest-8.4.2, pluggy-1.7.0
rootdir: D:\MLForge
collected 78 items

tests\test_api.py .........................                              [ 32%]
tests\test_data_ingestion.py ..........                                  [ 44%]
tests\test_environment.py ............                                   [ 60%]
tests\test_final_evaluation.py .....                                     [ 66%]
tests\test_models.py ......                                              [ 74%]
tests\test_preprocessing.py .........                                    [ 85%]
tests\test_tracking.py ...........                                       [100%]

====================== 78 passed, 14 warnings in 15.96s =======================
```

### Test Coverage Highlights
- **API Tests (25 tests):** Validation of `/health`, `/models`, `/metadata`, `/predict`; routing across all 3 architectures; dynamic column alignment; `pdays` sentinel feature engineering; strict schema rejection of `duration` and `y`; and permanent degraded-mode integration testing (`test_degraded_mode_all_endpoints_return_503`).
- **Data Ingestion Tests (10 tests):** Schema integrity, SHA-256 raw file validation, sentinel handling, and leakage checks.
- **Environment Tests (12 tests):** Python 3.11 verification, PyTorch CPU wheel enforcement, and dependency tree validation.
- **Preprocessing Tests (9 tests):** Stratified split disjointness, no test-set contamination, and 51-feature transformation shape verification.
- **Modeling Tests (6 tests):** Output tensor shapes, reproducibility seeds, and checkpoint serialization contracts.
- **Final Evaluation Tests (5 tests):** Frozen threshold enforcement, holdout metric validation, and generalization gap limits.
- **Tracking Tests (11 tests):** SQLite tracking URI initialization, artifact storage, and idempotent baseline run registration.

### Continuous Integration (GitHub Actions)
The repository includes a GitHub Actions CI workflow ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) running on `ubuntu-latest`:
- Checks out code and installs Python 3.11.
- Installs `uv` (0.13.1) and provisions virtual environment with cached dependencies from `requirements.lock`.
- Validates dependency tree consistency (`uv pip check`).
- Executes environment diagnostic check (`check_environment.py`).
- Runs the complete pytest suite with test coverage reporting (`pytest --cov=src`).

---

## Key Takeaways for Technical Interviewers

1. **Production Mindset over Notebook Experimentation:** Built as an engineered modular platform with strict separation of concerns, automated test coverage, and reproducible lockfiles.
2. **Defensive ML Governance:** Proactively audited and removed `duration` target leakage. Partitioned data with stratified holdouts and sealed test partitions before feature pipeline fitting.
3. **Imbalance Mastery via Threshold Optimization:** Rather than artificially distorting feature distributions with synthetic oversampling (SMOTE), models were optimized for maximum F1-score via post-training threshold tuning, raising accuracy to 88% and F1 to 0.49.
4. **Resilient Serving Architecture:** FastAPI application with lifespan management, dynamic feature transformation, dynamic latency tracking, and robust HTTP 503 degraded-mode fallback.
5. **Infrastructure Security & Supply Chain Hardening:** Multi-stage Docker container running as non-root (UID 10001) with an immutable root filesystem, serving-only dependency tree, and a SHA-256 cryptographic artifact manifest.
