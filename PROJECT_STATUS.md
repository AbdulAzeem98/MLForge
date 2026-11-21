# MLForge: Project Status & Handover Documentation

**Last Updated:** November 21, 2026 (19:05 IST)  
**Status:** Tasks 1–7 Completed & Verified.  
**Next Task:** **Task 8: Production Readiness / Monitoring**

---

## 1. Executive Summary & Handover Status

MLForge is an end-to-end, reproducible machine learning engineering platform built on a 64-bit Windows CPU environment.

- **Tasks 1 through 7 are 100% complete.**
- **No unfinished code changes or unsaved edits exist.**
- **All 69 unit and integration tests are passing (50 existing + 19 API tests).**

---

## 2. Summary of Completed Tasks (Tasks 1–6)

### Task 1: Environment & Project Architecture
- Established Python 3.11.17 64-bit virtual environment (`.venv`) using `uv`.
- Configured CPU-only PyTorch build (`torch==2.15.0+cpu`) via PyTorch CPU wheel index to prevent heavy CUDA overhead on Windows CPU.
- Pinned and managed dependencies via `requirements.txt` and fully locked `requirements.lock`.
- Created modular folder layout: `src/data`, `src/features`, `src/models`, `src/evaluation`, `src/tracking`, `src/api`, `src/utils`, `tests/`, `configs/`, `data/`, `experiments/`, `models/`.
- Verified environment health using `check_environment.py`.

### Task 2: Data Ingestion & Exploratory Data Analysis (EDA)
- Ingested UCI Bank Marketing dataset into `data/raw/bank-full.csv` (45,211 rows, 17 columns).
- Validated file integrity with SHA-256 hash (`f35d2d...`) recorded in `data/raw/ingestion_metadata.json`.
- Implemented robust data quality checks in `src/data/validate_data.py`: verified zero missing values, treated `"unknown"` as valid categorical placeholders, identified `pdays = -1` sentinel distribution (81.74%), and verified numerical bounds.
- **Leakage Governance:** Explicitly flagged `duration` (call duration) as target leakage for prospective marketing scoring, enforcing its exclusion from all future modeling pipelines.
- Generated comprehensive EDA reports and visualizations in `experiments/eda/`.

### Task 3: Preprocessing, Data Splitting & Feature Engineering
- Implemented 70/15/15 stratified train/val/test splitting on `y` (`src/data/split_data.py`).
  - Train: 31,647 rows (3,702 positive / 11.70%)
  - Validation: 6,782 rows (793 positive / 11.69%)
  - Holdout Test: 6,782 rows (794 positive / 11.71%)
  - Split indices stored in `data/processed/split_indices.json`.
- Enforced strict leakage exclusion by removing `duration`.
- Feature engineering in `src/features/build_features.py`:
  - `pdays_contacted` binary flag (`pdays != -1`).
  - `pdays_imputed` (sentinels mapped to median of contacted prospects).
  - Skew handling with logarithmic transforms (`log_campaign`, `log_previous`).
  - Interaction ratios (`balance_to_age_ratio`, `campaign_per_previous`).
- Fitted `ColumnTransformer` strictly on `X_train` with `RobustScaler` and `OneHotEncoder(handle_unknown='ignore')`. Transformed feature space: **51 dynamic columns**.
- Serialized pipeline to `models/preprocessor.joblib`, metadata to `data/processed/preprocessing_metadata.json`, and exported Parquet files.
- **Governance:** `X_test.parquet` and `y_test.parquet` remained isolated and untouched.

### Task 4: Baseline Model Training & Validation Optimization
- Trained three baseline classification models strictly on `X_train.parquet` and validated on `X_val.parquet`:
  1. **Logistic Regression**: `class_weight='balanced'`, L-BFGS solver, `max_iter=1000`.
  2. **Random Forest**: 100 estimators, `class_weight='balanced_subsample'`, `min_samples_leaf=5`, parallelized on CPU.
  3. **PyTorch Neural Network (MLP)**: 3-layer architecture (`51 -> 64 -> 32 -> 1`), BatchNorm1d, Dropout (0.3 / 0.2), ReLU activations, `BCEWithLogitsLoss` with positive class weighting, Adam optimizer (`lr=0.001`), ReduceLROnPlateau scheduler, EarlyStopping (`patience=10`, `min_delta=0.0001`, best epoch 11 of 21). Seeded reproducibility across Python, NumPy, PyTorch, and DataLoader.
- Selected classification thresholds by optimizing F1-score over `[0.05, 0.95]` grid on validation data only:
  - **Logistic Regression**: Threshold = `0.660` (Val F1: `0.4578`, ROC-AUC: `0.7850`, AP: `0.4447`)
  - **Random Forest**: Threshold = `0.635` (Val F1: `0.4908`, ROC-AUC: `0.7937`, AP: `0.4716`)
  - **PyTorch Neural Network**: Threshold = `0.685` (Val F1: `0.4776`, ROC-AUC: `0.7915`, AP: `0.4566`)
- Saved complete checkpoints to `models/logistic_regression.joblib`, `models/random_forest.joblib`, and `models/pytorch_model_best.pt`.
- Exported comparison reports and diagnostic plots to `experiments/baselines/`.

### Task 5: Final Holdout Test Evaluation
- Touched `X_test.parquet` and `y_test.parquet` for the first time with zero model retraining and zero preprocessing refitting.
- Verified validation-selected thresholds against Task 4 JSON: exactly matched `0.660`, `0.635`, and `0.685`.
- PyTorch inference executed using `model.eval()`, `torch.no_grad()`, `torch.sigmoid()`, maintaining exact sample order.
- Holdout test performance at validation-selected optimal thresholds:
  - **Logistic Regression (`0.660`)**: Test F1 = `0.4630` ($\Delta = +0.0052$), Test ROC-AUC = `0.7869` ($\Delta = +0.0019$), Test AP = `0.4485` ($\Delta = +0.0038$)
  - **Random Forest (`0.635`)**: Test F1 = `0.4990` ($\Delta = +0.0082$), Test ROC-AUC = `0.7984` ($\Delta = +0.0047$), Test AP = `0.4854` ($\Delta = +0.0138$)
  - **PyTorch Neural Network (`0.685`)**: Test F1 = `0.4897` ($\Delta = +0.0121$), Test ROC-AUC = `0.7963` ($\Delta = +0.0048$), Test AP = `0.4682` ($\Delta = +0.0116$)
- Confirmed stable generalization with minimal generalization gap ($\Delta = \text{Test} - \text{Val}$).
- Generated final evaluation report and plots in `experiments/final_evaluation/`.

### Task 6: MLflow Tracking & Compatibility Resolution
- Configured local SQLite tracking backend (`sqlite:///D:/MLForge/mlruns.db`) and local artifact store (`mlartifacts/`).
- Created dedicated experiment `MLForge_Bank_Marketing` (Experiment ID: `1`).
- Tracked all 3 models with complete parameters, validation metrics, frozen test metrics (`test_` prefix), diagnostic curve plots, and serialized model checkpoints.
- Registered run IDs:
  - Logistic Regression: `14b87d2fb2cc4f2a9a737ff7eae0360b`
  - Random Forest: `54ca60bc9ecd4bbd812da94483c02caf`
  - PyTorch Neural Network: `d00c8c6552b04a7aa9a7e5e4fb74bf24`
- Ensured idempotent run registration in `src/tracking/log_baselines.py`.
- **SQLAlchemy Compatibility Fix:** Pinned `sqlalchemy>=2.0,<2.1` (`2.0.58`) to resolve compatibility with `mlflow==2.22.5`, updated `requirements.txt` and `requirements.lock`, eliminated runtime monkey patching in `src/tracking/config.py`, verified standalone MLflow CLI process execution, and verified tracking integrity.

### Task 7: FastAPI Model Serving
- Implemented high-performance, asynchronous REST API using FastAPI and Pydantic v2.
- Input validation: accepts 15 raw customer features, enforces `extra="forbid"` to strictly reject target leakage (`duration`), ground truth labels (`y`), and arbitrary attributes.
- Feature engineering: derives `is_previously_contacted` and `pdays_recency` from `pdays`, drops raw `pdays`, and guarantees exact 16-column ordering matching the fitted `ColumnTransformer`'s `feature_names_in_`.
- Application-scoped lifecycle: `PredictionService` instantiated during FastAPI `lifespan` startup, stored in `app.state.prediction_service`, and injected into routes via `Depends(get_prediction_service)` (no global singletons).
- Multi-model inference: supports `logistic_regression`, `random_forest` (default), and `pytorch` (reconstructing checkpoint via `load_pytorch_checkpoint()` in `eval()` mode with `torch.no_grad()` and `torch.sigmoid()`).
- Decision thresholds: validation thresholds loaded from `experiments/baselines/model_comparison.json` (`0.660` for LR, `0.635` for RF, `0.685` for PyTorch).
- Latency tracking: dynamically timed inference per request without hardcoding.
- Endpoints: `GET /health`, `GET /models`, `GET /metadata`, and `POST /predict`.
- Test suite: 19 integration tests in `tests/test_api.py` covering all models, input validations, sentinel handling, error responses, and governance.

---

## 3. Project Architecture

```text
MLForge/
├── .venv/                              # Python 3.11.17 virtual environment (CPU-only)
├── configs/                            # Configuration files and placeholders
├── data/
│   ├── raw/
│   │   ├── bank-full.csv               # UCI Bank Marketing dataset (45,211 rows)
│   │   └── ingestion_metadata.json     # Ingestion timestamp, shape, and SHA-256 hash
│   └── processed/
│       ├── split_indices.json          # Stratified train/val/test row index mappings
│       ├── preprocessing_metadata.json # 51 transformed feature names and scaler params
│       ├── X_train.parquet             # Training features (31,647 x 51)
│       ├── y_train.parquet             # Training labels (31,647)
│       ├── X_val.parquet               # Validation features (6,782 x 51)
│       ├── y_val.parquet               # Validation labels (6,782)
│       ├── X_test.parquet              # Holdout test features (6,782 x 51)
│       └── y_test.parquet              # Holdout test labels (6,782)
├── experiments/
│   ├── eda/                            # EDA summary, quality reports, and distribution plots
│   ├── baselines/                      # Model comparison JSON/MD, PR/ROC curves, loss plots
│   └── final_evaluation/               # Holdout test metrics, report, confusion matrices, curves
├── mlartifacts/                        # Local file-based MLflow artifact repository
│   └── 1/                              # Experiment 1 logged artifacts (models, plots, metrics)
├── models/
│   ├── preprocessor.joblib             # Fitted Scikit-Learn ColumnTransformer pipeline
│   ├── logistic_regression.joblib      # Trained Logistic Regression model
│   ├── random_forest.joblib            # Trained Random Forest model (100 trees)
│   └── pytorch_model_best.pt           # PyTorch checkpoint (state_dict, config, best epoch)
├── src/
│   ├── __init__.py
│   ├── api/                            # Target module for Task 7 (FastAPI model serving)
│   │   └── __init__.py
│   ├── data/                           # Ingestion, validation, splitting, and EDA
│   │   ├── ingest_data.py
│   │   ├── validate_data.py
│   │   ├── split_data.py
│   │   └── explore_data.py
│   ├── features/                       # Feature engineering and transformation pipelines
│   │   ├── build_features.py
│   │   └── preprocessing.py
│   ├── models/                         # Model architectures, training routines, checkpoints
│   │   ├── pytorch_model.py
│   │   ├── random_forest.py
│   │   ├── train_baselines.py
│   │   └── train_pytorch.py
│   ├── evaluation/                     # Metric calculations and holdout evaluation
│   │   ├── metrics.py
│   │   ├── compare_models.py
│   │   └── evaluate_test.py
│   ├── tracking/                       # MLflow tracking configuration and run logger
│   │   ├── config.py
│   │   └── log_baselines.py
│   └── utils/
│       └── __init__.py
├── tests/                              # Comprehensive test suite (50 tests, all passing)
│   ├── test_environment.py             # Python version, imports, PyTorch CPU configuration
│   ├── test_data_ingestion.py          # Data validation, schema, sentinel handling, leakage checks
│   ├── test_preprocessing.py           # Split disjointness, feature transforms, no test leakage
│   ├── test_models.py                  # Model outputs, shapes, checkpoint integrity, reproducibility
│   ├── test_final_evaluation.py        # Holdout evaluation contracts, threshold verification
│   └── test_tracking.py                # MLflow SQLite initialization, artifact integrity, idempotency
├── check_environment.py                # Standalone environment verification script
├── mlruns.db                           # Local SQLite database backing MLflow tracking server
├── pytest.ini                          # Test runner configuration
├── requirements.txt                    # Project dependency specification
├── requirements.lock                   # Fully resolved uv lockfile
└── PROJECT_STATUS.md                   # This project handover and status document
```

---

## 4. Key File Paths

| Component | Path | Description |
| :--- | :--- | :--- |
| **Raw Data** | `data/raw/bank-full.csv` | Original UCI Bank Marketing dataset |
| **Ingestion Metadata** | `data/raw/ingestion_metadata.json` | Dataset shape and SHA-256 hash |
| **Split Indices** | `data/processed/split_indices.json` | Train (70%), Val (15%), Test (15%) indices |
| **Preprocessing Metadata** | `data/processed/preprocessing_metadata.json` | 51 feature column names, scaler parameters |
| **Fitted Preprocessor** | `models/preprocessor.joblib` | Serialized Scikit-Learn `ColumnTransformer` |
| **Logistic Regression Checkpoint** | `models/logistic_regression.joblib` | Trained Logistic Regression model |
| **Random Forest Checkpoint** | `models/random_forest.joblib` | Trained Random Forest model |
| **PyTorch Checkpoint** | `models/pytorch_model_best.pt` | PyTorch checkpoint dictionary |
| **Task 4 Comparison JSON** | `experiments/baselines/model_comparison.json` | Validation metrics and tuned thresholds |
| **Task 5 Holdout JSON** | `experiments/final_evaluation/test_metrics.json` | Final test metrics and generalization gaps |
| **Task 5 Evaluation Report** | `experiments/final_evaluation/test_report.md` | Generalization analysis report |
| **MLflow Database** | `mlruns.db` | Local SQLite backend store |
| **MLflow Artifacts** | `mlartifacts/` | Logged model checkpoints, plots, metadata |
| **Tracking Config** | `src/tracking/config.py` | URI setup and experiment retrieval helper |
| **Tracking Logger** | `src/tracking/log_baselines.py` | Idempotent baseline run registration script |

---

## 5. Environment & Dependency Specifications

- **Operating System:** Windows 10 (AMD64)
- **Python Version:** `3.11.17` (64-bit)
- **Key Installed Packages:**
  - `torch`: `2.15.0+cpu` (CPU build)
  - `scikit-learn`: `1.9.1`
  - `mlflow`: `2.22.5`
  - `sqlalchemy`: `2.0.58` (pinned `>=2.0,<2.1`)
  - `pandas`: `2.3.3`
  - `numpy`: `1.26.4`
  - `scipy`: `1.17.1`
  - `matplotlib`: `3.11.2`
  - `seaborn`: `0.13.2`
  - `fastapi`: `0.165.0`
  - `uvicorn`: `0.54.0`
  - `pydantic`: `2.14.3`
  - `pytest`: `8.4.2`
  - `pyarrow`: `19.0.1`
  - `joblib`: `1.6.0`

---

## 6. MLflow Tracking Configuration & Run Registry

- **Backend Store URI:** `sqlite:///D:/MLForge/mlruns.db`
- **Default Artifact Store:** `file:///D:/MLForge/mlartifacts`
- **Active Experiment:** `MLForge_Bank_Marketing` (ID: `1`)

### Registered Baseline Runs

| Model | Run ID | Status | Selected Threshold | Val F1 | Test F1 | Test ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | `14b87d2fb2cc4f2a9a737ff7eae0360b` | Finished | `0.660` | `0.4578` | `0.4630` | `0.7869` |
| **Random Forest** | `54ca60bc9ecd4bbd812da94483c02caf` | Finished | `0.635` | `0.4908` | `0.4990` | `0.7984` |
| **PyTorch Neural Network**| `d00c8c6552b04a7aa9a7e5e4fb74bf24` | Finished | `0.685` | `0.4776` | `0.4897` | `0.7963` |

---

## 7. Test Suite Status

Executing `.venv\Scripts\pytest.exe -v`:
- **Total Tests:** 69 passed in 25.33s
- **Breakdown by module:**
  - `tests/test_api.py`: 19 passed
  - `tests/test_data_ingestion.py`: 10 passed
  - `tests/test_environment.py`: 12 passed
  - `tests/test_preprocessing.py`: 9 passed
  - `tests/test_models.py`: 6 passed
  - `tests/test_final_evaluation.py`: 5 passed
  - `tests/test_tracking.py`: 8 passed

---

## 8. Exact Commands for Resumption & Verification

### 1. Run the Full Test Suite
```powershell
.venv\Scripts\pytest.exe -v
```

### 2. Verify MLflow Tracking CLI
```cmd
cmd /c "set MLFLOW_TRACKING_URI=sqlite:///D:/MLForge/mlruns.db && .venv\Scripts\mlflow.exe experiments search"
cmd /c "set MLFLOW_TRACKING_URI=sqlite:///D:/MLForge/mlruns.db && .venv\Scripts\mlflow.exe runs list --experiment-id 1"
```

### 3. Launch the MLflow Web UI
To open the MLflow dashboard locally on port 5000:
```powershell
.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///D:/MLForge/mlruns.db --port 5000 --host 127.0.0.1
```
*(Open `http://127.0.0.1:5000` in your web browser)*

### 4. Re-run Baseline Logging (Idempotency Check)
```powershell
.venv\Scripts\python.exe src/tracking/log_baselines.py
```

---

## 9. Next Steps: Task 7 (FastAPI Model Serving)

When resuming development, proceed directly to **Task 7**:
1. Implement FastAPI application in `src/api/` with input schema validation via Pydantic (`src/api/schemas.py`).
2. Integrate preprocessor pipeline (`models/preprocessor.joblib`) for runtime raw record transformation.
3. Serve inference endpoints for selected production model (or model routing) applying validation-tuned decision thresholds.
4. Add `/health`, `/predict`, and `/metadata` endpoints.
5. Create comprehensive API integration tests in `tests/test_api.py`.
