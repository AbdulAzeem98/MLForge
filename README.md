MLForge --- End-to-End Machine Learning Engineering Platform
An end-to-end machine learning engineering project that takes a
real-world tabular dataset through data ingestion, exploratory analysis,
leakage-aware preprocessing, model training, experiment tracking,
evaluation, and API-based inference.
MLForge compares traditional machine learning models with a PyTorch
neural network and exposes trained models through a FastAPI inference
service. The project emphasizes reproducibility, evaluation discipline,
test isolation, artifact integrity, and deployment preparation.
Repository:
AbdulAzeem98/MLForge
---
Project Overview
Problem: Predict whether a bank customer will subscribe to a term
deposit following a marketing campaign.
Dataset: UCI Bank Marketing --- Bank Full dataset
Task: Binary classification
Dataset size: 45,211 records and 17 original columns
Positive class: Approximately 11.7% of records
The project was designed to demonstrate the engineering lifecycle around
an ML model---not just model training---including data preparation,
model comparison, experiment tracking, testing, inference, and
deployment preparation.
Architecture
``` mermaid
flowchart TD
    A["UCI Bank Marketing Dataset"] --> B["Data Ingestion"]
    B --> C["Data Validation and EDA"]
    C --> D["Leakage-Aware Preprocessing"]
    D --> E["Stratified Train / Validation / Test Split"]
    E --> F["Model Training"]
    F --> G["Logistic Regression"]
    F --> H["Random Forest"]
    F --> I["PyTorch Neural Network"]
    G --> J["Validation Evaluation and Threshold Selection"]
    H --> J
    I --> J
    J --> K["Final Holdout Evaluation"]
    G --> L["MLflow Experiment Tracking"]
    H --> L
    I --> L
    J --> M["Model Artifacts and Evaluation Metadata"]
    M --> N["FastAPI Inference Service"]
    N --> O["Health / Models / Metadata / Predict"]
    P["pytest and GitHub Actions CI"] -.-> N
    P -.-> B
```
The diagram represents the implemented project workflow at a high
level.
Data Ingestion and Exploratory Analysis
Ingested the UCI Bank Marketing dataset and developed a repeatable
ingestion workflow.
Explored feature distributions, categorical values, data quality,
and target imbalance.
Identified `duration` as a leakage-prone feature and excluded it
from predictive modelling. Call duration is only known after the
call, so it would not be available for a genuine pre-call
prediction.
Treated `pdays = -1` as the indicator that a customer had not
previously been contacted.
Target distribution:
No subscription: 39,922 (88.30%)
Subscription: 5,289 (11.70%)
Leakage-Aware Preprocessing
Created a stratified 70/15/15 train-validation-test split using
random seed 42.
Split sizes: 31,647 training, 6,782 validation, and 6,782 test
records.
Fitted preprocessing transformations on training data only.
Used median imputation and standard scaling for numerical features.
Used constant-value imputation and one-hot encoding for categorical
features.
Engineered previous-contact indicators from `pdays`.
Excluded `duration` to avoid target leakage.
Transformed 16 predictive input features into 51 model-ready
features.
Fitting preprocessing only on the training partition prevents
information from validation or test data from influencing learned
imputation values, scaling parameters, or category encodings.
Model Development and Evaluation
Three classifiers were implemented and compared:
Logistic Regression
Random Forest
PyTorch feed-forward neural network
Class weighting was used to address target imbalance. The PyTorch model
used a two-hidden-layer network with ReLU activations, dropout,
`BCEWithLogitsLoss`, and early stopping.
Classification thresholds were selected using validation data rather
than assuming a fixed threshold of 0.5. The selected thresholds were
then held fixed for final holdout evaluation.
Validation results
Model                      ROC-AUC   Average Precision       F1
---
Logistic Regression         0.7720              0.4162   0.4490
Random Forest               0.7977              0.4621   0.4849
PyTorch Neural Network      0.7996              0.4455   0.4887
Final holdout results
Model                      ROC-AUC   Average Precision       F1
---
Logistic Regression         0.7682              0.4147   0.4347
Random Forest               0.7914              0.4512   0.4741
PyTorch Neural Network      0.7856              0.4311   0.4797
The best model depends on the metric:
PyTorch achieved the highest holdout F1 score: 0.4797.
Random Forest achieved the highest holdout ROC-AUC: 0.7914.
Random Forest achieved the highest holdout Average Precision:
0.4512.
This comparison illustrates why model selection should reflect the
evaluation objective rather than relying on a single metric. The final
holdout partition was kept separate from model selection and evaluated
using thresholds selected on validation data.
Selected classification thresholds
Model                      Threshold
---
Logistic Regression            0.660
Random Forest                  0.635
PyTorch Neural Network         0.685
MLflow Experiment Tracking
MLflow was integrated to record model experiment information, including
parameters and evaluation metrics. The project uses a SQLite tracking
backend and a local artifact store.
Tracking tests were isolated from local tracking state so that the test
suite does not depend on a developer's local MLflow database or artifact
directory.
FastAPI Inference Service
The FastAPI application provides four principal endpoints:
Endpoint          Purpose
---
`GET /health`     Reports service and model-artifact readiness
`GET /models`     Returns the available model catalog
`GET /metadata`   Returns model and preprocessing metadata
`POST /predict`   Generates a prediction
The service supports inference using Logistic Regression, Random Forest,
and the PyTorch model. When `model_name` is omitted, the API defaults to
Random Forest.
Degraded-mode behavior
If required model artifacts are missing or cannot be loaded, the
application can start in an unready state rather than incorrectly
reporting itself as healthy. The API endpoints return controlled HTTP
503 responses in this state.
A permanent regression test covers degraded-mode behavior across the API
endpoints.
After generating or obtaining the required local model artifacts, start
the API with:
``` bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
The interactive API documentation is available at:
``` text
http://127.0.0.1:8000/docs
```
Refer to the live `/docs` schema for the exact request fields and
accepted values.
Testing and Continuous Integration
The project includes tests for data workflows, model-related components,
tracking, and API behavior.
Engineering practices include:
Synthetic test data and fixtures.
Temporary directories for filesystem isolation.
Tests that do not rely on local production model artifacts.
Isolated MLflow tracking tests.
API integration tests.
Regression coverage for missing-artifact degraded mode.
Latest reported full test run: 78 passed, 0 failed.
GitHub Actions CI is configured for Ubuntu and Python 3.11, including
dependency installation, environment checks, and pytest execution.
The test result demonstrates the behavior covered by the test suite; it
does not replace load testing or live container verification.
Containerization and Deployment Preparation
The repository includes Docker deployment preparation:
Multi-stage Dockerfile based on Python 3.11 slim Bookworm.
Serving-specific dependency requirements and lockfile.
Non-root runtime user with UID/GID `10001:10001`.
CPU-oriented inference dependencies.
Model binaries kept outside the image and supplied through a
read-only runtime bind mount.
`.dockerignore` rules to exclude datasets, model binaries, local
environments, tests, and experiment outputs from ordinary build
context.
Runtime hardening documented through a read-only root filesystem,
temporary filesystem, dropped Linux capabilities, and
`no-new-privileges`.
Container health-check configuration.
Deployment status: Containerization configuration and deployment
documentation are prepared. A real Docker image build and live container
runtime verification have not yet been completed. The project should
therefore be described as Docker-prepared, not as successfully deployed
in a running container.
Model Artifact Integrity
A SHA-256 manifest records the expected byte size and digest for the
four production model artifacts.
The verification process checks both:
Actual file size against the manifest's `size_bytes`.
Actual SHA-256 digest against the manifest's `sha256`.
This detects mismatches between the current files and the recorded
manifest. A checksum match alone does not establish who created an
artifact or prove its provenance.
Technology Stack
Area                           Technologies
---
Language                       Python 3.11
Data processing                pandas, NumPy, SciPy
Traditional machine learning   scikit-learn
Deep learning                  PyTorch
Experiment tracking            MLflow, SQLite
API serving                    FastAPI, Uvicorn, Pydantic
Dependency management          uv
Testing                        pytest, pytest-cov
Continuous integration         GitHub Actions
Deployment preparation         Docker
Repository Structure
The main project areas include:
``` text
MLForge/
├── src/
│   ├── api/
│   └── tracking/
├── tests/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── experiments/
│   ├── baselines/
│   ├── eda/
│   └── final_evaluation/
├── .github/
│   └── workflows/
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements.lock
├── requirements-serve.txt
├── requirements-serve.lock
├── check_environment.py
├── PROJECT_STATUS.md
└── README.md
```
Generated datasets, trained model binaries, local MLflow stores, and
some experiment outputs are local runtime artifacts and are not all
tracked in Git. Consult `.gitignore` and `.dockerignore` for the
repository's actual tracking and build-context rules.
Getting Started
Prerequisites
Python 3.11
uv
Git
Clone the repository
``` bash
git clone https://github.com/AbdulAzeem98/MLForge.git
cd MLForge
```
Create a virtual environment
``` bash
uv venv
```
Activate it on Windows PowerShell:
``` powershell
.venv\Scripts\Activate.ps1
```
Activate it on Linux or macOS:
``` bash
source .venv/bin/activate
```
Install dependencies
``` bash
uv pip install -r requirements.lock
uv pip check
```
Follow the project's existing data ingestion, preprocessing, and
training workflow to create the local artifacts needed for inference.
Model binaries and generated data are not assumed to be present in a
fresh Git clone.
Run the tests
``` bash
pytest -q
```
Start the API
After the required model artifacts are available:
``` bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
Then visit `http://127.0.0.1:8000/docs`.
Limitations and Future Improvements
Potential next steps include:
Complete a real Docker image build and container runtime smoke test.
Add Docker image build and container verification to CI.
Deploy the inference service to a cloud platform.
Add structured logging, metrics, and tracing.
Perform load and concurrency testing.
Monitor input data and prediction drift.
Evaluate probability calibration.
Introduce model promotion, versioning, and rollback workflows.
Add artifact provenance and signed-manifest verification.
Optimize classification thresholds using business costs and
benefits.
These are future improvements, not claims of existing implementation.
Project Objective
MLForge demonstrates practical machine learning engineering across the
lifecycle: preparing data responsibly, training and evaluating models,
tracking experiments, testing inference behavior, verifying model
artifacts, and preparing a service for containerized deployment.
