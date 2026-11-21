# MLForge

An end-to-end Machine Learning training, evaluation, optimization, and inference platform designed for CPU environments.

---

## Project Structure

```text
MLForge/
├── .venv/                   # Python 3.11 virtual environment (git-ignored)
├── .gitignore               # Machine learning & Python gitignore rules
├── requirements.txt         # Core dependencies with CPU PyTorch pin
├── requirements.lock        # Reproducible exact dependency lock file
├── check_environment.py     # Standalone environment diagnostic script
├── README.md                # Project documentation and setup guide
├── configs/                 # Model and pipeline YAML/JSON configs
├── data/                    # Local datasets (git-ignored)
│   ├── raw/                 # Unmodified raw data
│   └── processed/           # Transformed features ready for training
├── models/                  # Saved weights, ONNX exports, and artifacts
├── src/                     # Core application source code
│   ├── data/                # Data loaders, ingestion, validation
│   ├── features/            # Feature transformations and pipelines
│   ├── models/              # Model architectures, trainers, hyperparameter tuning
│   ├── evaluation/          # Metrics, benchmarking, and error analysis
│   ├── api/                 # FastAPI inference endpoints & schemas
│   └── utils/               # Common utilities and logging
└── tests/                   # Pytest automated test suite
    └── test_environment.py  # Environment verification unit tests
```

---

## Environment Setup

### Prerequisites
- Windows 10/11
- Python 3.11 (Managed via `uv` or system installation)
- CPU only (no NVIDIA GPU or CUDA required)

### Activating the Virtual Environment

In PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

In Command Prompt (cmd):
```cmd
.venv\Scripts\activate.bat
```

### Dependency Installation

Dependencies are configured for CPU execution, using the official PyTorch CPU wheel index:
```powershell
pip install -r requirements.txt
```

---

## Verification

To run the environment diagnostic check:
```powershell
python check_environment.py
```

To run the automated tests:
```powershell
pytest -v
```
