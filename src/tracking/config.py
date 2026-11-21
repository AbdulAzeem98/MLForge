"""MLflow configuration and setup module for MLForge.

Configures a local SQLite tracking store (sqlite:///mlruns.db) and file-based artifact store (mlartifacts/).
Provides helper functions for experiment registration and client retrieval on Windows CPU.
"""

from pathlib import Path
from typing import Tuple

import mlflow
from mlflow.tracking import MlflowClient

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "mlruns.db"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "mlartifacts"
DEFAULT_EXPERIMENT_NAME = "MLForge_Bank_Marketing"


def get_tracking_uri(db_path: Path = DEFAULT_DB_PATH) -> str:
    """Return SQLite connection URI with normalized forward slashes."""
    db_path = Path(db_path).resolve()
    # SQLite URI format: sqlite:///C:/path/to/db.db
    return f"sqlite:///{db_path.as_posix()}"


def get_artifact_location(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> str:
    """Return local artifact directory URI with normalized forward slashes."""
    artifact_dir = Path(artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir.as_uri()


def setup_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    db_path: Path = DEFAULT_DB_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> Tuple[MlflowClient, str]:
    """Configure MLflow tracking URI, create or retrieve the experiment, and return client.

    Args:
        experiment_name: Name of target MLflow experiment.
        db_path: Path to SQLite database file.
        artifact_dir: Path to directory for logged artifacts.

    Returns:
        Tuple of (MlflowClient instance, experiment_id string).
    """
    tracking_uri = get_tracking_uri(db_path)
    artifact_uri = get_artifact_location(artifact_dir)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_uri,
        )
    else:
        experiment_id = experiment.experiment_id

    return client, experiment_id
