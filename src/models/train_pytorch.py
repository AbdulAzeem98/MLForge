"""PyTorch training module for MLForge — Model 3: Neural Network.

Implements mini-batch training on CPU with explicit forward pass, BCEWithLogitsLoss,
class-imbalance pos_weight derived strictly from y_train, Adam optimizer, early stopping
on validation loss, and full checkpoint serialization with map_location="cpu" compatibility.
Seeds Python, NumPy, PyTorch, and DataLoader generator for reproducibility.
"""

import sys
from pathlib import Path

# Add project root to sys.path to enable src.* imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import random
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import (
    benchmark_inference_latency,
    compute_classification_metrics,
    compute_threshold_independent_metrics,
    find_optimal_f1_threshold,
)
from src.models.pytorch_model import MLForgeClassifier, TabularDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> torch.Generator:
    """Seed Python, NumPy, and PyTorch for deterministic execution on CPU.

    Note on Nondeterminism:
        While CPU seeds guarantee identical parameter initialization, mini-batch shuffling,
        and pseudo-random operations, minor floating-point divergence can occasionally arise
        from non-deterministic parallel reduction summation in underlying OpenMP / Intel MKL threads.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # Allows optimized CPU kernels while keeping seeds deterministic
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def train_pytorch_model(
    processed_dir: Path = Path("data/processed"),
    checkpoint_path: Path = Path("models/pytorch_model_best.pt"),
    random_seed: int = 42,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs: int = 50,
    patience: int = 10,
    min_delta: float = 1e-4,
) -> Dict[str, Any]:
    """Train MLForgeClassifier using early stopping on validation loss.

    Args:
        processed_dir: Directory containing processed train and validation Parquet files.
        checkpoint_path: Destination path for full checkpoint serialization.
        random_seed: Random seed for initialization and batch shuffling.
        batch_size: Mini-batch sample size.
        learning_rate: Adam optimizer learning rate.
        weight_decay: L2 regularization penalty.
        max_epochs: Maximum training epochs.
        patience: Epochs to wait for validation loss improvement before early stopping.
        min_delta: Minimum validation loss drop considered an improvement.

    Returns:
        Dictionary containing trained model, validation predictions, loss history, and metrics.
    """
    processed_dir = Path(processed_dir)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Establish deterministic environment
    dl_generator = set_seed(random_seed)

    # 2. Load Training and Validation Sets (Strict validation-only governance: never load test)
    logger.info("Loading training and validation datasets for PyTorch...")
    X_train_df = pd.read_parquet(processed_dir / "X_train.parquet")
    y_train = pd.read_parquet(processed_dir / "y_train.parquet")["y"].values
    X_val_df = pd.read_parquet(processed_dir / "X_val.parquet")
    y_val = pd.read_parquet(processed_dir / "y_val.parquet")["y"].values

    X_train = X_train_df.values
    X_val = X_val_df.values
    input_dim = X_train.shape[1]

    # 3. Create Datasets and DataLoaders
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=dl_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # 4. Compute pos_weight exclusively from training labels
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_weight_val = float(n_neg / n_pos) if n_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)
    logger.info(
        f"Computed training pos_weight = {pos_weight_val:.4f} "
        f"from y_train (negatives={n_neg:,}, positives={n_pos:,})."
    )

    # 5. Initialize Model, Loss Function, and Optimizer
    model = MLForgeClassifier(
        input_dim=input_dim,
        hidden_dim1=128,
        hidden_dim2=64,
        dropout_rate=0.3,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # 6. Training Loop with Explicit Forward, Backward, and Early Stopping
    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state_dict = None

    logger.info(f"Initiating PyTorch training for up to {max_epochs} epochs with patience={patience}...")
    start_train_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * len(batch_x)

        epoch_train_loss = running_train_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_logits = model(val_x)
                v_loss = criterion(val_logits, val_y)
                running_val_loss += v_loss.item() * len(val_x)

        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)

        # Early Stopping Check
        if epoch_val_loss < (best_val_loss - min_delta):
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            patience_counter = 0
            # Clone best state dict
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best validation loss was {best_val_loss:.4f} at epoch {best_epoch}."
                )
                break

    train_duration_sec = time.perf_counter() - start_train_time
    logger.info(f"PyTorch training completed in {train_duration_sec:.3f} seconds across {epoch} epochs.")

    # 7. Restore Best State Dict
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # 8. Save Comprehensive Checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture_config": {
            "input_dim": input_dim,
            "hidden_dim1": 128,
            "hidden_dim2": 64,
            "dropout_rate": 0.3,
            "output_dim": 1,
        },
        "input_feature_count": input_dim,
        "preprocessing_artifact_reference": "models/preprocessor.joblib",
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 5),
        "total_epochs_trained": epoch,
        "training_config": {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "pos_weight": pos_weight_val,
            "random_seed": random_seed,
            "optimizer": "Adam",
            "loss_function": "BCEWithLogitsLoss",
        },
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved complete PyTorch checkpoint to {checkpoint_path}")

    # 9. Measure Validation Inference Latency
    latency_stats = benchmark_inference_latency(
        predict_fn=lambda x: model.predict_proba(x),
        X_sample=X_val,
        num_warmups=3,
        num_repeats=10,
    )

    # 10. Extract Validation Predictions & Compute Metrics
    y_val_prob = model.predict_proba(X_val)

    thresh_indep = compute_threshold_independent_metrics(y_val, y_val_prob)
    default_metrics = compute_classification_metrics(y_val, y_val_prob, threshold=0.50)
    best_thresh, best_f1 = find_optimal_f1_threshold(y_val, y_val_prob)
    tuned_metrics = compute_classification_metrics(y_val, y_val_prob, threshold=best_thresh)

    results = {
        "model_name": "PyTorch Neural Network",
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 5),
        "train_time_sec": round(train_duration_sec, 4),
        "latency": latency_stats,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "threshold_independent_metrics": thresh_indep,
        "default_threshold_metrics": default_metrics,
        "tuned_threshold_metrics": tuned_metrics,
        "validation_probabilities": y_val_prob.tolist(),
        "y_val": y_val.tolist(),
    }
    return results


def load_pytorch_checkpoint(
    checkpoint_path: Path = Path("models/pytorch_model_best.pt"),
) -> Tuple[MLForgeClassifier, Dict[str, Any]]:
    """Load complete PyTorch model checkpoint with map_location='cpu'.

    Args:
        checkpoint_path: Path to serialized .pt checkpoint.

    Returns:
        Tuple of (reconstructed MLForgeClassifier model, checkpoint metadata dict).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    arch = checkpoint["architecture_config"]
    model = MLForgeClassifier(
        input_dim=arch["input_dim"],
        hidden_dim1=arch["hidden_dim1"],
        hidden_dim2=arch["hidden_dim2"],
        dropout_rate=arch["dropout_rate"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


if __name__ == "__main__":
    train_pytorch_model()
