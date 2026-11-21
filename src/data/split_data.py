"""Data splitting module for MLForge.

Implements exact 70/15/15 stratified partitioning for binary classification.
Preserves original row identifiers separately for verification and auditability,
ensuring row identifiers are never included in feature matrices.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Container for stratified train, validation, and test datasets and metadata."""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]

    def verify_disjointness(self) -> bool:
        """Verify that train, validation, and test partitions share zero row identifiers."""
        set_train = set(self.train_indices)
        set_val = set(self.val_indices)
        set_test = set(self.test_indices)

        overlap_train_val = set_train & set_val
        overlap_train_test = set_train & set_test
        overlap_val_test = set_val & set_test

        has_overlap = bool(overlap_train_val or overlap_train_test or overlap_val_test)
        if has_overlap:
            logger.error(
                f"Split overlap detected! Train-Val: {len(overlap_train_val)}, "
                f"Train-Test: {len(overlap_train_test)}, Val-Test: {len(overlap_val_test)}"
            )
            return False

        total_split = len(set_train) + len(set_val) + len(set_test)
        all_unique = len(set_train | set_val | set_test)
        return total_split == all_unique


def stratified_split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    row_ids: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> SplitResult:
    """Split features and targets into 70/15/15 stratified partitions.

    Args:
        X: Feature DataFrame (excluding row identifiers).
        y: Target Series.
        row_ids: 1D array of original row identifiers preserved separately.
        train_ratio: Proportion of training data (0.70).
        val_ratio: Proportion of validation data (0.15).
        test_ratio: Proportion of test data (0.15).
        random_seed: Fixed random seed for reproducibility (42).

    Returns:
        SplitResult dataclass with partitioned sets and tracked row indices.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    n_samples = len(X)
    temp_ratio = val_ratio + test_ratio  # 0.30

    # Stage 1: Split into Train (70%) and Temp (30%)
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X,
        y,
        row_ids,
        test_size=temp_ratio,
        stratify=y,
        random_state=random_seed,
    )

    # Stage 2: Split Temp (30%) equally into Validation (15%) and Test (15%)
    # val_ratio / temp_ratio = 0.15 / 0.30 = 0.50
    test_fraction_of_temp = test_ratio / temp_ratio
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp,
        y_temp,
        idx_temp,
        test_size=test_fraction_of_temp,
        stratify=y_temp,
        random_state=random_seed,
    )

    result = SplitResult(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        train_indices=[int(i) for i in idx_train],
        val_indices=[int(i) for i in idx_val],
        test_indices=[int(i) for i in idx_test],
    )

    if not result.verify_disjointness():
        raise RuntimeError("Integrity check failed: Split sets are not mutually disjoint.")

    logger.info(
        f"Stratified split completed: Train={len(result.X_train):,} ({len(result.X_train)/n_samples:.1%}), "
        f"Val={len(result.X_val):,} ({len(result.X_val)/n_samples:.1%}), "
        f"Test={len(result.X_test):,} ({len(result.X_test)/n_samples:.1%})"
    )
    return result


def save_split_indices(
    split_result: SplitResult,
    output_path: Path = Path("data/processed/split_indices.json")
) -> None:
    """Save row identifiers for auditability and leakage checks."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "train_indices_count": len(split_result.train_indices),
        "val_indices_count": len(split_result.val_indices),
        "test_indices_count": len(split_result.test_indices),
        "train_indices": split_result.train_indices,
        "val_indices": split_result.val_indices,
        "test_indices": split_result.test_indices,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    logger.info(f"Saved separate split index manifest to {output_path}")
