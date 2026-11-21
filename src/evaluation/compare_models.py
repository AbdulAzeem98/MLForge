"""Model comparison and benchmarking orchestrator for MLForge.

Trains and evaluates Logistic Regression, Random Forest, and PyTorch Neural Network
on validation data. Computes threshold-independent metrics (ROC-AUC, PR-AUC), evaluates
at both default 0.50 and validation-tuned F1 thresholds, measures latency, and generates
publication-quality charts (loss curves, ROC, PR, confusion matrices) and comparison tables.
Enforces strict validation-only governance (never touches test data).
"""

import sys
from pathlib import Path

# Add project root to sys.path to enable src.* imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from src.models.train_baselines import train_logistic_regression
from src.models.random_forest import train_random_forest
from src.models.train_pytorch import train_pytorch_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
})


def plot_pytorch_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    best_epoch: int,
    output_path: Path,
) -> Path:
    """Plot PyTorch training and validation loss curves over epochs."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, "b-o", label="Training Loss", markersize=4, linewidth=1.5)
    plt.plot(epochs, val_losses, "r-s", label="Validation Loss", markersize=4, linewidth=1.5)
    plt.axvline(best_epoch, color="green", linestyle="--", linewidth=1.5, label=f"Best Checkpoint (Epoch {best_epoch})")

    plt.title("PyTorch Neural Network: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCEWithLogitsLoss")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_roc_curves(
    models_data: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Plot comparative ROC curves for all models."""
    plt.figure(figsize=(8, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for idx, m in enumerate(models_data):
        y_val = np.array(m["y_val"])
        y_prob = np.array(m["validation_probabilities"])
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        auc_score = m["threshold_independent_metrics"]["roc_auc"]
        plt.plot(fpr, tpr, color=colors[idx % len(colors)], linewidth=2,
                 label=f"{m['model_name']} (ROC-AUC = {auc_score:.4f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Validation ROC Curves Comparison")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_pr_curves(
    models_data: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Plot comparative Precision-Recall curves for all models."""
    plt.figure(figsize=(8, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    base_rate = float(np.mean(models_data[0]["y_val"]))

    for idx, m in enumerate(models_data):
        y_val = np.array(m["y_val"])
        y_prob = np.array(m["validation_probabilities"])
        prec, rec, _ = precision_recall_curve(y_val, y_prob)
        ap_score = m["threshold_independent_metrics"]["pr_auc"]
        plt.plot(rec, prec, color=colors[idx % len(colors)], linewidth=2,
                 label=f"{m['model_name']} (PR-AUC = {ap_score:.4f})")

    plt.axhline(base_rate, color="gray", linestyle="--", label=f"No-Skill Baseline ({base_rate:.2%})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Validation Precision-Recall Curves Comparison")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_confusion_matrices(
    models_data: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Plot confusion matrix heatmaps at both default and validation-tuned thresholds."""
    fig, axes = plt.subplots(len(models_data), 2, figsize=(11, 4 * len(models_data)))

    for i, m in enumerate(models_data):
        # 1. Default threshold (0.50)
        cm_def = m["default_threshold_metrics"]["confusion_matrix"]
        cm_def_arr = np.array([[cm_def["tn"], cm_def["fp"]], [cm_def["fn"], cm_def["tp"]]])
        sns.heatmap(cm_def_arr, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=axes[i, 0],
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"])
        axes[i, 0].set_title(f"{m['model_name']} — Default Threshold (0.50)\nF1 = {m['default_threshold_metrics']['f1']:.4f}")

        # 2. Validation-tuned threshold
        cm_tune = m["tuned_threshold_metrics"]["confusion_matrix"]
        cm_tune_arr = np.array([[cm_tune["tn"], cm_tune["fp"]], [cm_tune["fn"], cm_tune["tp"]]])
        sns.heatmap(cm_tune_arr, annot=True, fmt=",d", cmap="Greens", cbar=False, ax=axes[i, 1],
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"])
        t_val = m["tuned_threshold_metrics"]["threshold"]
        axes[i, 1].set_title(f"{m['model_name']} — Tuned Threshold ({t_val:.2f})\nF1 = {m['tuned_threshold_metrics']['f1']:.4f}")

    fig.suptitle("Validation Confusion Matrices: Default vs. Tuned Thresholds", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def generate_comparison_reports(
    models_data: List[Dict[str, Any]],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Generate Markdown and JSON summary comparison reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "model_comparison.md"
    json_path = output_dir / "model_comparison.json"

    # Clean JSON representation (exclude raw arrays for readability)
    json_summary = []
    for m in models_data:
        json_summary.append({
            "model_name": m["model_name"],
            "train_time_sec": m["train_time_sec"],
            "latency": m["latency"],
            "threshold_independent_metrics": m["threshold_independent_metrics"],
            "default_threshold_metrics": m["default_threshold_metrics"],
            "tuned_threshold_metrics": m["tuned_threshold_metrics"],
        })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)

    # Markdown Table
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# MLForge — Baseline Model Comparison & Evaluation Report\n\n")
        f.write("## Overview\n")
        f.write("- **Dataset:** UCI Bank Marketing (Validation Split: 6,782 records)\n")
        f.write("- **Data Governance:** Evaluated exclusively on validation data (`X_val.parquet`, `y_val.parquet`). Test data is held out completely.\n")
        f.write("- **Threshold Strategy:** ROC-AUC and PR-AUC are threshold-independent. Metrics are reported at both default (0.50) and validation F1-optimized thresholds.\n\n")

        f.write("## 1. Threshold-Independent & Efficiency Comparison\n\n")
        f.write("| Model | ROC-AUC | PR-AUC | Train Time (s) | Median Batch Latency (ms) | Per-Sample Latency (μs) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for m in json_summary:
            ti = m["threshold_independent_metrics"]
            lat = m["latency"]
            f.write(f"| **{m['model_name']}** | {ti['roc_auc']:.4f} | {ti['pr_auc']:.4f} | {m['train_time_sec']:.2f}s | "
                    f"{lat['median_batch_latency_ms']:.2f} ms | {lat['per_sample_latency_us']:.1f} μs |\n")

        f.write("\n## 2. Performance at Default Threshold (0.50)\n\n")
        f.write("| Model | Threshold | Accuracy | Precision | Recall | F1-Score | Confusion Matrix (TN / FP / FN / TP) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for m in json_summary:
            d = m["default_threshold_metrics"]
            cm = d["confusion_matrix"]
            f.write(f"| **{m['model_name']}** | {d['threshold']:.2f} | {d['accuracy']:.4f} | {d['precision']:.4f} | "
                    f"{d['recall']:.4f} | {d['f1']:.4f} | {cm['tn']} / {cm['fp']} / {cm['fn']} / {cm['tp']} |\n")

        f.write("\n## 3. Performance at Validation-Tuned F1 Threshold\n\n")
        f.write("| Model | Optimal Threshold | Accuracy | Precision | Recall | F1-Score | Confusion Matrix (TN / FP / FN / TP) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for m in json_summary:
            t = m["tuned_threshold_metrics"]
            cm = t["confusion_matrix"]
            f.write(f"| **{m['model_name']}** | {t['threshold']:.2f} | {t['accuracy']:.4f} | {t['precision']:.4f} | "
                    f"{t['recall']:.4f} | {t['f1']:.4f} | {cm['tn']} / {cm['fp']} / {cm['fn']} / {cm['tp']} |\n")

        f.write("\n## 4. Key Findings & Engineering Observations\n")
        f.write("1. **Class Imbalance & Threshold Tuning:** Because classes are heavily imbalanced (7.55:1), default thresholding (0.50) biases toward precision at the expense of recall or vice versa. Tuning the threshold strictly on validation F1 rebalances the trade-off without leaking test information.\n")
        f.write("2. **Latency vs Capacity:** Logistic Regression yields near-instant training and sub-millisecond per-sample latency. Random Forest provides strong tabular performance with parallel CPU induction (`n_jobs=-1`). PyTorch Neural Network enables early stopping on validation loss with dynamic feature sizing.\n")

    return md_path, json_path


def run_model_comparison_pipeline(
    experiments_dir: Path = Path("experiments/baselines"),
) -> Dict[str, Any]:
    """Execute complete model training, evaluation, plot generation, and reporting pipeline."""
    experiments_dir = Path(experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Step 1/3: Training Model 1 (Logistic Regression) ===")
    lr_results = train_logistic_regression()

    logger.info("=== Step 2/3: Training Model 2 (Random Forest) ===")
    rf_results = train_random_forest()

    logger.info("=== Step 3/3: Training Model 3 (PyTorch Neural Network) ===")
    nn_results = train_pytorch_model()

    models_data = [lr_results, rf_results, nn_results]

    logger.info("Generating comparative visualization artifacts...")
    plot_pytorch_loss_curves(
        train_losses=nn_results["train_losses"],
        val_losses=nn_results["val_losses"],
        best_epoch=nn_results["best_epoch"],
        output_path=experiments_dir / "pytorch_loss_curves.png",
    )
    plot_roc_curves(models_data, experiments_dir / "roc_curves.png")
    plot_pr_curves(models_data, experiments_dir / "pr_curves.png")
    plot_confusion_matrices(models_data, experiments_dir / "confusion_matrices.png")

    logger.info("Generating markdown and JSON comparison reports...")
    md_path, json_path = generate_comparison_reports(models_data, experiments_dir)

    logger.info(f"Model comparison pipeline completed. Artifacts saved to {experiments_dir}")
    return {
        "models": [m["model_name"] for m in models_data],
        "reports": [str(md_path), str(json_path)],
    }


if __name__ == "__main__":
    run_model_comparison_pipeline()
