# MLForge — Baseline Model Comparison & Evaluation Report

## Overview
- **Dataset:** UCI Bank Marketing (Validation Split: 6,782 records)
- **Data Governance:** Evaluated exclusively on validation data (`X_val.parquet`, `y_val.parquet`). Test data is held out completely.
- **Threshold Strategy:** ROC-AUC and PR-AUC are threshold-independent. Metrics are reported at both default (0.50) and validation F1-optimized thresholds.

## 1. Threshold-Independent & Efficiency Comparison

| Model | ROC-AUC | PR-AUC | Train Time (s) | Median Batch Latency (ms) | Per-Sample Latency (μs) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7720 | 0.4162 | 0.45s | 1.76 ms | 0.3 μs |
| **Random Forest** | 0.7977 | 0.4621 | 1.17s | 65.38 ms | 9.6 μs |
| **PyTorch Neural Network** | 0.7996 | 0.4455 | 20.87s | 3.87 ms | 0.6 μs |

## 2. Performance at Default Threshold (0.50)

| Model | Threshold | Accuracy | Precision | Recall | F1-Score | Confusion Matrix (TN / FP / FN / TP) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.50 | 0.7572 | 0.2698 | 0.6297 | 0.3778 | 4635 / 1353 / 294 / 500 |
| **Random Forest** | 0.50 | 0.8127 | 0.3401 | 0.6373 | 0.4435 | 5006 / 982 / 288 / 506 |
| **PyTorch Neural Network** | 0.50 | 0.7902 | 0.3104 | 0.6486 | 0.4199 | 4844 / 1144 / 279 / 515 |

## 3. Performance at Validation-Tuned F1 Threshold

| Model | Optimal Threshold | Accuracy | Precision | Recall | F1-Score | Confusion Matrix (TN / FP / FN / TP) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.66 | 0.8781 | 0.4767 | 0.4244 | 0.4490 | 5618 / 370 / 457 / 337 |
| **Random Forest** | 0.64 | 0.8794 | 0.4849 | 0.4849 | 0.4849 | 5579 / 409 / 409 / 385 |
| **PyTorch Neural Network** | 0.69 | 0.8763 | 0.4734 | 0.5050 | 0.4887 | 5542 / 446 / 393 / 401 |

## 4. Key Findings & Engineering Observations
1. **Class Imbalance & Threshold Tuning:** Because classes are heavily imbalanced (7.55:1), default thresholding (0.50) biases toward precision at the expense of recall or vice versa. Tuning the threshold strictly on validation F1 rebalances the trade-off without leaking test information.
2. **Latency vs Capacity:** Logistic Regression yields near-instant training and sub-millisecond per-sample latency. Random Forest provides strong tabular performance with parallel CPU induction (`n_jobs=-1`). PyTorch Neural Network enables early stopping on validation loss with dynamic feature sizing.
