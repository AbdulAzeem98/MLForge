# MLForge — Final Holdout Test Evaluation Report

## 1. Test Dataset Verification & Target Distribution
- **Test File:** `data/processed/X_test.parquet`, `data/processed/y_test.parquet`
- **Total Test Records:** 6,782
- **Negative Class (0 - no):** 5,989 (88.31%)
- **Positive Class (1 - yes):** 793 (11.69%)
- **Stratification Alignment:** Target positive rate (11.69%) matches the training (11.70%) and validation (11.71%) splits within 0.02%.

## 2. Threshold-Independent Metrics on Holdout Test Set

| Model | Test ROC-AUC | Test PR-AUC (AP) | Val ROC-AUC | Val PR-AUC | Δ ROC-AUC (Test - Val) | Δ PR-AUC (Test - Val) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7682 | 0.4147 | 0.7720 | 0.4162 | -0.0038 | -0.0015 |
| **Random Forest** | 0.7914 | 0.4512 | 0.7977 | 0.4621 | -0.0063 | -0.0109 |
| **PyTorch Neural Network** | 0.7856 | 0.4311 | 0.7996 | 0.4455 | -0.0140 | -0.0144 |

## 3. Performance at Default Threshold (0.50)

| Model | Threshold | Accuracy | Precision | Recall | F1-Score | Confusion Matrix (TN / FP / FN / TP) | Δ F1 (Test - Val) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.50 | 0.7561 | 0.2672 | 0.6230 | 0.3740 | 4634 / 1355 / 299 / 494 | -0.0038 |
| **Random Forest** | 0.50 | 0.8154 | 0.3416 | 0.6242 | 0.4416 | 5035 / 954 / 298 / 495 | -0.0019 |
| **PyTorch Neural Network** | 0.50 | 0.7974 | 0.3163 | 0.6305 | 0.4212 | 4908 / 1081 / 293 / 500 | +0.0013 |

## 4. Performance at Validation-Selected F1 Threshold

| Model | Applied Threshold | Accuracy | Precision | Recall | F1-Score | Confusion Matrix (TN / FP / FN / TP) | Δ F1 (Test - Val) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.660 | 0.8761 | 0.4661 | 0.4073 | 0.4347 | 5619 / 370 / 470 / 323 | -0.0143 |
| **Random Forest** | 0.635 | 0.8770 | 0.4741 | 0.4741 | 0.4741 | 5572 / 417 / 417 / 376 | -0.0108 |
| **PyTorch Neural Network** | 0.685 | 0.8756 | 0.4692 | 0.4905 | 0.4797 | 5549 / 440 / 404 / 389 | -0.0090 |

## 5. Comprehensive Validation vs. Test Generalization Analysis

| Model | Metric | Validation | Holdout Test | Generalization Gap (Δ = Test - Val) | Generalization Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | ROC-AUC | 0.7720 | 0.7682 | -0.0038 | Excellent stability (minimal variance across splits) |
| Logistic Regression | Average Precision (PR-AUC) | 0.4162 | 0.4147 | -0.0015 | Excellent stability (minimal variance across splits) |
| Logistic Regression | F1 (Default 0.50) | 0.3778 | 0.3740 | -0.0038 | Excellent stability (minimal variance across splits) |
| Logistic Regression | F1 (Validation Threshold) | 0.4490 | 0.4347 | -0.0143 | Excellent stability (minimal variance across splits) |
| Logistic Regression | Precision (Validation Threshold) | 0.4767 | 0.4661 | -0.0106 | Excellent stability (minimal variance across splits) |
| Logistic Regression | Recall (Validation Threshold) | 0.4244 | 0.4073 | -0.0171 | Mild attenuation (expected sampling variation across holdout partition) |
| Random Forest | ROC-AUC | 0.7977 | 0.7914 | -0.0063 | Excellent stability (minimal variance across splits) |
| Random Forest | Average Precision (PR-AUC) | 0.4621 | 0.4512 | -0.0109 | Excellent stability (minimal variance across splits) |
| Random Forest | F1 (Default 0.50) | 0.4435 | 0.4416 | -0.0019 | Excellent stability (minimal variance across splits) |
| Random Forest | F1 (Validation Threshold) | 0.4849 | 0.4741 | -0.0108 | Excellent stability (minimal variance across splits) |
| Random Forest | Precision (Validation Threshold) | 0.4849 | 0.4741 | -0.0108 | Excellent stability (minimal variance across splits) |
| Random Forest | Recall (Validation Threshold) | 0.4849 | 0.4741 | -0.0108 | Excellent stability (minimal variance across splits) |
| PyTorch Neural Network | ROC-AUC | 0.7996 | 0.7856 | -0.0140 | Excellent stability (minimal variance across splits) |
| PyTorch Neural Network | Average Precision (PR-AUC) | 0.4455 | 0.4311 | -0.0144 | Excellent stability (minimal variance across splits) |
| PyTorch Neural Network | F1 (Default 0.50) | 0.4199 | 0.4212 | +0.0013 | Excellent stability (minimal variance across splits) |
| PyTorch Neural Network | F1 (Validation Threshold) | 0.4887 | 0.4797 | -0.0090 | Excellent stability (minimal variance across splits) |
| PyTorch Neural Network | Precision (Validation Threshold) | 0.4734 | 0.4692 | -0.0042 | Excellent stability (minimal variance across splits) |
| PyTorch Neural Network | Recall (Validation Threshold) | 0.5050 | 0.4905 | -0.0145 | Excellent stability (minimal variance across splits) |

## 6. Data Governance & Integrity Certification
1. **Zero Retraining:** All models evaluated were loaded verbatim from `models/logistic_regression.joblib`, `models/random_forest.joblib`, and `models/pytorch_model_best.pt`.
2. **Zero Preprocessing Refit:** The holdout test features were preprocessed strictly using the preprocessor pipeline fitted during Task 3 on training data only.
3. **Zero Test Threshold Searching:** The exact thresholds (LR: 0.660, RF: 0.635, PyTorch: 0.685) were ingested from Task 4 validation artifacts. No search over test data was conducted.
4. **Finality:** These holdout test evaluations represent the unbiased terminal audit of baseline model performance on unseen data.
