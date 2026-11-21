# MLForge — Dataset Summary Report

## Overview
- **Dataset:** UCI Bank Marketing (Dataset ID: 222)
- **Total Records:** 45,211
- **Total Attributes:** 17 (7 numerical, 9 categorical, 1 target)
- **Target Column:** `y`
- **Target Imbalance:** `7.55:1 (no:yes)` (88.3% 'no', 11.7% 'yes')

## Numerical Attributes Summary

| Feature | Mean | Std | Min | Median | Max | Skewness | Outliers (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `age` | 40.94 | 10.62 | 18.0 | 39.0 | 95.0 | 0.68 | 1.08% |
| `balance` | 1,362.27 | 3,044.77 | -8,019.0 | 448.0 | 102,127.0 | 8.36 | 10.46% |
| `day` | 15.81 | 8.32 | 1.0 | 16.0 | 31.0 | 0.09 | 0.0% |
| `duration` | 258.16 | 257.53 | 0.0 | 180.0 | 4,918.0 | 3.14 | 7.16% |
| `campaign` | 2.76 | 3.1 | 1.0 | 2.0 | 63.0 | 4.9 | 6.78% |
| `pdays` | 40.2 | 100.13 | -1.0 | -1.0 | 871.0 | 2.62 | 18.26% |
| `previous` | 0.58 | 2.3 | 0.0 | 0.0 | 275.0 | 41.85 | 18.26% |

## Categorical Attributes Cardinality & Placeholders

| Feature | Cardinality | Most Frequent Category | Unknown Count | Unknown (%) |
| :--- | :--- | :--- | :--- | :--- |
| `job` | 12 | `blue-collar` (9,732) | 288 | 0.64% |
| `marital` | 3 | `married` (27,214) | 0 | 0.0% |
| `education` | 4 | `secondary` (23,202) | 1,857 | 4.11% |
| `default` | 2 | `no` (44,396) | 0 | 0.0% |
| `housing` | 2 | `yes` (25,130) | 0 | 0.0% |
| `loan` | 2 | `no` (37,967) | 0 | 0.0% |
| `contact` | 3 | `cellular` (29,285) | 13,020 | 28.8% |
| `month` | 12 | `may` (13,766) | 0 | 0.0% |
| `poutcome` | 4 | `unknown` (36,959) | 36,959 | 81.75% |

## Feature Insights & Modeling Governance
1. **Class Imbalance:** Only 11.7% of clients subscribed to the term deposit. Metrics such as ROC-AUC, PR-AUC, and F1-score must be used instead of raw accuracy.
2. **Target Leakage (`duration`):** The call duration feature has an intense correlation with the positive target (`r = 0.39`), because longer calls directly coincide with closing sales. In a real-world pre-call prospective scoring model, the duration of an unmade call is unknown (duration = 0). **`duration` must be dropped prior to pre-call model training.**
3. **Sentinel Value (`pdays = -1`):** 81.7% of records have `pdays = -1`, indicating no previous campaign contact. This should be engineered into an explicit binary indicator `was_previously_contacted` rather than imputed as numerical -1.
