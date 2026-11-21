# MLForge — Data Quality & Integrity Report

**Overall Dataset Status:** ✅ VALID

## Integrity Checklist
- **Schema Compliance:** All 17 columns match the expected types.
- **Standard Missing Values (NaN/Null):** 0 (0.00%)
- **Exact Duplicate Rows:** 0 (0.0%)
- **Domain Constraint Anomalies:** 0 anomalies detected.

## Categorical Placeholders ('unknown') Audit
| Column | 'unknown' Records | % of Dataset | Interpretation |
| :--- | :--- | :--- | :--- |
| `job` | 288 | 0.64% | Treated as explicit categorical category. |
| `education` | 1,857 | 4.11% | Treated as explicit categorical category. |
| `contact` | 13,020 | 28.8% | Treated as explicit categorical category. |
| `poutcome` | 36,959 | 81.75% | Treated as explicit categorical category. |

## Sentinel Values Audit
- **`pdays_-1`:** 36,954 occurrences (81.74%). Client was not previously contacted during any prior campaign.

## Data Governance Warnings
> ⚠️ **WARNING:** LEAKAGE WARNING: Feature 'duration' represents call duration in seconds. This attribute is unknown before a call is performed (the prospective prediction scenario) and strongly correlates with the target 'y'. As per UCI guidelines and best ML practices, 'duration' must be strictly excluded from prospective pre-call feature sets.

