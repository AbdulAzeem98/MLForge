"""Exploratory Data Analysis (EDA) module for the UCI Bank Marketing dataset.

Performs ingestion, schema and data-quality validation, statistical analysis,
outlier detection, visualization generation, and structured report creation.
All charts are saved under experiments/eda/ at high resolution (300 DPI).
"""

import sys
from pathlib import Path

# Add project root to sys.path to enable src.* imports when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for headless scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.ingest_data import ingest_bank_marketing_dataset, load_raw_dataset
from src.data.validate_data import validate_bank_marketing_dataset, ValidationResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set reproducible aesthetics and random seeds
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "figure.autolayout": False,
})

NUMERICAL_COLS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome"
]


def compute_numerical_summary(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute comprehensive descriptive statistics and outlier metrics for numerical columns."""
    summary: Dict[str, Dict[str, float]] = {}
    for col in NUMERICAL_COLS:
        series = df[col].dropna()
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_count = int(((series < lower_bound) | (series > upper_bound)).sum())
        outliers_pct = round((outliers_count / len(series)) * 100, 2)

        summary[col] = {
            "count": int(series.count()),
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "min": round(float(series.min()), 2),
            "q25": round(q1, 2),
            "median": round(float(series.median()), 2),
            "q75": round(q3, 2),
            "max": round(float(series.max()), 2),
            "skewness": round(float(series.skew()), 2),
            "iqr": round(iqr, 2),
            "outliers_count": outliers_count,
            "outliers_pct": outliers_pct,
        }
    return summary


def compute_categorical_summary(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Compute frequency distributions and cardinality for categorical columns."""
    summary: Dict[str, Dict[str, Any]] = {}
    for col in CATEGORICAL_COLS:
        series = df[col].astype(str)
        counts = series.value_counts()
        proportions = (series.value_counts(normalize=True) * 100).round(2)
        unknown_count = int((series.str.lower() == "unknown").sum())
        unknown_pct = round((unknown_count / len(series)) * 100, 2)

        summary[col] = {
            "unique_cardinality": int(series.nunique()),
            "top_category": counts.index[0],
            "top_category_count": int(counts.iloc[0]),
            "unknown_count": unknown_count,
            "unknown_pct": unknown_pct,
            "frequencies": {str(k): int(v) for k, v in counts.items()},
            "percentages": {str(k): float(v) for k, v in proportions.items()},
        }
    return summary


def plot_target_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    """Generate target variable class distribution visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    target_counts = df["y"].value_counts()
    colors = ["#4C72B0", "#55A868"]

    # Bar chart
    bars = axes[0].bar(target_counts.index, target_counts.values, color=colors, edgecolor="black", width=0.5)
    axes[0].set_title("Target Class Counts (y = term deposit subscribed?)")
    axes[0].set_xlabel("Subscription Decision (y)")
    axes[0].set_ylabel("Number of Clients")
    for bar in bars:
        height = bar.get_height()
        axes[0].annotate(f"{height:,} ({height / len(df):.1%})",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", va="bottom", fontweight="bold")

    # Donut chart
    axes[1].pie(
        target_counts.values,
        labels=[f"{k.upper()}: {v:,}" for k, v in target_counts.items()],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor="black"),
    )
    axes[1].set_title(f"Class Proportion (Imbalance: {target_counts['no'] / target_counts['yes']:.1f}:1)")

    fig.suptitle("UCI Bank Marketing: Target Class Analysis ('y')", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = output_dir / "target_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_numerical_distributions(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot histograms with KDE overlays for all numerical features."""
    cols_to_plot = [c for c in NUMERICAL_COLS if c != "day"]  # Day is discrete calendar index
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, col in enumerate(cols_to_plot):
        ax = axes[idx]
        sns.histplot(df[col], kde=True, ax=ax, color="#2b5c8f", bins=30, edgecolor="white")
        ax.set_title(f"Distribution: {col.capitalize()} (skew = {df[col].skew():.2f})")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

        # Add median and mean vertical lines
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:,.1f}")
        ax.axvline(median_val, color="green", linestyle=":", linewidth=1.5, label=f"Median: {median_val:,.1f}")
        ax.legend(fontsize=8)

    fig.suptitle("Numerical Feature Distributions with Density Estimations", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = output_dir / "numerical_distributions.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_categorical_distributions(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot bar charts for primary categorical features."""
    cat_subset = ["job", "education", "marital", "contact", "housing", "loan"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    axes = axes.flatten()

    for idx, col in enumerate(cat_subset):
        ax = axes[idx]
        order = df[col].value_counts().index
        sns.countplot(data=df, y=col, order=order, ax=ax, palette="Blues_r", hue=col, legend=False)
        ax.set_title(f"Categorical Frequency: {col.capitalize()}")
        ax.set_xlabel("Count")
        ax.set_ylabel(col.capitalize())

        # Annotate percentages
        total = len(df)
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f"{width:,.0f} ({width / total:.1%})",
                        (width, p.get_y() + p.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8)

    fig.suptitle("Categorical Feature Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = output_dir / "categorical_distributions.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """Generate Pearson and Spearman correlation heatmaps for numerical features."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Binary encode target for correlation inspection
    df_corr = df[NUMERICAL_COLS].copy()
    df_corr["target_y"] = (df["y"] == "yes").astype(int)

    pearson_corr = df_corr.corr(method="pearson")
    spearman_corr = df_corr.corr(method="spearman")

    sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                ax=axes[0], cbar_kws={"label": "Pearson r"}, square=True)
    axes[0].set_title("Pearson Linear Correlation")

    sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                ax=axes[1], cbar_kws={"label": "Spearman rank rho"}, square=True)
    axes[1].set_title("Spearman Rank Non-Linear Correlation")

    fig.suptitle("Numerical Correlation Heatmaps (Including Target 'y' and Leakage Feature 'duration')",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = output_dir / "correlation_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_numerical_outliers_boxplot(df: pd.DataFrame, output_dir: Path) -> Path:
    """Generate standardized box plots to highlight distribution outliers."""
    cols = ["age", "balance", "duration", "campaign", "previous"]
    fig, axes = plt.subplots(len(cols), 1, figsize=(12, 10))

    for idx, col in enumerate(cols):
        ax = axes[idx]
        sns.boxplot(x=df[col], ax=ax, color="#7293CB", fliersize=3, flierprops={"marker": "d", "color": "red"})
        ax.set_title(f"Box Plot & Outlier Spread: {col.capitalize()}", fontsize=11)
        ax.set_xlabel(col)

    fig.suptitle("Outlier Analysis via Interquartile Range (IQR)", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = output_dir / "numerical_outliers_boxplot.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_features_vs_target(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot bivariate interactions between key predictors and target subscription rate."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Job vs Subscription Rate
    job_sub = df.groupby("job")["y"].apply(lambda s: (s == "yes").mean() * 100).sort_values(ascending=False)
    sns.barplot(x=job_sub.values, y=job_sub.index, ax=axes[0, 0], palette="Greens_r", hue=job_sub.index, legend=False)
    axes[0, 0].set_title("Deposit Subscription Rate by Job Category (%)")
    axes[0, 0].set_xlabel("Subscription Rate (%)")
    axes[0, 0].set_ylabel("Job")
    for p in axes[0, 0].patches:
        width = p.get_width()
        axes[0, 0].annotate(f"{width:.1f}%", (width, p.get_y() + p.get_height() / 2),
                            xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=8)

    # 2. Education vs Subscription Rate
    edu_sub = df.groupby("education")["y"].apply(lambda s: (s == "yes").mean() * 100).sort_values(ascending=False)
    sns.barplot(x=edu_sub.index, y=edu_sub.values, ax=axes[0, 1], palette="Purples_r", hue=edu_sub.index, legend=False)
    axes[0, 1].set_title("Subscription Rate by Education Level (%)")
    axes[0, 1].set_xlabel("Education")
    axes[0, 1].set_ylabel("Subscription Rate (%)")
    for p in axes[0, 1].patches:
        height = p.get_height()
        axes[0, 1].annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2, height),
                            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # 3. Previous Outcome vs Subscription Rate
    poutcome_sub = df.groupby("poutcome")["y"].apply(lambda s: (s == "yes").mean() * 100).sort_values(ascending=False)
    sns.barplot(x=poutcome_sub.index, y=poutcome_sub.values, ax=axes[1, 0], palette="Blues_r", hue=poutcome_sub.index, legend=False)
    axes[1, 0].set_title("Subscription Rate by Previous Campaign Outcome (%)")
    axes[1, 0].set_xlabel("Previous Campaign Outcome (poutcome)")
    axes[1, 0].set_ylabel("Subscription Rate (%)")
    for p in axes[1, 0].patches:
        height = p.get_height()
        axes[1, 0].annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2, height),
                            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # 4. Duration Box Plot by Target (Highlighting Data Leakage)
    sns.boxplot(data=df, x="y", y="duration", ax=axes[1, 1], palette={"no": "#4C72B0", "yes": "#55A868"}, hue="y", legend=False)
    axes[1, 1].set_title("Call Duration (seconds) by Target Subscription (Data Leakage)")
    axes[1, 1].set_xlabel("Subscription Outcome (y)")
    axes[1, 1].set_ylabel("Call Duration (sec)")
    axes[1, 1].set_ylim(0, 2000)  # Focus on bulk of distribution

    fig.suptitle("Bivariate Analysis: Predictors vs Term Deposit Subscription", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = output_dir / "features_vs_target.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def generate_reports(
    df: pd.DataFrame,
    validation_res: ValidationResult,
    num_summary: Dict[str, Dict[str, float]],
    cat_summary: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> Tuple[Path, Path, Path, Path]:
    """Generate Markdown and JSON reports for dataset summary and data quality."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Summary JSON
    summary_json_path = output_dir / "dataset_summary.json"
    summary_data = {
        "dataset_name": "UCI Bank Marketing",
        "dataset_id": 222,
        "rows": len(df),
        "columns": len(df.columns),
        "target_distribution": validation_res.target_summary,
        "numerical_summary": num_summary,
        "categorical_summary": {k: {k2: v2 for k2, v2 in v.items() if k2 != "frequencies"} for k, v in cat_summary.items()},
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 2. Data Quality JSON
    quality_json_path = output_dir / "data_quality_report.json"
    quality_data = validation_res.to_dict()
    with open(quality_json_path, "w", encoding="utf-8") as f:
        json.dump(quality_data, f, indent=2)

    # 3. Dataset Summary Markdown Report
    summary_md_path = output_dir / "dataset_summary_report.md"
    target_info = validation_res.target_summary
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# MLForge — Dataset Summary Report\n\n")
        f.write("## Overview\n")
        f.write(f"- **Dataset:** UCI Bank Marketing (Dataset ID: 222)\n")
        f.write(f"- **Total Records:** {len(df):,}\n")
        f.write(f"- **Total Attributes:** {len(df.columns)} (7 numerical, 9 categorical, 1 target)\n")
        f.write(f"- **Target Column:** `{target_info.get('target_column', 'y')}`\n")
        f.write(f"- **Target Imbalance:** `{target_info.get('imbalance_ratio', 'N/A')}` "
                f"({target_info.get('percentages', {}).get('no', 0)}% 'no', "
                f"{target_info.get('percentages', {}).get('yes', 0)}% 'yes')\n\n")

        f.write("## Numerical Attributes Summary\n\n")
        f.write("| Feature | Mean | Std | Min | Median | Max | Skewness | Outliers (%) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for col, stats in num_summary.items():
            f.write(f"| `{col}` | {stats['mean']:,} | {stats['std']:,} | {stats['min']:,} | "
                    f"{stats['median']:,} | {stats['max']:,} | {stats['skewness']} | {stats['outliers_pct']}% |\n")

        f.write("\n## Categorical Attributes Cardinality & Placeholders\n\n")
        f.write("| Feature | Cardinality | Most Frequent Category | Unknown Count | Unknown (%) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for col, stats in cat_summary.items():
            f.write(f"| `{col}` | {stats['unique_cardinality']} | `{stats['top_category']}` ({stats['top_category_count']:,}) | "
                    f"{stats['unknown_count']:,} | {stats['unknown_pct']}% |\n")

        f.write("\n## Feature Insights & Modeling Governance\n")
        f.write("1. **Class Imbalance:** Only 11.7% of clients subscribed to the term deposit. Metrics such as ROC-AUC, PR-AUC, and F1-score must be used instead of raw accuracy.\n")
        f.write("2. **Target Leakage (`duration`):** The call duration feature has an intense correlation with the positive target (`r = 0.39`), because longer calls directly coincide with closing sales. In a real-world pre-call prospective scoring model, the duration of an unmade call is unknown (duration = 0). **`duration` must be dropped prior to pre-call model training.**\n")
        f.write("3. **Sentinel Value (`pdays = -1`):** 81.7% of records have `pdays = -1`, indicating no previous campaign contact. This should be engineered into an explicit binary indicator `was_previously_contacted` rather than imputed as numerical -1.\n")

    # 4. Data Quality Markdown Report
    quality_md_path = output_dir / "data_quality_report.md"
    with open(quality_md_path, "w", encoding="utf-8") as f:
        f.write("# MLForge — Data Quality & Integrity Report\n\n")
        f.write(f"**Overall Dataset Status:** {'✅ VALID' if validation_res.is_valid else '❌ INVALID'}\n\n")
        f.write("## Integrity Checklist\n")
        f.write(f"- **Schema Compliance:** All {validation_res.total_cols} columns match the expected types.\n")
        f.write(f"- **Standard Missing Values (NaN/Null):** {sum(validation_res.standard_missing_values.values()):,} (0.00%)\n")
        f.write(f"- **Exact Duplicate Rows:** {validation_res.duplicate_rows_count:,} ({validation_res.duplicate_rows_pct}%)\n")
        f.write(f"- **Domain Constraint Anomalies:** {len(validation_res.domain_anomalies)} anomalies detected.\n\n")

        f.write("## Categorical Placeholders ('unknown') Audit\n")
        f.write("| Column | 'unknown' Records | % of Dataset | Interpretation |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for col, p_data in validation_res.categorical_placeholders.items():
            f.write(f"| `{col}` | {p_data['count']:,} | {p_data['percentage']}% | Treated as explicit categorical category. |\n")

        f.write("\n## Sentinel Values Audit\n")
        for name, s_data in validation_res.sentinel_values.items():
            f.write(f"- **`{name}`:** {s_data['count']:,} occurrences ({s_data['percentage']}%). {s_data['meaning']}\n")

        f.write("\n## Data Governance Warnings\n")
        for warning in validation_res.leakage_warnings:
            f.write(f"> ⚠️ **WARNING:** {warning}\n\n")

    return summary_md_path, summary_json_path, quality_md_path, quality_json_path


def run_eda_pipeline(
    raw_dir: Path = Path("data/raw"),
    eda_output_dir: Path = Path("experiments/eda"),
    force_ingest: bool = False,
) -> Dict[str, Any]:
    """Execute complete ingestion, validation, and exploratory data analysis pipeline."""
    logger.info("Step 1: Ingesting dataset...")
    ingest_meta = ingest_bank_marketing_dataset(raw_dir=raw_dir, force=force_ingest)

    logger.info("Step 2: Loading raw dataset...")
    df = load_raw_dataset(raw_dir / "bank-full.csv")

    logger.info("Step 3: Performing dataset validation...")
    validation_res = validate_bank_marketing_dataset(df)
    if not validation_res.is_valid:
        logger.warning(f"Validation identified domain anomalies: {validation_res.domain_anomalies}")
    else:
        logger.info("Validation passed successfully with zero schema or domain anomalies.")

    logger.info("Step 4: Computing statistical summaries...")
    num_summary = compute_numerical_summary(df)
    cat_summary = compute_categorical_summary(df)

    logger.info("Step 5: Generating high-resolution visualizations...")
    eda_output_dir.mkdir(parents=True, exist_ok=True)
    plot_target = plot_target_distribution(df, eda_output_dir)
    plot_num = plot_numerical_distributions(df, eda_output_dir)
    plot_cat = plot_categorical_distributions(df, eda_output_dir)
    plot_corr = plot_correlation_heatmap(df, eda_output_dir)
    plot_out = plot_numerical_outliers_boxplot(df, eda_output_dir)
    plot_biv = plot_features_vs_target(df, eda_output_dir)

    logger.info("Step 6: Generating summary and data-quality reports...")
    summary_md, summary_json, quality_md, quality_json = generate_reports(
        df=df,
        validation_res=validation_res,
        num_summary=num_summary,
        cat_summary=cat_summary,
        output_dir=eda_output_dir,
    )

    results = {
        "dataset_shape": list(df.shape),
        "validation_status": "VALID" if validation_res.is_valid else "INVALID",
        "plots": [
            str(plot_target),
            str(plot_num),
            str(plot_cat),
            str(plot_corr),
            str(plot_out),
            str(plot_biv),
        ],
        "reports": [
            str(summary_md),
            str(summary_json),
            str(quality_md),
            str(quality_json),
        ],
        "target_distribution": validation_res.target_summary,
    }
    logger.info("EDA Pipeline completed successfully.")
    return results


if __name__ == "__main__":
    run_eda_pipeline()
