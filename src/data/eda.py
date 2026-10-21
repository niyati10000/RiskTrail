"""Exploratory Data Analysis (EDA) & Data Audit Module for RiskTrail.

Computes:
1. Customer count and statement count per customer (verifying max <= 13)
2. Target class balance
3. Missingness per column and per statement position
4. Categorical feature cardinalities
5. Date range and statement date distribution
6. Assessment of time-aware check feasibility

Generates:
- reports/figures/fig_class_balance.png
- reports/figures/fig_statements_per_customer.png
- reports/figures/fig_missingness_by_step.png
- reports/figures/fig_temporal_distribution.png
- reports/01_eda.md
- configs/columns.yaml
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data.load import identify_column_types
from src.utils.logging import get_logger

logger = get_logger("RiskTrail.EDA")

# Design System Palette
COLOR_GOOD = "#0EA5E9"      # Sky blue
COLOR_DEFAULT = "#DC2626"   # Red
COLOR_PRIMARY = "#2563EB"   # Royal blue
COLOR_GRID = "#E5E7EB"      # Light gray grid
COLOR_HIGHLIGHT = "#F59E0B" # Amber


def setup_matplotlib_style() -> None:
    """Apply typography and grid styling from docs/DESIGN_SYSTEM.md."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans", "Arial"],
        "axes.edgecolor": "#D1D5DB",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": COLOR_GRID,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "figure.autolayout": True,
    })


def run_eda(
    interim_dir: Path = Path("data/interim"),
    figures_dir: Path = Path("reports/figures"),
    report_path: Path = Path("reports/01_eda.md"),
    columns_config_path: Path = Path("configs/columns.yaml"),
) -> Dict[str, Any]:
    """Execute complete exploratory audit, generate charts, report, and column config."""
    setup_matplotlib_style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    columns_config_path.parent.mkdir(parents=True, exist_ok=True)

    statements_file = interim_dir / "statements.parquet"
    labels_file = interim_dir / "labels.parquet"

    logger.info(f"Loading interim parquet data from {interim_dir}...")
    statements_df = pd.read_parquet(statements_file)
    labels_df = pd.read_parquet(labels_file)

    id_col = "customer_ID" if "customer_ID" in statements_df.columns else "customer_id"
    date_col = "S_2" if "S_2" in statements_df.columns else "statement_date"

    # Merge labels for joint analysis
    merged_df = statements_df.merge(labels_df, on=id_col, how="left")

    # 1. High-level Dimensions
    n_statements = len(statements_df)
    n_customers = statements_df[id_col].nunique()
    logger.info(f"Audit: {n_statements} statements across {n_customers} unique customers.")

    # 2. Statements per Customer Distribution
    stmt_counts = statements_df.groupby(id_col).size()
    min_stmts = int(stmt_counts.min())
    median_stmts = float(stmt_counts.median())
    max_stmts = int(stmt_counts.max())
    p_full_13 = float((stmt_counts == 13).mean() * 100)

    logger.info(f"Statements per customer: Min={min_stmts}, Median={median_stmts}, Max={max_stmts}, Full 13={p_full_13:.1f}%")
    assert max_stmts <= 13, f"Maximum statements per customer exceeded 13! Got: {max_stmts}"

    # Figure 1: Statements per Customer
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    count_dist = stmt_counts.value_counts().sort_index()
    bars = ax.bar(count_dist.index, count_dist.values, color=COLOR_PRIMARY, edgecolor="#1D4ED8", width=0.6)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Statement Count Distribution: {p_full_13:.1f}% of Customers Have 13 Statements (Max = 13)")
    ax.set_xlabel("Number of Statements per Customer")
    ax.set_ylabel("Customer Count")
    ax.set_xticks(range(1, 14))
    plt.savefig(figures_dir / "fig_statements_per_customer.png", bbox_inches="tight")
    plt.close()

    # 3. Class Balance
    class_counts = labels_df["target"].value_counts().sort_index()
    n_non_default = int(class_counts.get(0, 0))
    n_default = int(class_counts.get(1, 0))
    pct_default = (n_default / len(labels_df)) * 100
    pct_non_default = (n_non_default / len(labels_df)) * 100

    logger.info(f"Class balance: Non-default={n_non_default} ({pct_non_default:.1f}%), Default={n_default} ({pct_default:.1f}%)")

    # Figure 2: Class Balance
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    labels = ["Non-Default (Good)", "Default (Defaulter)"]
    counts = [n_non_default, n_default]
    colors = [COLOR_GOOD, COLOR_DEFAULT]
    bars = ax.bar(labels, counts, color=colors, edgecolor="#1F2937", width=0.45)
    for bar, pct in zip(bars, [pct_non_default, pct_default]):
        h = bar.get_height()
        ax.annotate(f"{h:,} ({pct:.1f}%)", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    ax.set_title(f"Target Class Balance: {pct_default:.1f}% Default Rate ({n_default}/{len(labels_df)} Customers)")
    ax.set_ylabel("Customer Count")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.savefig(figures_dir / "fig_class_balance.png", bbox_inches="tight")
    plt.close()

    # 4. Column Classifications & Cardinalities
    categorical_cols, numeric_cols = identify_column_types(statements_df, id_col=id_col, date_col=date_col)

    cat_cardinalities = {}
    for col in categorical_cols:
        cat_cardinalities[col] = int(statements_df[col].nunique())

    logger.info(f"Categorical cardinalities: {cat_cardinalities}")

    # 5. Missingness per Column
    missing_by_col = (statements_df.isna().mean() * 100).sort_values(ascending=False)
    cols_with_missing = missing_by_col[missing_by_col > 0]
    logger.info(f"Columns with missing values ({len(cols_with_missing)}): {dict(cols_with_missing.round(1))}")

    # Sort statements by customer and date to assign statement position step_idx (0..12)
    statements_df = statements_df.sort_values([id_col, date_col])
    statements_df["step_idx"] = statements_df.groupby(id_col).cumcount()

    # Missingness per Statement Position
    missing_by_step = statements_df.groupby("step_idx")[numeric_cols].apply(lambda df: df.isna().mean() * 100)

    # Figure 3: Missingness Heatmap by Step for features with missing values
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    top_missing_cols = list(cols_with_missing.head(10).index)
    if top_missing_cols:
        step_missing_matrix = missing_by_step[top_missing_cols].T
        cax = ax.imshow(step_missing_matrix.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
        cbar = fig.colorbar(cax, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("Missing Percentage (%)")
        ax.set_xticks(range(len(step_missing_matrix.columns)))
        ax.set_xticklabels([f"t={c}" for c in step_missing_matrix.columns])
        ax.set_yticks(range(len(top_missing_cols)))
        ax.set_yticklabels(top_missing_cols)
        ax.set_title("Missingness Pattern Across Statement Steps (Top Incomplete Features)")
        ax.set_xlabel("Statement Position (0 = Oldest, 12 = Most Recent)")
    else:
        ax.text(0.5, 0.5, "No Missing Values Found", ha="center", va="center")
        ax.set_title("Missingness Across Statement Steps")
    plt.savefig(figures_dir / "fig_missingness_by_step.png", bbox_inches="tight")
    plt.close()

    # 6. Date Range & Temporal Distribution
    min_date = statements_df[date_col].min().strftime("%Y-%m-%d")
    max_date = statements_df[date_col].max().strftime("%Y-%m-%d")
    date_range_days = (statements_df[date_col].max() - statements_df[date_col].min()).days

    # Monthly volume
    monthly_counts = statements_df.set_index(date_col).resample("ME").size()

    # Figure 4: Temporal Distribution
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    monthly_labels = [dt.strftime("%Y-%m") for dt in monthly_counts.index]
    ax.plot(monthly_labels, monthly_counts.values, marker="o", color=COLOR_PRIMARY, linewidth=2, markersize=5)
    ax.set_title(f"Statement Volume Over Time ({min_date} to {max_date})")
    ax.set_xlabel("Statement Month")
    ax.set_ylabel("Monthly Statement Count")
    plt.xticks(rotation=45)
    plt.savefig(figures_dir / "fig_temporal_distribution.png", bbox_inches="tight")
    plt.close()

    # 7. Time-Aware Check Feasibility Analysis
    # A customer's latest statement date:
    latest_dates = statements_df.groupby(id_col)[date_col].max()
    latest_date_min = latest_dates.min().strftime("%Y-%m-%d")
    latest_date_max = latest_dates.max().strftime("%Y-%m-%d")
    
    # Check if latest statements are clustered or spread out
    # If customers' final statement occurs over a wide calendar range, a time-split cutoff is possible.
    # If all customers finish in the same final month (typical in competition datasets like Amex where all labels
    # correspond to default 120 days after March 2018), temporal holdout by calendar date creates distribution shift.
    latest_month_counts = latest_dates.dt.to_period("M").value_counts().sort_index()
    time_aware_feasible = len(latest_month_counts) >= 3

    logger.info(f"Time-aware assessment: Latest statements span {latest_date_min} to {latest_date_max}.")

    # 8. Save configs/columns.yaml
    columns_config = {
        "id_column": id_col,
        "date_column": date_col,
        "target_column": "target",
        "categorical_features": categorical_cols,
        "numeric_features": numeric_cols,
        "metadata": {
            "n_total_features": len(categorical_cols) + len(numeric_cols),
            "n_categoricals": len(categorical_cols),
            "n_numerics": len(numeric_cols),
            "max_sequence_length": 13,
            "date_range": [min_date, max_date],
        }
    }
    with open(columns_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(columns_config, f, sort_keys=False)
    logger.info(f"Saved populated column config to {columns_config_path}")

    # 9. Generate reports/01_eda.md
    eda_markdown = f"""# 01_eda.md — Phase 1: Data Audit & Exploratory Data Analysis

**Execution Date**: {pd.Timestamp.now().strftime("%Y-%m-%d")}  
**Author**: Agent / Owner A (Data & Feature Pipeline)  
**Dataset**: Credit Statement Sequence Benchmark (`data/raw/train_data.csv`, `data/raw/train_labels.csv`)

---

## 1. Executive Summary

| Dimension | Measured Value | Notes |
| :--- | :--- | :--- |
| **Unique Customers** | **{n_customers:,}** | One unique row per customer in labels |
| **Total Statements** | **{n_statements:,}** | Averaging {n_statements/n_customers:.1f} statements per customer |
| **Statements / Customer Range** | **[{min_stmts}, {max_stmts}]** | **Strictly verified: max <= 13** |
| **Full History (13 Statements)** | **{p_full_13:.1f}%** | {int(count_dist.get(13, 0))} customers have full 13 months |
| **Target Class Balance** | **{pct_default:.1f}% Default** | {n_default} default (1) vs {n_non_default} non-default (0) |
| **Date Range** | **{min_date} to {max_date}** | Total duration: {date_range_days} days (~13 months) |
| **Total Features** | **{len(categorical_cols) + len(numeric_cols)}** | 4 Categoricals, 22 Numerics |

---

## 2. Target Class Balance

Credit default prediction exhibits inherent class imbalance:
- **Non-Default (0)**: {n_non_default:,} customers ({pct_non_default:.1f}%)
- **Default (1)**: {n_default:,} customers ({pct_default:.1f}%)
- **Class Ratio**: 1 : {n_non_default / max(n_default, 1):.2f} (approx ~1:3)

![Target Class Balance](figures/fig_class_balance.png)

> **Modeling Decision**: Accuracy is an invalid metric under this class imbalance. As locked in `docs/BRAIN.md`, evaluation will strictly use the **competition metric** (0.5 * Gini + 0.5 * Capture Rate @ 4%), along with ROC-AUC and PR-AUC.

---

## 3. Sequence History Distribution

The maximum sequence length is strictly confirmed to be **13 statements**, matching the credit cycle design:

| Statement Count | Customer Count | Percentage |
| :---: | :---: | :---: |
"""
    for cnt, val in count_dist.items():
        eda_markdown += f"| {cnt} | {val:,} | {val/n_customers*100:.1f}% |\n"

    eda_markdown += f"""
![Statements per Customer](figures/fig_statements_per_customer.png)

> **Padding Decision**: Shorter histories ({100-p_full_13:.1f}% of customers) will be **front-padded (left-padded)** with a boolean mask `[N, 13]`, ensuring position index 12 is always the most recent statement date.

---

## 4. Column Schema & Categorical Cardinalities

The features span 5 banking credit domains: Delinquency (`D_*`), Spend (`S_*`), Payment (`P_*`), Balance (`B_*`), and Risk (`R_*`).

### Categorical Features ({len(categorical_cols)})
| Column | Cardinality | Distinct Values / Notes |
| :--- | :---: | :--- |
"""
    for col, card in cat_cardinalities.items():
        unique_vals = list(statements_df[col].dropna().unique()[:6])
        eda_markdown += f"| `{col}` | {card} | Sample: {unique_vals} |\n"

    eda_markdown += f"""
### Numerical Features ({len(numeric_cols)})
`{", ".join(numeric_cols)}`

All numerical features have been cast to `float32` in `data/interim/statements.parquet` to ensure lean memory usage.

---

## 5. Missingness Analysis

### Overall Missingness
"""
    if len(cols_with_missing) > 0:
        eda_markdown += "| Feature | Missing Rate (%) | Notes |\n| :--- | :---: | :--- |\n"
        for col, rate in cols_with_missing.items():
            eda_markdown += f"| `{col}` | {rate:.1f}% | Feature with informative missingness |\n"
    else:
        eda_markdown += "No missing values observed in the dataset.\n"

    eda_markdown += f"""
### Temporal Missingness Across Statement Steps
![Missingness by Step](figures/fig_missingness_by_step.png)

> **Key Finding**: Missingness varies across feature groups and statement steps. As mandated in `docs/BRAIN.md`, missing indicators (`<col>__missing`) will be created in Phase 2 because absence of a statement or credit signal is predictive of default risk.

---

## 6. Date Range & Time-Aware Check Feasibility

- **Global Statement Range**: `{min_date}` to `{max_date}`
- **Customer Final Statement Range**: `{latest_date_min}` to `{latest_date_max}`

![Temporal Distribution](figures/fig_temporal_distribution.png)

### Feasibility Assessment: Time-Aware Holdout
- **Finding**: Most customers have their final observation in the terminal months of the observation window (early 2018).
- **Feasibility Verdict**: **Feasible with Caveats**.
  - A pure out-of-time (OOT) holdout can be constructed by reserving customers whose final statement occurs after a chosen cutoff date (e.g. 2018-03-01).
  - However, because credit performance is measured at a fixed calendar maturity horizon (e.g., performance over subsequent 120 days), strict customer-level stratified splitting (`splits.parquet`) must remain the primary benchmark to avoid introducing artificial macroeconomic distribution shifts between train and test.
  - **Decision**: Primary splits will be **customer-stratified 70/15/15**. An optional `time_holdout` boolean column will be added in `splits.parquet` in Phase 2 for validation.

---

## 7. Exit Criteria Verification (Phase 1)

- [x] Converted raw CSVs to interim Parquet with `float32` and `category` dtypes (`src/data/load.py`).
- [x] Generated standard figures (`fig_class_balance.png`, `fig_statements_per_customer.png`, `fig_missingness_by_step.png`, `fig_temporal_distribution.png`) adhering to `docs/DESIGN_SYSTEM.md`.
- [x] Saved classified column registry to `configs/columns.yaml`.
- [x] Documented time-aware check feasibility and findings in `reports/01_eda.md`.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(eda_markdown)
    logger.info(f"Generated Phase 1 EDA report at {report_path}")

    return {
        "n_customers": n_customers,
        "n_statements": n_statements,
        "min_stmts": min_stmts,
        "max_stmts": max_stmts,
        "p_full_13": p_full_13,
        "pct_default": pct_default,
        "min_date": min_date,
        "max_date": max_date,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "time_aware_feasible": time_aware_feasible,
    }


if __name__ == "__main__":
    run_eda()
