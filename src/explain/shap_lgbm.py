"""SHAP TreeExplainer interpretability and adverse-action reason codes for LightGBM.

Computes exact TreeExplainer SHAP values on tabular features, generates the top-20
global summary plot (fig_shap_summary.png), and extracts lender-grade adverse action
reason codes (top-5 positive risk drivers) for declined loan applicants.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils.logging import get_logger

logger = get_logger("RiskTrail.Explain.SHAP")

# Visual Design Palette from docs/DESIGN_SYSTEM.md
GRID_COLOR = "#E5E7EB"
GRAY_COLOR = "#6B7280"
RISK_COLOR = "#DC2626"
GOOD_COLOR = "#0EA5E9"

FEATURE_DESCRIPTIONS = {
    "D_39": "Delinquency: Days past due / revolving delinquency indicator",
    "D_41": "Delinquency: Secondary delinquency balance",
    "D_42": "Delinquency: High-delinquency historical ratio",
    "D_43": "Delinquency: Cumulative delinquency past 180 days",
    "D_44": "Delinquency: Recent severe delinquency flag",
    "P_2": "Payment: Ratio of total payments to balance",
    "P_3": "Payment: Minimum payment ratio",
    "P_4": "Payment: Total payment amount velocity",
    "B_1": "Balance: Total revolving credit balance",
    "B_2": "Balance: Credit card statement balance ratio",
    "B_3": "Balance: Ratio of balance to credit limit",
    "B_4": "Balance: Current credit line utilization",
    "B_5": "Balance: Unsecured loan balance",
    "B_9": "Balance: High-utilization balance segment",
    "S_3": "Spend: Total statement purchase spend",
    "S_5": "Spend: Discretionary retail spend velocity",
    "S_6": "Spend: Cash advance / ATM transaction volume",
    "S_7": "Spend: Travel and entertainment spend",
    "S_8": "Spend: Recurring utility / auto-pay debit spend",
    "R_1": "Risk: In-house proprietary behavioral risk score",
    "R_2": "Risk: Secondary external credit bureau risk score",
    "R_3": "Risk: Recent inquiry and credit velocity score",
    "D_63": "Categorical: Delinquency status category code",
    "D_64": "Categorical: Debt-to-income categorization",
    "B_30": "Categorical: Bankcard trade status category",
    "B_38": "Categorical: Secondary credit line tier",
}


def describe_feature(feat_name: str) -> str:
    """Provide a human-readable, lender-grade explanation for an aggregated feature."""
    parts = feat_name.split("__")
    base = parts[0]
    agg = parts[1] if len(parts) > 1 else ""

    base_desc = FEATURE_DESCRIPTIONS.get(base, f"Account metric {base}")
    agg_desc_map = {
        "last": "Most recent statement value",
        "first": "Oldest statement baseline value",
        "mean": "13-month average level",
        "std": "Volatility over time",
        "min": "Historical minimum level",
        "max": "Historical peak level",
        "delta_last_first": "Net change from oldest to most recent statement",
        "slope": "Directional monthly growth trajectory (slope)",
        "delta_last_mean": "Deviation of recent statement from historical average",
        "nunique": "Number of category shifts",
        "count": "Total statement count",
    }
    agg_desc = agg_desc_map.get(agg, agg)
    return f"{base_desc} ({agg_desc})"


def compute_shap_explanations(
    model_path: Union[str, Path] = PROJECT_ROOT / "results" / "lgbm" / "model.txt",
    data_dir: Union[str, Path] = PROJECT_ROOT / "data" / "processed",
    output_dir: Union[str, Path] = PROJECT_ROOT / "results" / "lgbm",
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """Compute TreeExplainer SHAP values on test split and persist explanation artifacts.

    Returns:
        shap_values: Array of SHAP values [150, 214].
        X_test: Test features DataFrame (150 rows).
        feature_names: List of 214 feature column names.
    """
    model_path = Path(model_path)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    booster = lgb.Booster(model_file=str(model_path))

    tab_df = pd.read_parquet(data_dir / "tabular.parquet")
    splits_df = pd.read_parquet(data_dir / "splits.parquet")

    split_id_col = "customer_ID" if "customer_ID" in splits_df.columns else "customer_id"
    tab_id_col = "customer_id" if "customer_id" in tab_df.columns else "customer_ID"

    test_cids = splits_df[splits_df["split"] == "test"][split_id_col].values
    test_tab = tab_df.set_index(tab_id_col).loc[test_cids]

    feature_cols = [c for c in test_tab.columns if c not in ["target"]]
    X_test = test_tab[feature_cols]

    logger.info(f"Computing SHAP TreeExplainer values for {len(X_test)} test applicants across {len(feature_cols)} features...")
    explainer = shap.TreeExplainer(booster)
    raw_sv = explainer.shap_values(X_test)

    # In binary classification, LightGBM TreeExplainer outputs list [neg_shap, pos_shap] or single array
    if isinstance(raw_sv, list):
        shap_values = raw_sv[1] if len(raw_sv) > 1 else raw_sv[0]
    else:
        shap_values = raw_sv

    shap_values = np.asarray(shap_values, dtype=np.float32)

    # Persist SHAP values and feature index
    np.save(output_dir / "shap_values.npy", shap_values)
    with open(output_dir / "shap_feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    logger.info(f"Saved SHAP explanations to {output_dir}")
    return shap_values, X_test, feature_cols


def plot_fig_shap_summary(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    output_path: Path = PROJECT_ROOT / "reports" / "figures" / "fig_shap_summary.png",
    max_display: int = 20,
) -> None:
    """Generate publication-grade SHAP summary bar chart matching docs/DESIGN_SYSTEM.md."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute mean absolute SHAP values per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:max_display]
    top_indices = top_indices[::-1]  # ascending for horizontal bar chart

    feature_names = [X_test.columns[i] for i in top_indices]
    values = [mean_abs_shap[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    bars = ax.barh(
        range(len(values)),
        values,
        color="#2563EB",  # LightGBM blue
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
    )

    for i, bar in enumerate(bars):
        w = bar.get_width()
        ax.annotate(
            f" {w:.3f}",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            va="center",
            fontsize=8.5,
            color="#1F2937",
        )

    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(feature_names, fontsize=9)
    ax.set_xlabel("Mean Absolute SHAP Value (|log-odds contribution|)", fontsize=10)
    ax.set_title(
        "Top 20 Tabular Risk Drivers: Delinquency Balances and Trajectory Dominate",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlim(0.0, max(values) * 1.15)
    ax.grid(axis="x", linestyle="--", alpha=0.5, color=GRID_COLOR)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def get_customer_reason_codes(
    customer_id: str,
    shap_values: Optional[np.ndarray] = None,
    X_test: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Extract top-k adverse action reason codes for a specific customer.

    Features with the largest positive SHAP values are the primary contributors
    pushing the model's prediction toward default (adverse credit decision).

    Args:
        customer_id: Target customer identifier.
        shap_values: Precomputed SHAP array or None (loads from disk).
        X_test: Feature DataFrame or None.
        feature_cols: Feature names or None.
        top_k: Number of adverse action reason codes to return (default 5).

    Returns:
        List[Dict[str, Any]]: Ordered list of top adverse action reason codes.
    """
    if shap_values is None or X_test is None or feature_cols is None:
        shap_values, X_test, feature_cols = compute_shap_explanations()

    if customer_id not in X_test.index:
        raise KeyError(f"Customer '{customer_id}' not found in evaluated test set.")

    row_idx = X_test.index.get_loc(customer_id)
    cust_shap = shap_values[row_idx]
    cust_feat_values = X_test.loc[customer_id]

    # Rank features by positive SHAP contribution (increasing risk)
    pos_indices = np.argsort(cust_shap)[::-1]
    top_pos_indices = [idx for idx in pos_indices if cust_shap[idx] > 0][:top_k]

    reasons: List[Dict[str, Any]] = []
    for rank, idx in enumerate(top_pos_indices, start=1):
        feat = feature_cols[idx]
        val = float(cust_feat_values[feat])
        impact = float(cust_shap[idx])
        desc = describe_feature(feat)
        reasons.append({
            "rank": rank,
            "feature": feat,
            "feature_value": round(val, 4),
            "shap_impact": round(impact, 4),
            "adverse_action_reason": desc,
            "feature_group": feat.split("_")[0],
        })

    return reasons


def format_adverse_action_notice(
    customer_id: str,
    risk_score: float,
    cutoff: float,
    reason_codes: List[Dict[str, Any]],
) -> str:
    """Format a compliance-ready, lender-grade adverse action statement for a declined customer."""
    lines = [
        "=" * 85,
        "CREDIT DECISION NOTICE — ADVERSE ACTION STATEMENT",
        "=" * 85,
        f"Applicant ID:          {customer_id}",
        f"Application Decision:  DECLINED (Credit Policy Cutoff: {cutoff:.3f})",
        f"Assessed Risk Score:   {risk_score:.4f} (High Default Likelihood)",
        "-" * 85,
        "PRIMARY ADVERSE ACTION REASONS (Ordered by Quantitative Impact):",
        "-" * 85,
    ]

    for r in reason_codes:
        lines.append(
            f"{r['rank']}. {r['feature']:<26} | Impact: +{r['shap_impact']:.3f} | Value: {r['feature_value']:<8} | {r['adverse_action_reason']}"
        )

    lines.append("=" * 85)
    return "\n".join(lines)


def run_shap_pipeline() -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """Execute full LightGBM SHAP explainability pipeline."""
    logger.info("Starting Phase 7: LightGBM SHAP Explainability Pipeline")
    shap_values, X_test, feature_cols = compute_shap_explanations()
    plot_fig_shap_summary(shap_values, X_test)
    logger.info("SHAP explainability pipeline completed successfully.")
    return shap_values, X_test, feature_cols


if __name__ == "__main__":
    run_shap_pipeline()
