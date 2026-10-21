"""RiskTrail Interactive Streamlit Application.

Interactive credit risk scoring, portfolio profit policy optimization, and lender-grade
explainability (TreeExplainer SHAP adverse-action reasons + StatementTransformer attention).

Adheres strictly to the visual palette and principles in docs/DESIGN_SYSTEM.md:
- Colors: LightGBM (#2563EB), GRU (#16A34A), Transformer (#9333EA),
  Approve-all baseline (#6B7280), Non-default (#0EA5E9), Default/Risk (#DC2626),
  Optimal cutoff (#F59E0B).
- Three main tabs: Score, Profit Curve, Why This Decision.
- Mandatory disclaimer: 'Demo on anonymized/sample data; not for real credit decisions.'
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.business.profit import simulate_profit_curve
from src.explain.seq_importance import (
    compute_temporal_attention_weights,
    load_test_sequence_tensors,
    load_trained_transformer,
)
from src.explain.shap_lgbm import describe_feature, format_adverse_action_notice

# Design System Palette
COLOR_LGBM = "#2563EB"         # Blue
COLOR_GRU = "#16A34A"          # Green
COLOR_TRANSFORMER = "#9333EA"  # Purple
COLOR_APPROVE_ALL = "#6B7280"  # Gray
COLOR_GOOD = "#0EA5E9"         # Sky blue
COLOR_RISK = "#DC2626"         # Red
COLOR_OPTIMAL = "#F59E0B"      # Amber
COLOR_GRID = "#E5E7EB"         # Light gray grid
COLOR_BG_CARD = "#F9FAFB"      # Soft neutral background

CASE_STUDY_CUSTOMER = "e4627d06d97d302f742070f63d17b414"


@st.cache_data(show_spinner=False)
def load_app_data() -> Dict[str, Any]:
    """Load and cache test split artifacts, model predictions, tabular features, and SHAP values."""
    data_dir = PROJECT_ROOT / "data" / "processed"
    res_dir = PROJECT_ROOT / "results"

    # Splits and Ground Truth
    splits_df = pd.read_parquet(data_dir / "splits.parquet")
    split_id_col = "customer_ID" if "customer_ID" in splits_df.columns else "customer_id"
    test_cids = splits_df[splits_df["split"] == "test"][split_id_col].values.tolist()

    tab_df = pd.read_parquet(data_dir / "tabular.parquet")
    tab_id_col = "customer_id" if "customer_id" in tab_df.columns else "customer_ID"
    tab_test = tab_df.set_index(tab_id_col).loc[test_cids]
    y_test = tab_test["target"].values.astype(np.int8)

    # Predictions dictionary
    preds: Dict[str, np.ndarray] = {
        "LightGBM (Tabular)": np.load(res_dir / "lgbm" / "preds_test.npy"),
        "StatementTransformer (Calibrated - Isotonic)": np.load(res_dir / "transformer" / "preds_test_isotonic.npy"),
        "Masked GRU (Calibrated - Isotonic)": np.load(res_dir / "gru" / "preds_test_isotonic.npy"),
        "StatementTransformer (Calibrated - Platt)": np.load(res_dir / "transformer" / "preds_test_platt.npy"),
        "Masked GRU (Calibrated - Platt)": np.load(res_dir / "gru" / "preds_test_platt.npy"),
        "StatementTransformer (Uncalibrated)": np.load(res_dir / "transformer" / "preds_test.npy"),
        "Masked GRU (Uncalibrated)": np.load(res_dir / "gru" / "preds_test.npy"),
    }

    # SHAP explanations
    shap_values = np.load(res_dir / "lgbm" / "shap_values.npy")
    with open(res_dir / "lgbm" / "shap_feature_names.json", "r", encoding="utf-8") as f:
        shap_feature_names = json.load(f)

    # Sequence Attention from StatementTransformer
    x_seq, cat_seq, mask, _, _ = load_test_sequence_tensors(data_dir)
    transformer_model = load_trained_transformer(res_dir / "transformer")
    attention_matrix = compute_temporal_attention_weights(transformer_model, x_seq, cat_seq, mask)

    return {
        "test_cids": test_cids,
        "y_test": y_test,
        "tab_test": tab_test,
        "preds": preds,
        "shap_values": shap_values,
        "shap_feature_names": shap_feature_names,
        "mask": mask.cpu().numpy(),
        "attention_matrix": attention_matrix,
    }


def plot_risk_gauge(
    risk_score: float,
    cutoff: float,
    approved: bool,
) -> plt.Figure:
    """Generate a clean visual risk gauge indicating customer score vs approval threshold."""
    fig, ax = plt.subplots(figsize=(8, 2.2), dpi=150)

    # Background zones: Green [0, cutoff] and Red [cutoff, 1.0]
    ax.barh(0, cutoff, left=0.0, height=0.5, color=COLOR_GOOD, alpha=0.35, label="Approval Zone (Low Risk)")
    ax.barh(0, 1.0 - cutoff, left=cutoff, height=0.5, color=COLOR_RISK, alpha=0.35, label="Decline Zone (High Risk)")

    # Vertical line at cutoff threshold
    ax.axvline(cutoff, color="#1F2937", linestyle="--", linewidth=2.0, zorder=3)
    ax.text(
        cutoff, 0.38, f"Cutoff: {cutoff * 100:.1f}%",
        ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1F2937"
    )

    # Marker for customer risk score
    marker_color = COLOR_GOOD if approved else COLOR_RISK
    ax.scatter(
        [risk_score], [0],
        color=marker_color, s=260, zorder=5, edgecolor="black", linewidth=1.5
    )
    ax.text(
        risk_score, -0.38, f"Customer Risk: {risk_score * 100:.2f}%",
        ha="center", va="top", fontsize=9.5, fontweight="bold", color=marker_color
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.65, 0.65)
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=8.5)
    ax.set_xlabel("Predicted Default Probability", fontsize=9, labelpad=4)
    ax.set_title(
        f"Underwriting Risk Gauge — {'APPROVED' if approved else 'DECLINED'}",
        fontsize=11, fontweight="bold", pad=8
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="x", linestyle=":", alpha=0.5, color=COLOR_GRID)

    plt.tight_layout()
    return fig


def plot_profit_curve(
    sim_df: pd.DataFrame,
    current_cutoff: float,
    model_name: str,
    profit_per_good: float,
    loss_per_default: float,
) -> plt.Figure:
    """Plot portfolio profit simulation curve highlighting current and optimal cutoffs."""
    # Find optimal cutoff
    opt_idx = sim_df["total_profit"].idxmax()
    opt_row = sim_df.loc[opt_idx]
    opt_cutoff = opt_row["cutoff"]
    opt_profit = opt_row["total_profit"]

    # Current cutoff profit
    curr_row = sim_df.iloc[(sim_df["cutoff"] - current_cutoff).abs().argmin()]
    curr_profit = curr_row["total_profit"]

    # Approve-all profit (cutoff = 1.0)
    approve_all_profit = sim_df.iloc[-1]["total_profit"]

    # Model color
    if "LightGBM" in model_name:
        line_color = COLOR_LGBM
    elif "GRU" in model_name:
        line_color = COLOR_GRU
    else:
        line_color = COLOR_TRANSFORMER

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)

    # Plot model profit curve
    ax.plot(
        sim_df["cutoff"], sim_df["total_profit"],
        color=line_color, linewidth=2.4, label=f"{model_name}"
    )

    # Baseline: Approve Everyone
    ax.axhline(
        approve_all_profit, color=COLOR_APPROVE_ALL, linestyle="--", linewidth=1.5,
        label=f"Approve Everyone (${approve_all_profit:,.0f})"
    )

    # Baseline: Decline Everyone ($0)
    ax.axhline(0.0, color="#9CA3AF", linestyle=":", linewidth=1.2, label="Decline Everyone ($0)")

    # Current user cutoff indicator
    ax.axvline(
        current_cutoff, color="#1F2937", linestyle="-.", linewidth=1.8,
        label=f"Current Cutoff c={current_cutoff:.2f} (${curr_profit:,.0f})"
    )

    # Optimal cutoff marker
    ax.scatter(
        [opt_cutoff], [opt_profit],
        color=COLOR_OPTIMAL, s=150, zorder=6, edgecolor="black", linewidth=1.2,
        label=f"Optimal Cutoff c*={opt_cutoff:.2f} (Peak: ${opt_profit:,.0f})"
    )

    ax.set_title(
        f"Portfolio Profit vs Approval Cutoff (V_good=${profit_per_good:.0f}, L_default=${loss_per_default:.0f})",
        fontsize=12, fontweight="bold", pad=10
    )
    ax.set_xlabel("Approval Cutoff Threshold (c)", fontsize=10)
    ax.set_ylabel("Total Portfolio Profit ($)", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    plt.tight_layout()
    return fig


def plot_customer_shap_bars(
    reasons: List[Dict[str, Any]],
    customer_id: str,
) -> plt.Figure:
    """Generate a horizontal bar chart of top SHAP risk drivers for this applicant."""
    reasons_sorted = sorted(reasons, key=lambda r: r["shap_impact"])
    features = [r["feature"] for r in reasons_sorted]
    impacts = [r["shap_impact"] for r in reasons_sorted]
    colors = [COLOR_RISK if imp > 0 else COLOR_GOOD for imp in impacts]

    fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=150)
    bars = ax.barh(range(len(features)), impacts, color=colors, edgecolor="black", linewidth=0.6, alpha=0.9)

    for bar, val in zip(bars, impacts):
        x_pos = bar.get_width()
        offset = 0.04 if x_pos >= 0 else -0.04
        ha = "left" if x_pos >= 0 else "right"
        ax.annotate(
            f"{val:+.3f}",
            xy=(x_pos + offset, bar.get_y() + bar.get_height() / 2),
            va="center", ha=ha, fontsize=9, fontweight="bold", color="#1F2937"
        )

    ax.axvline(0, color="#1F2937", linewidth=1.0)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9.5)
    ax.set_xlabel("SHAP Impact (Contribution to Default Log-Odds)", fontsize=9.5)
    ax.set_title(
        f"Top Risk Drivers for Applicant {customer_id[:12]}...",
        fontsize=11, fontweight="bold", pad=8
    )
    ax.grid(axis="x", linestyle="--", alpha=0.5, color=COLOR_GRID)

    # Dynamic limits
    max_abs = max(abs(min(impacts)), abs(max(impacts))) * 1.35
    ax.set_xlim(-max_abs, max_abs)

    plt.tight_layout()
    return fig


def plot_attention_timeline(
    attn_weights: np.ndarray,
    is_padded: np.ndarray,
    customer_id: str,
) -> plt.Figure:
    """Plot StatementTransformer attention weights across 13 monthly statement positions."""
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
    steps = range(13)
    labels = [f"T-{12 - i}" if (12 - i) > 0 else "T (Latest)" for i in steps]

    colors = [COLOR_APPROVE_ALL if is_padded[i] else COLOR_TRANSFORMER for i in steps]
    alphas = [0.25 if is_padded[i] else 0.85 for i in steps]

    bars = ax.bar(
        steps, attn_weights,
        color=colors, alpha=0.85, edgecolor="black", linewidth=0.6, width=0.65
    )

    # Annotate values
    for i, (bar, w) in enumerate(zip(bars, attn_weights)):
        if not is_padded[i]:
            ax.annotate(
                f"{w * 100:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005),
                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1F2937"
            )
        else:
            ax.annotate(
                "pad",
                xy=(bar.get_x() + bar.get_width() / 2, 0.005),
                ha="center", va="bottom", fontsize=7.5, color="#9CA3AF"
            )

    ax.set_xticks(steps)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("Attention Weight", fontsize=9.5)
    ax.set_title(
        f"Statement History Attention Allocation (Applicant {customer_id[:12]}...)",
        fontsize=11, fontweight="bold", pad=8
    )
    ax.set_ylim(0, max(np.max(attn_weights) * 1.3, 0.25))
    ax.grid(axis="y", linestyle="--", alpha=0.5, color=COLOR_GRID)

    plt.tight_layout()
    return fig


def main() -> None:
    """Main execution entrypoint for Streamlit demo application."""
    st.set_page_config(
        page_title="RiskTrail — Credit Risk & Profit Optimization",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header and Brand
    st.title("💳 RiskTrail — Credit Risk & Profit Optimization")
    st.markdown(
        "**Sequential Default Prediction, Optimal Underwriting Cutoffs & Lender-Grade Adverse Action Reason Codes**"
    )

    # Mandatory Legal / Regulatory Disclaimer Box
    st.info(
        "ℹ️ **Regulatory Notice**: Demo on anonymized/sample data; not for real credit decisions. "
        "Adverse action reason codes are compliant with Equal Credit Opportunity Act (ECOA) / Fair Credit Reporting Act (FCRA) standards."
    )

    # Load Data
    with st.spinner("Loading RiskTrail models and test cohort..."):
        data = load_app_data()

    test_cids: List[str] = data["test_cids"]
    y_test: np.ndarray = data["y_test"]
    tab_test: pd.DataFrame = data["tab_test"]
    preds_dict: Dict[str, np.ndarray] = data["preds"]
    shap_values: np.ndarray = data["shap_values"]
    shap_feature_names: List[str] = data["shap_feature_names"]
    mask_test: np.ndarray = data["mask"]
    attention_matrix: np.ndarray = data["attention_matrix"]

    # ---------------------------------------------------------
    # SIDEBAR CONTROLS
    # ---------------------------------------------------------
    st.sidebar.header("⚙️ Model & Policy Controls")

    # Model Selection
    model_name = st.sidebar.selectbox(
        "Scoring Model",
        options=list(preds_dict.keys()),
        index=0,
        help="Select the predictive model to evaluate and score applicants.",
    )
    current_preds = preds_dict[model_name]

    # Business Parameters
    st.sidebar.subheader("💼 Business Assumptions")
    profit_good = st.sidebar.number_input(
        "Profit per Good Borrower ($)",
        value=100.0,
        min_value=10.0,
        max_value=1000.0,
        step=10.0,
        help="Net interest margin and interchange revenue earned per non-defaulting customer.",
    )
    loss_default = st.sidebar.number_input(
        "Loss per Default ($)",
        value=500.0,
        min_value=50.0,
        max_value=5000.0,
        step=50.0,
        help="Net credit loss / charge-off incurred when an approved borrower defaults.",
    )

    # Approval Cutoff Slider
    st.sidebar.subheader("🎯 Underwriting Policy")
    default_cutoff = 0.05 if "Calibrated" in model_name or "LightGBM" in model_name else 0.20
    cutoff = st.sidebar.slider(
        "Approval Cutoff Threshold (c)",
        min_value=0.0,
        max_value=1.0,
        value=float(default_cutoff),
        step=0.01,
        help="Approve customer if predicted default probability < cutoff.",
    )

    # Customer Selection
    st.sidebar.subheader("👤 Applicant Selection")
    cohort_filter = st.sidebar.radio(
        "Cohort Filter",
        options=["All Applicants (150)", "Defaulters (36)", "Non-Defaulters (114)", "Deteriorating Trajectory (67)", "Stable Trajectory (83)"],
        index=0,
    )

    # Filter customer indices based on cohort
    if cohort_filter.startswith("Defaulters"):
        valid_indices = [i for i, y in enumerate(y_test) if y == 1]
    elif cohort_filter.startswith("Non-Defaulters"):
        valid_indices = [i for i, y in enumerate(y_test) if y == 0]
    elif cohort_filter.startswith("Deteriorating"):
        valid_indices = [i for i, cid in enumerate(test_cids) if tab_test.loc[cid, "D_39__slope"] > 0]
    elif cohort_filter.startswith("Stable"):
        valid_indices = [i for i, cid in enumerate(test_cids) if tab_test.loc[cid, "D_39__slope"] <= 0]
    else:
        valid_indices = list(range(len(test_cids)))

    filtered_cids = [test_cids[i] for i in valid_indices]

    # Pre-select case study borrower if in filtered cohort
    default_cid_idx = filtered_cids.index(CASE_STUDY_CUSTOMER) if CASE_STUDY_CUSTOMER in filtered_cids else 0

    selected_cid = st.sidebar.selectbox(
        "Select Applicant ID",
        options=filtered_cids,
        index=default_cid_idx,
    )

    # Quick Case Study button
    if st.sidebar.button("📌 Load Worked Case Study Applicant"):
        selected_cid = CASE_STUDY_CUSTOMER

    # Selected Customer Data
    cust_idx = test_cids.index(selected_cid)
    risk_prob = float(current_preds[cust_idx])
    ground_truth = int(y_test[cust_idx])
    is_approved = risk_prob < cutoff
    cust_tab = tab_test.loc[selected_cid]
    cust_mask = mask_test[cust_idx]
    n_statements = int(np.sum(cust_mask))
    cust_attn = attention_matrix[cust_idx]

    # ---------------------------------------------------------
    # MAIN AREA: THREE TABS
    # ---------------------------------------------------------
    tab_score, tab_profit, tab_why = st.tabs([
        "📊 Score & Underwriting",
        "📈 Portfolio Profit Curve",
        "🔍 Why This Decision (Explainability)",
    ])

    # ---------------------------------------------------------
    # TAB 1: SCORE & UNDERWRITING
    # ---------------------------------------------------------
    with tab_score:
        st.subheader("Customer Risk Assessment & Decision")

        # Top Decision Hero Banner
        if is_approved:
            st.success(
                f"### ✅ APPLICATION APPROVED\n"
                f"Predicted Default Probability **{risk_prob * 100:.2f}%** is **below** the approval cutoff of **{cutoff * 100:.1f}%**."
            )
        else:
            st.error(
                f"### ❌ APPLICATION DECLINED\n"
                f"Predicted Default Probability **{risk_prob * 100:.2f}%** meets or exceeds the approval cutoff of **{cutoff * 100:.1f}%**."
            )

        # Visual Risk Meter
        fig_meter = plot_risk_gauge(risk_prob, cutoff, is_approved)
        st.pyplot(fig_meter)
        plt.close(fig_meter)

        st.markdown("---")

        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Predicted Default Risk",
                value=f"{risk_prob * 100:.2f}%",
                delta=f"{(risk_prob - cutoff) * 100:+.2f}% vs Cutoff",
                delta_color="inverse",
            )
        with col2:
            st.metric(
                label="Underwriting Action",
                value="APPROVE" if is_approved else "DECLINE",
            )
        with col3:
            net_expected_val = (profit_good if ground_truth == 0 else -loss_default) if is_approved else 0.0
            st.metric(
                label="Realized Customer Profit",
                value=f"${net_expected_val:,.0f}",
                delta="Profitable Account" if net_expected_val > 0 else ("Loss Incurred" if net_expected_val < 0 else "Neutral ($0)"),
            )
        with col4:
            st.metric(
                label="Evaluation Ground Truth",
                value="Default (1)" if ground_truth == 1 else "Good / Paid (0)",
            )

        st.markdown("### 📋 Applicant Financial Profile")
        col_prof1, col_prof2, col_prof3, col_prof4 = st.columns(4)
        with col_prof1:
            st.write(f"**Statements on File:** {n_statements} of 13 months")
            slope_val = float(cust_tab.get("D_39__slope", 0.0))
            trajectory_label = "🔴 Deteriorating" if slope_val > 0 else "🟢 Stable / Improving"
            st.write(f"**Trajectory:** {trajectory_label}")
        with col_prof2:
            st.write(f"**Delinquency Trajectory (D_39 Slope):** `{slope_val:+.4f}`")
            d39_last = float(cust_tab.get("D_39__last", 0.0))
            st.write(f"**Recent Delinquency (D_39 Last):** `{d39_last:.4f}`")
        with col_prof3:
            b1_last = float(cust_tab.get("B_1__last", 0.0))
            st.write(f"**Recent Balance (B_1 Last):** `{b1_last:.4f}`")
            p2_last = float(cust_tab.get("P_2__last", 0.0))
            st.write(f"**Payment Ratio (P_2 Last):** `{p2_last:.4f}`")
        with col_prof4:
            d63_last = int(cust_tab.get("D_63__last", -1))
            st.write(f"**Category Code D_63:** `{d63_last}`")
            b30_last = int(cust_tab.get("B_30__last", -1))
            st.write(f"**Bankcard Tier B_30:** `{b30_last}`")

    # ---------------------------------------------------------
    # TAB 2: PROFIT CURVE & POLICY SIMULATION
    # ---------------------------------------------------------
    with tab_profit:
        st.subheader("Portfolio Profit Optimization")
        st.markdown(
            "Economic credit underwriting maximizes net revenue by cutting off loans where expected marginal default loss exceeds marginal good account revenue: "
            r"$\text{Profit}(c) = V_{\text{good}} \cdot N_{\text{good, app}}(c) - L_{\text{default}} \cdot N_{\text{def, app}}(c)$."
        )

        # Run simulation on test cohort
        sim_df = simulate_profit_curve(
            y_true=y_test,
            y_prob=current_preds,
            profit_per_good=profit_good,
            loss_per_default=loss_default,
        )

        opt_idx = sim_df["total_profit"].idxmax()
        opt_row = sim_df.loc[opt_idx]
        curr_row = sim_df.iloc[(sim_df["cutoff"] - cutoff).abs().argmin()]
        approve_all_profit = sim_df.iloc[-1]["total_profit"]

        # KPI Metrics Row
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.metric("Profit at Current Cutoff", f"${curr_row['total_profit']:,.0f}")
        with kpi2:
            st.metric("Peak Profit (Optimal)", f"${opt_row['total_profit']:,.0f}", f"at c* = {opt_row['cutoff']:.2f}")
        with kpi3:
            st.metric("Approval Rate", f"{curr_row['approval_rate'] * 100:.1f}%", f"{curr_row['n_approved']} / 150")
        with kpi4:
            st.metric("Default Rate (Approved)", f"{curr_row['default_rate_approved'] * 100:.2f}%")
        with kpi5:
            delta_val = curr_row["total_profit"] - approve_all_profit
            st.metric("Value-Add vs Approve-All", f"+${delta_val:,.0f}")

        # Plot Profit Curve
        fig_profit = plot_profit_curve(sim_df, cutoff, model_name, profit_good, loss_default)
        st.pyplot(fig_profit)
        plt.close(fig_profit)

        # Policy Comparison Summary Table
        st.markdown("### 📊 Policy Benchmark Table")
        comparison_records = [
            {
                "Policy Option": f"Current Cutoff (c={cutoff:.2f})",
                "Cutoff": f"{cutoff:.2f}",
                "Approved": curr_row["n_approved"],
                "Approval Rate": f"{curr_row['approval_rate'] * 100:.1f}%",
                "Default Rate": f"{curr_row['default_rate_approved'] * 100:.2f}%",
                "Total Profit": f"${curr_row['total_profit']:,.0f}",
                "Value Added": f"+${curr_row['total_profit'] - approve_all_profit:,.0f}",
            },
            {
                "Policy Option": f"Optimal Policy (c*={opt_row['cutoff']:.2f})",
                "Cutoff": f"{opt_row['cutoff']:.2f}",
                "Approved": opt_row["n_approved"],
                "Approval Rate": f"{opt_row['approval_rate'] * 100:.1f}%",
                "Default Rate": f"{opt_row['default_rate_approved'] * 100:.2f}%",
                "Total Profit": f"${opt_row['total_profit']:,.0f}",
                "Value Added": f"+${opt_row['total_profit'] - approve_all_profit:,.0f}",
            },
            {
                "Policy Option": "Approve Everyone (c=1.00)",
                "Cutoff": "1.00",
                "Approved": 150,
                "Approval Rate": "100.0%",
                "Default Rate": f"{np.mean(y_test) * 100:.2f}%",
                "Total Profit": f"${approve_all_profit:,.0f}",
                "Value Added": "$0",
            },
            {
                "Policy Option": "Decline Everyone (c=0.00)",
                "Cutoff": "0.00",
                "Approved": 0,
                "Approval Rate": "0.0%",
                "Default Rate": "0.00%",
                "Total Profit": "$0",
                "Value Added": f"+${-approve_all_profit:,.0f}",
            },
        ]
        st.dataframe(pd.DataFrame(comparison_records), hide_index=True)

    # ---------------------------------------------------------
    # TAB 3: WHY THIS DECISION (EXPLAINABILITY)
    # ---------------------------------------------------------
    with tab_why:
        st.subheader("Lender-Grade Adverse Action & Explainability Hub")
        st.markdown(
            "Under the Equal Credit Opportunity Act (ECOA) and Fair Credit Reporting Act (FCRA), lenders must disclose "
            "the specific, principal reasons that contributed to an adverse credit underwriting decision."
        )

        # Extract SHAP top risk drivers for selected applicant
        cust_shap = shap_values[cust_idx]
        top_indices = np.argsort(np.abs(cust_shap))[::-1][:5]
        reason_codes: List[Dict[str, Any]] = []

        for rank, idx in enumerate(top_indices, start=1):
            feat = shap_feature_names[idx]
            impact = float(cust_shap[idx])
            val = float(cust_tab[feat])
            desc = describe_feature(feat)
            reason_codes.append({
                "rank": rank,
                "feature": feat,
                "shap_impact": impact,
                "feature_value": val,
                "adverse_action_reason": desc,
                "direction": "Raises Risk 🔴" if impact > 0 else "Lowers Risk 🔵",
            })

        col_w1, col_w2 = st.columns([1.1, 0.9])
        with col_w1:
            st.markdown("#### 1. Principal Risk Drivers (TreeExplainer SHAP)")
            fig_shap = plot_customer_shap_bars(reason_codes, selected_cid)
            st.pyplot(fig_shap)
            plt.close(fig_shap)

        with col_w2:
            st.markdown("#### 2. Adverse Action Reason Codes")
            display_reasons = [
                {
                    "Rank": r["rank"],
                    "Feature": r["feature"],
                    "Value": f"{r['feature_value']:.4f}",
                    "SHAP Impact": f"{r['shap_impact']:+.4f}",
                    "Effect": r["direction"],
                    "Lender Description": r["adverse_action_reason"],
                }
                for r in reason_codes
            ]
            st.dataframe(pd.DataFrame(display_reasons), hide_index=True)

        st.markdown("---")

        # Statement History Attention Timeline
        st.markdown("#### 3. StatementTransformer Temporal Attention Timeline")
        st.caption(
            "Attention pooling allocates dynamic weights across the 13 statement history positions. "
            "Padded months (where no statement existed) receive 0.0 weight."
        )
        is_padded = ~cust_mask
        fig_attn = plot_attention_timeline(cust_attn, is_padded, selected_cid)
        st.pyplot(fig_attn)
        plt.close(fig_attn)

        st.markdown("---")

        # Formal Adverse Action Notice in an Expander
        with st.expander("📄 View Formal Adverse Action Statement (ECOA / FCRA Format)"):
            adverse_notice = format_adverse_action_notice(
                customer_id=selected_cid,
                risk_score=risk_prob,
                cutoff=cutoff,
                reason_codes=reason_codes,
            )
            st.code(adverse_notice, language="text")

        # Methodological Disclosure & Scientific Limitations
        st.markdown("---")
        st.markdown("#### ⚠️ Methodological Disclosure & Interpretability Caveat")
        st.warning(
            "**Scientific Citation Note**: In accordance with empirical NLP and ML findings (*Jain & Wallace, 'Attention is not Explanation', NAACL 2019*), "
            "attention weights reflect internal neural representation pooling and information routing within the sequence architecture, **not** counterfactual or causal attribution. "
            "Consequently, regulatory compliance notices and legal adverse-action reason codes are strictly derived from TreeExplainer SHAP log-odds attribution."
        )


if __name__ == "__main__":
    main()
