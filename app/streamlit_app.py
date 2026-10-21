"""RiskTrail — Interactive Credit Default Risk & Profit Optimization Dashboard.

Streamlit demo application built strictly in accordance with docs/DESIGN_SYSTEM.md:
- Visual Palette: LightGBM (#2563EB), Non-default (#0EA5E9), Default/Risk (#DC2626),
  Optimal cutoff (#F59E0B), Baseline (#6B7280), Grid (#E5E7EB).
- Self-Contained: Uses only a small synthetic sample (app/sample_data.parquet) and the
  saved LightGBM model (app/model.txt); contains NO real customer identifiers.
- Layout: Sidebar with cutoff slider and profit/loss inputs.
- Three Tabs:
    1. Score: Risk probability gauge, real-time APPROVE/DECLINE decision, financial profile.
    2. Profit Curve: Expected portfolio profit simulation, optimal cutoff, baseline comparison.
    3. Why This Decision: Top-5 SHAP adverse-action reasons (red/blue bar chart), statement
       timeline, formal compliance notice (ECOA/FCRA).
- Mandatory educational and regulatory disclaimers.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Application and Data Paths (self-contained within app/)
APP_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_PATH = APP_DIR / "sample_data.parquet"
SAMPLE_STMTS_PATH = APP_DIR / "sample_statements.parquet"
SAMPLE_SHAP_PATH = APP_DIR / "sample_shap.npy"
SHAP_FEAT_PATH = APP_DIR / "shap_features.json"
MODEL_PATH = APP_DIR / "model.txt"

# Design System Color Palette (docs/DESIGN_SYSTEM.md)
COLOR_LGBM = "#2563EB"         # Blue
COLOR_GOOD = "#0EA5E9"         # Sky blue (non-default / lowers risk)
COLOR_RISK = "#DC2626"         # Red (default / raises risk)
COLOR_OPTIMAL = "#F59E0B"      # Amber (optimal cutoff marker)
COLOR_APPROVE_ALL = "#6B7280"  # Gray dashed baseline
COLOR_GRID = "#E5E7EB"         # Gridlines

CASE_STUDY_SYNTH_ID = "SYNTH_0045"

FEATURE_DESCRIPTIONS: Dict[str, str] = {
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


@st.cache_data(show_spinner=False)
def load_synthetic_demo_data() -> Dict[str, Any]:
    """Load cached synthetic sample dataset and precomputed SHAP values."""
    if not (SAMPLE_DATA_PATH.exists() and SAMPLE_SHAP_PATH.exists() and SHAP_FEAT_PATH.exists()):
        raise FileNotFoundError(
            f"Synthetic sample files missing in {APP_DIR}. "
            "Please run 'python src/data/create_sample.py' to generate them."
        )

    tab_df = pd.read_parquet(SAMPLE_DATA_PATH)
    stmts_df = pd.read_parquet(SAMPLE_STMTS_PATH)
    shap_values = np.load(SAMPLE_SHAP_PATH)
    with open(SHAP_FEAT_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    customer_ids = tab_df["customer_id"].tolist()
    y_true = tab_df["target"].values.astype(np.int8)
    preds = tab_df["predicted_prob"].values.astype(np.float64)

    return {
        "tab_df": tab_df.set_index("customer_id"),
        "stmts_df": stmts_df,
        "shap_values": shap_values,
        "feature_names": feature_names,
        "customer_ids": customer_ids,
        "y_true": y_true,
        "preds": preds,
    }


def simulate_profit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    profit_per_good: float,
    loss_per_default: float,
    cutoffs: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Sweep approval cutoffs and compute expected portfolio profit."""
    if cutoffs is None:
        cutoffs = np.linspace(0.0, 1.0, 201)

    N = len(y_true)
    records: List[Dict[str, float]] = []

    for c in cutoffs:
        approved = y_prob < c
        n_app = int(np.sum(approved))
        app_rate = n_app / N

        n_good_app = int(np.sum(approved & (y_true == 0)))
        n_def_app = int(np.sum(approved & (y_true == 1)))
        def_rate_app = (n_def_app / n_app) if n_app > 0 else 0.0

        total_profit = (n_good_app * profit_per_good) - (n_def_app * loss_per_default)

        records.append({
            "cutoff": round(float(c), 4),
            "n_approved": n_app,
            "approval_rate": round(float(app_rate), 4),
            "n_good_approved": n_good_app,
            "n_def_approved": n_def_app,
            "default_rate_approved": round(float(def_rate_app), 4),
            "total_profit": round(float(total_profit), 2),
        })

    return pd.DataFrame(records)


def plot_risk_gauge_fig(risk_score: float, cutoff: float, approved: bool) -> plt.Figure:
    """Generate visual risk gauge indicating customer score vs approval threshold."""
    fig, ax = plt.subplots(figsize=(8.2, 2.3), dpi=150)

    # Approval zone (green) and Decline zone (red)
    ax.barh(0, cutoff, left=0.0, height=0.5, color=COLOR_GOOD, alpha=0.35, label="Approval Zone (Low Risk)")
    ax.barh(0, 1.0 - cutoff, left=cutoff, height=0.5, color=COLOR_RISK, alpha=0.35, label="Decline Zone (High Risk)")

    # Vertical threshold line
    ax.axvline(cutoff, color="#1F2937", linestyle="--", linewidth=2.0, zorder=3)
    ax.text(
        cutoff, 0.38, f"Cutoff: {cutoff * 100:.1f}%",
        ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1F2937"
    )

    # Customer score pointer
    marker_color = COLOR_GOOD if approved else COLOR_RISK
    ax.scatter([risk_score], [0], color=marker_color, s=260, zorder=5, edgecolor="black", linewidth=1.5)
    ax.text(
        risk_score, -0.38, f"Score: {risk_score * 100:.2f}%",
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


def plot_profit_curve_fig(
    sim_df: pd.DataFrame,
    current_cutoff: float,
    profit_good: float,
    loss_default: float,
) -> plt.Figure:
    """Plot portfolio profit simulation curve highlighting current and optimal cutoffs."""
    opt_idx = sim_df["total_profit"].idxmax()
    opt_row = sim_df.loc[opt_idx]
    opt_cutoff = opt_row["cutoff"]
    opt_profit = opt_row["total_profit"]

    curr_row = sim_df.iloc[(sim_df["cutoff"] - current_cutoff).abs().argmin()]
    curr_profit = curr_row["total_profit"]
    approve_all_profit = sim_df.iloc[-1]["total_profit"]

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)

    # Model curve
    ax.plot(sim_df["cutoff"], sim_df["total_profit"], color=COLOR_LGBM, linewidth=2.4, label="LightGBM Policy")

    # Baselines
    ax.axhline(
        approve_all_profit, color=COLOR_APPROVE_ALL, linestyle="--", linewidth=1.5,
        label=f"Approve Everyone (${approve_all_profit:,.0f})"
    )
    ax.axhline(0.0, color="#9CA3AF", linestyle=":", linewidth=1.2, label="Decline Everyone ($0)")

    # Current cutoff indicator
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
        f"Portfolio Profit vs Approval Cutoff (V_good=${profit_good:.0f}, L_default=${loss_default:.0f})",
        fontsize=12, fontweight="bold", pad=10
    )
    ax.set_xlabel("Approval Cutoff Threshold (c)", fontsize=10)
    ax.set_ylabel("Total Portfolio Profit ($)", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    plt.tight_layout()
    return fig


def plot_customer_shap_fig(reasons: List[Dict[str, Any]], customer_id: str) -> plt.Figure:
    """Generate horizontal bar chart of top SHAP risk drivers for applicant."""
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
    ax.set_xlabel("SHAP Impact (Log-Odds Contribution to Default Risk)", fontsize=9.5)
    ax.set_title(f"Principal Risk Drivers: Applicant {customer_id}", fontsize=11, fontweight="bold", pad=8)
    ax.grid(axis="x", linestyle="--", alpha=0.5, color=COLOR_GRID)

    max_abs = max(abs(min(impacts)), abs(max(impacts))) * 1.35
    ax.set_xlim(-max_abs, max_abs)

    plt.tight_layout()
    return fig


def plot_statement_trajectory_fig(customer_stmts: pd.DataFrame, customer_id: str) -> plt.Figure:
    """Generate 13-month statement trajectory timeline for D_39 and B_1."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.8, 4.4), sharex=True, dpi=150)

    steps = customer_stmts["step"].values
    labels = customer_stmts["month_label"].values
    d39 = customer_stmts["D_39"].values
    b1 = customer_stmts["B_1"].values
    is_pad = customer_stmts["is_padded"].values

    # Delinquency trajectory (D_39)
    ax1.plot(steps, d39, marker="o", color=COLOR_RISK, linewidth=2.0, label="D_39 (Delinquency Level)")
    for s, v, pad in zip(steps, d39, is_pad):
        if not pad:
            ax1.annotate(f"{v:.1f}", xy=(s, v + 0.5), ha="center", fontsize=7.5, color="#1F2937")
    ax1.set_ylabel("D_39 Level", fontsize=9)
    ax1.set_title(f"13-Month Historical Statement Trajectory (Applicant {customer_id})", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)
    ax1.legend(loc="upper left", fontsize=8)

    # Balance trajectory (B_1)
    ax2.plot(steps, b1, marker="s", color=COLOR_LGBM, linewidth=2.0, label="B_1 (Revolving Balance Ratio)")
    for s, v, pad in zip(steps, b1, is_pad):
        if not pad:
            ax2.annotate(f"{v:.2f}", xy=(s, v + 0.03), ha="center", fontsize=7.5, color="#1F2937")
    ax2.set_xticks(steps)
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
    ax2.set_ylabel("B_1 Ratio", fontsize=9)
    ax2.set_xlabel("Statement Month", fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)
    ax2.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    return fig


def main() -> None:
    """Streamlit application main layout and interaction logic."""
    st.set_page_config(
        page_title="RiskTrail — Credit Risk & Profit Optimization",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header and Branding
    st.title("💳 RiskTrail — Credit Risk & Profit Optimization")
    st.markdown(
        "**Credit Default Risk Prediction, Profit-Maximizing Underwriting Policies & Lender-Grade Adverse Action Reason Codes**"
    )

    # Educational Use Disclaimer (MANDATORY per docs/SECURITY_AND_ACCESS.md)
    st.info(
        "ℹ️ **Educational & Research Use Disclaimer**: This application operates exclusively on a "
        "**small synthetic, anonymized sample** and a saved LightGBM benchmark model. It is designed for "
        "methodological evaluation, educational demonstration, and research purposes. "
        "**It is NOT certified for real-world automated credit underwriting or consumer credit decisions.**"
    )

    # Load synthetic dataset
    with st.spinner("Loading synthetic credit demo sample..."):
        demo_data = load_synthetic_demo_data()

    tab_df: pd.DataFrame = demo_data["tab_df"]
    stmts_df: pd.DataFrame = demo_data["stmts_df"]
    shap_values: np.ndarray = demo_data["shap_values"]
    feature_names: List[str] = demo_data["feature_names"]
    customer_ids: List[str] = demo_data["customer_ids"]
    y_true: np.ndarray = demo_data["y_true"]
    preds: np.ndarray = demo_data["preds"]

    # ---------------------------------------------------------
    # SIDEBAR CONTROLS
    # ---------------------------------------------------------
    st.sidebar.header("⚙️ Policy & Economic Controls")

    # Economic Assumptions
    st.sidebar.subheader("💼 Lender Economics")
    profit_good = st.sidebar.number_input(
        "Profit per Good Account ($)",
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
    cutoff = st.sidebar.slider(
        "Approval Cutoff Threshold (c)",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01,
        help="Approve customer if predicted default probability < cutoff.",
    )

    # Applicant Selection
    st.sidebar.subheader("👤 Applicant Selector")
    cohort_filter = st.sidebar.radio(
        "Filter Cohort",
        options=["All Applicants (50)", "Confirmed Defaulters (14)", "Good / Paid (36)", "Deteriorating (18)", "Stable (32)"],
        index=0,
    )

    # Filter indices based on cohort
    if cohort_filter.startswith("Confirmed Defaulters"):
        valid_indices = [i for i, y in enumerate(y_true) if y == 1]
    elif cohort_filter.startswith("Good"):
        valid_indices = [i for i, y in enumerate(y_true) if y == 0]
    elif cohort_filter.startswith("Deteriorating"):
        valid_indices = [i for i, cid in enumerate(customer_ids) if tab_df.loc[cid, "D_39__slope"] > 0]
    elif cohort_filter.startswith("Stable"):
        valid_indices = [i for i, cid in enumerate(customer_ids) if tab_df.loc[cid, "D_39__slope"] <= 0]
    else:
        valid_indices = list(range(len(customer_ids)))

    filtered_cids = [customer_ids[i] for i in valid_indices]

    # Pre-select case study borrower
    default_cid_idx = filtered_cids.index(CASE_STUDY_SYNTH_ID) if CASE_STUDY_SYNTH_ID in filtered_cids else 0

    selected_cid = st.sidebar.selectbox(
        "Select Synthetic Applicant ID",
        options=filtered_cids,
        index=default_cid_idx,
    )

    if st.sidebar.button("📌 Load Worked Case Study Applicant (High Delinquency)"):
        selected_cid = CASE_STUDY_SYNTH_ID

    # Current Applicant Metrics
    cust_idx = customer_ids.index(selected_cid)
    risk_prob = float(preds[cust_idx])
    ground_truth = int(y_true[cust_idx])
    is_approved = risk_prob < cutoff
    cust_tab = tab_df.loc[selected_cid]
    cust_stmts = stmts_df[stmts_df["customer_id"] == selected_cid]
    n_statements = int((~cust_stmts["is_padded"]).sum())

    # ---------------------------------------------------------
    # MAIN AREA: THREE TABS
    # ---------------------------------------------------------
    tab_score, tab_profit, tab_why = st.tabs([
        "📊 Score",
        "📈 Profit Curve",
        "🔍 Why This Decision",
    ])

    # ---------------------------------------------------------
    # TAB 1: SCORE
    # ---------------------------------------------------------
    with tab_score:
        st.subheader("Synthetic Applicant Risk Score & Decision")

        # Top Decision Banner
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

        # Risk Gauge Meter
        fig_meter = plot_risk_gauge_fig(risk_prob, cutoff, is_approved)
        st.pyplot(fig_meter)
        plt.close(fig_meter)

        st.markdown("---")

        # KPI Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Predicted Default Risk",
                value=f"{risk_prob * 100:.2f}%",
                delta=f"{(risk_prob - cutoff) * 100:+.2f}% vs Cutoff",
                delta_color="inverse",
            )
        with col2:
            st.metric(label="Underwriting Action", value="APPROVE" if is_approved else "DECLINE")
        with col3:
            realized_profit = (profit_good if ground_truth == 0 else -loss_default) if is_approved else 0.0
            st.metric(
                label="Realized Customer Profit",
                value=f"${realized_profit:,.0f}",
                delta="Profitable" if realized_profit > 0 else ("Charge-Off Loss" if realized_profit < 0 else "Neutral ($0)"),
            )
        with col4:
            st.metric(
                label="Synthetic Ground Truth",
                value="Default (1)" if ground_truth == 1 else "Good Account (0)",
            )

        st.markdown("### 📋 Account Financial Summary")
        cp1, cp2, cp3, cp4 = st.columns(4)
        with cp1:
            st.write(f"**Statements on File:** {n_statements} of 13 months")
            slope_v = float(cust_tab.get("D_39__slope", 0.0))
            traj_text = "🔴 Deteriorating" if slope_v > 0 else "🟢 Stable"
            st.write(f"**Trajectory:** {traj_text}")
        with cp2:
            st.write(f"**Delinquency Slope (D_39):** `{slope_v:+.4f}`")
            st.write(f"**Recent Delinquency (D_39 Last):** `{float(cust_tab.get('D_39__last', 0.0)):.2f}`")
        with cp3:
            st.write(f"**Revolving Balance (B_1 Last):** `{float(cust_tab.get('B_1__last', 0.0)):.4f}`")
            st.write(f"**Payment Ratio (P_2 Mean):** `{float(cust_tab.get('P_2__mean', 0.0)):.4f}`")
        with cp4:
            st.write(f"**Delinquency Code (D_63):** `{int(cust_tab.get('D_63__last', 0))}`")
            st.write(f"**Bankcard Tier (B_30):** `{int(cust_tab.get('B_30__last', 0))}`")

    # ---------------------------------------------------------
    # TAB 2: PROFIT CURVE
    # ---------------------------------------------------------
    with tab_profit:
        st.subheader("Portfolio Profit vs Approval Cutoff Simulation")
        st.markdown(
            "Economic lending decisions balance good borrower interest margin against default loss: "
            r"$\text{Profit}(c) = V_{\text{good}} \cdot N_{\text{good, app}}(c) - L_{\text{default}} \cdot N_{\text{def, app}}(c)$."
        )

        sim_df = simulate_profit(y_true, preds, profit_good, loss_default)
        opt_idx = sim_df["total_profit"].idxmax()
        opt_row = sim_df.loc[opt_idx]
        curr_row = sim_df.iloc[(sim_df["cutoff"] - cutoff).abs().argmin()]
        approve_all_profit = sim_df.iloc[-1]["total_profit"]

        # KPI Metrics Row
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.metric("Profit at Current Cutoff", f"${curr_row['total_profit']:,.0f}")
        with k2:
            st.metric("Peak Profit (Optimal)", f"${opt_row['total_profit']:,.0f}", f"at c* = {opt_row['cutoff']:.2f}")
        with k3:
            st.metric("Approval Rate", f"{curr_row['approval_rate'] * 100:.1f}%", f"{curr_row['n_approved']} / {len(customer_ids)}")
        with k4:
            st.metric("Default Rate (Approved)", f"{curr_row['default_rate_approved'] * 100:.2f}%")
        with k5:
            delta_p = curr_row["total_profit"] - approve_all_profit
            st.metric("Value-Add vs Approve-All", f"+${delta_p:,.0f}")

        # Profit Curve Figure
        fig_prof = plot_profit_curve_fig(sim_df, cutoff, profit_good, loss_default)
        st.pyplot(fig_prof)
        plt.close(fig_prof)

        # Policy Benchmarks Table
        st.markdown("### 📊 Policy Benchmark Table")
        benchmarks = [
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
                "Approved": len(customer_ids),
                "Approval Rate": "100.0%",
                "Default Rate": f"{np.mean(y_true) * 100:.2f}%",
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
        st.dataframe(pd.DataFrame(benchmarks), hide_index=True)

    # ---------------------------------------------------------
    # TAB 3: WHY THIS DECISION
    # ---------------------------------------------------------
    with tab_why:
        st.subheader("Lender-Grade Adverse Action & Explainability Hub")
        st.markdown(
            "Under the Equal Credit Opportunity Act (ECOA) and Fair Credit Reporting Act (FCRA), lenders must disclose "
            "the principal reasons that contributed to an adverse credit underwriting decision."
        )

        # Extract top SHAP risk drivers
        cust_shap = shap_values[cust_idx]
        top_indices = np.argsort(np.abs(cust_shap))[::-1][:5]
        reason_codes: List[Dict[str, Any]] = []

        for rank, idx in enumerate(top_indices, start=1):
            feat = feature_names[idx]
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
            fig_shap = plot_customer_shap_fig(reason_codes, selected_cid)
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

        # 13-Month Statement Trajectory
        st.markdown("#### 3. 13-Month Historical Statement Trajectory")
        st.caption("Visualizing the monthly evolution of delinquency (D_39) and revolving balance (B_1) leading up to the application.")
        fig_traj = plot_statement_trajectory_fig(cust_stmts, selected_cid)
        st.pyplot(fig_traj)
        plt.close(fig_traj)

        st.markdown("---")

        # Formal Adverse Action Notice in an Expander
        with st.expander("📄 View Formal Adverse Action Statement (ECOA / FCRA Format)"):
            lines = [
                "=" * 85,
                "CREDIT DECISION NOTICE — ADVERSE ACTION STATEMENT",
                f"Applicant Identifier: {selected_cid}",
                f"Application Date: Most Recent Statement T",
                "=" * 85,
                "",
                "DECISION OUTCOME: DECLINED" if not is_approved else "DECISION OUTCOME: APPROVED",
                f"Model Assessed Default Risk: {risk_prob * 100:.2f}%",
                f"Policy Underwriting Cutoff:  {cutoff * 100:.2f}%",
                "",
                "KEY FACTORS CONTRIBUTING TO ADVERSE DECISION:",
            ]
            for r in reason_codes:
                impact_label = f"+{r['shap_impact']:.4f}" if r['shap_impact'] > 0 else f"{r['shap_impact']:.4f}"
                lines.append(f"  [{r['rank']}] {r['adverse_action_reason']}")
                lines.append(f"      Feature: {r['feature']} = {r['feature_value']:.4f} (SHAP Contribution: {impact_label} log-odds)")
                lines.append("")
            lines.extend([
                "=" * 85,
                "DISCLOSURE UNDER EQUAL CREDIT OPPORTUNITY ACT (ECOA) & FAIR CREDIT REPORTING ACT (FCRA):",
                "Our credit scoring system evaluated the applicant's 13-month credit performance history.",
                "The reasons listed above represent the principal factors influencing this underwriting determination.",
                "=" * 85,
            ])
            st.code("\n".join(lines), language="text")

        # Methodological Disclosure
        st.markdown("---")
        st.caption(
            "**Notice**: TreeExplainer SHAP values represent exact local game-theoretic feature attributions in log-odds space. "
            "Educational demo operating on synthetic sample data."
        )


if __name__ == "__main__":
    main()
