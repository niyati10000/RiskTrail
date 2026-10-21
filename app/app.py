"""Streamlit Demo Application for Credit Default Risk Prediction & Profit Optimization.

Educational and evaluation interface:
- Sidebar: cutoff slider, profit/loss assumptions
- Tabs: Score (risk probability & decision), Profit curve, Why this decision (top SHAP reason codes)
"""

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="RiskTrail Demo", layout="wide")
    st.title("RiskTrail — Credit Risk & Profit Optimization")
    st.caption("Educational demo on anonymized/sample data; not for real credit decisions.")

    st.sidebar.header("Decision Parameters")
    cutoff = st.sidebar.slider("Approval Cutoff", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    profit_good = st.sidebar.number_input("Profit per Good Customer ($)", value=100.0, step=10.0)
    loss_default = st.sidebar.number_input("Loss per Default ($)", value=500.0, step=50.0)

    tab_score, tab_profit, tab_why = st.tabs(["Score", "Profit Curve", "Why This Decision"])

    with tab_score:
        st.subheader("Customer Risk Assessment")
        st.info("Model scoring pipeline will be connected in Phase 8.")

    with tab_profit:
        st.subheader("Portfolio Profit vs Approval Cutoff")
        st.info("Profit simulation curves will be visualized here.")

    with tab_why:
        st.subheader("Lender Adverse-Action Reason Codes")
        st.info("SHAP reason codes and temporal attention weights will be shown here.")


if __name__ == "__main__":
    main()
