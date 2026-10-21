"""Unit and integration tests for the Streamlit demo application and synthetic sample (Phase 8b).

Verifies synthetic data loader contracts, visual plot generation (gauge, profit curves, SHAP bars,
13-month statement trajectory), underwriting decision logic, and integration contracts.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from app.streamlit_app import (
    CASE_STUDY_SYNTH_ID,
    describe_feature,
    load_synthetic_demo_data,
    plot_customer_shap_fig,
    plot_profit_curve_fig,
    plot_risk_gauge_fig,
    plot_statement_trajectory_fig,
    simulate_profit,
)
from src.data.create_sample import generate_synthetic_sample


def test_synthetic_sample_artifacts_exist():
    """Verify that all synthetic sample artifacts exist in app/ directory."""
    app_dir = Path(__file__).resolve().parents[1] / "app"
    assert (app_dir / "sample_data.parquet").exists()
    assert (app_dir / "sample_statements.parquet").exists()
    assert (app_dir / "sample_shap.npy").exists()
    assert (app_dir / "shap_features.json").exists()
    assert (app_dir / "model.txt").exists()
    assert (app_dir / "requirements.txt").exists()


def test_load_synthetic_demo_data_contract():
    """Verify load_synthetic_demo_data returns valid shapes without real customer PII."""
    data = load_synthetic_demo_data()
    required_keys = [
        "tab_df", "stmts_df", "shap_values", "feature_names",
        "customer_ids", "y_true", "preds"
    ]
    for key in required_keys:
        assert key in data, f"Missing key '{key}' in load_synthetic_demo_data output"

    assert len(data["customer_ids"]) == 50
    assert len(data["y_true"]) == 50
    assert len(data["preds"]) == 50
    assert data["tab_df"].shape[0] == 50
    assert data["shap_values"].shape == (50, 214)
    assert len(data["feature_names"]) == 214
    assert data["stmts_df"].shape[0] == 50 * 13

    # Verify no real customer hashes exist (all synthetic identifiers)
    for cid in data["customer_ids"]:
        assert cid.startswith("SYNTH_"), f"Unexpected customer ID format: {cid}"


def test_synthetic_case_study_customer():
    """Verify designated case study customer SYNTH_0045 has high delinquency and valid features."""
    data = load_synthetic_demo_data()
    assert CASE_STUDY_SYNTH_ID in data["customer_ids"]
    cust_idx = data["customer_ids"].index(CASE_STUDY_SYNTH_ID)

    assert data["y_true"][cust_idx] == 1  # Defaulter
    assert data["preds"][cust_idx] > 0.5   # High predicted default risk

    cust_tab = data["tab_df"].loc[CASE_STUDY_SYNTH_ID]
    assert cust_tab["D_39__slope"] > 0    # Positive delinquency trajectory


def test_plot_risk_gauge_fig_generation():
    """Verify plot_risk_gauge_fig produces valid figures for both approved and declined cases."""
    fig_app = plot_risk_gauge_fig(risk_score=0.015, cutoff=0.05, approved=True)
    assert isinstance(fig_app, plt.Figure)
    plt.close(fig_app)

    fig_dec = plot_risk_gauge_fig(risk_score=0.85, cutoff=0.05, approved=False)
    assert isinstance(fig_dec, plt.Figure)
    plt.close(fig_dec)


def test_plot_profit_curve_fig_generation():
    """Verify profit curve figure creation across different cutoff thresholds."""
    data = load_synthetic_demo_data()
    sim_df = simulate_profit(
        y_true=data["y_true"],
        y_prob=data["preds"],
        profit_per_good=100.0,
        loss_per_default=500.0,
    )

    assert len(sim_df) == 201
    assert "total_profit" in sim_df.columns

    fig = plot_profit_curve_fig(
        sim_df=sim_df,
        current_cutoff=0.05,
        profit_good=100.0,
        loss_default=500.0,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_customer_shap_fig_generation():
    """Verify SHAP horizontal bar chart figure generation."""
    reasons = [
        {"feature": "D_39__last", "shap_impact": 1.45, "feature_value": 15.2},
        {"feature": "D_39__slope", "shap_impact": 0.88, "feature_value": 0.65},
        {"feature": "P_2__mean", "shap_impact": -0.75, "feature_value": 0.40},
        {"feature": "B_1__last", "shap_impact": 0.42, "feature_value": 0.80},
        {"feature": "D_39__mean", "shap_impact": 0.35, "feature_value": 12.0},
    ]
    fig = plot_customer_shap_fig(reasons, customer_id=CASE_STUDY_SYNTH_ID)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_statement_trajectory_fig_generation():
    """Verify statement trajectory figure creation across 13 monthly steps."""
    data = load_synthetic_demo_data()
    cust_stmts = data["stmts_df"][data["stmts_df"]["customer_id"] == CASE_STUDY_SYNTH_ID]

    fig = plot_statement_trajectory_fig(cust_stmts, customer_id=CASE_STUDY_SYNTH_ID)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_describe_feature_human_readable():
    """Verify that describe_feature provides lender-grade friendly explanations."""
    desc = describe_feature("D_39__last")
    assert "Delinquency" in desc
    assert "Most recent statement value" in desc

    desc_slope = describe_feature("D_39__slope")
    assert "trajectory" in desc_slope.lower() or "slope" in desc_slope.lower()


def test_app_and_streamlit_app_imports():
    """Verify that both app/app.py and app/streamlit_app.py expose callable main functions."""
    import app.app as app_module
    import app.streamlit_app as st_module

    assert hasattr(app_module, "main") and callable(app_module.main)
    assert hasattr(st_module, "main") and callable(st_module.main)
