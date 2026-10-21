"""Unit and integration tests for the Streamlit demo application (Phase 8b).

Verifies data loaders, visual plot generation (gauge, profit curves, SHAP bars, attention timeline),
underwriting decision logic, and integration contracts.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from app.app import (
    CASE_STUDY_CUSTOMER,
    load_app_data,
    plot_attention_timeline,
    plot_customer_shap_bars,
    plot_profit_curve,
    plot_risk_gauge,
)
from src.business.profit import simulate_profit_curve


def test_load_app_data_contract():
    """Verify that load_app_data returns all required artifacts with consistent shapes."""
    data = load_app_data()
    required_keys = [
        "test_cids", "y_test", "tab_test", "preds", "shap_values",
        "shap_feature_names", "mask", "attention_matrix"
    ]
    for key in required_keys:
        assert key in data, f"Missing key '{key}' in load_app_data output"

    assert len(data["test_cids"]) == 150
    assert len(data["y_test"]) == 150
    assert data["tab_test"].shape[0] == 150
    assert data["shap_values"].shape == (150, 214)
    assert len(data["shap_feature_names"]) == 214
    assert data["mask"].shape == (150, 13)
    assert data["attention_matrix"].shape == (150, 13)

    # Check model predictions
    expected_models = [
        "LightGBM (Tabular)",
        "StatementTransformer (Calibrated - Isotonic)",
        "Masked GRU (Calibrated - Isotonic)",
        "StatementTransformer (Uncalibrated)",
        "Masked GRU (Uncalibrated)",
    ]
    for m in expected_models:
        assert m in data["preds"]
        assert data["preds"][m].shape == (150,)


def test_case_study_customer_presence():
    """Verify worked case study borrower is present in test set and has valid attributes."""
    data = load_app_data()
    assert CASE_STUDY_CUSTOMER in data["test_cids"]
    cust_idx = data["test_cids"].index(CASE_STUDY_CUSTOMER)

    # Borrower is a confirmed defaulter with 13 statements
    assert data["y_test"][cust_idx] == 1
    assert data["mask"][cust_idx].sum() == 13
    assert np.isclose(data["attention_matrix"][cust_idx].sum(), 1.0, atol=1e-4)


def test_plot_risk_gauge_generation():
    """Verify that plot_risk_gauge generates valid matplotlib figures for both decisions."""
    fig_app = plot_risk_gauge(risk_score=0.02, cutoff=0.05, approved=True)
    assert isinstance(fig_app, plt.Figure)
    plt.close(fig_app)

    fig_dec = plot_risk_gauge(risk_score=0.88, cutoff=0.05, approved=False)
    assert isinstance(fig_dec, plt.Figure)
    plt.close(fig_dec)


def test_plot_profit_curve_generation():
    """Verify profit curve figure creation across different cutoff thresholds."""
    data = load_app_data()
    sim_df = simulate_profit_curve(
        y_true=data["y_test"],
        y_prob=data["preds"]["LightGBM (Tabular)"],
        profit_per_good=100.0,
        loss_per_default=500.0,
    )

    fig = plot_profit_curve(
        sim_df=sim_df,
        current_cutoff=0.05,
        model_name="LightGBM (Tabular)",
        profit_per_good=100.0,
        loss_per_default=500.0,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_customer_shap_bars_generation():
    """Verify SHAP bar chart generation with mixed positive and negative impacts."""
    reasons = [
        {"feature": "D_39__last", "shap_impact": 1.45, "feature_value": 0.85},
        {"feature": "D_39__mean", "shap_impact": 0.95, "feature_value": 0.62},
        {"feature": "P_2__mean", "shap_impact": -0.82, "feature_value": 0.44},
        {"feature": "B_1__last", "shap_impact": 0.41, "feature_value": 0.35},
        {"feature": "R_1__max", "shap_impact": -0.15, "feature_value": 0.12},
    ]
    fig = plot_customer_shap_bars(reasons, customer_id=CASE_STUDY_CUSTOMER)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_attention_timeline_generation():
    """Verify attention timeline plot generation with padded and valid statements."""
    data = load_app_data()
    cust_idx = data["test_cids"].index(CASE_STUDY_CUSTOMER)
    attn = data["attention_matrix"][cust_idx]
    is_padded = ~data["mask"][cust_idx]

    fig = plot_attention_timeline(attn, is_padded, customer_id=CASE_STUDY_CUSTOMER)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_app_module_import():
    """Verify that app.app imports cleanly and exposes the main function."""
    import app.app as demo_app
    assert hasattr(demo_app, "main")
    assert callable(demo_app.main)
