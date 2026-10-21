"""Unit tests for Phase 6: Business Profit Simulation & Cutoff Policy Optimization.

Validates:
1. Mathematical profit simulation on toy cases with analytically known profits.
2. Optimal cutoff identification.
3. Simulation across all models on test split (LightGBM, GRU, Transformer, baselines).
4. Loss-to-profit sensitivity analysis (2x to 20x).
5. Generation and integrity of reports/figures/fig_profit_by_cutoff.png.
6. Completeness of reports/06_business.md.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.business.profit import (
    find_optimal_cutoff,
    run_business_simulation,
    run_sensitivity_analysis,
    simulate_all_models,
    simulate_profit_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_simulate_profit_curve_toy():
    """Verify profit curve calculations on analytical toy example."""
    y_true = np.array([0, 0, 1], dtype=np.int8)
    y_prob = np.array([0.1, 0.2, 0.8], dtype=np.float64)
    V = 100.0
    L = 500.0
    cutoffs = np.array([0.05, 0.5, 0.9])

    df = simulate_profit_curve(y_true, y_prob, profit_per_good=V, loss_per_default=L, cutoffs=cutoffs)

    # Cutoff 0.05 -> approves none
    row0 = df[df["cutoff"] == 0.05].iloc[0]
    assert row0["n_approved"] == 0
    assert row0["total_profit"] == 0.0

    # Cutoff 0.5 -> approves 2 good borrowers
    row1 = df[df["cutoff"] == 0.5].iloc[0]
    assert row1["n_approved"] == 2
    assert row1["n_good_approved"] == 2
    assert row1["n_def_approved"] == 0
    assert row1["default_rate_approved"] == 0.0
    assert row1["total_profit"] == 200.0

    # Cutoff 0.9 -> approves all 3 (2 good, 1 bad)
    row2 = df[df["cutoff"] == 0.9].iloc[0]
    assert row2["n_approved"] == 3
    assert row2["n_good_approved"] == 2
    assert row2["n_def_approved"] == 1
    assert row2["default_rate_approved"] == pytest.approx(1 / 3, abs=1e-4)
    assert row2["total_profit"] == (2 * 100.0) - (1 * 500.0)  # -300.0


def test_find_optimal_cutoff_toy():
    """Verify find_optimal_cutoff selects the maximum profit threshold."""
    y_true = np.array([0, 0, 1], dtype=np.int8)
    y_prob = np.array([0.1, 0.2, 0.8], dtype=np.float64)
    cutoffs = np.array([0.05, 0.5, 0.9])

    df = simulate_profit_curve(y_true, y_prob, profit_per_good=100.0, loss_per_default=500.0, cutoffs=cutoffs)
    opt = find_optimal_cutoff(df)

    assert opt["optimal_cutoff"] == 0.5
    assert opt["max_profit"] == 200.0
    assert opt["n_approved"] == 2
    assert opt["default_rate_approved"] == 0.0


def test_simulate_all_models_on_test_split():
    """Verify simulate_all_models evaluates test split correctly against baselines."""
    sim_data = simulate_all_models()

    assert sim_data["n_test"] == 150
    assert sim_data["n_good_total"] == 114
    assert sim_data["n_def_total"] == 36

    # Approve everyone baseline: 114 * 100 - 36 * 500 = -6600
    assert sim_data["approve_all_profit"] == -6600.0
    # Oracle ceiling: 114 * 100 = 11400
    assert sim_data["oracle_profit"] == 11400.0

    summary_df = sim_data["summary_table"]
    assert len(summary_df) >= 5

    # Check LightGBM, Calibrated GRU, and Calibrated Transformer
    for m_key in ["lgbm", "gru", "transformer"]:
        row = summary_df[summary_df["model_key"] == m_key].iloc[0]
        assert row["max_profit"] == 11400.0
        assert row["approval_rate"] == 0.76
        assert row["n_approved"] == 114
        assert row["default_rate_approved"] == 0.0
        assert row["value_add_vs_approve_all"] == 18000.0


def test_sensitivity_analysis_grid():
    """Verify loss/profit sensitivity calculations over 2x to 20x ratios."""
    sim_data = simulate_all_models()
    ratios = [2.0, 5.0, 20.0]

    sens_df = run_sensitivity_analysis(
        y_test=sim_data["y_test"],
        model_predictions=sim_data["raw_predictions"],
        profit_per_good=100.0,
        ratios=ratios,
        cutoffs=sim_data["cutoffs"],
    )

    assert len(sens_df) == 3

    # R = 2.0x (L = $200): Approve all profit = 114*100 - 36*200 = 4200
    r2 = sens_df[sens_df["loss_profit_ratio"] == "2.0x"].iloc[0]
    assert r2["approve_all_profit"] == 4200.0
    assert r2["lgbm_max_profit"] == 11400.0
    assert r2["lgbm_value_add"] == 7200.0

    # R = 5.0x (L = $500): Approve all profit = -6600
    r5 = sens_df[sens_df["loss_profit_ratio"] == "5.0x"].iloc[0]
    assert r5["approve_all_profit"] == -6600.0
    assert r5["lgbm_max_profit"] == 11400.0
    assert r5["lgbm_value_add"] == 18000.0

    # R = 20.0x (L = $2000): Approve all profit = 114*100 - 36*2000 = -60600
    r20 = sens_df[sens_df["loss_profit_ratio"] == "20.0x"].iloc[0]
    assert r20["approve_all_profit"] == -60600.0
    assert r20["lgbm_max_profit"] == 11400.0
    assert r20["lgbm_value_add"] == 72000.0


def test_business_simulation_pipeline_and_artifacts():
    """Verify run_business_simulation generates reports/06_business.md and fig_profit_by_cutoff."""
    sim_data = run_business_simulation()
    assert sim_data is not None

    fig_path = PROJECT_ROOT / "reports" / "figures" / "fig_profit_by_cutoff.png"
    assert fig_path.exists()
    assert fig_path.stat().st_size > 5000

    report_path = PROJECT_ROOT / "reports" / "06_business.md"
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Business Profit Simulation & Cutoff Policy Report" in content
    assert "LightGBM" in content
    assert "Masked GRU" in content
    assert "Transformer" in content
    assert "Approve Everyone" in content
    assert "11,400" in content
