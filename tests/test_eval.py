"""Unit tests for Phase 5: Cross-Model Evaluation, Calibration, and Segment Analysis.

Validates:
1. Expected Calibration Error (ECE) and calibration curve calculations.
2. ModelCalibrator fitting and prediction (Platt scaling and Isotonic regression).
3. Cross-model comparison table compilation from results directory.
4. Trend segmentation analysis (deteriorating vs stable customer cohorts).
5. Generation and integrity of figures (fig_model_comparison, fig_calibration, fig_trend_segment).
6. Existence and completeness of reports/05_evaluation.md.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.eval.calibration import (
    ModelCalibrator,
    compute_calibration_curve,
    compute_expected_calibration_error,
)
from src.eval.compare import (
    analyze_trend_segments,
    calibrate_all_models,
    compare_models,
    load_ground_truth_splits,
    run_evaluation_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_compute_expected_calibration_error_perfect():
    """Verify ECE is 0.0 for perfectly calibrated predictions."""
    y_true = np.array([0, 0, 1, 1], dtype=np.int8)
    y_prob = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_compute_expected_calibration_error_miscalibrated():
    """Verify ECE reflects calibration error for biased probabilities."""
    y_true = np.array([0, 0, 0, 0], dtype=np.int8)
    y_prob = np.array([0.8, 0.8, 0.8, 0.8], dtype=np.float32)
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert ece == pytest.approx(0.8, abs=1e-5)


def test_calibration_curve_outputs():
    """Verify compute_calibration_curve returns expected tuple structures."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int8)
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.85], dtype=np.float32)

    prob_true, prob_pred, brier, ece = compute_calibration_curve(y_true, y_prob, n_bins=4)
    assert len(prob_true) == len(prob_pred)
    assert len(prob_true) > 0
    assert 0.0 <= brier <= 1.0
    assert 0.0 <= ece <= 1.0


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_model_calibrator_fit_and_predict(method):
    """Verify ModelCalibrator fits on validation and improves calibration on shifted predictions."""
    np.random.seed(42)
    N = 200
    # Synthetic targets with default rate 20%
    y_val = (np.random.rand(N) < 0.2).astype(np.int8)
    # Probability shifted upward (class-weighting bias)
    p_val = np.clip(y_val * 0.7 + 0.3 + np.random.randn(N) * 0.05, 0.0, 1.0)

    calibrator = ModelCalibrator(method=method)
    calibrator.fit(p_val, y_val)

    # Test predictions
    N_test = 100
    y_test = (np.random.rand(N_test) < 0.2).astype(np.int8)
    p_test = np.clip(y_test * 0.7 + 0.3 + np.random.randn(N_test) * 0.05, 0.0, 1.0)

    p_cal = calibrator.predict(p_test)
    assert p_cal.shape == p_test.shape
    assert (p_cal >= 0.0).all() and (p_cal <= 1.0).all()

    diag = calibrator.evaluate(p_test, p_cal, y_test)
    assert diag["brier_calibrated"] < diag["brier_uncalibrated"]


def test_load_ground_truth_splits():
    """Verify test and validation split alignment with processed parquet files."""
    y_valid, y_test, test_tab = load_ground_truth_splits()
    assert len(y_valid) == 150
    assert len(y_test) == 150
    assert len(test_tab) == 150
    assert y_test.sum() == 36  # Exactly 36 defaulters in 150 test customers (24%)


def test_compare_models_compilation():
    """Verify compare_models produces valid DataFrame with all benchmark models."""
    df = compare_models()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3

    expected_models = {"LightGBM", "Masked GRU", "Transformer"}
    assert set(df["model"].tolist()) == expected_models

    for _, row in df.iterrows():
        assert 0.0 <= row["comp_metric_test"] <= 1.0
        assert 0.0 <= row["roc_auc_test"] <= 1.0
        assert 0.0 <= row["pr_auc_test"] <= 1.0
        assert 0.0 <= row["brier_uncalibrated"] <= 1.0
        assert 0.0 <= row["brier_isotonic"] <= 1.0


def test_trend_segment_analysis_properties():
    """Verify trend segmentation logic partitions into deteriorating and stable cohorts."""
    seg_summary = analyze_trend_segments()
    assert "deteriorating" in seg_summary
    assert "stable" in seg_summary

    det = seg_summary["deteriorating"]
    stable = seg_summary["stable"]

    # Sum of segment customers equals 150 test customers
    assert det["n_customers"] + stable["n_customers"] == 150
    # All 36 defaulters reside in deteriorating cohort
    assert det["n_defaults"] == 36
    assert stable["n_defaults"] == 0
    assert det["default_rate"] > 0.5
    assert stable["default_rate"] == 0.0

    # Models evaluated on deteriorating cohort
    for m in ["lgbm", "gru", "transformer"]:
        assert m in det["models"]
        assert det["models"][m]["competition_metric"] > 0.5


def test_pipeline_execution_and_artifacts_integrity():
    """Verify run_evaluation_pipeline generates all figures and report."""
    comp_df = run_evaluation_pipeline()
    assert len(comp_df) == 3

    # Check figures generated in reports/figures/
    fig_comp = PROJECT_ROOT / "reports" / "figures" / "fig_model_comparison.png"
    fig_cal = PROJECT_ROOT / "reports" / "figures" / "fig_calibration.png"
    fig_seg = PROJECT_ROOT / "reports" / "figures" / "fig_trend_segment.png"

    assert fig_comp.exists() and fig_comp.stat().st_size > 5000
    assert fig_cal.exists() and fig_cal.stat().st_size > 5000
    assert fig_seg.exists() and fig_seg.stat().st_size > 5000

    # Check report written
    report = PROJECT_ROOT / "reports" / "05_evaluation.md"
    assert report.exists()
    content = report.read_text(encoding="utf-8")
    assert "Executive Summary & Honest Findings" in content
    assert "LightGBM" in content
    assert "Masked GRU" in content
    assert "Transformer" in content
    assert "Deteriorating" in content
    assert "Calibration" in content
