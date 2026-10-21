"""Unit tests for evaluation metrics module (src/eval/metrics.py).

Validates:
1. Perfect classification metrics (ROC-AUC=1.0, Gini=1.0, PR-AUC=1.0).
2. Worst-case / inverted predictions (ROC-AUC=0.0, Gini=-1.0).
3. Random baseline predictions (ROC-AUC=0.5, Gini=0.0).
4. Default capture rate at top 4% on controlled 100-sample arrays with exact manual solutions.
5. Exact Brier calibration score calculation.
6. Official weighted metrics (normalized_gini_weighted, capture_rate_at_k_weighted).
7. Comprehensive metric suite output dictionary.
"""

import numpy as np
import pytest

from src.eval.metrics import (
    brier,
    capture_rate_at_k,
    capture_rate_at_k_weighted,
    competition_metric,
    competition_metric_weighted,
    compute_all_metrics,
    normalized_gini,
    normalized_gini_weighted,
    pr_auc,
    roc_auc,
)


def test_perfect_predictions():
    """Verify metrics for perfect ranking and calibrated probabilities."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0.05, 0.1, 0.15, 0.2, 0.8, 0.85, 0.9, 0.95])

    assert np.isclose(roc_auc(y_true, y_pred), 1.0)
    assert np.isclose(pr_auc(y_true, y_pred), 1.0)
    assert np.isclose(normalized_gini(y_true, y_pred), 1.0)
    assert np.isclose(normalized_gini_weighted(y_true, y_pred), 1.0)
    assert brier(y_true, y_pred) < 0.05


def test_inverted_worst_case_predictions():
    """Verify metrics when predictions are completely inverted."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.9, 0.8, 0.2, 0.1])

    assert np.isclose(roc_auc(y_true, y_pred), 0.0)
    assert np.isclose(normalized_gini(y_true, y_pred), -1.0)
    assert np.isclose(capture_rate_at_k(y_true, y_pred, k=0.5), 0.0)


def test_random_predictions():
    """Verify that uninformative random predictions give Gini ~ 0.0."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    assert np.isclose(roc_auc(y_true, y_pred), 0.5)
    assert np.isclose(normalized_gini(y_true, y_pred), 0.0)


def test_capture_rate_at_top_4_percent_exact_math():
    """Verify default capture rate at top 4% on controlled 100-sample datasets.

    N = 100, k = 0.04 -> top 4 items.
    Total positives = 10.
    """
    # 10 positives, 90 negatives
    y_true = np.zeros(100, dtype=int)
    y_true[90:] = 1  # 10 defaults

    # Case A: Top 4 predictions contain exactly 4 defaulters -> 4 / 10 = 0.40
    y_pred_a = np.linspace(0.01, 1.0, 100)
    # The highest 4 predictions (indices 96, 97, 98, 99) are all ones in y_true
    assert np.isclose(capture_rate_at_k(y_true, y_pred_a, k=0.04), 0.40)

    # Case B: Top 4 predictions contain exactly 2 defaulters -> 2 / 10 = 0.20
    y_pred_b = np.copy(y_pred_a)
    # Swap top 2 values to indices that are 0 in y_true
    y_pred_b[98], y_pred_b[0] = y_pred_b[0], y_pred_b[98]
    y_pred_b[99], y_pred_b[1] = y_pred_b[1], y_pred_b[99]
    assert np.isclose(capture_rate_at_k(y_true, y_pred_b, k=0.04), 0.20)

    # Case C: Top 4 predictions contain 0 defaulters -> 0.0
    y_pred_c = 1.0 - y_pred_a
    assert np.isclose(capture_rate_at_k(y_true, y_pred_c, k=0.04), 0.0)


def test_competition_metric_arithmetic():
    """Verify competition_metric = 0.5 * (normalized_gini + capture_rate)."""
    y_true = np.zeros(100, dtype=int)
    y_true[90:] = 1
    y_pred = np.linspace(0.01, 1.0, 100)

    gini = normalized_gini(y_true, y_pred)
    cap4 = capture_rate_at_k(y_true, y_pred, k=0.04)
    comp = competition_metric(y_true, y_pred, k=0.04)

    assert np.isclose(comp, 0.5 * (gini + cap4))


def test_brier_score_exact_math():
    """Verify exact Brier score calculation: mean((y_pred - y_true)^2)."""
    y_true = np.array([1, 0])
    y_pred = np.array([0.8, 0.2])
    # (0.8 - 1)^2 = 0.04; (0.2 - 0)^2 = 0.04 -> mean = 0.04
    assert np.isclose(brier(y_true, y_pred), 0.04)


def test_weighted_metrics_on_subsampled_population():
    """Verify that official weighted metrics handle 20x negative weights properly."""
    # 20 negatives, 5 positives
    y_true = np.array([0] * 20 + [1] * 5)
    # Perfect model
    y_pred = y_true.astype(float)

    gini_w = normalized_gini_weighted(y_true, y_pred, weight_neg=20.0)
    cap_w = capture_rate_at_k_weighted(y_true, y_pred, k=0.04, weight_neg=20.0)
    comp_w = competition_metric_weighted(y_true, y_pred, k=0.04, weight_neg=20.0)

    assert np.isclose(gini_w, 1.0)
    assert cap_w >= 0.0
    assert comp_w <= 1.0


def test_compute_all_metrics_structure():
    """Verify dictionary structure and keys returned by compute_all_metrics."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.65, 0.15, 0.85])

    metrics = compute_all_metrics(y_true, y_pred)

    expected_keys = {
        "competition_metric",
        "normalized_gini",
        "capture_rate_at_4pct",
        "competition_metric_weighted",
        "normalized_gini_weighted",
        "capture_rate_at_4pct_weighted",
        "roc_auc",
        "pr_auc",
        "brier",
    }
    assert expected_keys.issubset(set(metrics.keys()))
    for k, val in metrics.items():
        assert isinstance(val, float)
        assert not np.isnan(val)
