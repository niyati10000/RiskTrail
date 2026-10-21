"""Evaluation metrics for credit risk default prediction models.

Implements:
1. Standard unweighted metrics:
   - normalized_gini: 2 * ROC_AUC - 1
   - capture_rate_at_k: proportion of all actual defaults captured in the top k (default 4%) highest-risk predictions
   - competition_metric: 0.5 * (normalized_gini + capture_rate_at_4pct)
   - roc_auc: Area Under the Receiver Operating Characteristic Curve
   - pr_auc: Area Under the Precision-Recall Curve (Average Precision)
   - brier: Mean squared probability error (Brier score loss)

2. Official competition weighted metrics:
   - Weights negative samples by 20x to account for 5% negative subsampling in competition data
   - normalized_gini_weighted: Lorentz curve AUC over sample weights normalized by maximum achievable Gini
   - capture_rate_at_k_weighted: Defaults captured within the top 4% of total sample weight
   - competition_metric_weighted: 0.5 * (normalized_gini_weighted + capture_rate_at_k_weighted)
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC).

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_pred: Predicted default probabilities in [0, 1].

    Returns:
        float: ROC-AUC score in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_pred))


def pr_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Area Under the Precision-Recall Curve (Average Precision score).

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_pred: Predicted default probabilities in [0, 1].

    Returns:
        float: PR-AUC score in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true))
    return float(average_precision_score(y_true, y_pred))


def brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Brier score loss (mean squared calibration error).

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_pred: Predicted default probabilities in [0, 1].

    Returns:
        float: Brier score (lower is better, 0 is perfect).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(brier_score_loss(y_true, y_pred))


def normalized_gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute standard unweighted normalized Gini coefficient: 2 * ROC_AUC - 1.

    Properties:
        Perfect model: 1.0
        Random model: 0.0
        Inverted model: -1.0

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_pred: Predicted probabilities in [0, 1].

    Returns:
        float: Normalized Gini coefficient in [-1, 1].
    """
    return float(2.0 * roc_auc(y_true, y_pred) - 1.0)


def capture_rate_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: float = 0.04) -> float:
    """Compute fraction of total actual defaults captured within the top k highest-risk predictions.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_pred: Predicted default probabilities.
        k: Top population fraction cutoff (default 0.04 = top 4%).

    Returns:
        float: Capture rate in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0 or np.sum(y_true) == 0:
        return 0.0

    # Sort descending by predicted probability
    order = np.argsort(-y_pred, kind="mergesort")
    y_sorted = y_true[order]

    # Number of candidates in the top k fraction (at least 1 if k > 0 and n > 0)
    top_k_count = int(np.ceil(k * n))
    top_k_count = min(max(top_k_count, 1), n)

    defaults_captured = np.sum(y_sorted[:top_k_count])
    total_defaults = np.sum(y_true)
    return float(defaults_captured / total_defaults)


def competition_metric(y_true: np.ndarray, y_pred: np.ndarray, k: float = 0.04) -> float:
    """Compute unweighted competition evaluation metric: 0.5 * (normalized_gini + capture_rate@k).

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_pred: Predicted default probabilities.
        k: Top fraction for capture rate (default 0.04).

    Returns:
        float: Competition score in [-0.5, 1.0].
    """
    gini = normalized_gini(y_true, y_pred)
    cap = capture_rate_at_k(y_true, y_pred, k=k)
    return float(0.5 * (gini + cap))


def _weighted_gini_score(y_true: np.ndarray, y_pred: np.ndarray, weight_neg: float = 20.0) -> float:
    """Internal helper to compute raw weighted Gini area with negative sample weights."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df = df.sort_values("y_pred", ascending=False, kind="mergesort").reset_index(drop=True)
    df["weight"] = np.where(df["y_true"] == 0, weight_neg, 1.0)

    total_weight = df["weight"].sum()
    total_pos = (df["y_true"] * df["weight"]).sum()
    if total_pos == 0 or total_weight == 0:
        return 0.0

    df["random"] = (df["weight"] / total_weight).cumsum()
    df["cum_pos"] = (df["y_true"] * df["weight"]).cumsum()
    df["lorentz"] = df["cum_pos"] / total_pos
    df["gini"] = (df["lorentz"] - df["random"]) * df["weight"]
    return float(df["gini"].sum())


def normalized_gini_weighted(y_true: np.ndarray, y_pred: np.ndarray, weight_neg: float = 20.0) -> float:
    """Compute official competition normalized weighted Gini with negative weights.

    In the competition dataset, negative non-defaulters were subsampled at 5%, so negatives
    are re-weighted 20x to restore true population prevalence.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_pred: Predicted default probabilities.
        weight_neg: Multiplier for negative samples (default 20.0).

    Returns:
        float: Normalized weighted Gini coefficient in [-1, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = _weighted_gini_score(y_true, y_true, weight_neg=weight_neg)
    if denom == 0.0:
        return 0.0
    num = _weighted_gini_score(y_true, y_pred, weight_neg=weight_neg)
    return float(num / denom)


def capture_rate_at_k_weighted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: float = 0.04,
    weight_neg: float = 20.0,
) -> float:
    """Compute official competition default capture rate at top k% of weighted population.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_pred: Predicted default probabilities.
        k: Top weighted population fraction cutoff (default 0.04).
        weight_neg: Multiplier for negative samples (default 20.0).

    Returns:
        float: Weighted capture rate in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total_defaults = np.sum(y_true)
    if total_defaults == 0:
        return 0.0

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df = df.sort_values("y_pred", ascending=False, kind="mergesort").reset_index(drop=True)
    df["weight"] = np.where(df["y_true"] == 0, weight_neg, 1.0)
    df["cum_weight"] = df["weight"].cumsum()

    cutoff_weight = k * df["weight"].sum()
    df_top_slice = df[df["cum_weight"] <= cutoff_weight]

    # If the first item alone exceeds cutoff, include at least the first item
    if len(df_top_slice) == 0 and len(df) > 0:
        df_top_slice = df.iloc[:1]

    captured_defaults = df_top_slice["y_true"].sum()
    return float(captured_defaults / total_defaults)


def competition_metric_weighted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: float = 0.04,
    weight_neg: float = 20.0,
) -> float:
    """Compute official competition metric with 20x negative weighting: 0.5 * (G_weighted + D_weighted).

    Args:
        y_true: Binary ground truth labels.
        y_pred: Predicted probabilities.
        k: Top population fraction for capture rate (default 0.04).
        weight_neg: Multiplier for negative samples (default 20.0).

    Returns:
        float: Weighted competition metric.
    """
    gini_w = normalized_gini_weighted(y_true, y_pred, weight_neg=weight_neg)
    cap_w = capture_rate_at_k_weighted(y_true, y_pred, k=k, weight_neg=weight_neg)
    return float(0.5 * (gini_w + cap_w))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: float = 0.04,
    weight_neg: float = 20.0,
) -> Dict[str, float]:
    """Compute complete metric suite for model benchmarking.

    Args:
        y_true: Binary ground truth labels.
        y_pred: Predicted probabilities.
        k: Top fraction for default capture rate (default 0.04).
        weight_neg: Multiplier for negative samples (default 20.0).

    Returns:
        Dict[str, float]: Dictionary of all evaluation metrics rounded to 5 decimal places.
    """
    return {
        "competition_metric": round(competition_metric(y_true, y_pred, k=k), 5),
        "normalized_gini": round(normalized_gini(y_true, y_pred), 5),
        "capture_rate_at_4pct": round(capture_rate_at_k(y_true, y_pred, k=k), 5),
        "competition_metric_weighted": round(
            competition_metric_weighted(y_true, y_pred, k=k, weight_neg=weight_neg), 5
        ),
        "normalized_gini_weighted": round(
            normalized_gini_weighted(y_true, y_pred, weight_neg=weight_neg), 5
        ),
        "capture_rate_at_4pct_weighted": round(
            capture_rate_at_k_weighted(y_true, y_pred, k=k, weight_neg=weight_neg), 5
        ),
        "roc_auc": round(roc_auc(y_true, y_pred), 5),
        "pr_auc": round(pr_auc(y_true, y_pred), 5),
        "brier": round(brier(y_true, y_pred), 5),
    }
