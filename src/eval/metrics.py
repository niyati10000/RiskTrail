"""Evaluation metrics for credit risk default prediction.

Implements normalized Gini, default capture rate at 4%, competition metric,
ROC-AUC, PR-AUC, and Brier score.
"""

from typing import Dict
import numpy as np


def normalized_gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute normalized Gini coefficient (2 * ROC_AUC - 1)."""
    raise NotImplementedError("Phase 5 implementation pending.")


def capture_rate_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: float = 0.04) -> float:
    """Compute default capture rate within the top k (default 4%) highest-risk predictions."""
    raise NotImplementedError("Phase 5 implementation pending.")


def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute primary competition evaluation metric: 0.5 * (normalized_gini + capture_rate@4%)."""
    raise NotImplementedError("Phase 5 implementation pending.")


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive metric suite: competition_metric, gini, capture_at_4pct, roc_auc, pr_auc, brier."""
    raise NotImplementedError("Phase 5 implementation pending.")
