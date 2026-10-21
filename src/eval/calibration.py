"""Probability calibration assessment and correction module.

Generates reliability curves, Brier scores, Expected Calibration Error (ECE),
and fits Isotonic Regression and Platt Scaling (logistic calibration) on validation
splits to correct prior shift in class-weighted sequence models.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) across empirical probability bins.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Args:
        y_true: Ground truth binary targets {0, 1}.
        y_prob: Predicted default probabilities in [0, 1].
        n_bins: Number of equal-width probability bins.

    Returns:
        float: Expected Calibration Error in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    N = len(y_true)
    if N == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / N) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compute reliability diagram coordinates, Brier score, and ECE.

    Args:
        y_true: Ground truth binary targets {0, 1}.
        y_prob: Predicted probabilities in [0, 1].
        n_bins: Number of discretization bins.
        strategy: 'uniform' or 'quantile'.

    Returns:
        prob_true: Empirical fraction of positives in each active bin.
        prob_pred: Mean predicted probability in each active bin.
        brier: Brier score loss.
        ece: Expected Calibration Error.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )
    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    return prob_true, prob_pred, brier, ece


class ModelCalibrator:
    """Fits and applies post-hoc probability calibration (Platt scaling or Isotonic Regression)."""

    def __init__(self, method: str = "isotonic"):
        """Initialize calibrator.

        Args:
            method: 'isotonic' for non-parametric isotonic regression,
                    or 'platt' for logistic regression on logit scale.
        """
        self.method = method.lower()
        self.calibrator: Optional[Any] = None
        self._fitted = False

    def fit(self, p_val: np.ndarray, y_val: np.ndarray) -> "ModelCalibrator":
        """Fit calibration mapping on validation split predictions and ground truth labels.

        Args:
            p_val: Validation predicted probabilities in [0, 1].
            y_val: Validation binary ground truth targets {0, 1}.

        Returns:
            self: Fitted calibrator instance.
        """
        p_val = np.asarray(p_val, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.float64)

        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            self.calibrator.fit(p_val, y_val)
        elif self.method == "platt":
            # Map probabilities to logit scale with defensive clipping
            eps = 1e-6
            p_clipped = np.clip(p_val, eps, 1.0 - eps)
            logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            self.calibrator.fit(logits, y_val)
        else:
            raise ValueError(f"Unknown calibration method '{self.method}'. Choose 'isotonic' or 'platt'.")

        self._fitted = True
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        """Apply fitted calibration transform to target probabilities.

        Args:
            p: Predicted probabilities in [0, 1].

        Returns:
            np.ndarray: Calibrated probabilities in [0, 1].
        """
        if not self._fitted or self.calibrator is None:
            raise RuntimeError("ModelCalibrator must be fitted before predict() is called.")

        p = np.asarray(p, dtype=np.float64)
        if self.method == "isotonic":
            p_cal = self.calibrator.predict(p)
            return np.clip(p_cal, 0.0, 1.0).astype(np.float32)
        elif self.method == "platt":
            eps = 1e-6
            p_clipped = np.clip(p, eps, 1.0 - eps)
            logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
            p_cal = self.calibrator.predict_proba(logits)[:, 1]
            return np.clip(p_cal, 0.0, 1.0).astype(np.float32)

        return p.astype(np.float32)

    def evaluate(
        self,
        p_uncal: np.ndarray,
        p_cal: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """Compute comparative calibration diagnostics before and after calibration."""
        _, _, brier_uncal, ece_uncal = compute_calibration_curve(y_true, p_uncal, n_bins=n_bins)
        _, _, brier_cal, ece_cal = compute_calibration_curve(y_true, p_cal, n_bins=n_bins)

        return {
            "brier_uncalibrated": round(brier_uncal, 5),
            "brier_calibrated": round(brier_cal, 5),
            "ece_uncalibrated": round(ece_uncal, 5),
            "ece_calibrated": round(ece_cal, 5),
            "brier_improvement_pct": round(
                (brier_uncal - brier_cal) / max(brier_uncal, 1e-6) * 100.0, 2
            ),
        }
