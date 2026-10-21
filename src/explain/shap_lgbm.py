"""SHAP explainability for LightGBM.

Computes TreeExplainer SHAP values, generates global summary plots, and derives
adverse-action reason codes (top positive risk contributors) for declined applicants.
"""

from pathlib import Path
from typing import Any, List
import pandas as pd


def compute_shap_explanations(
    model: Any,
    X: pd.DataFrame,
    output_dir: Path,
    top_k_reasons: int = 5,
) -> pd.DataFrame:
    """Compute SHAP values, plot beeswarm summary, and extract customer reason codes."""
    raise NotImplementedError("Phase 7 implementation pending.")
