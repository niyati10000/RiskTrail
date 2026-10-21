"""LightGBM tabular baseline model training and inference.

Trains on customer-level tabular features with early stopping on validation split.
Saves predictions (preds_valid.npy, preds_test.npy), model.txt, metrics.json, and config.json.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np


def train_lgbm(
    config: Dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    seed: int = 42,
) -> Dict[str, float]:
    """Train LightGBM tabular model with early stopping.

    Args:
        config: Model hyperparameter configuration dictionary.
        data_dir: Directory containing tabular.parquet and splits.parquet.
        output_dir: Directory to save model, predictions, and metrics.
        seed: Random seed.

    Returns:
        Dict[str, float]: Validation and test metric scores.
    """
    raise NotImplementedError("Phase 3 implementation pending.")
