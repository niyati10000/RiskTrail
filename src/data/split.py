"""Customer-level stratified splitting module.

Splits data strictly at the customer level (never row-level) stratified on default target,
with optional time-aware holdout. Outputs splits.parquet as the single source of truth.
"""

from pathlib import Path
from typing import Optional
import pandas as pd


def create_customer_splits(
    labels_df: pd.DataFrame,
    customer_id_col: str = "customer_id",
    target_col: str = "target",
    train_size: float = 0.70,
    valid_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Create customer-level stratified splits (train, valid, test).

    Args:
        labels_df: DataFrame containing customer IDs and binary target labels.
        customer_id_col: Column name identifying unique customers.
        target_col: Column name for default binary label.
        train_size: Proportion for training set.
        valid_size: Proportion for validation set.
        test_size: Proportion for test set.
        seed: Random seed for reproducibility.
        output_path: Optional path to save splits.parquet.

    Returns:
        pd.DataFrame: Split assignments per customer_id.
    """
    raise NotImplementedError("Phase 2 implementation pending.")
