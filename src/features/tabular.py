"""Tabular feature engineering module.

Computes customer-level summary aggregates:
last, first, mean, std, min, max, delta_last_first, slope/trend, delta_last_mean.
Outputs tabular.parquet for tree-based models like LightGBM.
"""

from typing import Sequence
import pandas as pd


def generate_tabular_features(
    df: pd.DataFrame,
    customer_id_col: str,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Generate per-customer statistical aggregates across statement history.

    Args:
        df: Statement-level cleaned DataFrame sorted by customer and date.
        customer_id_col: Column name identifying the customer.
        feature_cols: List of numerical features to aggregate.

    Returns:
        pd.DataFrame: Tabular DataFrame with one row per customer.
    """
    raise NotImplementedError("Phase 2 implementation pending.")
