"""Sequence feature engineering module.

Constructs 3D tensors [N, 13, F] with key padding mask [N, 13] for sequential models.
Padding convention: pad at the front (left/older steps), ensuring the last step
is always the customer's most recent statement.
"""

from typing import Tuple, Sequence
import numpy as np
import pandas as pd


def generate_sequence_tensors(
    df: pd.DataFrame,
    customer_id_col: str,
    feature_cols: Sequence[str],
    max_len: int = 13,
    padding_side: str = "front",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sequence arrays [N, seq_len, F], boolean mask [N, seq_len], and customer IDs.

    Args:
        df: Cleaned statement data sorted by customer and date.
        customer_id_col: Column name identifying the customer.
        feature_cols: Numeric and encoded feature columns.
        max_len: Maximum history length (default 13).
        padding_side: Side to pad ('front' for older steps).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (X_seq, mask, customer_ids)
    """
    raise NotImplementedError("Phase 2 implementation pending.")
