"""Data loading module.

Converts raw statement-level data and labels into interim parquet format
with optimized dtypes (e.g., float32, category).
"""

from pathlib import Path
from typing import Optional
import pandas as pd


def load_raw_data(
    statements_path: Path,
    labels_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load raw statement data and optional labels into interim format.

    Args:
        statements_path: Path to raw statements CSV or Parquet file.
        labels_path: Optional path to raw customer labels file.

    Returns:
        pd.DataFrame: Loaded statement-level dataframe.
    """
    raise NotImplementedError("Phase 1 implementation pending.")
