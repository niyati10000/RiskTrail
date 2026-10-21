"""Data loading module.

Converts raw CSV statement-level data and labels into optimized interim Parquet files
using float32 for continuous features and category dtypes for categorical features.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.utils.logging import get_logger

logger = get_logger("RiskTrail.Load")

# Standard categorical features defined by the Amex benchmark specification
STANDARD_AMEX_CATEGORICALS = {
    "B_30", "B_38", "D_114", "D_116", "D_117", "D_120",
    "D_126", "D_63", "D_64", "D_66", "D_68"
}


def identify_column_types(
    df: pd.DataFrame,
    id_col: str = "customer_ID",
    date_col: str = "S_2",
) -> Tuple[List[str], List[str]]:
    """Partition feature columns into categorical and numerical features.

    Args:
        df: Input raw statement dataframe.
        id_col: Column name of customer identifier.
        date_col: Column name of statement date.

    Returns:
        Tuple[List[str], List[str]]: (categorical_cols, numeric_cols)
    """
    excluded = {id_col, date_col, "target"}
    candidate_cols = [c for c in df.columns if c not in excluded]

    categorical_cols = []
    numeric_cols = []

    for col in candidate_cols:
        # Check standard categorical list or string/category dtype
        if (
            col in STANDARD_AMEX_CATEGORICALS
            or pd.api.types.is_string_dtype(df[col])
            or isinstance(df[col].dtype, pd.CategoricalDtype)
        ):
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    return categorical_cols, numeric_cols


def convert_raw_to_interim(
    raw_dir: Path = Path("data/raw"),
    interim_dir: Path = Path("data/interim"),
    statements_filename: str = "train_data.csv",
    labels_filename: str = "train_labels.csv",
) -> Tuple[Path, Path]:
    """Convert raw statement CSVs and label CSVs to optimized Parquet files.

    Args:
        raw_dir: Directory containing raw CSV files.
        interim_dir: Directory to store interim Parquet files.
        statements_filename: Name of the statement CSV file.
        labels_filename: Name of the customer labels CSV file.

    Returns:
        Tuple[Path, Path]: Paths to saved (statements_parquet, labels_parquet).
    """
    raw_dir = Path(raw_dir)
    interim_dir = Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)

    statements_csv = raw_dir / statements_filename
    labels_csv = raw_dir / labels_filename

    if not statements_csv.exists():
        raise FileNotFoundError(f"Raw statements file not found at: {statements_csv}")
    if not labels_csv.exists():
        raise FileNotFoundError(f"Raw labels file not found at: {labels_csv}")

    logger.info(f"Loading raw statements from {statements_csv}...")
    statements_df = pd.read_csv(statements_csv)
    initial_mem = statements_df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(f"Loaded {len(statements_df)} statement rows. Memory: {initial_mem:.2f} MB")

    # Standardize column naming if needed
    id_col = "customer_ID" if "customer_ID" in statements_df.columns else "customer_id"
    date_col = "S_2" if "S_2" in statements_df.columns else "statement_date"

    # Identify categorical and numeric columns
    categorical_cols, numeric_cols = identify_column_types(statements_df, id_col=id_col, date_col=date_col)

    logger.info(f"Identified {len(categorical_cols)} categorical ({categorical_cols}) and {len(numeric_cols)} numeric features.")

    # Cast datatypes for efficiency
    for col in numeric_cols:
        statements_df[col] = statements_df[col].astype(np.float32)

    for col in categorical_cols:
        statements_df[col] = statements_df[col].astype("category")

    # Date parsing
    statements_df[date_col] = pd.to_datetime(statements_df[date_col])

    optimized_mem = statements_df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(f"Memory after casting to float32 & category: {optimized_mem:.2f} MB ({100*(1 - optimized_mem/initial_mem):.1f}% reduction)")

    # Save statements to Parquet
    statements_out = interim_dir / "statements.parquet"
    statements_df.to_parquet(statements_out, engine="pyarrow", index=False)
    logger.info(f"Saved statements parquet to {statements_out}")

    # Load and optimize labels
    logger.info(f"Loading raw labels from {labels_csv}...")
    labels_df = pd.read_csv(labels_csv)
    labels_df["target"] = labels_df["target"].astype(np.int8)
    labels_out = interim_dir / "labels.parquet"
    labels_df.to_parquet(labels_out, engine="pyarrow", index=False)
    logger.info(f"Saved labels parquet ({len(labels_df)} customers) to {labels_out}")

    return statements_out, labels_out


if __name__ == "__main__":
    convert_raw_to_interim()
