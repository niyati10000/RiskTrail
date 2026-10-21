"""Data cleaning, categorical encoding, and imputation module.

Guarantees zero data leakage by fitting all imputation statistics (medians)
and categorical value maps strictly on the TRAIN split customers.

Adds:
- Categorical encoding: integer codes (0..K-1), with -1 for missing or unseen values
- Numeric imputation: median values computed strictly on train split rows
- Missing indicators: `<col>__missing` int8 binary indicators for all columns with missing data
- Sequence metadata:
  - `step_idx`: int8 sequential position after sorting by customer and statement date (0..12)
  - `n_statements`: int8 total statement count per customer
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.utils.io import load_yaml, save_json
from src.utils.logging import get_logger

logger = get_logger("RiskTrail.Clean")


def clean_statements(
    statements_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    categorical_cols: List[str],
    numeric_cols: List[str],
    customer_id_col: str = "customer_ID",
    date_col: str = "S_2",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean statement dataframe, encoding categoricals and imputing numerics on train only.

    Args:
        statements_df: Raw or loaded statement DataFrame.
        splits_df: Splits DataFrame with [customer_id_col, 'split'].
        categorical_cols: List of categorical feature names.
        numeric_cols: List of numeric feature names.
        customer_id_col: Column name identifying unique customers.
        date_col: Column name of statement timestamp.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and fitted preprocessing artifacts.
    """
    logger.info("Starting statement data cleaning pipeline...")
    df = statements_df.copy()

    # Sort statements by customer and date ascending for proper sequence tracking
    df = df.sort_values([customer_id_col, date_col]).reset_index(drop=True)

    # Add sequence metadata
    df["step_idx"] = df.groupby(customer_id_col).cumcount().astype(np.int8)
    cust_stmt_counts = df.groupby(customer_id_col).size().rename("n_statements")
    df = df.join(cust_stmt_counts, on=customer_id_col)
    df["n_statements"] = df["n_statements"].astype(np.int8)

    # Identify train split customers for leakage-free fitting
    train_cust_ids = set(splits_df[splits_df["split"] == "train"][customer_id_col])
    train_mask = df[customer_id_col].isin(train_cust_ids)
    train_statements = df[train_mask]

    logger.info(f"Fitted set: {len(train_cust_ids)} train customers ({len(train_statements)} train statement rows).")

    # Step 1: Missing Indicators (<col>__missing)
    # Check all candidate feature columns that have any missing values across the dataset
    all_feature_cols = categorical_cols + numeric_cols
    cols_with_missing = [c for c in all_feature_cols if df[c].isna().any()]
    logger.info(f"Creating missing indicators for {len(cols_with_missing)} columns: {cols_with_missing}")

    for col in cols_with_missing:
        indicator_name = f"{col}__missing"
        df[indicator_name] = df[col].isna().astype(np.int8)

    # Step 2: Categorical Encoding (fit on TRAIN only)
    # Codes: 0..K-1 for observed categories in train, -1 for missing or unseen
    category_mappings: Dict[str, Dict[str, int]] = {}
    for col in categorical_cols:
        # Get unique non-null categories from train split
        # Convert to string to handle mixed or discrete float types consistently
        train_vals = train_statements[col].dropna()
        if isinstance(train_vals.dtype, pd.CategoricalDtype):
            unique_train_cats = [str(c) for c in train_vals.cat.categories if c in train_vals.values]
        else:
            unique_train_cats = sorted([str(x) for x in train_vals.unique()])

        mapping = {cat_val: idx for idx, cat_val in enumerate(unique_train_cats)}
        category_mappings[col] = mapping

        # Map full dataframe: convert to string for mapping, fill unmapped/NaN with -1
        col_str = df[col].astype(str).where(df[col].notna(), other=np.nan)
        encoded_series = col_str.map(mapping).fillna(-1).astype(np.int16)
        df[col] = encoded_series
        logger.info(f"Encoded categorical '{col}': {len(mapping)} train classes (unseen/missing -> -1)")

    # Step 3: Numeric Imputation (medians fitted on TRAIN only)
    train_medians: Dict[str, float] = {}
    for col in numeric_cols:
        train_col_clean = train_statements[col].dropna()
        if len(train_col_clean) > 0:
            median_val = float(train_col_clean.median())
        else:
            median_val = 0.0
        train_medians[col] = median_val

        # Impute missing values with train median
        df[col] = df[col].fillna(median_val).astype(np.float32)

    # Verification: Ensure no numeric NaNs remain
    nan_remaining = df[numeric_cols].isna().sum().sum()
    if nan_remaining > 0:
        raise ValueError(f"Found {nan_remaining} remaining NaNs after numeric imputation!")

    logger.info("Imputation and categorical encoding completed with zero remaining NaNs in numeric features.")

    artifacts = {
        "categorical_mappings": category_mappings,
        "numeric_train_medians": train_medians,
        "columns_with_missing": cols_with_missing,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "n_train_customers": len(train_cust_ids),
        "n_total_customers": df[customer_id_col].nunique(),
        "n_total_statements": len(df),
    }

    return df, artifacts


def run_clean_pipeline(
    interim_dir: Path = Path("data/interim"),
    processed_dir: Path = Path("data/processed"),
    columns_config_path: Path = Path("configs/columns.yaml"),
) -> Tuple[Path, Path]:
    """Execute full statement cleaning and save statements_clean.parquet and clean_artifacts.json."""
    interim_dir = Path(interim_dir)
    processed_dir = Path(processed_dir)

    statements_path = interim_dir / "statements.parquet"
    splits_path = processed_dir / "splits.parquet"

    if not statements_path.exists():
        raise FileNotFoundError(f"Interim statements file not found at: {statements_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"Processed splits file not found at: {splits_path}")

    # Load column definitions
    columns_config = load_yaml(columns_config_path)
    cat_cols = columns_config.get("categorical_features", [])
    num_cols = columns_config.get("numeric_features", [])
    id_col = columns_config.get("id_column", "customer_ID")
    date_col = columns_config.get("date_column", "S_2")

    statements_df = pd.read_parquet(statements_path)
    splits_df = pd.read_parquet(splits_path)

    clean_df, artifacts = clean_statements(
        statements_df=statements_df,
        splits_df=splits_df,
        categorical_cols=cat_cols,
        numeric_cols=num_cols,
        customer_id_col=id_col,
        date_col=date_col,
    )

    # Save cleaned statements to data/interim/statements_clean.parquet
    output_statements_path = interim_dir / "statements_clean.parquet"
    clean_df.to_parquet(output_statements_path, engine="pyarrow", index=False)
    logger.info(f"Saved statements_clean.parquet ({len(clean_df)} rows, {len(clean_df.columns)} columns) to {output_statements_path}")

    # Save artifacts for reproducible inference
    artifacts_path = interim_dir / "clean_artifacts.json"
    save_json(artifacts, artifacts_path)
    logger.info(f"Saved cleaning artifacts to {artifacts_path}")

    return output_statements_path, artifacts_path


if __name__ == "__main__":
    run_clean_pipeline()
