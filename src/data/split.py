"""Customer-level stratified splitting module.

Splits data strictly at the customer level (never row-level) stratified on default target,
with an optional time-aware holdout flag. Outputs splits.parquet as the single source of truth.
"""

import sys
from pathlib import Path
from typing import Optional, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import load_yaml
from src.utils.logging import get_logger

logger = get_logger("RiskTrail.Split")


def create_customer_splits(
    labels_df: pd.DataFrame,
    statements_df: Optional[pd.DataFrame] = None,
    customer_id_col: str = "customer_ID",
    target_col: str = "target",
    train_size: float = 0.70,
    valid_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Create customer-level stratified splits (train, valid, test) with optional time_holdout flag.

    Args:
        labels_df: DataFrame containing customer IDs and binary target labels.
        statements_df: Optional statement DataFrame used to derive customer latest statement dates.
        customer_id_col: Column name identifying unique customers.
        target_col: Column name for default binary label.
        train_size: Proportion for training set (e.g. 0.70).
        valid_size: Proportion for validation set (e.g. 0.15).
        test_size: Proportion for test set (e.g. 0.15).
        seed: Random seed for reproducibility.
        output_path: Optional path to save splits.parquet.

    Returns:
        pd.DataFrame: Split assignments per customer_id with columns [customer_id, split, time_holdout].
    """
    total_ratio = train_size + valid_size + test_size
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split proportions must sum to 1.0 (got {total_ratio})")

    # Standardize column name if necessary
    if customer_id_col not in labels_df.columns:
        matching = [c for c in labels_df.columns if c.lower() == customer_id_col.lower()]
        if matching:
            customer_id_col = matching[0]
        else:
            raise KeyError(f"Customer ID column '{customer_id_col}' not found in labels DataFrame")

    if target_col not in labels_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in labels DataFrame")

    # Ensure unique customer IDs
    if labels_df[customer_id_col].duplicated().any():
        raise ValueError("labels_df contains duplicate customer IDs!")

    n_customers = len(labels_df)
    logger.info(f"Splitting {n_customers} unique customers (seed={seed}, ratios: {train_size}/{valid_size}/{test_size})...")

    # Step 1: Split into Train vs (Valid + Test)
    temp_size = valid_size + test_size
    train_df, temp_df = train_test_split(
        labels_df[[customer_id_col, target_col]],
        test_size=temp_size,
        stratify=labels_df[target_col],
        random_state=seed,
    )

    # Step 2: Split (Valid + Test) proportionally into Valid and Test
    relative_test_size = test_size / temp_size
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df[target_col],
        random_state=seed,
    )

    # Assign split label
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    valid_df["split"] = "valid"
    test_df["split"] = "test"

    splits_df = pd.concat([train_df, valid_df, test_df], axis=0).reset_index(drop=True)

    # Step 3: Compute optional time_holdout flag
    # If statements are provided, customers whose latest statement is in the latest temporal quantile
    # are flagged for optional time-aware out-of-time sensitivity testing.
    splits_df["time_holdout"] = False
    if statements_df is not None:
        stmt_id_col = customer_id_col if customer_id_col in statements_df.columns else "customer_ID"
        stmt_date_col = "S_2" if "S_2" in statements_df.columns else "statement_date"
        if stmt_id_col in statements_df.columns and stmt_date_col in statements_df.columns:
            latest_dates = statements_df.groupby(stmt_id_col)[stmt_date_col].max()
            date_threshold = latest_dates.quantile(0.85)  # latest 15% of statement dates
            oot_customers = set(latest_dates[latest_dates >= date_threshold].index)
            splits_df["time_holdout"] = splits_df[customer_id_col].isin(oot_customers)
            logger.info(f"Flagged {splits_df['time_holdout'].sum()} customers as time_holdout (latest date >= {date_threshold})")

    # Logging verification
    for split_name in ["train", "valid", "test"]:
        sub = splits_df[splits_df["split"] == split_name]
        default_rate = sub[target_col].mean() * 100
        logger.info(f"Split '{split_name}': {len(sub)} customers ({len(sub)/n_customers*100:.1f}%), Default rate: {default_rate:.2f}%")

    # Drop target column so splits.parquet contains only mapping metadata
    result_df = splits_df[[customer_id_col, "split", "time_holdout"]].copy()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(output_path, engine="pyarrow", index=False)
        logger.info(f"Saved splits to {output_path}")

    return result_df


def run_split_pipeline(
    data_config_path: Path = Path("configs/data.yaml"),
    interim_dir: Path = Path("data/interim"),
    processed_dir: Path = Path("data/processed"),
) -> Path:
    """Run full splitting pipeline using project configs and interim files."""
    data_config = load_yaml(data_config_path)
    seed = data_config.get("seed", 42)
    split_ratios = data_config.get("split_ratios", {"train": 0.70, "valid": 0.15, "test": 0.15})
    output_file = processed_dir / "splits.parquet"

    labels_file = interim_dir / "labels.parquet"
    statements_file = interim_dir / "statements.parquet"

    if not labels_file.exists():
        raise FileNotFoundError(f"Interim labels file not found at: {labels_file}")

    labels_df = pd.read_parquet(labels_file)
    statements_df = pd.read_parquet(statements_file) if statements_file.exists() else None

    create_customer_splits(
        labels_df=labels_df,
        statements_df=statements_df,
        customer_id_col="customer_ID",
        target_col="target",
        train_size=split_ratios.get("train", 0.70),
        valid_size=split_ratios.get("valid", 0.15),
        test_size=split_ratios.get("test", 0.15),
        seed=seed,
        output_path=output_file,
    )
    return output_file


if __name__ == "__main__":
    run_split_pipeline()
