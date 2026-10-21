"""Tabular feature engineering module.

Vectorized customer-level summary aggregations across statement sequences:
- Numeric features (9 aggs): last, first, mean, std, min, max, delta_last_first, slope, delta_last_mean
- Categorical features (4 aggs): last, first, nunique, count
- Double-underscore column naming format: {feature}__{agg}

Produces data/processed/tabular.parquet containing customer_id, aggregated features (float32 / int16), and target.
Guarantees strictly vectorized groupby computations with zero Python loops over customers.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.utils.io import load_yaml
from src.utils.logging import get_logger

logger = get_logger("RiskTrail.Features.Tabular")


def compute_vectorized_slopes(
    df: pd.DataFrame,
    customer_id_col: str,
    numeric_cols: Sequence[str],
    step_col: str,
    grouped_size: pd.Series,
    mean_y_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute simple linear regression slope across time steps for all customers and features.

    Vectorized implementation:
        Slope = (sum(x_i * y_i) - n * x_bar * y_bar) / sum((x_i - x_bar)^2)
        Given x_i = step_idx in [0, n - 1]:
            x_bar = (n - 1) / 2
            sum((x_i - x_bar)^2) = n * (n^2 - 1) / 12
            numerator = sum(x_i * y_i) - [n * (n - 1) / 2] * y_bar

    Args:
        df: Statement DataFrame with customer_id, numeric features, and step column.
        customer_id_col: Column name identifying the customer.
        numeric_cols: List of numeric feature names.
        step_col: Column name representing chronological step index (0..n-1).
        grouped_size: Series with customer statement counts (index = customer_id).
        mean_y_df: DataFrame with mean values for each numeric feature (index = customer_id).

    Returns:
        pd.DataFrame: Slopes DataFrame indexed by customer_id with columns {feature}__slope.
    """
    n_cust = grouped_size.values.astype(np.float64)[:, np.newaxis]  # shape (N, 1)
    denom = (n_cust * (n_cust**2 - 1.0)) / 12.0  # shape (N, 1)

    # Compute step * feature products in vectorized matrix form
    step_vals = df[step_col].values[:, np.newaxis]
    feature_matrix = df[list(numeric_cols)].values
    xy_matrix = feature_matrix * step_vals

    xy_df = pd.DataFrame(xy_matrix, columns=list(numeric_cols))
    xy_df[customer_id_col] = df[customer_id_col].values
    sum_xy = xy_df.groupby(customer_id_col, sort=True).sum().reindex(mean_y_df.index)

    # Numerator calculation
    n_term = (n_cust * (n_cust - 1.0)) / 2.0  # shape (N, 1)
    mean_y_matrix = mean_y_df.values  # shape (N, F)
    numerator = sum_xy.values - (n_term * mean_y_matrix)

    # Safe division: if n <= 1 or denom <= 0, slope is 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = np.where(denom > 0, numerator / denom, 0.0)
        # Catch any residual NaN/inf values defensively
        slopes = np.nan_to_num(slopes, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    slope_cols = [f"{col}__slope" for col in numeric_cols]
    return pd.DataFrame(slopes, index=mean_y_df.index, columns=slope_cols)


def generate_tabular_features(
    df: pd.DataFrame,
    customer_id_col: str,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    step_col: str = "step_idx",
    date_col: Optional[str] = None,
    output_id_col: str = "customer_id",
) -> pd.DataFrame:
    """Generate per-customer statistical aggregates across statement history.

    Computes:
    - Numeric: last, first, mean, std, min, max, delta_last_first, slope, delta_last_mean
    - Categorical: last, first, nunique, count

    Args:
        df: Statement DataFrame sorted by customer and date/step_idx.
        customer_id_col: Column name identifying the customer in df.
        numeric_cols: List of numerical features to aggregate.
        categorical_cols: List of categorical features to aggregate.
        step_col: Column name for sequential statement index.
        date_col: Optional column name for statement timestamp.
        output_id_col: Column name for customer ID in output DataFrame.

    Returns:
        pd.DataFrame: Tabular DataFrame with one row per customer.
    """
    logger.info(
        f"Computing tabular summary features for {df[customer_id_col].nunique()} customers "
        f"across {len(df)} statements..."
    )

    # Ensure statements are strictly ordered chronologically
    sort_cols = [customer_id_col]
    if date_col and date_col in df.columns:
        sort_cols.append(date_col)
    elif step_col in df.columns:
        sort_cols.append(step_col)

    df_sorted = df.sort_values(sort_cols).reset_index(drop=True)

    # Ensure step_idx exists
    if step_col not in df_sorted.columns:
        df_sorted[step_col] = df_sorted.groupby(customer_id_col).cumcount().astype(np.int8)

    grouped = df_sorted.groupby(customer_id_col, sort=True)
    cust_sizes = grouped.size()

    # 1. Base numeric aggregations (last, first, mean, std, min, max)
    logger.info(f"Aggregating {len(numeric_cols)} numeric features (last, first, mean, std, min, max)...")
    base_num = grouped[list(numeric_cols)].agg(["last", "first", "mean", "std", "min", "max"])
    base_num.columns = [f"{col}__{stat}" for col, stat in base_num.columns]

    # Fill sample std NaN with 0.0 (e.g. single statement customers)
    std_col_names = [f"{col}__std" for col in numeric_cols]
    base_num[std_col_names] = base_num[std_col_names].fillna(0.0)

    # 2. Derived numeric deltas (delta_last_first, delta_last_mean)
    logger.info("Computing derived deltas: delta_last_first and delta_last_mean...")
    deltas_dict: Dict[str, pd.Series] = {}
    for col in numeric_cols:
        deltas_dict[f"{col}__delta_last_first"] = (
            base_num[f"{col}__last"] - base_num[f"{col}__first"]
        ).astype(np.float32)
        deltas_dict[f"{col}__delta_last_mean"] = (
            base_num[f"{col}__last"] - base_num[f"{col}__mean"]
        ).astype(np.float32)
    deltas_df = pd.DataFrame(deltas_dict, index=base_num.index)

    # 3. Vectorized slope calculation across all features and customers
    logger.info(f"Computing vectorized linear slopes for {len(numeric_cols)} features...")
    mean_cols = [f"{col}__mean" for col in numeric_cols]
    mean_y_df = base_num[mean_cols]
    slopes_df = compute_vectorized_slopes(
        df=df_sorted,
        customer_id_col=customer_id_col,
        numeric_cols=numeric_cols,
        step_col=step_col,
        grouped_size=cust_sizes,
        mean_y_df=mean_y_df,
    )

    # 4. Categorical aggregations (last, first, nunique, count)
    logger.info(f"Aggregating {len(categorical_cols)} categorical features (last, first, nunique, count)...")
    cat_aggs = grouped[list(categorical_cols)].agg(["last", "first", "nunique", "count"])
    cat_aggs.columns = [f"{col}__{stat}" for col, stat in cat_aggs.columns]

    # Assemble columns in clean, deterministic order
    ordered_feature_names: List[str] = []
    for c in numeric_cols:
        for stat in [
            "last",
            "first",
            "mean",
            "std",
            "min",
            "max",
            "delta_last_first",
            "slope",
            "delta_last_mean",
        ]:
            ordered_feature_names.append(f"{c}__{stat}")
    for c in categorical_cols:
        for stat in ["last", "first", "nunique", "count"]:
            ordered_feature_names.append(f"{c}__{stat}")

    # Concatenate all aggregate blocks at once to prevent dataframe fragmentation
    features_df = pd.concat([base_num, deltas_df, slopes_df, cat_aggs], axis=1)[ordered_feature_names]

    # Cast dtypes: strictly float32 for continuous features, int16 for categorical stats
    for col in features_df.columns:
        if features_df[col].dtype == "float64":
            features_df[col] = features_df[col].astype(np.float32)
        elif features_df[col].dtype in ["int64", "int32"]:
            features_df[col] = features_df[col].astype(np.int16)

    # Reset customer index to column
    features_df = features_df.reset_index().rename(columns={customer_id_col: output_id_col})
    return features_df


def build_tabular_dataset(
    statements_path: Union[str, Path] = PROJECT_ROOT / "data" / "interim" / "statements_clean.parquet",
    labels_path: Union[str, Path] = PROJECT_ROOT / "data" / "interim" / "labels.parquet",
    columns_config_path: Union[str, Path] = PROJECT_ROOT / "configs" / "columns.yaml",
    output_path: Union[str, Path] = PROJECT_ROOT / "data" / "processed" / "tabular.parquet",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build, save, and profile customer-level tabular dataset.

    Args:
        statements_path: Path to cleaned statements parquet file.
        labels_path: Path to customer labels parquet file.
        columns_config_path: Path to columns YAML configuration.
        output_path: Destination path for tabular parquet output.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Result DataFrame and metadata metrics.
    """
    statements_path = Path(statements_path)
    labels_path = Path(labels_path)
    columns_config_path = Path(columns_config_path)
    output_path = Path(output_path)

    logger.info(f"Loading cleaned statements from {statements_path}...")
    statements_df = pd.read_parquet(statements_path)

    logger.info(f"Loading labels from {labels_path}...")
    labels_df = pd.read_parquet(labels_path)

    cols_cfg = load_yaml(columns_config_path)
    id_col = cols_cfg["id_column"]
    target_col = cols_cfg["target_column"]
    numeric_cols = cols_cfg["numeric_features"]
    categorical_cols = cols_cfg["categorical_features"]

    # Generate customer-level aggregates
    tabular_df = generate_tabular_features(
        df=statements_df,
        customer_id_col=id_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        step_col="step_idx",
        date_col=cols_cfg.get("date_column", "S_2"),
        output_id_col="customer_id",
    )

    # Standardize label DataFrame and merge target
    labels_standardized = labels_df.rename(columns={id_col: "customer_id"})
    if target_col in labels_standardized.columns:
        tabular_df = tabular_df.merge(
            labels_standardized[["customer_id", target_col]],
            on="customer_id",
            how="left",
        )
        tabular_df[target_col] = tabular_df[target_col].astype(np.int8)
    else:
        raise KeyError(f"Target column '{target_col}' not found in {labels_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tabular_df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(f"Saved tabular dataset to {output_path}")

    # Compute profiling metrics
    n_rows, n_cols = tabular_df.shape
    mem_bytes = tabular_df.memory_usage(deep=True).sum()
    mem_mb = mem_bytes / (1024.0 * 1024.0)

    # Invariant checks
    assert not tabular_df["customer_id"].duplicated().any(), "Duplicate customer_id detected in tabular output!"
    assert tabular_df.isna().sum().sum() == 0, f"Detected unexpected NaNs in tabular output: {tabular_df.isna().sum().to_dict()}"

    stats = {
        "n_rows": n_rows,
        "n_columns": n_cols,
        "memory_mb": round(mem_mb, 4),
        "n_numeric_features": len(numeric_cols),
        "n_numeric_aggs": len(numeric_cols) * 9,
        "n_categorical_features": len(categorical_cols),
        "n_categorical_aggs": len(categorical_cols) * 4,
        "output_path": str(output_path),
    }

    print("\n" + "=" * 60)
    print("RiskTrail Tabular Feature Engineering Complete")
    print("=" * 60)
    print(f"Total customers (rows)    : {n_rows:,}")
    print(f"Total columns             : {n_cols}")
    print(f"Memory footprint (RAM)    : {mem_mb:.2f} MB")
    print(f"Numeric features          : {len(numeric_cols)} (9 aggregates each -> {len(numeric_cols) * 9} cols)")
    print(f"Categorical features      : {len(categorical_cols)} (4 aggregates each -> {len(categorical_cols) * 4} cols)")
    print(f"Identifiers & labels      : customer_id, target")
    print(f"Saved artifact            : {output_path}")
    print("=" * 60 + "\n")

    return tabular_df, stats


if __name__ == "__main__":
    build_tabular_dataset()
