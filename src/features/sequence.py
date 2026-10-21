"""Sequence feature engineering module.

Constructs 3D tensors [N, 13, F] with boolean key padding masks [N, 13] for PyTorch sequence models
(GRU/LSTM and Transformer).
Padding convention:
    Pad at the front (left/older steps), ensuring index 12 is always the customer's most recent statement.
    mask is a boolean array: True = real statement, False = padding step.

Produces in data/processed/:
- X_seq.npy: float32 [N, 13, F] standardized continuous features
- cat_seq.npy: int64 [N, 13, C] categorical indices for embeddings
- mask.npy: bool [N, 13] boolean padding mask
- y.npy: int8 [N] default labels aligned with ids.npy
- ids.npy: string [N] customer IDs defining row order
- scaler.pkl / scaler_params.json: StandardScaler fitted strictly on TRAIN split statements
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.io import load_yaml
from src.utils.logging import get_logger

logger = get_logger("RiskTrail.Features.Sequence")


def build_sequence_arrays(
    df: pd.DataFrame,
    customer_id_col: str,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    customer_order: Sequence[str],
    max_len: int = 13,
    padding_side: str = "front",
    step_col: str = "step_idx",
    date_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct 3D sequence tensors and boolean padding mask aligned to customer_order.

    Args:
        df: Statement DataFrame sorted chronologically per customer.
        customer_id_col: Column identifying unique customers.
        numeric_cols: Continuous feature column names (already standardized).
        categorical_cols: Categorical feature column names (integer coded).
        customer_order: List of customer IDs defining row ordering.
        max_len: Maximum sequence length (default 13).
        padding_side: Side to apply padding ('front' for older steps).
        step_col: Sequential statement position column (0..k-1).
        date_col: Optional statement timestamp column.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            X_seq: float32 [N, max_len, len(numeric_cols)]
            cat_seq: int64 [N, max_len, len(categorical_cols)]
            mask: bool [N, max_len] (True = real statement, False = padding)
    """
    if padding_side != "front":
        raise ValueError(f"Padding side '{padding_side}' not supported. Locked convention is 'front'.")

    # Sort statements chronologically
    sort_cols = [customer_id_col]
    if date_col and date_col in df.columns:
        sort_cols.append(date_col)
    elif step_col in df.columns:
        sort_cols.append(step_col)

    df_sorted = df.sort_values(sort_cols).reset_index(drop=True)

    # Compute sequential step index and statement counts per customer
    cumcounts = df_sorted.groupby(customer_id_col).cumcount()
    sizes = df_sorted.groupby(customer_id_col)[customer_id_col].transform("size")

    # If sequences exceed max_len, keep the most recent max_len statements
    if (sizes > max_len).any():
        keep_mask = cumcounts >= (sizes - max_len)
        df_sorted = df_sorted[keep_mask].reset_index(drop=True)
        cumcounts = df_sorted.groupby(customer_id_col).cumcount()
        sizes = df_sorted.groupby(customer_id_col)[customer_id_col].transform("size")

    n_cust = len(customer_order)
    n_num = len(numeric_cols)
    n_cat = len(categorical_cols)

    X_seq = np.zeros((n_cust, max_len, n_num), dtype=np.float32)
    cat_seq = np.zeros((n_cust, max_len, n_cat), dtype=np.int64)
    mask = np.zeros((n_cust, max_len), dtype=bool)

    # Vectorized array assignment via coordinate mapping
    cust_to_row = {cid: idx for idx, cid in enumerate(customer_order)}
    row_indices = df_sorted[customer_id_col].map(cust_to_row).values
    time_indices = (max_len - sizes.values + cumcounts.values).astype(int)

    # Only assign rows for customers present in customer_order
    valid_cust_mask = ~np.isnan(row_indices.astype(float)) if row_indices.dtype == object else (row_indices >= 0)
    if not np.all(valid_cust_mask):
        row_indices = row_indices[valid_cust_mask].astype(int)
        time_indices = time_indices[valid_cust_mask]
        df_valid = df_sorted[valid_cust_mask]
    else:
        row_indices = row_indices.astype(int)
        df_valid = df_sorted

    X_seq[row_indices, time_indices, :] = df_valid[list(numeric_cols)].values.astype(np.float32)
    cat_seq[row_indices, time_indices, :] = df_valid[list(categorical_cols)].values.astype(np.int64)
    mask[row_indices, time_indices] = True

    return X_seq, cat_seq, mask


def build_sequence_dataset(
    statements_path: Union[str, Path] = PROJECT_ROOT / "data" / "interim" / "statements_clean.parquet",
    labels_path: Union[str, Path] = PROJECT_ROOT / "data" / "interim" / "labels.parquet",
    splits_path: Union[str, Path] = PROJECT_ROOT / "data" / "processed" / "splits.parquet",
    columns_config_path: Union[str, Path] = PROJECT_ROOT / "configs" / "columns.yaml",
    output_dir: Union[str, Path] = PROJECT_ROOT / "data" / "processed",
    max_len: int = 13,
    padding_side: str = "front",
) -> Dict[str, Any]:
    """Build, standardize, and persist sequence tensors and masks with zero data leakage.

    Fits StandardScaler strictly on statements from TRAIN customers.
    Persists:
    - X_seq.npy [N, 13, F] float32
    - cat_seq.npy [N, 13, C] int64
    - mask.npy [N, 13] bool
    - y.npy [N] int8
    - ids.npy [N] string
    - scaler.pkl and scaler_params.json

    Returns:
        Dict[str, Any]: Metadata summary and tensor profiling statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading cleaned statements from {statements_path}...")
    statements_df = pd.read_parquet(statements_path)

    logger.info(f"Loading labels from {labels_path}...")
    labels_df = pd.read_parquet(labels_path)

    logger.info(f"Loading customer splits from {splits_path}...")
    splits_df = pd.read_parquet(splits_path)

    cols_cfg = load_yaml(columns_config_path)
    id_col = cols_cfg["id_column"]
    target_col = cols_cfg["target_column"]
    numeric_cols = cols_cfg["numeric_features"]
    categorical_cols = cols_cfg["categorical_features"]
    date_col = cols_cfg.get("date_column", "S_2")

    # Determine canonical customer ID ordering (sorted lexicographically matching tabular.parquet)
    unique_customer_ids = sorted(statements_df[id_col].unique().tolist())
    n_customers = len(unique_customer_ids)
    logger.info(f"Identified {n_customers} unique customers.")

    # 1. Zero-leakage standardization: Fit scaler strictly on TRAIN split statements
    split_id_col = "customer_ID" if "customer_ID" in splits_df.columns else "customer_id"
    train_cust_set = set(splits_df[splits_df["split"] == "train"][split_id_col])
    train_statements_mask = statements_df[id_col].isin(train_cust_set)
    train_statements = statements_df[train_statements_mask]

    logger.info(
        f"Fitting StandardScaler on {len(train_statements)} TRAIN statements "
        f"across {len(train_cust_set)} train customers..."
    )
    scaler = StandardScaler()
    scaler.fit(train_statements[numeric_cols].values)

    # Persist fitted scaler (pickle + json metadata)
    scaler_pkl_path = output_dir / "scaler.pkl"
    with open(scaler_pkl_path, "wb") as f:
        pickle.dump(scaler, f)

    scaler_params = {
        col: {"mean": float(m), "scale": float(s), "var": float(v)}
        for col, m, s, v in zip(numeric_cols, scaler.mean_, scaler.scale_, scaler.var_)
    }
    scaler_json_path = output_dir / "scaler_params.json"
    with open(scaler_json_path, "w", encoding="utf-8") as f:
        json.dump(scaler_params, f, indent=2)

    logger.info(f"Saved fitted scaler to {scaler_pkl_path} and {scaler_json_path}")

    # Standardize numeric features across all statements
    df_transformed = statements_df.copy()
    df_transformed[numeric_cols] = scaler.transform(statements_df[numeric_cols].values).astype(np.float32)

    # 2. Build sequence tensors, categorical embeddings index tensor, and mask
    logger.info("Constructing 3D sequence arrays with front-padding...")
    X_seq, cat_seq, mask = build_sequence_arrays(
        df=df_transformed,
        customer_id_col=id_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        customer_order=unique_customer_ids,
        max_len=max_len,
        padding_side=padding_side,
        step_col="step_idx",
        date_col=date_col,
    )

    # 3. Align customer labels (y.npy) with ids.npy
    label_id_col = "customer_ID" if "customer_ID" in labels_df.columns else "customer_id"
    labels_indexed = labels_df.set_index(label_id_col)
    y = labels_indexed.loc[unique_customer_ids, target_col].values.astype(np.int8)
    ids = np.array(unique_customer_ids, dtype=object)

    # 4. Invariant verifications
    assert X_seq.shape == (n_customers, max_len, len(numeric_cols))
    assert cat_seq.shape == (n_customers, max_len, len(categorical_cols))
    assert mask.shape == (n_customers, max_len)
    assert len(y) == n_customers
    assert len(ids) == n_customers

    # Check mask sum matches n_statements per customer
    cust_stmt_counts = statements_df.groupby(id_col)["n_statements"].first().loc[unique_customer_ids].values
    mask_sums = mask.sum(axis=1)
    mismatches = np.where(mask_sums != cust_stmt_counts)[0]
    assert len(mismatches) == 0, f"Mask sum mismatch for {len(mismatches)} customers!"

    # Check padding positions are strictly zeroed
    assert np.all(X_seq[~mask] == 0.0), "Non-zero values found in padded X_seq positions!"
    assert np.all(cat_seq[~mask] == 0), "Non-zero values found in padded cat_seq positions!"

    # 5. Persist array artifacts
    x_seq_path = output_dir / "X_seq.npy"
    cat_seq_path = output_dir / "cat_seq.npy"
    mask_path = output_dir / "mask.npy"
    y_path = output_dir / "y.npy"
    ids_path = output_dir / "ids.npy"

    np.save(x_seq_path, X_seq)
    np.save(cat_seq_path, cat_seq)
    np.save(mask_path, mask)
    np.save(y_path, y)
    np.save(ids_path, ids)

    logger.info("Successfully persisted all sequence arrays to disk.")

    summary = {
        "n_customers": n_customers,
        "max_sequence_length": max_len,
        "n_numeric_features": len(numeric_cols),
        "n_categorical_features": len(categorical_cols),
        "X_seq_shape": list(X_seq.shape),
        "X_seq_dtype": str(X_seq.dtype),
        "cat_seq_shape": list(cat_seq.shape),
        "cat_seq_dtype": str(cat_seq.dtype),
        "mask_shape": list(mask.shape),
        "mask_dtype": str(mask.dtype),
        "y_shape": list(y.shape),
        "y_dtype": str(y.dtype),
        "ids_shape": list(ids.shape),
        "x_seq_path": str(x_seq_path),
        "cat_seq_path": str(cat_seq_path),
        "mask_path": str(mask_path),
        "y_path": str(y_path),
        "ids_path": str(ids_path),
        "scaler_pkl_path": str(scaler_pkl_path),
    }

    # Save artifact metadata
    with open(output_dir / "sequence_artifacts.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("RiskTrail Sequence Tensors Construction Complete")
    print("=" * 60)
    print(f"Total customers (N)         : {n_customers:,}")
    print(f"Max sequence length (T)     : {max_len}")
    print(f"Numeric features (F)        : {len(numeric_cols)}")
    print(f"Categorical features (C)    : {len(categorical_cols)}")
    print(f"X_seq tensor shape          : {X_seq.shape} ({X_seq.dtype})")
    print(f"cat_seq tensor shape        : {cat_seq.shape} ({cat_seq.dtype})")
    print(f"mask array shape            : {mask.shape} ({mask.dtype})")
    print(f"Labels array shape          : {y.shape} ({y.dtype})")
    print(f"Customer IDs array shape    : {ids.shape} ({ids.dtype})")
    print(f"Padding convention          : Front-padded (index 12 = most recent statement)")
    print(f"Standardizer                : Fitted strictly on {len(train_cust_set)} TRAIN customers")
    print(f"Saved artifacts directory   : {output_dir}")
    print("=" * 60 + "\n")

    return summary


if __name__ == "__main__":
    build_sequence_dataset()
