"""Unit tests for sequence feature engineering module (src/features/sequence.py).

Validates:
1. Production array existence, shapes, and dtypes:
   - X_seq.npy: [1000, 13, 22] float32
   - cat_seq.npy: [1000, 13, 4] int64
   - mask.npy: [1000, 13] bool
   - y.npy: [1000] int8
   - ids.npy: [1000] object / string
2. Alignment invariant: Identical customer row order across all sequence arrays and tabular.parquet.
3. Mask invariant: mask.sum(axis=1) == n_statements for every single customer.
4. Front-padding convention:
   - Real statements are placed at the end (index 12 is always True).
   - Padded positions (where mask == False) have strictly 0.0 in X_seq and 0 in cat_seq.
5. Leakage prevention:
   - StandardScaler fitted strictly on statements belonging to train split customers.
   - Train statements have mean ~ 0 and std ~ 1.
6. Synthetic controlled tests on toy sequence histories (varying lengths).
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.sequence import build_sequence_arrays, build_sequence_dataset
from src.utils.io import load_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "columns.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


@pytest.fixture
def columns_config():
    """Load column configuration."""
    return load_yaml(CONFIG_PATH)


@pytest.fixture
def sequence_arrays():
    """Load all persisted sequence arrays and artifacts."""
    x_path = PROCESSED_DIR / "X_seq.npy"
    cat_path = PROCESSED_DIR / "cat_seq.npy"
    mask_path = PROCESSED_DIR / "mask.npy"
    y_path = PROCESSED_DIR / "y.npy"
    ids_path = PROCESSED_DIR / "ids.npy"

    assert x_path.exists(), f"Missing {x_path}"
    assert cat_path.exists(), f"Missing {cat_path}"
    assert mask_path.exists(), f"Missing {mask_path}"
    assert y_path.exists(), f"Missing {y_path}"
    assert ids_path.exists(), f"Missing {ids_path}"

    return {
        "X_seq": np.load(x_path),
        "cat_seq": np.load(cat_path),
        "mask": np.load(mask_path),
        "y": np.load(y_path),
        "ids": np.load(ids_path, allow_pickle=True),
    }


def test_sequence_shapes_and_dtypes(sequence_arrays, columns_config):
    """Verify exact tensor dimensions and types across all persisted arrays."""
    n_cust = 1000
    seq_len = 13
    n_num = len(columns_config["numeric_features"])  # 22
    n_cat = len(columns_config["categorical_features"])  # 4

    X_seq = sequence_arrays["X_seq"]
    cat_seq = sequence_arrays["cat_seq"]
    mask = sequence_arrays["mask"]
    y = sequence_arrays["y"]
    ids = sequence_arrays["ids"]

    assert X_seq.shape == (n_cust, seq_len, n_num)
    assert X_seq.dtype == np.float32

    assert cat_seq.shape == (n_cust, seq_len, n_cat)
    assert cat_seq.dtype == np.int64

    assert mask.shape == (n_cust, seq_len)
    assert mask.dtype == bool

    assert y.shape == (n_cust,)
    assert y.dtype == np.int8

    assert ids.shape == (n_cust,)
    assert len(set(ids)) == n_cust


def test_mask_sum_equals_n_statements(sequence_arrays):
    """Verify that mask.sum(1) exactly equals each customer's actual statement count."""
    stmts_df = pd.read_parquet(INTERIM_DIR / "statements_clean.parquet")
    mask = sequence_arrays["mask"]
    ids = sequence_arrays["ids"]

    # Customer statement counts indexed by customer_ID
    cust_stmt_counts = stmts_df.groupby("customer_ID")["n_statements"].first().loc[ids].values
    mask_sums = mask.sum(axis=1)

    assert np.array_equal(mask_sums, cust_stmt_counts), (
        "Discrepancy detected between mask.sum(axis=1) and customer n_statements!"
    )
    assert mask_sums.min() >= 2
    assert mask_sums.max() <= 13


def test_front_padding_invariants(sequence_arrays):
    """Verify front-padding convention: real steps at the back, padding zeros at front."""
    X_seq = sequence_arrays["X_seq"]
    cat_seq = sequence_arrays["cat_seq"]
    mask = sequence_arrays["mask"]

    # 1. Most recent statement (index 12 / -1) must always be True
    assert np.all(mask[:, -1] == True), "Index -1 (most recent statement) must be True for all customers!"

    # 2. Customers with < 13 statements must have index 0 as False (front padded)
    short_seq_mask = mask.sum(axis=1) < 13
    assert np.all(mask[short_seq_mask, 0] == False), "Older steps must be padded at the front!"

    # 3. All padding positions must be strictly 0.0 in X_seq and 0 in cat_seq
    pad_positions = ~mask
    assert np.all(X_seq[pad_positions] == 0.0), "X_seq contains non-zero values in padded positions!"
    assert np.all(cat_seq[pad_positions] == 0), "cat_seq contains non-zero values in padded positions!"


def test_row_order_alignment_with_tabular_parquet(sequence_arrays):
    """Verify that ids.npy, y.npy, and sequence tensors align 1-to-1 with tabular.parquet."""
    tabular_df = pd.read_parquet(PROCESSED_DIR / "tabular.parquet")
    ids = sequence_arrays["ids"]
    y = sequence_arrays["y"]

    # Customer ID alignment
    assert np.array_equal(ids, tabular_df["customer_id"].values), (
        "Row order of ids.npy does not match tabular.parquet customer_id!"
    )

    # Label alignment
    assert np.array_equal(y, tabular_df["target"].values), (
        "Row order of y.npy does not match tabular.parquet target!"
    )


def test_standardizer_fitted_strictly_on_train_split(columns_config):
    """Verify that the saved StandardScaler was fitted strictly on train customers without leakage."""
    scaler_path = PROCESSED_DIR / "scaler.pkl"
    assert scaler_path.exists(), f"Scaler not found at {scaler_path}"

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    stmts_df = pd.read_parquet(INTERIM_DIR / "statements_clean.parquet")
    splits_df = pd.read_parquet(PROCESSED_DIR / "splits.parquet")

    split_id_col = "customer_ID" if "customer_ID" in splits_df.columns else "customer_id"
    train_cids = set(splits_df[splits_df["split"] == "train"][split_id_col])
    train_stmts = stmts_df[stmts_df["customer_ID"].isin(train_cids)]
    all_stmts = stmts_df

    numeric_cols = columns_config["numeric_features"]

    # Compute reference statistics
    train_means = train_stmts[numeric_cols].mean().values
    all_means = all_stmts[numeric_cols].mean().values

    # Scaler mean must match train statistics exactly
    assert np.allclose(scaler.mean_, train_means, atol=1e-5), (
        "Saved scaler mean does not match train split statistics!"
    )

    # Scaler mean must differ from the full dataset mean (leakage detector)
    assert not np.allclose(scaler.mean_, all_means, atol=1e-5), (
        "Scaler mean identically matches full dataset mean, indicating potential test leakage!"
    )


def test_build_sequence_arrays_synthetic_toy():
    """Verify build_sequence_arrays on controlled synthetic toy sequences."""
    # Cust 1: 2 statements, Cust 2: 3 statements
    toy_df = pd.DataFrame({
        "cid": ["c1", "c1", "c2", "c2", "c2"],
        "step": [0, 1, 0, 1, 2],
        "num_f": [10.0, 20.0, 1.0, 2.0, 3.0],
        "cat_f": [5, 6, 7, 8, 9],
    })

    X, C, M = build_sequence_arrays(
        df=toy_df,
        customer_id_col="cid",
        numeric_cols=["num_f"],
        categorical_cols=["cat_f"],
        customer_order=["c1", "c2"],
        max_len=4,  # Use max_len = 4 for simple testing
        padding_side="front",
        step_col="step",
    )

    assert X.shape == (2, 4, 1)
    assert C.shape == (2, 4, 1)
    assert M.shape == (2, 4)

    # Cust 1 has 2 statements -> indices [2, 3] should be True, [0, 1] False
    assert list(M[0]) == [False, False, True, True]
    assert np.isclose(X[0, 2, 0], 10.0)
    assert np.isclose(X[0, 3, 0], 20.0)
    assert X[0, 0, 0] == 0.0
    assert X[0, 1, 0] == 0.0
    assert list(C[0, :, 0]) == [0, 0, 5, 6]

    # Cust 2 has 3 statements -> indices [1, 2, 3] True, [0] False
    assert list(M[1]) == [False, True, True, True]
    assert np.isclose(X[1, 1, 0], 1.0)
    assert np.isclose(X[1, 2, 0], 2.0)
    assert np.isclose(X[1, 3, 0], 3.0)
    assert X[1, 0, 0] == 0.0
    assert list(C[1, :, 0]) == [0, 7, 8, 9]
