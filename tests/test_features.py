"""Unit tests for tabular feature engineering module (src/features/tabular.py).

Validates:
1. Shape, schema, and column counts for tabular.parquet (1,000 customers, 216 columns).
2. Naming convention: {feature}__{agg} for all numeric and categorical features.
3. Strict float32 casting for continuous features and absence of NaNs / Infs.
4. Mathematical correctness of all aggregations on controlled synthetic data:
   - last, first, mean, std, min, max
   - delta_last_first, delta_last_mean
   - slope (comparing closed-form vectorized formulation to numpy.polyfit reference)
5. Edge cases:
   - Single-statement customer (n=1): slope=0.0, std=0.0 without NaN or divide-by-zero
   - Constant sequence (zero variance): slope=0.0, std=0.0
   - Decreasing sequences with negative slopes
6. Categorical aggregations: last, first, nunique, count.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.tabular import (
    build_tabular_dataset,
    compute_vectorized_slopes,
    generate_tabular_features,
)
from src.utils.io import load_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "columns.yaml"
TABULAR_PATH = PROJECT_ROOT / "data" / "processed" / "tabular.parquet"


@pytest.fixture
def columns_config():
    """Load column configuration."""
    return load_yaml(CONFIG_PATH)


@pytest.fixture
def tabular_df():
    """Load processed tabular parquet file."""
    assert TABULAR_PATH.exists(), f"Processed tabular dataset not found at {TABULAR_PATH}"
    return pd.read_parquet(TABULAR_PATH)


def test_tabular_parquet_shape_and_keys(tabular_df, columns_config):
    """Verify customer-level tabular dataset shape and key columns."""
    assert len(tabular_df) == 1000
    assert not tabular_df["customer_id"].duplicated().any()

    # Column count check: 1 (customer_id) + 22*9 (numeric) + 4*4 (cat) + 1 (target) = 216
    n_numeric = len(columns_config["numeric_features"])
    n_cat = len(columns_config["categorical_features"])
    expected_cols = 1 + (n_numeric * 9) + (n_cat * 4) + 1
    assert tabular_df.shape[1] == expected_cols == 216

    assert "customer_id" in tabular_df.columns
    assert "target" in tabular_df.columns
    assert tabular_df["customer_id"].iloc[0] is not None
    assert set(tabular_df["target"].unique()).issubset({0, 1})
    assert tabular_df["target"].dtype == np.int8


def test_tabular_column_naming_convention(tabular_df, columns_config):
    """Verify strict adherence to {feature}__{agg} naming convention."""
    numeric_aggs = [
        "last",
        "first",
        "mean",
        "std",
        "min",
        "max",
        "delta_last_first",
        "slope",
        "delta_last_mean",
    ]
    categorical_aggs = ["last", "first", "nunique", "count"]

    for feat in columns_config["numeric_features"]:
        for agg in numeric_aggs:
            col_name = f"{feat}__{agg}"
            assert col_name in tabular_df.columns, f"Missing expected numeric aggregate column: {col_name}"
            assert tabular_df[col_name].dtype == np.float32, f"Column {col_name} should be float32, got {tabular_df[col_name].dtype}"

    for feat in columns_config["categorical_features"]:
        for agg in categorical_aggs:
            col_name = f"{feat}__{agg}"
            assert col_name in tabular_df.columns, f"Missing expected categorical aggregate column: {col_name}"
            assert tabular_df[col_name].dtype in [np.int16, np.int32, np.int64], f"Column {col_name} should be integer"


def test_tabular_no_missing_or_infinite_values(tabular_df):
    """Ensure zero NaN, null, or Inf values exist in tabular features."""
    nan_count = tabular_df.isna().sum().sum()
    assert nan_count == 0, f"Found {nan_count} unexpected NaN values in tabular dataset."

    numeric_cols = tabular_df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(tabular_df[numeric_cols].values).sum()
    assert inf_count == 0, f"Found {inf_count} unexpected Inf values in tabular dataset."


def test_tabular_memory_footprint(tabular_df):
    """Verify that the tabular dataset memory footprint remains strictly under 2.0 MB."""
    mem_mb = tabular_df.memory_usage(deep=True).sum() / (1024.0 * 1024.0)
    assert mem_mb < 2.0, f"Memory footprint exceeded expected threshold: {mem_mb:.2f} MB"
    assert mem_mb > 0.4, f"Memory footprint unusually low: {mem_mb:.2f} MB"


def test_synthetic_math_aggregations():
    """Verify mathematical exactness of all 9 numeric aggregations on toy data."""
    # Construct a toy dataframe with 2 customers
    # Customer A: 3 steps [0, 1, 2], values [10.0, 20.0, 30.0] -> slope = 10.0
    # Customer B: 4 steps [0, 1, 2, 3], values [100.0, 80.0, 60.0, 40.0] -> slope = -20.0
    toy_df = pd.DataFrame({
        "customer_ID": ["cust_A", "cust_A", "cust_A", "cust_B", "cust_B", "cust_B", "cust_B"],
        "step_idx": [0, 1, 2, 0, 1, 2, 3],
        "val_num": [10.0, 20.0, 30.0, 100.0, 80.0, 60.0, 40.0],
        "val_cat": [1, 2, 1, 3, 3, 3, 4],
    })

    result = generate_tabular_features(
        df=toy_df,
        customer_id_col="customer_ID",
        numeric_cols=["val_num"],
        categorical_cols=["val_cat"],
        step_col="step_idx",
        output_id_col="customer_id",
    )

    cust_a = result[result["customer_id"] == "cust_A"].iloc[0]
    assert np.isclose(cust_a["val_num__last"], 30.0)
    assert np.isclose(cust_a["val_num__first"], 10.0)
    assert np.isclose(cust_a["val_num__mean"], 20.0)
    assert np.isclose(cust_a["val_num__min"], 10.0)
    assert np.isclose(cust_a["val_num__max"], 30.0)
    assert np.isclose(cust_a["val_num__delta_last_first"], 20.0)
    assert np.isclose(cust_a["val_num__delta_last_mean"], 10.0)
    assert np.isclose(cust_a["val_num__slope"], 10.0)
    assert cust_a["val_cat__last"] == 1
    assert cust_a["val_cat__first"] == 1
    assert cust_a["val_cat__nunique"] == 2
    assert cust_a["val_cat__count"] == 3

    cust_b = result[result["customer_id"] == "cust_B"].iloc[0]
    assert np.isclose(cust_b["val_num__last"], 40.0)
    assert np.isclose(cust_b["val_num__first"], 100.0)
    assert np.isclose(cust_b["val_num__mean"], 70.0)
    assert np.isclose(cust_b["val_num__min"], 40.0)
    assert np.isclose(cust_b["val_num__max"], 100.0)
    assert np.isclose(cust_b["val_num__delta_last_first"], -60.0)
    assert np.isclose(cust_b["val_num__delta_last_mean"], -30.0)
    assert np.isclose(cust_b["val_num__slope"], -20.0)
    assert cust_b["val_cat__last"] == 4
    assert cust_b["val_cat__first"] == 3
    assert cust_b["val_cat__nunique"] == 2
    assert cust_b["val_cat__count"] == 4


def test_edge_cases_single_statement_and_constant():
    """Verify robust handling of single-statement and zero-variance customers."""
    toy_df = pd.DataFrame({
        "customer_ID": ["cust_single", "cust_const", "cust_const", "cust_const"],
        "step_idx": [0, 0, 1, 2],
        "val_num": [55.5, 7.0, 7.0, 7.0],
        "val_cat": [0, 1, 1, 1],
    })

    result = generate_tabular_features(
        df=toy_df,
        customer_id_col="customer_ID",
        numeric_cols=["val_num"],
        categorical_cols=["val_cat"],
        step_col="step_idx",
        output_id_col="customer_id",
    )

    # Customer with single statement (n=1)
    single = result[result["customer_id"] == "cust_single"].iloc[0]
    assert np.isclose(single["val_num__last"], 55.5)
    assert np.isclose(single["val_num__first"], 55.5)
    assert np.isclose(single["val_num__mean"], 55.5)
    assert np.isclose(single["val_num__std"], 0.0)  # Must be filled with 0.0, not NaN
    assert np.isclose(single["val_num__delta_last_first"], 0.0)
    assert np.isclose(single["val_num__delta_last_mean"], 0.0)
    assert np.isclose(single["val_num__slope"], 0.0)  # Must be 0.0 without divide by zero
    assert single["val_cat__count"] == 1
    assert single["val_cat__nunique"] == 1

    # Customer with constant value (zero variance)
    const = result[result["customer_id"] == "cust_const"].iloc[0]
    assert np.isclose(const["val_num__std"], 0.0)
    assert np.isclose(const["val_num__delta_last_first"], 0.0)
    assert np.isclose(const["val_num__delta_last_mean"], 0.0)
    assert np.isclose(const["val_num__slope"], 0.0)
    assert const["val_cat__nunique"] == 1
    assert const["val_cat__count"] == 3


def test_vectorized_slope_matches_numpy_polyfit():
    """Verify that vectorized slope formulation matches numpy.polyfit on varied sequence lengths."""
    np.random.seed(42)
    rows = []
    expected_slopes = {}

    for i in range(15):
        cid = f"cust_{i:02d}"
        seq_len = np.random.randint(2, 14)  # 2 to 13 statements
        x = np.arange(seq_len)
        true_slope = np.random.uniform(-10.0, 10.0)
        intercept = np.random.uniform(5.0, 50.0)
        noise = np.random.normal(0, 0.5, size=seq_len)
        y = intercept + (true_slope * x) + noise

        # Reference slope via numpy.polyfit (degree 1)
        ref_slope, _ = np.polyfit(x, y, 1)
        expected_slopes[cid] = float(ref_slope)

        for step, val in zip(x, y):
            rows.append({
                "customer_ID": cid,
                "step_idx": step,
                "feat": float(val),
                "cat": 1,
            })

    test_df = pd.DataFrame(rows)
    result = generate_tabular_features(
        df=test_df,
        customer_id_col="customer_ID",
        numeric_cols=["feat"],
        categorical_cols=["cat"],
        step_col="step_idx",
        output_id_col="customer_id",
    )

    for cid, exp_slope in expected_slopes.items():
        computed_slope = float(result.loc[result["customer_id"] == cid, "feat__slope"].iloc[0])
        assert np.isclose(computed_slope, exp_slope, atol=1e-4), (
            f"Slope mismatch for {cid}: computed {computed_slope}, expected {exp_slope}"
        )
