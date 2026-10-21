"""Tests for customer-level stratified splitting and leakage-free data cleaning."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.split import create_customer_splits
from src.data.clean import clean_statements


@pytest.fixture
def sample_data():
    """Create a controlled toy dataset of customers and statements."""
    np.random.seed(42)
    n_custs = 200
    cids = [f"cust_{i:03d}" for i in range(n_custs)]
    targets = np.random.choice([0, 1], size=n_custs, p=[0.75, 0.25])
    labels_df = pd.DataFrame({"customer_ID": cids, "target": targets})

    records = []
    dates = pd.date_range("2017-03-01", periods=13, freq="30D")
    for cid in cids:
        n_stmts = int(np.random.randint(3, 14))
        for step, dt in enumerate(dates[:n_stmts]):
            records.append({
                "customer_ID": cid,
                "S_2": dt,
                "feature_num1": float(np.random.normal(10, 2)) if np.random.rand() > 0.2 else np.nan,
                "feature_num2": float(np.random.normal(50, 5)),
                "feature_cat1": np.random.choice(["A", "B", "C"]) if np.random.rand() > 0.1 else np.nan,
            })
    statements_df = pd.DataFrame(records)
    return labels_df, statements_df


def test_no_customer_leakage_across_splits(sample_data) -> None:
    """Verify that no customer appears in more than one split."""
    labels_df, statements_df = sample_data
    splits_df = create_customer_splits(
        labels_df=labels_df,
        statements_df=statements_df,
        customer_id_col="customer_ID",
        target_col="target",
        train_size=0.70,
        valid_size=0.15,
        test_size=0.15,
        seed=42,
    )

    train_cids = set(splits_df[splits_df["split"] == "train"]["customer_ID"])
    valid_cids = set(splits_df[splits_df["split"] == "valid"]["customer_ID"])
    test_cids = set(splits_df[splits_df["split"] == "test"]["customer_ID"])

    # Mutually exclusive: pairwise intersections must be empty
    assert len(train_cids.intersection(valid_cids)) == 0, "Leakage detected between train and valid!"
    assert len(train_cids.intersection(test_cids)) == 0, "Leakage detected between train and test!"
    assert len(valid_cids.intersection(test_cids)) == 0, "Leakage detected between valid and test!"

    # Exhaustive: union equals all customers
    all_split_cids = train_cids.union(valid_cids).union(test_cids)
    all_label_cids = set(labels_df["customer_ID"])
    assert all_split_cids == all_label_cids, "Missing customers in split assignments!"
    assert len(splits_df) == len(labels_df)


def test_stratification_within_tolerance(sample_data) -> None:
    """Verify that target default rates in each split match the overall rate within tolerance."""
    labels_df, statements_df = sample_data
    overall_default_rate = labels_df["target"].mean()

    splits_df = create_customer_splits(
        labels_df=labels_df,
        statements_df=statements_df,
        customer_id_col="customer_ID",
        target_col="target",
        train_size=0.70,
        valid_size=0.15,
        test_size=0.15,
        seed=42,
    )
    merged = splits_df.merge(labels_df, on="customer_ID")

    for split_name in ["train", "valid", "test"]:
        split_rate = merged[merged["split"] == split_name]["target"].mean()
        # Stratification should be preserved within 2.5% tolerance
        assert abs(split_rate - overall_default_rate) < 0.025, (
            f"Split '{split_name}' default rate {split_rate:.3f} deviates from overall {overall_default_rate:.3f}"
        )


def test_split_proportions(sample_data) -> None:
    """Verify split ratios match expected proportions (70/15/15) within 1.5% margin."""
    labels_df, statements_df = sample_data
    n_total = len(labels_df)

    splits_df = create_customer_splits(
        labels_df=labels_df,
        statements_df=statements_df,
        customer_id_col="customer_ID",
        target_col="target",
        train_size=0.70,
        valid_size=0.15,
        test_size=0.15,
        seed=42,
    )

    train_pct = (splits_df["split"] == "train").sum() / n_total
    valid_pct = (splits_df["split"] == "valid").sum() / n_total
    test_pct = (splits_df["split"] == "test").sum() / n_total

    assert abs(train_pct - 0.70) <= 0.015
    assert abs(valid_pct - 0.15) <= 0.015
    assert abs(test_pct - 0.15) <= 0.015


def test_split_determinism(sample_data) -> None:
    """Verify that running split with the same seed produces identical customer sets."""
    labels_df, statements_df = sample_data
    split1 = create_customer_splits(labels_df, seed=99)
    split2 = create_customer_splits(labels_df, seed=99)
    pd.testing.assert_frame_equal(split1, split2)


def test_time_holdout_flag_integrity(sample_data) -> None:
    """Verify that time_holdout flag is boolean and marks the latest customers."""
    labels_df, statements_df = sample_data
    splits_df = create_customer_splits(
        labels_df=labels_df,
        statements_df=statements_df,
        customer_id_col="customer_ID",
        target_col="target",
        seed=42,
    )
    assert "time_holdout" in splits_df.columns
    assert splits_df["time_holdout"].dtype == bool
    assert splits_df["time_holdout"].sum() > 0
    assert splits_df["time_holdout"].sum() < len(splits_df)


def test_clean_fits_on_train_only_no_leakage() -> None:
    """Verify that cleaning statistics (medians) are fitted strictly on train customers."""
    splits_data = pd.DataFrame({
        "customer_ID": ["train_1", "train_2", "test_1"],
        "split": ["train", "train", "test"],
    })

    # Train customers have feature values 10 and 20 (median = 15).
    # Test customer has an extreme outlier 1,000,000 and a missing value.
    statements = pd.DataFrame({
        "customer_ID": ["train_1", "train_2", "test_1", "test_1"],
        "S_2": pd.to_datetime(["2018-01-01", "2018-01-01", "2018-01-01", "2018-02-01"]),
        "num_col": [10.0, 20.0, 1000000.0, np.nan],
        "cat_col": ["A", "B", "A", "UNSEEN_NEW_CAT"],
    })

    clean_df, artifacts = clean_statements(
        statements_df=statements,
        splits_df=splits_data,
        categorical_cols=["cat_col"],
        numeric_cols=["num_col"],
        customer_id_col="customer_ID",
        date_col="S_2",
    )

    # Imputed median for num_col must be strictly 15.0 (train median), NOT contaminated by 1,000,000
    assert artifacts["numeric_train_medians"]["num_col"] == 15.0
    # The missing value in test_1's second row must be filled with the train median (15.0)
    assert clean_df.loc[clean_df["customer_ID"] == "test_1", "num_col"].iloc[1] == 15.0

    # Unseen category 'UNSEEN_NEW_CAT' not in train must be mapped to -1
    unseen_code = clean_df.loc[clean_df["customer_ID"] == "test_1", "cat_col"].iloc[1]
    assert unseen_code == -1

    # Missing indicator must flag the missing entry correctly
    assert "num_col__missing" in clean_df.columns
    test_1_missing = clean_df.loc[clean_df["customer_ID"] == "test_1", "num_col__missing"].tolist()
    assert test_1_missing == [0, 1]
    train_missing = clean_df.loc[clean_df["customer_ID"].isin(["train_1", "train_2"]), "num_col__missing"].tolist()
    assert train_missing == [0, 0]


def test_processed_splits_parquet_production() -> None:
    """Verify production data/processed/splits.parquet adheres to all project rules."""
    splits_path = Path("data/processed/splits.parquet")
    assert splits_path.exists(), "data/processed/splits.parquet missing!"

    splits_df = pd.read_parquet(splits_path)
    assert "customer_ID" in splits_df.columns
    assert "split" in splits_df.columns
    assert set(splits_df["split"].unique()) == {"train", "valid", "test"}

    # No duplicates
    assert not splits_df["customer_ID"].duplicated().any()

    # Pairwise disjoint
    train_set = set(splits_df[splits_df["split"] == "train"]["customer_ID"])
    valid_set = set(splits_df[splits_df["split"] == "valid"]["customer_ID"])
    test_set = set(splits_df[splits_df["split"] == "test"]["customer_ID"])
    assert len(train_set.intersection(valid_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(valid_set.intersection(test_set)) == 0
