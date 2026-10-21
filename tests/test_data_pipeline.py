"""Tests for data loading, casting, and exploratory analysis in Phase 1."""

from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import yaml

from src.data.load import convert_raw_to_interim, identify_column_types
from src.data.eda import run_eda


def test_identify_column_types() -> None:
    """Verify automatic classification of string, categorical, and float columns."""
    sample_df = pd.DataFrame({
        "customer_ID": ["c1", "c2"],
        "S_2": ["2018-01-01", "2018-02-01"],
        "D_63": ["CR", "CO"],
        "B_30": [0.0, 1.0],
        "P_2": [0.8, 0.7],
        "B_1": [0.1, 0.2],
        "target": [0, 1],
    })
    cat_cols, num_cols = identify_column_types(sample_df)
    assert "D_63" in cat_cols
    assert "B_30" in cat_cols
    assert "P_2" in num_cols
    assert "B_1" in num_cols
    assert "customer_ID" not in cat_cols and "customer_ID" not in num_cols
    assert "target" not in cat_cols and "target" not in num_cols


def test_convert_raw_to_interim_types(tmp_path: Path) -> None:
    """Verify raw CSV conversion produces float32 numerics and category categoricals."""
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    raw_dir.mkdir()

    raw_statements = pd.DataFrame({
        "customer_ID": ["c1", "c1", "c2"],
        "S_2": ["2018-01-01", "2018-02-01", "2018-01-15"],
        "D_63": ["CR", "CR", "CO"],
        "P_2": [0.85, 0.75, 0.90],
    })
    raw_labels = pd.DataFrame({
        "customer_ID": ["c1", "c2"],
        "target": [1, 0],
    })
    raw_statements.to_csv(raw_dir / "train_data.csv", index=False)
    raw_labels.to_csv(raw_dir / "train_labels.csv", index=False)

    stmts_out, labels_out = convert_raw_to_interim(
        raw_dir=raw_dir,
        interim_dir=interim_dir,
    )
    assert stmts_out.exists()
    assert labels_out.exists()

    df_stmts = pd.read_parquet(stmts_out)
    assert df_stmts["P_2"].dtype == np.float32
    assert isinstance(df_stmts["D_63"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_datetime64_any_dtype(df_stmts["S_2"])

    df_labels = pd.read_parquet(labels_out)
    assert df_labels["target"].dtype == np.int8


def test_interim_data_invariants() -> None:
    """Verify the project interim data invariants match specifications."""
    interim_dir = Path("data/interim")
    statements_file = interim_dir / "statements.parquet"
    labels_file = interim_dir / "labels.parquet"

    assert statements_file.exists(), "Interim statements.parquet missing"
    assert labels_file.exists(), "Interim labels.parquet missing"

    df_stmts = pd.read_parquet(statements_file)
    df_labels = pd.read_parquet(labels_file)

    # Customers in statements must match labels
    stmt_custs = set(df_stmts["customer_ID"].unique())
    label_custs = set(df_labels["customer_ID"].unique())
    assert stmt_custs == label_custs

    # Max statements per customer <= 13
    stmt_counts = df_stmts.groupby("customer_ID").size()
    assert stmt_counts.max() <= 13
    assert stmt_counts.min() >= 1


def test_columns_yaml_validity() -> None:
    """Verify configs/columns.yaml is populated and matches interim dataset columns."""
    config_path = Path("configs/columns.yaml")
    assert config_path.exists()

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["id_column"] == "customer_ID"
    assert config["date_column"] == "S_2"
    assert config["target_column"] == "target"
    assert len(config["categorical_features"]) > 0
    assert len(config["numeric_features"]) > 0


def test_eda_artifacts_exist() -> None:
    """Verify all figures and report from Phase 1 exist."""
    fig_dir = Path("reports/figures")
    assert (fig_dir / "fig_class_balance.png").exists()
    assert (fig_dir / "fig_statements_per_customer.png").exists()
    assert (fig_dir / "fig_missingness_by_step.png").exists()
    assert (fig_dir / "fig_temporal_distribution.png").exists()
    assert Path("reports/01_eda.md").exists()
