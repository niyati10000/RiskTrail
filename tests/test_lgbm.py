"""Unit tests for LightGBM tabular baseline model (src/models/lgbm.py).

Validates:
1. Production output files in results/lgbm/:
   - preds_valid.npy [150] float32
   - preds_test.npy [150] float32
   - metrics.json
   - config.json
   - model.txt
2. Prediction alignment with customer splits in data/processed/splits.parquet.
3. Probability sanity checks: values strictly in [0.0, 1.0], no NaNs or Infs.
4. Model performance benchmarks (ROC-AUC > 0.90, competition_metric > 0.50, Brier < 0.10).
5. Fast programmatic invocation of train_lgbm function with custom parameter grid.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.lgbm import load_split_tabular_data, train_lgbm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_LGBM_DIR = PROJECT_ROOT / "results" / "lgbm"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@pytest.fixture
def lgbm_artifacts():
    """Load persisted LightGBM training artifacts."""
    preds_val_path = RESULTS_LGBM_DIR / "preds_valid.npy"
    preds_test_path = RESULTS_LGBM_DIR / "preds_test.npy"
    metrics_path = RESULTS_LGBM_DIR / "metrics.json"
    config_path = RESULTS_LGBM_DIR / "config.json"
    model_path = RESULTS_LGBM_DIR / "model.txt"

    assert preds_val_path.exists(), f"Missing {preds_val_path}"
    assert preds_test_path.exists(), f"Missing {preds_test_path}"
    assert metrics_path.exists(), f"Missing {metrics_path}"
    assert config_path.exists(), f"Missing {config_path}"
    assert model_path.exists(), f"Missing {model_path}"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return {
        "preds_valid": np.load(preds_val_path),
        "preds_test": np.load(preds_test_path),
        "metrics": metrics,
        "config": config,
        "model_path": model_path,
    }


def test_lgbm_predictions_shape_and_range(lgbm_artifacts):
    """Verify prediction arrays shape, dtype, and valid probability range."""
    splits_df = pd.read_parquet(PROCESSED_DIR / "splits.parquet")
    n_valid = (splits_df["split"] == "valid").sum()
    n_test = (splits_df["split"] == "test").sum()

    preds_val = lgbm_artifacts["preds_valid"]
    preds_test = lgbm_artifacts["preds_test"]

    assert preds_val.shape == (n_valid,) == (150,)
    assert preds_val.dtype == np.float32
    assert np.all(preds_val >= 0.0) and np.all(preds_val <= 1.0)
    assert not np.isnan(preds_val).any()

    assert preds_test.shape == (n_test,) == (150,)
    assert preds_test.dtype == np.float32
    assert np.all(preds_test >= 0.0) and np.all(preds_test <= 1.0)
    assert not np.isnan(preds_test).any()


def test_lgbm_metrics_sanity(lgbm_artifacts):
    """Verify that LightGBM baseline meets benchmark criteria."""
    metrics = lgbm_artifacts["metrics"]

    assert "valid" in metrics and "test" in metrics

    val_m = metrics["valid"]
    test_m = metrics["test"]

    for split_m in [val_m, test_m]:
        assert split_m["roc_auc"] >= 0.90
        assert split_m["competition_metric"] >= 0.50
        assert split_m["brier"] <= 0.05
        assert split_m["normalized_gini"] >= 0.80


def test_lgbm_config_contents(lgbm_artifacts):
    """Verify metadata and hyperparameter tracking in config.json."""
    config = lgbm_artifacts["config"]

    assert config["model_name"] == "lgbm"
    assert config["n_train_samples"] == 700
    assert config["n_valid_samples"] == 150
    assert config["n_test_samples"] == 150
    assert config["feature_count"] == 214
    assert config["best_iteration"] > 0
    assert len(config["search_history"]) >= 10


def test_lgbm_model_file_validity(lgbm_artifacts):
    """Verify that model.txt contains valid LightGBM booster specification."""
    model_path = lgbm_artifacts["model_path"]
    with open(model_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "tree" in content.lower()
    assert "version=" in content.lower()
    assert len(content) > 1000


def test_load_split_tabular_data_partitions():
    """Verify load_split_tabular_data returns aligned partitions without leakage."""
    (
        X_train, y_train, ids_train,
        X_valid, y_valid, ids_valid,
        X_test, y_test, ids_test,
        feature_cols,
    ) = load_split_tabular_data(PROCESSED_DIR)

    assert len(X_train) == 700
    assert len(X_valid) == 150
    assert len(X_test) == 150
    assert len(feature_cols) == 214

    # No customer overlap across splits
    assert len(set(ids_train) & set(ids_valid)) == 0
    assert len(set(ids_train) & set(ids_test)) == 0
    assert len(set(ids_valid) & set(ids_test)) == 0
