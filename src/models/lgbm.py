"""LightGBM tabular baseline model training and inference.

Trains a gradient-boosted decision tree baseline on customer-level tabular features (data/processed/tabular.parquet).
Features:
- Stratified customer split training (train/valid/test from splits.parquet)
- Early stopping on validation split
- Systematic hyperparameter grid search over learning rates, leaves, and subsampling
- Complete evaluation metrics logging (competition metric, Gini, capture@4%, ROC-AUC, PR-AUC, Brier)
- Saves artifacts in results/lgbm/:
  - preds_valid.npy [150] float32
  - preds_test.npy [150] float32
  - metrics.json
  - config.json
  - model.txt
"""

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.eval.metrics import competition_metric, compute_all_metrics
from src.utils.io import load_yaml, save_json
from src.utils.logging import get_logger
from src.utils.seed import seed_everything

# Suppress lightgbm deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = get_logger("RiskTrail.Models.LightGBM")


def get_default_param_grid() -> List[Dict[str, Any]]:
    """Return default parameter grid for tabular baseline tuning (~12 trials)."""
    return [
        {"learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 20, "colsample_bytree": 0.8},
        {"learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 20, "colsample_bytree": 0.8},
        {"learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 10, "colsample_bytree": 0.7},
        {"learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 20, "colsample_bytree": 0.8},
        {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 10, "colsample_bytree": 0.8},
        {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20, "colsample_bytree": 0.8},
        {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 30, "colsample_bytree": 0.9},
        {"learning_rate": 0.08, "num_leaves": 15, "min_child_samples": 20, "colsample_bytree": 0.8},
        {"learning_rate": 0.08, "num_leaves": 31, "min_child_samples": 10, "colsample_bytree": 0.7},
        {"learning_rate": 0.08, "num_leaves": 31, "min_child_samples": 20, "colsample_bytree": 0.8},
        {"learning_rate": 0.10, "num_leaves": 15, "min_child_samples": 20, "colsample_bytree": 0.8},
        {"learning_rate": 0.10, "num_leaves": 31, "min_child_samples": 20, "colsample_bytree": 0.8},
    ]


def load_split_tabular_data(
    data_dir: Path,
) -> Tuple[
    pd.DataFrame, pd.Series, List[str],
    pd.DataFrame, pd.Series, List[str],
    pd.DataFrame, pd.Series, List[str],
    List[str],
]:
    """Load tabular and split data and partition by split, maintaining exact split row order.

    Args:
        data_dir: Directory containing tabular.parquet and splits.parquet.

    Returns:
        X_train, y_train, ids_train,
        X_valid, y_valid, ids_valid,
        X_test, y_test, ids_test,
        feature_cols
    """
    tabular_path = data_dir / "tabular.parquet"
    splits_path = data_dir / "splits.parquet"

    if not tabular_path.exists():
        raise FileNotFoundError(f"Missing tabular dataset at {tabular_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing splits dataset at {splits_path}")

    tabular_df = pd.read_parquet(tabular_path)
    splits_df = pd.read_parquet(splits_path)

    split_id_col = "customer_ID" if "customer_ID" in splits_df.columns else "customer_id"

    # Identify split customer ID ordering
    train_cids = splits_df[splits_df["split"] == "train"][split_id_col].tolist()
    valid_cids = splits_df[splits_df["split"] == "valid"][split_id_col].tolist()
    test_cids = splits_df[splits_df["split"] == "test"][split_id_col].tolist()

    feature_cols = [c for c in tabular_df.columns if c not in ["customer_id", "target"]]

    tab_indexed = tabular_df.set_index("customer_id")

    df_train = tab_indexed.loc[train_cids].reset_index()
    df_valid = tab_indexed.loc[valid_cids].reset_index()
    df_test = tab_indexed.loc[test_cids].reset_index()

    X_train, y_train = df_train[feature_cols], df_train["target"]
    X_valid, y_valid = df_valid[feature_cols], df_valid["target"]
    X_test, y_test = df_test[feature_cols], df_test["target"]

    return (
        X_train, y_train, train_cids,
        X_valid, y_valid, valid_cids,
        X_test, y_test, test_cids,
        feature_cols,
    )


def train_lgbm(
    config: Optional[Dict[str, Any]] = None,
    data_dir: Union[str, Path] = PROJECT_ROOT / "data" / "processed",
    output_dir: Union[str, Path] = PROJECT_ROOT / "results" / "lgbm",
    seed: int = 42,
    param_grid: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Train LightGBM tabular baseline model with early stopping and hyperparameter search.

    Args:
        config: Model hyperparameter configuration dictionary or None (loads from configs/lgbm.yaml).
        data_dir: Directory containing tabular.parquet and splits.parquet.
        output_dir: Directory to save model, predictions, and metrics.
        seed: Random seed.
        param_grid: Optional list of parameter dicts to search over.

    Returns:
        Dict[str, Any]: Validation and test metric scores and run metadata.
    """
    seed_everything(seed)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base configuration
    if config is None:
        config_path = PROJECT_ROOT / "configs" / "lgbm.yaml"
        base_config = load_yaml(config_path) if config_path.exists() else {}
    else:
        base_config = config.copy()

    start_time = time.time()
    logger.info("Loading tabular datasets and partitions...")
    (
        X_train, y_train, ids_train,
        X_valid, y_valid, ids_valid,
        X_test, y_test, ids_test,
        feature_cols,
    ) = load_split_tabular_data(data_dir)

    logger.info(
        f"Data partitioned: {len(X_train)} train, {len(X_valid)} valid, {len(X_test)} test "
        f"across {len(feature_cols)} features."
    )

    grid = param_grid or get_default_param_grid()
    early_stopping_rounds = base_config.get("early_stopping_rounds", 50)
    n_estimators = base_config.get("n_estimators", 1500)
    subsample = base_config.get("subsample", 0.8)

    best_score = -float("inf")
    best_params: Dict[str, Any] = {}
    best_model: Optional[lgb.LGBMClassifier] = None
    search_history: List[Dict[str, Any]] = []

    eval_metric = base_config.get("metric", "binary_logloss")
    logger.info(f"Starting hyperparameter grid search ({len(grid)} trials, eval_metric={eval_metric})...")

    for idx, trial_params in enumerate(grid, start=1):
        # Merge base configuration with trial hyperparameters
        model_params = {
            "objective": "binary",
            "metric": eval_metric,
            "boosting_type": base_config.get("boosting_type", "gbdt"),
            "n_estimators": n_estimators,
            "subsample": subsample,
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
            **trial_params,
        }

        clf = lgb.LGBMClassifier(**model_params)
        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=eval_metric,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        p_valid = clf.predict_proba(X_valid)[:, 1].astype(np.float32)
        val_comp_metric = competition_metric(y_valid.values, p_valid)
        best_iter = clf.best_iteration_

        search_history.append({
            "trial": idx,
            "params": trial_params,
            "best_iteration": int(best_iter),
            "valid_competition_metric": round(val_comp_metric, 5),
        })

        logger.info(
            f"Trial {idx:02d}/{len(grid)}: comp_metric={val_comp_metric:.5f}, "
            f"best_iter={best_iter}, params={trial_params}"
        )

        if val_comp_metric > best_score:
            best_score = val_comp_metric
            best_params = model_params
            best_model = clf

    assert best_model is not None, "Model fitting failed across all trials!"
    logger.info(f"Grid search complete. Best validation competition metric: {best_score:.5f}")

    # Generate aligned predictions for valid and test splits
    preds_valid = best_model.predict_proba(X_valid)[:, 1].astype(np.float32)
    preds_test = best_model.predict_proba(X_test)[:, 1].astype(np.float32)

    # Compute comprehensive evaluation metrics
    metrics_valid = compute_all_metrics(y_valid.values, preds_valid)
    metrics_test = compute_all_metrics(y_test.values, preds_test)

    total_time = round(time.time() - start_time, 2)

    # Persist artifacts
    preds_valid_path = output_dir / "preds_valid.npy"
    preds_test_path = output_dir / "preds_test.npy"
    metrics_path = output_dir / "metrics.json"
    config_path = output_dir / "config.json"
    model_path = output_dir / "model.txt"

    np.save(preds_valid_path, preds_valid)
    np.save(preds_test_path, preds_test)

    # Save model text booster
    best_model.booster_.save_model(str(model_path))

    # Save metrics JSON
    metrics_payload = {
        "valid": metrics_valid,
        "test": metrics_test,
    }
    save_json(metrics_payload, metrics_path)

    # Save config JSON
    config_payload = {
        "model_name": "lgbm",
        "best_params": {
            k: v for k, v in best_params.items() if isinstance(v, (int, float, str, bool))
        },
        "best_iteration": int(best_model.best_iteration_),
        "seed": seed,
        "feature_count": len(feature_cols),
        "n_train_samples": len(X_train),
        "n_valid_samples": len(X_valid),
        "n_test_samples": len(X_test),
        "training_time_seconds": total_time,
        "search_history": search_history,
    }
    save_json(config_payload, config_path)

    logger.info(f"Saved all results to {output_dir}")

    # Print formatted metric table
    _print_metric_table(metrics_valid, metrics_test, best_model.best_iteration_, total_time)

    return {
        "valid": metrics_valid,
        "test": metrics_test,
        "best_iteration": int(best_model.best_iteration_),
        "output_dir": str(output_dir),
    }


def _print_metric_table(
    val_m: Dict[str, float],
    test_m: Dict[str, float],
    best_iter: int,
    duration: float,
) -> None:
    """Print clean metric report table."""
    print("\n" + "=" * 80)
    print("RiskTrail Phase 3: LightGBM Tabular Baseline Results")
    print("=" * 80)
    print(f"Optimal Trees (early stopped): {best_iter} | Training duration: {duration}s\n")
    print(f"{'Evaluation Metric':<32} | {'Validation (N=150)':<20} | {'Test (N=150)':<20}")
    print("-" * 80)
    print(f"{'Competition Metric (unweighted)':<32} | {val_m['competition_metric']:<20.5f} | {test_m['competition_metric']:<20.5f}")
    print(f"{'Normalized Gini':<32} | {val_m['normalized_gini']:<20.5f} | {test_m['normalized_gini']:<20.5f}")
    print(f"{'Capture Rate @ 4%':<32} | {val_m['capture_rate_at_4pct']:<20.5f} | {test_m['capture_rate_at_4pct']:<20.5f}")
    print(f"{'ROC-AUC':<32} | {val_m['roc_auc']:<20.5f} | {test_m['roc_auc']:<20.5f}")
    print(f"{'PR-AUC':<32} | {val_m['pr_auc']:<20.5f} | {test_m['pr_auc']:<20.5f}")
    print(f"{'Brier Score (lower is better)':<32} | {val_m['brier']:<20.5f} | {test_m['brier']:<20.5f}")
    print("-" * 80)
    print(f"{'Official Weighted Comp Metric':<32} | {val_m['competition_metric_weighted']:<20.5f} | {test_m['competition_metric_weighted']:<20.5f}")
    print(f"{'Official Weighted Gini':<32} | {val_m['normalized_gini_weighted']:<20.5f} | {test_m['normalized_gini_weighted']:<20.5f}")
    print(f"{'Official Weighted Capture @ 4%':<32} | {val_m['capture_rate_at_4pct_weighted']:<20.5f} | {test_m['capture_rate_at_4pct_weighted']:<20.5f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    train_lgbm()
