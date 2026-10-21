"""Synthetic sample data generator for RiskTrail demo and Hugging Face Spaces deployment.

Generates a realistic, small synthetic dataset (50 accounts) that:
1. Contains NO real customer identifiers or raw unmasked customer records.
2. Emulates empirical credit risk distributions (low risk, borderline, high risk/delinquency).
3. Matches the exact 214 tabular feature schema required by the trained LightGBM model.
4. Precomputes default probabilities and SHAP TreeExplainer attribution values.
5. Emulates 13-month statement histories for visual temporal explainability.
6. Saves self-contained artifacts to app/ for lightweight, zero-token Hugging Face deployment.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from src.utils.logging import get_logger

logger = get_logger("RiskTrail.Data.CreateSample")


def generate_synthetic_sample(
    n_samples: int = 50,
    seed: int = 42,
    output_dir: Path = PROJECT_ROOT / "app",
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, List[str]]:
    """Generate 50 synthetic applicant profiles with tabular features and 13-month histories.

    Args:
        n_samples: Number of synthetic accounts to create (default 50).
        seed: Random seed for reproducibility.
        output_dir: Directory where synthetic artifacts will be saved.

    Returns:
        tab_df: Tabular feature DataFrame [n_samples, 214 + metadata].
        stmts_df: Statement history DataFrame [n_samples * 13, features].
        shap_values: Precomputed SHAP attribution matrix [n_samples, 214].
        feature_names: List of 214 feature column names.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)

    # Load feature column names from trained LightGBM model
    model_path = PROJECT_ROOT / "results" / "lgbm" / "model.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained LightGBM model not found at {model_path}")

    with open(PROJECT_ROOT / "results" / "lgbm" / "shap_feature_names.json", "r", encoding="utf-8") as f:
        feature_names: List[str] = json.load(f)

    # 1. Generate synthetic customer IDs and ground-truth risk profiles
    # 32 Good accounts (64%), 8 Borderline accounts (16%), 10 Defaulters (20%)
    customer_ids = [f"SYNTH_{i+1:04d}" for i in range(n_samples)]
    targets = np.zeros(n_samples, dtype=np.int8)
    targets[40:50] = 1   # Last 10 are confirmed defaulters
    targets[32:40] = rng.choice([0, 1], size=8, p=[0.5, 0.5])  # Borderline mix

    # 2. Synthesize base risk levels: 0.0 (prime) to 1.0 (subprime/default)
    risk_latents = np.zeros(n_samples, dtype=np.float32)
    risk_latents[:32] = rng.uniform(0.01, 0.15, size=32)
    risk_latents[32:40] = rng.uniform(0.20, 0.45, size=8)
    risk_latents[40:] = rng.uniform(0.70, 0.98, size=10)

    # Make applicant SYNTH_0045 the designated worked case study applicant
    risk_latents[44] = 0.92
    targets[44] = 1

    # 3. Populate 214 tabular features using synthetic generators
    X_dict: Dict[str, np.ndarray] = {}

    for feat in feature_names:
        parts = feat.split("__")
        base = parts[0]
        agg = parts[1] if len(parts) > 1 else ""

        # Delinquency features (D_39, D_41, etc.)
        if base.startswith("D_"):
            if base == "D_39":
                scale = 16.0
                base_val = risk_latents * scale + rng.normal(0, 0.5, size=n_samples)
                base_val = np.clip(base_val, 0.0, 30.0)
            elif base in ["D_63", "D_64"]:  # Categorical
                cat_choice = np.where(risk_latents > 0.5, rng.choice([2, 3], size=n_samples), rng.choice([0, 1], size=n_samples))
                X_dict[feat] = cat_choice.astype(np.float32)
                continue
            else:
                base_val = risk_latents * 4.0 + rng.exponential(0.2, size=n_samples)

        # Balance features (B_1, B_2, etc.)
        elif base.startswith("B_"):
            if base in ["B_30", "B_38"]:  # Categorical
                cat_choice = np.where(risk_latents > 0.5, rng.choice([1, 2], size=n_samples), 0)
                X_dict[feat] = cat_choice.astype(np.float32)
                continue
            else:
                base_val = risk_latents * 0.9 + rng.uniform(0.02, 0.15, size=n_samples)

        # Payment features (P_2, P_3, etc.)
        elif base.startswith("P_"):
            # High payment ratio is good (inversely proportional to risk)
            base_val = (1.0 - risk_latents) * 0.85 + rng.normal(0.1, 0.05, size=n_samples)
            base_val = np.clip(base_val, 0.0, 1.5)

        # Spend and Risk features (S_, R_)
        elif base.startswith("S_"):
            base_val = rng.uniform(0.01, 0.6, size=n_samples) * (1.0 + risk_latents * 0.5)
        elif base.startswith("R_"):
            base_val = risk_latents * 0.8 + rng.normal(0.05, 0.02, size=n_samples)
        else:
            base_val = rng.normal(0.5, 0.2, size=n_samples)

        # Apply aggregation modifiers
        if agg == "last":
            vals = base_val + (risk_latents - 0.2) * 1.5
        elif agg == "first":
            vals = base_val * 0.8
        elif agg == "mean":
            vals = base_val
        elif agg == "std":
            vals = np.abs(base_val * 0.25 + rng.uniform(0.01, 0.05, size=n_samples))
        elif agg == "min":
            vals = np.maximum(0.0, base_val * 0.5)
        elif agg == "max":
            vals = base_val * 1.4 + 0.1
        elif agg == "delta_last_first":
            vals = (risk_latents - 0.2) * 2.0 + rng.normal(0, 0.1, size=n_samples)
        elif agg == "slope":
            # Defaulters have strong positive trajectory / slope
            vals = (risk_latents - 0.25) * 0.9 + rng.normal(0, 0.05, size=n_samples)
        elif agg == "delta_last_mean":
            vals = (risk_latents - 0.3) * 0.8
        elif agg in ["nunique", "count"]:
            vals = np.full(n_samples, 13.0 if agg == "count" else 2.0)
        else:
            vals = base_val

        X_dict[feat] = np.nan_to_num(vals.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)

    # Build Tabular DataFrame
    tab_df = pd.DataFrame(X_dict, index=customer_ids)
    tab_df.index.name = "customer_id"

    # 4. Score using LightGBM model
    booster = lgb.Booster(model_file=str(model_path))
    preds = booster.predict(tab_df[feature_names])
    tab_df["target"] = targets
    tab_df["predicted_prob"] = preds.astype(np.float32)

    # 5. Precompute SHAP values
    logger.info("Computing TreeExplainer SHAP values on synthetic sample...")
    explainer = shap.TreeExplainer(booster)
    raw_shap = explainer.shap_values(tab_df[feature_names])
    if isinstance(raw_shap, list):
        shap_values = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
    else:
        shap_values = raw_shap
    shap_values = np.asarray(shap_values, dtype=np.float32)

    # 6. Generate 13-month statement timelines
    stmt_records: List[Dict[str, Any]] = []
    for i, cid in enumerate(customer_ids):
        r_lat = risk_latents[i]
        # Most customers have 13 statements, a few have 9-11
        valid_stmts = 13 if i % 6 != 0 else rng.randint(9, 13)
        pad_count = 13 - valid_stmts

        for step in range(13):
            month_label = f"T-{12 - step}" if (12 - step) > 0 else "T (Latest)"
            is_pad = step < pad_count

            if is_pad:
                d39 = 0.0
                b1 = 0.0
                p2 = 0.0
            else:
                progress = (step - pad_count) / max(1, valid_stmts - 1)
                d39 = float(max(0.0, (r_lat * 15.0) * progress + rng.normal(0, 0.4)))
                b1 = float(min(1.0, max(0.0, (r_lat * 0.9) * progress + 0.05 + rng.normal(0, 0.02))))
                p2 = float(max(0.0, (1.0 - r_lat) * (1.2 - 0.4 * progress) + rng.normal(0, 0.05)))

            stmt_records.append({
                "customer_id": cid,
                "step": step,
                "month_label": month_label,
                "is_padded": is_pad,
                "D_39": round(d39, 4),
                "B_1": round(b1, 4),
                "P_2": round(p2, 4),
            })

    stmts_df = pd.DataFrame(stmt_records)

    # 7. Persist self-contained artifacts
    tab_path = output_dir / "sample_data.parquet"
    stmt_path = output_dir / "sample_statements.parquet"
    shap_path = output_dir / "sample_shap.npy"
    feat_path = output_dir / "shap_features.json"
    model_dest = output_dir / "model.txt"

    tab_df.reset_index().to_parquet(tab_path, index=False)
    stmts_df.to_parquet(stmt_path, index=False)
    np.save(shap_path, shap_values)
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    # Copy model.txt to app/ for standalone deployment
    if not model_dest.exists():
        shutil.copy2(model_path, model_dest)

    logger.info(f"Synthetic sample successfully generated at {output_dir}:")
    logger.info(f"  - {tab_path.name}: {len(tab_df)} applicants with {len(feature_names)} features")
    logger.info(f"  - {stmt_path.name}: {len(stmts_df)} monthly statement records")
    logger.info(f"  - {shap_path.name}: {shap_values.shape} SHAP matrix")
    logger.info(f"  - {model_dest.name}: LightGBM booster copied")

    return tab_df, stmts_df, shap_values, feature_names


if __name__ == "__main__":
    generate_synthetic_sample()
