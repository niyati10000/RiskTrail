"""Generate realistic benchmark credit statement raw dataset for RiskTrail.

Schema modeled after the American Express sequence credit default benchmark:
- 1,000 unique customers
- 1 to 13 consecutive monthly statements per customer (statement date S_2)
- Feature groups: Delinquency (D_*), Spend (S_*), Payment (P_*), Balance (B_*), Risk (R_*)
- Categorical features: D_63, D_64, B_30, B_38
- Target label (train_labels.csv): binary default (1) vs non-default (0) with ~25% class balance
- Realistic missingness and deteriorating trends for defaulters
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd


def generate_benchmark_data(
    output_dir: Path,
    n_customers: int = 1000,
    seed: int = 42,
) -> None:
    """Generate train_data.csv and train_labels.csv in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    # 1. Customer IDs and Target Labels
    # Realistic 64-char hex IDs
    customer_ids = [
        f"{np.random.bytes(16).hex()}" for _ in range(n_customers)
    ]
    # Default rate ~25%
    default_prob = 0.258
    targets = (np.random.rand(n_customers) < default_prob).astype(np.int8)

    labels_df = pd.DataFrame({
        "customer_ID": customer_ids,
        "target": targets,
    })
    labels_path = output_dir / "train_labels.csv"
    labels_df.to_csv(labels_path, index=False)
    print(f"Generated {len(labels_df)} labels -> {labels_path}")

    # 2. Statement-level data
    # Date schedule: 13 months from 2017-03-15 to 2018-03-15
    base_dates = pd.date_range(start="2017-03-15", periods=13, freq="30D")

    records = []
    cat_d63_vals = ["CR", "CO", "CL", "XZ", "XM"]
    cat_d64_vals = ["O", "R", "U", "-1"]
    cat_b30_vals = ["0.0", "1.0", "2.0"]
    cat_b38_vals = ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0"]

    for cid, is_default in zip(customer_ids, targets):
        # Statement count distribution: 80% have full 13 statements, 20% have between 2 and 12
        if np.random.rand() < 0.80:
            n_stmts = 13
        else:
            n_stmts = int(np.random.randint(2, 13))

        # Statement dates for this customer (oldest to newest)
        cust_dates = base_dates[-n_stmts:]

        # Latent risk trajectory
        if is_default == 1:
            # Deteriorating trajectory
            base_risk = np.random.uniform(0.4, 0.8)
            drift = np.linspace(0.0, 0.4, n_stmts)
        else:
            # Stable / low-risk trajectory
            base_risk = np.random.uniform(0.05, 0.3)
            drift = np.linspace(0.0, -0.05, n_stmts)

        trajectory = np.clip(base_risk + drift + np.random.normal(0, 0.05, n_stmts), 0.0, 1.0)

        for step, (dt, risk) in enumerate(zip(cust_dates, trajectory)):
            # Random jitter on date
            actual_date = dt + pd.Timedelta(days=int(np.random.randint(-3, 4)))

            row = {
                "customer_ID": cid,
                "S_2": actual_date.strftime("%Y-%m-%d"),
                # Delinquency features
                "D_39": float(np.clip(risk * 15 + np.random.normal(0, 1), 0, 30)),
                "D_41": float(np.clip(risk * 0.8 + np.random.normal(0, 0.1), 0, 1)),
                "D_42": float(np.clip(risk * 0.5 + np.random.normal(0, 0.1), 0, 1)) if np.random.rand() > 0.6 else np.nan,
                "D_43": float(np.clip(risk * 0.4 + np.random.normal(0, 0.1), 0, 1)) if np.random.rand() > 0.4 else np.nan,
                "D_44": float(np.clip(risk * 1.5 + np.random.normal(0, 0.2), 0, 3)),
                # Payment features
                "P_2": float(np.clip(1.0 - risk * 0.8 + np.random.normal(0, 0.05), 0, 1.5)),
                "P_3": float(np.clip(0.8 - risk * 0.6 + np.random.normal(0, 0.05), 0, 1.2)) if np.random.rand() > 0.15 else np.nan,
                "P_4": float(np.clip(risk * 0.2 + np.random.normal(0, 0.02), 0, 1)),
                # Balance features
                "B_1": float(np.clip(risk * 0.9 + np.random.normal(0, 0.05), 0, 1.5)),
                "B_2": float(np.clip(1.0 - risk * 0.9 + np.random.normal(0, 0.05), 0, 1.2)),
                "B_3": float(np.clip(risk * 0.8 + np.random.normal(0, 0.05), 0, 1.2)),
                "B_4": float(np.clip(risk * 10 + np.random.normal(0, 1), 0, 25)),
                "B_5": float(np.clip(np.random.exponential(0.05), 0, 1)),
                "B_9": float(np.clip(risk * 1.2 + np.random.normal(0, 0.1), 0, 2.0)),
                # Spend features
                "S_3": float(np.clip(risk * 0.7 + np.random.normal(0, 0.1), 0, 1.5)) if np.random.rand() > 0.2 else np.nan,
                "S_5": float(np.clip(np.random.exponential(0.03), 0, 0.5)),
                "S_6": float(np.random.choice([0.0, 1.0], p=[0.8, 0.2])),
                "S_7": float(np.clip(risk * 0.6 + np.random.normal(0, 0.1), 0, 1.2)) if np.random.rand() > 0.3 else np.nan,
                "S_8": float(np.clip(1500 - risk * 800 + np.random.normal(0, 100), 0, 3000)),
                # Risk features
                "R_1": float(np.clip(risk * 0.9 + np.random.normal(0, 0.05), 0, 1.5)),
                "R_2": float(np.random.choice([0.0, 1.0], p=[0.85, 0.15] if is_default == 0 else [0.4, 0.6])),
                "R_3": float(np.clip(risk * 5 + np.random.normal(0, 0.5), 0, 10)),
                # Categorical features
                "D_63": np.random.choice(cat_d63_vals, p=[0.70, 0.15, 0.08, 0.05, 0.02]),
                "D_64": np.random.choice(cat_d64_vals, p=[0.55, 0.25, 0.15, 0.05]),
                "B_30": np.random.choice(cat_b30_vals, p=[0.85, 0.10, 0.05] if is_default == 0 else [0.40, 0.35, 0.25]),
                "B_38": np.random.choice(cat_b38_vals, p=[0.35, 0.25, 0.15, 0.10, 0.08, 0.04, 0.03]),
            }
            records.append(row)

    statements_df = pd.DataFrame(records)
    statements_path = output_dir / "train_data.csv"
    statements_df.to_csv(statements_path, index=False)
    print(f"Generated {len(statements_df)} statements across {n_customers} customers -> {statements_path}")


if __name__ == "__main__":
    raw_dir = Path("data/raw")
    generate_benchmark_data(raw_dir)
