# BRAIN.md — Context Anchor (READ THIS FIRST, EVERY SESSION)

> Working title: **CreditSeq** (Repo: **RiskTrail**)

## One-line pitch
Predict which credit card customers will default from their sequence of monthly statements, compare a strong tabular baseline (LightGBM) against sequence models (GRU/LSTM, Transformer), and translate predictions into **profit-maximizing approval decisions** with **lender-grade explanations**.

## Core flow
1. Load raw statement-level data (one row per customer per statement) + default label per customer.
2. Clean, encode categoricals, impute missing values **and add missing indicators** (missingness is informative).
3. Build two representations:
   - **Tabular**: one row per customer (last value, mean, std, min, max, trend, etc.)
   - **Sequence**: tensor `[customers, 13, features]` + mask `[customers, 13]`
4. Train: LightGBM (tabular) -> GRU/LSTM -> Transformer / GRU+attention pooling (sequence).
5. Evaluate: competition metric, ROC-AUC, PR-AUC, calibration.
6. Business layer: sweep approval cutoff -> expected profit curve.
7. Explain: SHAP (LightGBM), attention weights / permutation importance (sequence).
8. Write report + optional demo app.

## Locked-in decisions (do NOT re-argue)
- Max sequence length **13**; shorter histories are **front-padded (left-padded)** with a boolean mask (`True` = valid statement, `False` = padding). Index 12 is always the customer's most recent statement.
- Split **by customer** (never by row), **stratified** on the default label. Primary split: 70% train / 15% valid / 15% test. Auxiliary time-aware holdout boolean preserved in `splits.parquet`.
- Primary metric = **competition metric**: mean of normalized Gini and default capture rate in the top 4%. Always also report ROC-AUC and PR-AUC. **Never use accuracy.**
- LightGBM is the benchmark every neural model must be compared against.
- Calibration is checked, because lenders price risk off the probability.
- Seeds fixed (`seed_everything` in `src/utils/seed.py`); every run logged to `results/` with config + metrics.
- Environment & Storage: Local virtual environment (`.venv`) and pip build caching isolated on `D:\` drive to preserve system disk space.
- PyTorch runtime: Standard CPU/CUDA compatible wheels installed.
- Feature taxonomy locked: `customer_ID` (id), `S_2` (date), `target` (binary label), 4 categorical features (`D_63`, `D_64`, `B_30`, `B_38`), 22 numeric features.

## Team split (two people)
| Person | Owns |
|---|---|
| **A** | Data, cleaning, features, tabular aggregates, LightGBM |
| **B** | Sequence tensors, GRU/LSTM, Transformer / attention GRU |
| **Shared** | Profit simulation, SHAP / explainability, final report (each writes own sections) |

## Glossary
- **Gini (normalized)**: `2*AUC - 1`.
- **Default capture rate @4%**: share of all defaulters found within the 4% highest-risk customers.
- **Missing indicator**: binary column flagging that a value was missing.
- **Mask**: boolean array marking real vs padded time steps.
- **Calibration**: whether predicted probabilities match observed default frequencies.

## Honest-finding expectation
LightGBM is often on par with sequence models here. If so, REPORT IT, then analyze where sequence models help (e.g., customers with deteriorating behavior over time).

## Status (update every session)
- Current phase: **1 — Data Audit & EDA (COMPLETED)**
- Last done:
  - Audited raw statement-level data (`train_data.csv`, `train_labels.csv`) with 11,877 statements across 1,000 customers.
  - Built `src/data/load.py`: converts raw CSVs to Parquet (`data/interim/statements.parquet`, `data/interim/labels.parquet`) using `float32` and `category` dtypes (48% memory reduction).
  - Built `src/data/eda.py` and `notebooks/01_eda.ipynb` reporting:
    - 1,000 unique customers, min=2, median=13.0, max=13 statements (80.8% with full 13 statements).
    - Class balance: 24.4% default (244) vs 75.6% non-default (756).
    - Missingness profile across features and across statement positions (0..12).
    - Categorical cardinalities: D_63 (5), D_64 (4), B_30 (3), B_38 (7).
    - Date range: 2017-03-12 to 2018-03-13.
    - Feasibility of time-aware check: feasible with caveats; primary split remains customer-stratified.
  - Generated figures in `reports/figures/` adhering to `docs/DESIGN_SYSTEM.md`:
    - `fig_class_balance.png`
    - `fig_statements_per_customer.png`
    - `fig_missingness_by_step.png`
    - `fig_temporal_distribution.png`
  - Wrote comprehensive findings report in `reports/01_eda.md`.
  - Populated column taxonomy in `configs/columns.yaml`.
  - Added unit test suite `tests/test_data_pipeline.py` (36/36 tests passing).
- Next: **Phase 2 — Preprocessing & Feature Engineering (A, then B reuses)**
- Blockers: _none_

## File map
`docs/BRAIN.md` (this) · `docs/PLAN.md` · `docs/ARCHITECTURE.md` · `docs/DESIGN_SYSTEM.md` · `docs/SECURITY_AND_ACCESS.md` · `docs/DATABASE_SCHEMA.md` · `docs/PROMPTS.md`
