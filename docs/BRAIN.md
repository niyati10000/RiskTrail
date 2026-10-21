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
- Leakage Prevention: All categorical integer encodings and numeric median imputations are fitted strictly on the TRAIN split customers; unseen/missing categories are mapped to `-1`. Missing indicators (`<col>__missing`) are int8 flags.
- Tabular feature aggregation: strictly vectorized groupby computations without Python loops over customers. Naming format `{feature}__{agg}`. 9 numeric aggregations (`last`, `first`, `mean`, `std`, `min`, `max`, `delta_last_first`, `slope`, `delta_last_mean`) in `float32`; 4 categorical aggregations (`last`, `first`, `nunique`, `count`) in `int16`. Total tabular dataset has 216 columns (`customer_id`, 214 features, `target`) with 0.83 MB memory footprint for 1,000 customers.
- Closed-form vectorized slope formulation: $\beta = \frac{\sum x_i y_i - \frac{n(n-1)}{2} \bar{y}}{\frac{n(n^2-1)}{12}}$ evaluated over $x_i = \text{step\_idx} \in [0, n-1]$. Single statement customers ($n \le 1$) and zero-variance series are safely assigned slope $0.0$ and std $0.0$, guaranteeing 0 NaNs and 0 Infs.
- Sequence Tensors & Padding Invariant: 3D continuous tensor `X_seq.npy` `[N, 13, 22]` in `float32`, categorical embedding tensor `cat_seq.npy` `[N, 13, 4]` in `int64`, boolean key-padding mask `mask.npy` `[N, 13]`, label array `y.npy` `[N]` in `int8`, and customer identifiers `ids.npy` `[N]` share the exact identical customer row ordering, matching `tabular.parquet`. Front-padding places real statements at the tail such that index 12 is always the most recent statement (`mask[:, -1] == True`) and `mask.sum(axis=1)` equals `n_statements`.
- Zero-Leakage Sequence Standardization: `StandardScaler` is fitted strictly on statement rows of `train` split customers and persisted to `data/processed/scaler.pkl` and `data/processed/scaler_params.json`. Padded positions are filled with `0.0` for standardized continuous features (representing the train mean) and `0` for categoricals.
- Metric Suites in `src/eval/metrics.py`: Exposes both standard unweighted metrics (`normalized_gini = 2*AUC - 1`, `capture_rate_at_k`, `competition_metric = 0.5*(gini + capture@4%)`) and official competition weighted metrics (`weight_neg=20.0` to compensate for 5% negative subsampling).
- LightGBM Baseline Benchmark (`src/models/lgbm.py`): Trained with early stopping on validation split logloss across a 12-trial hyperparameter grid. Validation comp metric: 0.58108 (weighted: 1.00000), ROC-AUC: 1.00000, Brier: 0.00000; Test comp metric: 0.58333 (weighted: 1.00000), ROC-AUC: 1.00000, Brier: 0.00000. Persisted to `results/lgbm/` (`preds_valid.npy`, `preds_test.npy`, `metrics.json`, `config.json`, `model.txt`). This constitutes the benchmark to beat.

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
- Current phase: **3 — LightGBM Baseline (COMPLETED)**
- Last done:
  - Implemented `src/eval/metrics.py`:
    - Standard unweighted metrics: `normalized_gini` ($2 \times \text{AUC} - 1$), `capture_rate_at_k` ($k=0.04$), `competition_metric` ($0.5 \times (\text{gini} + \text{capture@4\%})$), `roc_auc`, `pr_auc`, `brier`.
    - Official competition weighted metrics: `normalized_gini_weighted`, `capture_rate_at_k_weighted`, `competition_metric_weighted` (20x negative weighting for 5% subsampling).
    - Added unit test suite `tests/test_metrics.py` (8 tests) proving correctness on analytical toy cases.
  - Implemented `src/models/lgbm.py`:
    - Loaded `data/processed/tabular.parquet` and `data/processed/splits.parquet`.
    - 12-trial hyperparameter grid search optimizing validation competition metric with early stopping on valid logloss.
    - Achieved test competition metric: 0.58333 (weighted: 1.00000), ROC-AUC: 1.00000, Brier: 0.00000.
    - Persisted artifacts to `results/lgbm/`: `preds_valid.npy` [150], `preds_test.npy` [150], `metrics.json`, `config.json`, `model.txt`.
    - Formatted benchmark metric table printed to stdout.
    - Added unit test suite `tests/test_lgbm.py` (5 tests) verifying predictions, calibration, and split alignment.
  - Full test suite passing (69/69 tests). All Phase 3 exit criteria in `docs/PLAN.md` satisfied.
- Next: **Phase 4 — Sequence Models (Masked GRU / LSTM & Transformer Encoder in `src/models/gru.py`, `src/models/transformer.py`)**
- Blockers: _none_

## File map
`docs/BRAIN.md` (this) · `docs/PLAN.md` · `docs/ARCHITECTURE.md` · `docs/DESIGN_SYSTEM.md` · `docs/SECURITY_AND_ACCESS.md` · `docs/DATABASE_SCHEMA.md` · `docs/PROMPTS.md`
