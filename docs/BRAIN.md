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
- Current phase: **2 — Preprocessing & Feature Engineering (Part 1 COMPLETED)**
- Last done:
  - Implemented `src/data/split.py`: generated customer-stratified splits (70% train / 15% valid / 15% test) saved to `data/processed/splits.parquet` with auxiliary `time_holdout` boolean flag.
  - Implemented `src/data/clean.py`:
    - Zero data leakage: fits median imputers and categorical mappings strictly on train split statements.
    - Encoded categoricals (`D_63`, `D_64`, `B_30`, `B_38`) into integer codes (0..K-1, with -1 for unseen/missing).
    - Added `<col>__missing` int8 indicators for features with missing values (`D_42`, `D_43`, `P_3`, `S_3`, `S_7`).
    - Added sequence metadata: `step_idx` (0..12) and `n_statements` (2..13).
    - Saved `data/interim/statements_clean.parquet` and `data/interim/clean_artifacts.json`.
  - Implemented unit test suite `tests/test_split.py` proving:
    - Zero customer overlap across splits (`train ∩ valid = ∅`, `train ∩ test = ∅`, `valid ∩ test = ∅`).
    - Stratification preserved within ±0.4% across all splits.
    - Imputer statistics are strictly isolated to train without test leakage.
  - Full test suite passing (43/43 tests).
- Next: **Phase 2 (Part 2) — Feature Engineering (`tabular.py` aggregates & `sequence.py` tensors `[N, 13, F]`)**
- Blockers: _none_

## File map
`docs/BRAIN.md` (this) · `docs/PLAN.md` · `docs/ARCHITECTURE.md` · `docs/DESIGN_SYSTEM.md` · `docs/SECURITY_AND_ACCESS.md` · `docs/DATABASE_SCHEMA.md` · `docs/PROMPTS.md`
