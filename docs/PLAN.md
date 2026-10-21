# PLAN.md — Phased Roadmap

Each phase produces something demoable and has **exit criteria**. Do not start a phase until the previous one's exit criteria are met. Owner: **A** = data/LightGBM, **B** = sequence models, **S** = shared.

---

## Phase 0 — Setup & Brain (S)
- Create repo, folders (see ARCHITECTURE.md), `requirements.txt`, config file, seeds util.
- Drop the 7 brain files into `/docs`.
- **Exit:** `python -c "import lightgbm, torch, shap, sklearn"` works; repo pushed; BRAIN.md status updated.

## Phase 1 — Data Audit & EDA (A)
- Load data, check shape, dtypes, customers count, statements per customer (expect up to 13).
- Class balance, missingness by column and by statement position, categorical cardinality, date range.
- Decide time-aware check feasibility.
- **Exit:** `reports/01_eda.md` + figures; list of columns by type (categorical / numeric / date / id) saved to `configs/columns.yaml`.

## Phase 2 — Preprocessing & Feature Engineering (A, then B reuses)
- Encode categoricals, impute, add missing indicators.
- Build **tabular features**: last, mean, std, min, max, first, last-minus-first, slope/trend, last-minus-mean.
- Build **sequence tensors** `[N, 13, F]` + mask `[N, 13]` (B can start in parallel from the cleaned parquet).
- Stratified customer-level split saved as `splits.parquet` (train/valid/test + optional time-aware holdout).
- **Exit:** `data/processed/` has tabular parquet, `X_seq.npy`, `mask.npy`, `y.npy`, `splits.parquet`; unit test proves no customer appears in two splits.

## Phase 3 — LightGBM Baseline (A)
- Train with early stopping on validation; modest hyperparameter search.
- Log metrics (competition metric, ROC-AUC, PR-AUC).
- Save predictions for valid/test.
- **Exit:** `results/lgbm/` with model, preds, metrics.json. This is the number to beat.

## Phase 4 — Sequence Models (B)
- 4a. GRU/LSTM with masking (packed sequence or mask-aware pooling).
- 4b. Transformer encoder with learned positional encoding **or** GRU with attention pooling.
- Class-imbalance handling (weighted loss or sampling), early stopping, LR schedule.
- **Exit:** `results/gru/`, `results/transformer/` with preds + metrics.json using the same splits.

## Phase 5 — Evaluation & Calibration (S)
- Unified evaluator: competition metric, ROC-AUC, PR-AUC on identical test split for all models.
- Calibration curves + Brier score; apply isotonic/Platt calibration if needed.
- Where-sequence-helps analysis: segment customers by trend (deteriorating vs stable) and compare models.
- **Exit:** `reports/05_evaluation.md` with a comparison table and calibration plots.

## Phase 6 — Business Profit Simulation (S)
- Assumptions in config: profit per good customer, loss per default.
- Sweep approval cutoff 0..1; compute total profit for each model.
- Compare against **approve everyone** and the **LightGBM baseline**; mark the profit-maximizing cutoff.
- **Exit:** profit-vs-cutoff chart, table of optimal cutoffs, sensitivity to the profit/loss ratio.

## Phase 7 — Explainability (S)
- SHAP on LightGBM: global summary, plus per-customer reason codes for rejections.
- Sequence model: attention weights over months and/or permutation importance (by feature group and by time step).
- Which months and features drive risk.
- **Exit:** `reports/07_explainability.md`, figures, one worked "why was this customer rejected" example.

## Phase 8 — Report, Demo & Deploy (S)
- Report: each person writes their own sections (data/LightGBM vs sequence models); shared sections for evaluation, business, explainability.
- Optional Streamlit demo: upload/select a customer -> risk score, decision at chosen cutoff, top reasons.
- Optional deploy to Hugging Face Spaces with code on GitHub.
- **Exit:** final report PDF, README with reproduction steps, demo link (if built).

---

## Timeline template (adjust to your deadline)
| Week | Phases |
|---|---|
| 1 | 0, 1, 2 |
| 2 | 3, 4 (parallel) |
| 3 | 5, 6, 7 |
| 4 | 8 + buffer |

## Risk list
- Memory blowup on full data -> use parquet, float32, column subsets, or sample customers for dev.
- Sequence model underperforms LightGBM -> expected; report honestly.
- Leakage via row-level split -> guarded by the split test in Phase 2.
