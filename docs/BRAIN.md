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
- Masked Recurrent Sequence Model (`src/models/gru.py`): Categorical embeddings (cardinalities [5, 4, 3, 7] -> 16 dims) + numeric projection (22 -> 48 dims) -> configurable (Bi)GRU/LSTM -> mask-aware pooling (last valid state + masked mean + masked max -> 192 dims) -> MLP -> sigmoid. Vectorized left-alignment gather and input sanitization guarantee zero influence from padded steps (mathematical invariance proven by unit test with diff == 0.0). Trained with AdamW, weighted BCE (`pos_weight = 3.09`), CosineAnnealingLR, and early stopping on competition metric. Artifacts persisted in `results/gru/`: `preds_valid.npy`, `preds_test.npy`, `metrics.json`, `config.json`, `model.pt`. Validation comp metric: 0.58108; Test comp metric: 0.58333.
- Transformer Encoder & Attention Pooling (`src/models/transformer.py`): Continuous projection (22 -> 48 dims) + categorical embeddings (4 features -> 16 dims) combined into `d_model = 64` + learned positional encodings across 13 monthly statement positions -> 2-layer Pre-LN TransformerEncoder (`src_key_padding_mask=~mask`) or GRU/LSTM recurrent backbone with right-aligned state restoration -> Attention Pooling (parameterized query network weighting valid statements with padded statements clamped strictly to 0.0 attention weight) -> MLP classification head -> Sigmoid default probability. Exposes normalized `[B, 13]` attention weights via `get_attention_weights()` for Phase 7 temporal explainability. Total trainable parameters: 77,645. Trained with AdamW (`lr=5e-4`), weighted BCE (`pos_weight=3.09`), CosineAnnealingLR, early stopping on `competition_metric`. Persisted artifacts to `results/transformer/`: `preds_valid.npy` [150], `preds_test.npy` [150], `metrics.json`, `config.json`, `model.pt`. Validation comp metric: 0.58108; Test comp metric: 0.58333.
- Cross-Model Evaluation, Calibration & Trend Segmentation (`src/eval/compare.py`, `src/eval/calibration.py`): LightGBM matches sequence models (Masked GRU, StatementTransformer) on overall ranking discrimination (Test Comp Metric = 0.5833, ROC-AUC = 1.0000, PR-AUC = 1.0000). Class-weighted sequence models exhibit positive prior probability shift, which is corrected via post-hoc Platt scaling and Isotonic regression fitted strictly on the validation split (reducing GRU Brier score from 0.0310 to 0.0000, and Transformer Brier from 0.0364 to 0.0001). Customer trajectory segmentation by delinquency slope (`D_39__slope > 0`) concentrates 100% of test defaulters into the deteriorating cohort ($N=67$, 36 defaults), while calibrated models assign 0.00% predicted risk to the stable non-default cohort ($N=83$). Artifacts and figures persisted to `reports/figures/` (`fig_model_comparison.png`, `fig_calibration.png`, `fig_trend_segment.png`) and report written to `reports/05_evaluation.md`.
- Business Profit Simulation & Cutoff Policy Optimization (`src/business/profit.py`, `configs/business.yaml`): Evaluates economic credit policies under $V_{\text{good}} = \$100$ net margin and $L_{\text{default}} = \$500$ charge-off loss across approval cutoffs $c \in [0, 1]$. While naive "Approve Everyone" yields a $-\$6,600$ loss on the test cohort ($N=150$, 114 good, 36 defaults), deploying LightGBM, Calibrated GRU, or Calibrated Transformer achieves the theoretical Oracle ceiling of $+\$11,400$ ($+\$18,000$ value-add) by cleanly approving all 114 good accounts (76.0% approval rate) with zero defaults. Uncalibrated neural sequence models require higher cutoffs ($c^* \approx 0.195 - 0.265$) due to class-weighting logit shift, whereas Isotonic Calibration restores optimal operational cutoffs to $c^* \le 0.05$. Loss/profit sensitivity analysis across 2x to 20x ratios proves consistent cutoff stability and model parity. Artifacts persisted to `reports/figures/fig_profit_by_cutoff.png` and `reports/06_business.md`.
- Model Explainability & Compliance Reason Codes (`src/explain/shap_lgbm.py`, `src/explain/seq_importance.py`): LightGBM feature attribution via TreeExplainer identifies delinquency features (`D_39__last`, `D_39__mean`, `D_39__max`, `D_39__slope`) as primary risk drivers (>75% importance), generating regulatory top-5 adverse-action reason codes for declined applicants. StatementTransformer attention pooling exposes temporal attention across statement history, demonstrating strong recency weighting peaking at statement $T$ (17.7%) and $T-2$ (10.2%). Permutation importance on the sequence model confirms that destroying the Balance group ($B$) and the latest statement ($T$) causes the largest direct competition metric drops. Critical methodological caveat documented: attention weights represent internal information pooling, not counterfactual or causal attribution. Artifacts persisted to `reports/figures/` (`fig_shap_summary.png`, `fig_attention_by_month.png`) and report written to `reports/07_explainability.md`.

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
- Current phase: **7 — Explainability & Adverse Action (COMPLETED)**
- Last done:
  - Implemented `src/explain/shap_lgbm.py`:
    - TreeExplainer SHAP calculations for LightGBM, generating `reports/figures/fig_shap_summary.png` (top 20 features).
    - `get_customer_reason_codes` and `format_adverse_action_notice` extracting regulatory top-5 positive risk drivers with human-readable descriptions.
  - Implemented `src/explain/seq_importance.py`:
    - Temporal attention extraction from `StatementTransformer` comparing defaulters vs non-defaulters (`reports/figures/fig_attention_by_month.png`), proving sharp recency concentration at statement $T$ (17.7%).
    - Permutation feature importance by functional feature group (D, B, P, S, R) and by statement time step.
    - End-to-end worked case study for declined applicant `e4627d06d97d302f742070f63d17b414` aligning SHAP reasons with temporal attention weights.
    - Formulated critical scientific warning: attention weights reflect internal pooling routing, not counterfactual or causal attribution.
    - Authored comprehensive report in `reports/07_explainability.md`.
    - Added unit test suite `tests/test_explain.py` (7 tests).
  - Full test suite passing (123/123 tests). All Phase 7 exit criteria in `docs/PLAN.md` satisfied.
- Next: **Phase 8 — Final Report, Demo & Deployment (`README.md`, Streamlit demo `app.py`, final packaging)**
- Blockers: _none_

## File map
`docs/BRAIN.md` (this) · `docs/PLAN.md` · `docs/ARCHITECTURE.md` · `docs/DESIGN_SYSTEM.md` · `docs/SECURITY_AND_ACCESS.md` · `docs/DATABASE_SCHEMA.md` · `docs/PROMPTS.md`
