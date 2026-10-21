# ARCHITECTURE.md — Tech & Module Breakdown

## Stack
- Python 3.10+
- Data: pandas (or polars), pyarrow/parquet, numpy
- Tabular model: LightGBM
- Sequence models: PyTorch
- Metrics/calibration: scikit-learn
- Explainability: SHAP (LightGBM), attention weights / permutation importance (PyTorch)
- Charts: matplotlib / plotly
- Optional demo: Streamlit
- Config: YAML; experiment logs: JSON + CSV in `results/`

## Repo layout
```
creditseq/
├─ docs/                  # the brain files
├─ configs/
│  ├─ columns.yaml        # column types: id, date, categorical, numeric, label
│  ├─ data.yaml           # paths, seq_len=13, padding side, seed
│  ├─ lgbm.yaml
│  ├─ gru.yaml
│  ├─ transformer.yaml
│  └─ business.yaml       # profit_per_good, loss_per_default
├─ data/
│  ├─ raw/                # untouched downloads (gitignored)
│  ├─ interim/            # cleaned parquet
│  └─ processed/          # tabular.parquet, X_seq.npy, mask.npy, y.npy, splits.parquet
├─ src/
│  ├─ data/      load.py, clean.py, split.py
│  ├─ features/  tabular.py, sequence.py
│  ├─ models/    lgbm.py, gru.py, transformer.py
│  ├─ eval/      metrics.py, calibration.py, compare.py
│  ├─ business/  profit.py
│  ├─ explain/   shap_lgbm.py, seq_importance.py
│  └─ utils/     seed.py, logging.py, io.py
├─ notebooks/             # exploration only, no logic that isn't in src/
├─ results/<model>/       # preds_valid.npy, preds_test.npy, metrics.json, config.json
├─ reports/               # one md per phase + figures/
├─ tests/                 # split leakage, metric, shape tests
├─ app/                   # optional Streamlit demo
└─ README.md
```

## Data pipeline
1. `load.py` -> raw to parquet in `interim/` (float32, category dtypes).
2. `clean.py` -> encode categoricals (ordinal/label for trees, embedding index for NN), impute (median for numeric, sentinel for categorical), add `<col>__missing` indicators.
3. `split.py` -> customer-level stratified split -> `splits.parquet` (`customer_id`, `split`). Optional time-aware holdout by latest statement date.
4. `features/tabular.py` -> per-customer aggregates.
5. `features/sequence.py` -> tensor `[N, 13, F]`, mask `[N, 13]`. **Padding convention: pad at the front (older steps), so the last step is always the most recent statement** (change here if you choose otherwise and update BRAIN.md).

## Models
**LightGBM** — binary objective, early stopping on valid, `scale_pos_weight` or default with calibration after.

**GRU/LSTM** — input projection -> (Bi)GRU/LSTM -> masked last-state or masked mean/max pooling -> MLP -> sigmoid. Categorical features via embeddings.

**Transformer encoder** — input projection + learned positional encoding (13 positions) -> N encoder layers with key-padding mask -> attention pooling or CLS token -> MLP head. Alternative: GRU + attention pooling (simpler, interpretable weights).

Training defaults: AdamW, weighted BCE or focal loss, OneCycle/cosine LR, gradient clipping, early stopping on competition metric, fixed seeds.

## Evaluation module
`metrics.py` exposes:
- `normalized_gini(y, p)`
- `capture_rate_at_k(y, p, k=0.04)`
- `competition_metric(y, p)` = 0.5 * (gini + capture@4%)
- `roc_auc`, `pr_auc`, `brier`

> If using the original competition data, the official metric weights negatives (subsampled at 5%) by 20 — implement the weighted version to match leaderboard numbers; keep the unweighted version for internal comparison and say which you used in the report.

## Business module
`profit.py`: given `y`, `p`, `profit_per_good`, `loss_per_default`, sweep cutoffs `t`; approve if `p < t`;
`profit(t) = profit_per_good * (#approved & non-default) - loss_per_default * (#approved & default)`.
Baselines: approve-everyone, LightGBM. Output: curve, argmax cutoff, sensitivity grid over the loss/profit ratio.

## Explainability module
- `shap_lgbm.py`: TreeExplainer, summary plot, top-N reason codes per customer.
- `seq_importance.py`: attention weights per time step (if attention model), permutation importance by time step and feature group.

## Experiment contract
Every model script: reads its YAML, uses `splits.parquet`, writes `results/<model>/{preds_valid.npy, preds_test.npy, metrics.json, config.json}`. `compare.py` reads only these files, so models stay decoupled.

## Testing
- No customer in more than one split.
- Tensor shapes and mask sums match statement counts.
- Metric functions on toy examples with known answers.
