# DATABASE_SCHEMA.md — Data Layout & Schemas

There is deliberately **no server database**. Data lives in parquet/npy files; experiment results live in a small SQLite file (optional) or JSON/CSV. This keeps the project reproducible and easy to run on any machine.

## 1. Raw data (`data/raw/`)
**statements** — one row per customer per statement
| Column | Type | Notes |
|---|---|---|
| `customer_id` | string | hash/ID, never displayed |
| `statement_date` | date | used for ordering + optional time-aware check |
| `<feature_*>` | float / category | numeric and categorical features (names listed in `configs/columns.yaml`) |

**labels** — one row per customer
| Column | Type | Notes |
|---|---|---|
| `customer_id` | string | PK, joins to statements |
| `target` | int8 | 1 = default, 0 = non-default |

> Column names depend on your dataset; fill the real ones into `configs/columns.yaml` in Phase 1.

## 2. Interim (`data/interim/`)
`statements_clean.parquet` — same rows as raw, plus:
- encoded categoricals (int codes, `-1` for unseen/missing)
- imputed numerics
- `<col>__missing` int8 indicator columns
- `step_idx` int8 (0..12, position after sorting by date within customer)
- `n_statements` int8 (per customer, repeated)

## 3. Processed (`data/processed/`)
| File | Shape / schema | Purpose |
|---|---|---|
| `tabular.parquet` | `[N, 1 + A + 1]`: `customer_id`, aggregate features, `target` | LightGBM input |
| `X_seq.npy` | `float32 [N, 13, F]` | sequence input (padded) |
| `cat_seq.npy` | `int64 [N, 13, C]` | categorical indices for embeddings |
| `mask.npy` | `bool [N, 13]` | True = real statement, False = padding |
| `y.npy` | `int8 [N]` | labels, same row order as the arrays above |
| `ids.npy` | `str [N]` | row-order to `customer_id` mapping |
| `splits.parquet` | `customer_id`, `split` in {train, valid, test}, optional `time_holdout` bool | single source of truth for splits |

**Invariant:** every array above has the same row order, defined by `ids.npy`.

### Aggregate feature naming (tabular)
`{feature}__{agg}` where `agg` in `last, first, mean, std, min, max, delta_last_first, slope, delta_last_mean`.

## 4. Results (`results/<model>/`)
| File | Content |
|---|---|
| `preds_valid.npy`, `preds_test.npy` | float32 probabilities, aligned to split order from `splits.parquet` |
| `metrics.json` | `{competition_metric, gini, capture_at_4pct, roc_auc, pr_auc, brier}` for valid and test |
| `config.json` | hyperparameters, seed, git commit, feature set version, training time |
| `model.*` | `model.txt` (LightGBM) or `model.pt` (state_dict) |

## 5. Optional experiment log (SQLite: `results/experiments.db`)
```sql
CREATE TABLE runs (
  run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  model_name    TEXT NOT NULL,
  started_at    TEXT NOT NULL,
  git_commit    TEXT,
  seed          INTEGER,
  config_json   TEXT NOT NULL
);

CREATE TABLE metrics (
  run_id        INTEGER REFERENCES runs(run_id),
  split         TEXT NOT NULL,          -- valid | test
  metric        TEXT NOT NULL,          -- competition_metric, roc_auc, pr_auc, ...
  value         REAL NOT NULL,
  PRIMARY KEY (run_id, split, metric)
);

CREATE TABLE profit_curves (
  run_id        INTEGER REFERENCES runs(run_id),
  cutoff        REAL NOT NULL,
  profit        REAL NOT NULL,
  approval_rate REAL NOT NULL,
  PRIMARY KEY (run_id, cutoff)
);
```
Use it only if JSON/CSV becomes painful; the file-based contract in ARCHITECTURE.md is sufficient.

## 6. Business config (`configs/business.yaml`)
```yaml
profit_per_good: 100      # assumption, units: currency per good customer
loss_per_default: 500     # assumption, units: currency per defaulter
cutoff_grid: 0.0:1.0:0.005
```
Values are **assumptions**; the report includes a sensitivity grid over the loss/profit ratio.

## 7. Why no users table
No accounts, no logins, no persistence of user input: the demo is stateless. If you later add one, put it in a separate SQLite file and update SECURITY_AND_ACCESS.md.
