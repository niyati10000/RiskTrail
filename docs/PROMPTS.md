# PROMPTS.md — Prompt Templates & Execution Log

Standard prompts and instructions for running each phase of the **RiskTrail** project.

---

## Phase 0 — Setup & Environment Verification
```text
Verify repo setup:
1. Ensure all packages (pandas, pyarrow, numpy, scikit-learn, lightgbm, torch, shap, matplotlib, pyyaml, pytest, streamlit) import without error.
2. Verify utils (seed_everything, load_yaml, save_json, get_logger).
3. Run tests/test_smoke.py.
```

## Phase 1 — Data Audit & Exploratory Data Analysis (A)
```text
Perform initial data audit on statement-level data:
- Inspect row count, unique customers, distribution of statements per customer (up to 13).
- Assess missingness across columns and temporal positions.
- Examine target class balance.
- Generate figures and report in reports/01_eda.md.
- Output column classification to configs/columns.yaml.
```

## Phase 2 — Preprocessing & Feature Engineering (A & B)
```text
Execute data preprocessing pipeline:
- Ordinal/embedding encoding for categoricals, median/indicator imputation for numerics.
- Construct tabular summary aggregates -> data/processed/tabular.parquet.
- Construct 3D sequence tensors [N, 13, F] and key padding masks [N, 13] -> X_seq.npy, mask.npy.
- Generate customer-stratified split -> splits.parquet (single source of truth).
- Validate split isolation (zero customer leakage across train/valid/test).
```

## Phase 3 — LightGBM Tabular Baseline (A)
```text
Train LightGBM tabular benchmark:
- Train with early stopping on validation split using competition metric.
- Save outputs to results/lgbm/: model.txt, preds_valid.npy, preds_test.npy, metrics.json, config.json.
- Establish the performance baseline to beat.
```

## Phase 4 — Sequence Modeling (GRU & Transformer) (B)
```text
Train recurrent and attention-based sequence models:
- 4a: Train MaskedGRUModel with packed/masked pooling.
- 4b: Train StatementTransformer with learned positional encodings and key-padding mask.
- Save results to results/gru/ and results/transformer/ adhering to the experiment contract.
```

## Phase 5 — Unified Evaluation & Calibration (Shared)
```text
Conduct comprehensive evaluation:
- Benchmark all models on the identical test split: competition metric, normalized Gini, capture@4%, ROC-AUC, PR-AUC, Brier score.
- Compute reliability diagrams / calibration curves.
- Perform segment analysis: compare sequence models vs LightGBM on deteriorating vs stable customers.
- Write reports/05_evaluation.md.
```

## Phase 6 — Business Profit Simulation (Shared)
```text
Simulate business lending economics:
- Using configs/business.yaml (profit_per_good, loss_per_default).
- Sweep approval cutoffs [0.0, 1.0].
- Compute profit curves, optimal approval cutoff, and sensitivity grid across loss/profit ratios.
- Output fig_profit_by_cutoff and summary findings.
```

## Phase 7 — Explainability & Adverse Action (Shared)
```text
Generate model explanations:
- SHAP TreeExplainer for LightGBM with global summary and top-5 reason codes per applicant.
- Attention weights analysis across statement months for sequence models.
- Document adverse-action explanations in reports/07_explainability.md.
```

## Phase 8 — Final Report & Demo Application (Shared)
```text
Finalize deliverables:
- Compile comprehensive project report.
- Launch and verify Streamlit interactive demo (app/app.py).
- Validate end-to-end reproduction steps.
```
