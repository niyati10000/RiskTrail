# RiskTrail — Credit Risk Sequence Modeling & Profit Optimization

> **Credit Default Risk Prediction from Monthly Statement Sequences, Tabular vs Sequence Architecture Benchmarking, and Profit-Maximizing Approval Policies.**

---

> [!IMPORTANT]
> **Educational & Research Use Disclaimer**: This repository is designed exclusively for educational, benchmarking, and research purposes. It is **not** certified for real-world automated credit decisions or consumer underwriting.

---

## Overview

RiskTrail benchmarks credit risk prediction on customer-level statement histories (up to 13 consecutive monthly statements per customer):
1. **Tabular Benchmark (LightGBM)**: Statistical aggregations across statement history (`last`, `first`, `mean`, `std`, `min`, `max`, `trend/slope`, `delta`).
2. **Deep Sequence Models (PyTorch)**: Recurrent (`MaskedGRUModel`) and Attention-based (`StatementTransformer`) architectures capturing temporal deterioration patterns.
3. **Lender Economics & Calibration**: Evaluates probability calibration and translates risk probabilities into optimal lending cutoff curves that maximize net portfolio profit:
   $$\text{Profit}(t) = \text{profit\_per\_good} \times N_{\text{approved, good}} - \text{loss\_per\_default} \times N_{\text{approved, default}}$$
4. **Lender-Grade Explainability**: SHAP adverse-action reason codes for rejected applicants and temporal attention importance.

---

## Repository Architecture

```text
RiskTrail/
├── docs/                  # Project brain files (context, architecture, schema, plan)
│   ├── BRAIN.md           # Master context anchor & session status
│   ├── ARCHITECTURE.md    # System design & module contracts
│   ├── DATABASE_SCHEMA.md # Data schemas & array invariants
│   ├── DESIGN_SYSTEM.md   # Visualization palette & demo standards
│   ├── PLAN.md            # Phased roadmap & exit gates
│   ├── SECURITY_AND_ACCESS.md # Governance, secrets, & ethical lending
│   └── PROMPTS.md         # Phased execution prompts & audit trail
├── configs/               # YAML configurations
│   ├── columns.yaml       # Column classifications (id, date, categoricals, numerics)
│   ├── data.yaml          # Paths, max sequence length (13), padding side, seed
│   ├── lgbm.yaml          # LightGBM hyperparameters
│   ├── gru.yaml           # GRU architecture & training hyperparams
│   ├── transformer.yaml   # Transformer architecture & training hyperparams
│   └── business.yaml      # Profit/loss assumptions & cutoff sweep grid
├── data/                  # Data directories (gitignored)
│   ├── raw/               # Untouched original statement data & labels
│   ├── interim/           # Cleaned and encoded Parquet statements
│   └── processed/         # Tabular parquet, sequence tensors, splits.parquet
├── src/                   # Core modular package
│   ├── data/              # load.py, clean.py, split.py
│   ├── features/          # tabular.py, sequence.py
│   ├── models/            # lgbm.py, gru.py, transformer.py
│   ├── eval/              # metrics.py, calibration.py, compare.py
│   ├── business/          # profit.py
│   ├── explain/           # shap_lgbm.py, seq_importance.py
│   └── utils/             # seed.py, logging.py, io.py
├── notebooks/             # Exploratory analysis only (no core pipeline logic)
├── results/               # Standardized experiment runs (<model>/preds, metrics.json)
├── reports/               # Markdown reports per phase + figures/
├── tests/                 # Unit, smoke, and split leakage verification tests
│   └── test_smoke.py      # Dependency import & utility smoke test
├── app/                   # Streamlit interactive decision demo
├── requirements.txt       # Pinned production & research dependencies
├── .env.example           # Environment variables template
├── .gitignore             # Secrets, data, cache, and prediction ignores
└── README.md              # Project onboarding & reproduction guide
```

---

## Clean Setup Instructions

### 1. Prerequisites
- **Python**: 3.10+ (tested on Python 3.11.9)
- **Git**

### 2. Clone Repository
```bash
git clone <repo-url>
cd RiskTrail
```

### 3. Create & Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Windows (CMD):
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
Install all pinned dependencies:
```bash
pip install -r requirements.txt
```

*(Note for CPU-only environments: PyTorch CPU wheels can alternatively be installed via `pip install torch --index-url https://download.pytorch.org/whl/cpu`)*

### 5. Setup Environment Configuration
```bash
cp .env.example .env
```
*(On Windows PowerShell: `Copy-Item .env.example .env`)*

### 6. Run Tests & Model Pipelines
Run the complete automated pytest verification suite (smoke, data pipeline, split isolation, tabular features, sequence tensors, metrics, LightGBM):
```bash
pytest -v
```

Execute the tabular, sequence, and LightGBM baseline pipelines:
```bash
python src/features/tabular.py
python src/features/sequence.py
python src/models/lgbm.py
```

---

## Development Workflow & Roadmap

The project executes strictly in sequential phases according to `docs/PLAN.md`. Each phase must pass its explicit exit criteria before proceeding:
- **Phase 0 (Completed)**: Setup, repo architecture, pinned dependencies, seed and I/O utilities, smoke tests, brain documentation.
- **Phase 1 (Completed)**: Data Audit & EDA (`reports/01_eda.md`, `configs/columns.yaml`).
- **Phase 2 (Completed)**: Preprocessing, Missing Indicators, Customer-Stratified Splits (`splits.parquet`), Tabular Aggregates (`tabular.parquet`), and Front-Padded Sequence Tensors (`X_seq.npy`, `cat_seq.npy`, `mask.npy`, `y.npy`, `ids.npy`, `scaler.pkl`).
- **Phase 3 (Completed)**: LightGBM Tabular Baseline (`results/lgbm/`: `preds_valid.npy`, `preds_test.npy`, `metrics.json`, `config.json`, `model.txt`).
- **Phase 4**: Sequence Models (Masked GRU & Transformer) (`results/gru/`, `results/transformer/`).
- **Phase 4**: Sequence Models (Masked GRU & Transformer) (`results/gru/`, `results/transformer/`).
- **Phase 5**: Cross-Model Evaluation, Normalized Gini, Capture Rate @ 4%, Calibration Curves.
- **Phase 6**: Business Profit Optimization & Approval Cutoff Sweeps.
- **Phase 7**: Explainability, Adverse Action Reason Codes & Temporal Attention Analysis.
- **Phase 8**: Executive Report & Interactive Streamlit Demo.

---

## License & Responsible Data Handling
- Do not commit or distribute raw/processed customer statements or prediction arrays containing customer IDs.
- For data licensing and competition terms, refer to the source dataset agreement.
