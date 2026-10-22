---
title: RiskTrail Credit Risk & Profit Optimization
emoji: 💳
colorFrom: blue
colorTo: purple
sdk: streamlit
app_file: app/streamlit_app.py
pinned: false
license: mit
---

# RiskTrail — Credit Risk Sequence Modeling & Profit Optimization

> **Credit Default Risk Prediction from Monthly Statement Sequences, Tabular vs Sequence Architecture Benchmarking, Profit-Maximizing Approval Policies, and Lender-Grade Explainability.**

---

> [!IMPORTANT]
> **Educational & Research Use Disclaimer**: This repository is designed exclusively for educational, benchmarking, and research purposes. It is **not** certified for real-world automated credit decisions or consumer underwriting.

---

## Overview

RiskTrail benchmarks credit risk prediction on customer-level statement histories (up to 13 consecutive monthly statements per customer):
1. **Tabular Benchmark (LightGBM)**: Statistical aggregations across statement history (`last`, `first`, `mean`, `std`, `min`, `max`, `trend/slope`, `delta`).
2. **Deep Sequence Models (PyTorch)**: Recurrent (`MaskedGRUModel`) and Attention-based (`StatementTransformer`) architectures capturing temporal deterioration patterns with strict zero-leakage front-padding.
3. **Probability Calibration**: Evaluates probability calibration (Platt scaling & Isotonic regression) to correct neural prior shift caused by class weighting (`pos_weight = 3.09`).
4. **Lender Economics & Business Optimization**: Translates predicted default probabilities into profit-maximizing lending cutoffs under bank revenue assumptions:
   $$\text{Profit}(c) = V_{\text{good}} \cdot N_{\text{good, app}}(c) - L_{\text{default}} \cdot N_{\text{def, app}}(c)$$
5. **Lender-Grade Explainability**: TreeExplainer SHAP top-5 adverse-action reason codes for rejected applicants and temporal statement attention allocation from `StatementTransformer`.
6. **Executive Report & Interactive Demo**: Institutional 8-section synthesis (`reports/FINAL_REPORT.md`) and interactive Streamlit web dashboard (`app/streamlit_app.py`).

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
│   ├── PROMPTS.md         # Phased execution prompts & audit trail
│   └── HF_SPACES_DEPLOY.md # Hugging Face Spaces deployment & checklist
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
│   ├── data/              # load.py, clean.py, split.py, create_sample.py
│   ├── features/          # tabular.py, sequence.py
│   ├── models/            # lgbm.py, gru.py, transformer.py
│   ├── eval/              # metrics.py, calibration.py, compare.py
│   ├── business/          # profit.py
│   ├── explain/           # shap_lgbm.py, seq_importance.py
│   └── utils/             # seed.py, logging.py, io.py
├── notebooks/             # Exploratory analysis only (no core pipeline logic)
├── results/               # Standardized experiment runs (<model>/preds, metrics.json)
│   ├── lgbm/              # Baseline tree model, preds, SHAP arrays
│   ├── gru/               # Masked GRU weights, calibrated & uncalibrated preds
│   └── transformer/       # StatementTransformer weights, attention, permutation importance
├── reports/               # Markdown reports per phase + figures/
│   ├── figures/           # 10 publication-standard figures
│   ├── 01_eda.md          # Exploratory data audit
│   ├── 05_evaluation.md   # Cross-model evaluation & calibration report
│   ├── 06_business.md     # Business profit optimization report
│   ├── 07_explainability.md # Adverse-action SHAP & attention report
│   └── FINAL_REPORT.md    # Consolidated 8-section institutional research document
├── tests/                 # Unit, smoke, and integration test suite (134 tests)
├── app/                   # Streamlit demo & synthetic artifacts
│   ├── streamlit_app.py   # Main demo interface (Score, Profit Curve, Why Tabs)
│   ├── app.py             # Entrypoint wrapper
│   ├── sample_data.parquet # Anonymized synthetic customer sample (50 accounts)
│   ├── sample_statements.parquet # Synthetic 13-month statement histories
│   ├── sample_shap.npy    # Precomputed SHAP values for sample accounts
│   ├── shap_features.json # 214 feature names list
│   ├── model.txt          # Portable LightGBM booster model
│   └── requirements.txt   # Pinned dependencies for Hugging Face Spaces
├── requirements.txt       # Pinned production & research dependencies
├── .env.example           # Environment variables template
├── .gitignore             # Secrets, data, cache, and prediction ignores
└── README.md              # Project onboarding & reproduction guide
```

---

## Clean Setup & Reproduction Instructions

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

On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Note for CPU-only environments: PyTorch CPU wheels can alternatively be installed via `pip install torch --index-url https://download.pytorch.org/whl/cpu`)*

### 5. Setup Environment Configuration
```bash
cp .env.example .env
```
*(On Windows PowerShell: `Copy-Item .env.example .env`)*

### 6. Run Full Verification Test Suite
Execute the complete automated test suite (134 tests covering data integrity, split isolation, padding invariance, metrics, models, business simulation, explainability, report structure, and demo app):
```bash
pytest -v
```

---

## End-to-End Execution Pipeline

To run or reproduce all phases from scratch:

```bash
# 1. Feature Engineering (Tabular aggregations and Front-Padded Sequence Tensors)
python src/features/tabular.py
python src/features/sequence.py

# 2. Model Training & Validation
python src/models/lgbm.py
python src/models/gru.py
python src/models/transformer.py

# 3. Model Evaluation, Probability Calibration & Trend Segmentation
python src/eval/compare.py

# 4. Business Profit Cutoff Optimization & Sensitivity Sweeps
python src/business/profit.py

# 5. Explainability (SHAP TreeExplainer, Reason Codes & Statement Attention)
python src/explain/shap_lgbm.py
python src/explain/seq_importance.py

# 6. Generate Synthetic Demo Sample for Standalone / Cloud Deployment
python src/data/create_sample.py
```

---

## Interactive Streamlit Demo Application

Launch the interactive underwriting decision dashboard locally:

```bash
streamlit run app/streamlit_app.py
```
*(or run `streamlit run app/app.py`)*

### Key Capabilities of the Demo
1. **Score & Underwriting Tab**:
   - Select any synthetic applicant or load the designated case study borrower (`SYNTH_0045`).
   - Dynamic **Underwriting Risk Gauge** indicating customer score vs approval threshold.
   - Real-time **APPROVED** or **DECLINED** policy decision banner at chosen cutoff.
   - Financial trajectory metrics (delinquency slope $\beta$, revolving balance $B\_1$, payment ratio $P\_2$).
2. **Profit Curve Tab**:
   - Interactive portfolio profit simulation curve across cutoffs $c \in [0, 1]$.
   - Visual comparison against the "Approve Everyone" baseline and peak profit at optimal cutoff $c^*$.
   - Policy KPI benchmarking table (Current vs Optimal vs Approve-All vs Decline-All).
3. **Why This Decision (Explainability) Tab**:
   - Top-5 risk drivers horizontal bar chart (red = raises risk, blue = lowers risk) using TreeExplainer SHAP.
   - Structured adverse-action reason codes with lender-grade descriptions.
   - 13-statement temporal trajectory chart showing delinquency and balance evolution over time.
   - Formal adverse-action notice formatted for FCRA / ECOA regulatory audits.

---

## Hugging Face Spaces Deployment & Security Checklist

This repository is pre-configured for one-click deployment to **Hugging Face Spaces** via the Streamlit SDK.

### Security Checklist (per `docs/SECURITY_AND_ACCESS.md`)
- [x] **No Raw or Customer Data**: All real customer data directories (`data/raw/`, `data/interim/`, `data/processed/`) are strictly gitignored.
- [x] **Synthetic Sample Only**: The demo uses only `app/sample_data.parquet` (50 synthetic records generated by `src/data/create_sample.py`).
- [x] **Zero Secrets or API Keys**: No Kaggle credentials, Hugging Face tokens (`HF_TOKEN`), or private keys exist in any tracked files or git history.
- [x] **Model Portability**: The saved LightGBM benchmark model is bundled at `app/model.txt` (text format, zero pickle vulnerabilities).
- [x] **Educational Use Disclaimer**: The application prominently displays the mandatory educational and non-commercial disclaimer.

### Deployment Commands
```bash
# 1. Add your Hugging Face Space as a git remote
git remote add space https://huggingface.co/spaces/<YOUR_HF_USERNAME>/risktrail-demo

# 2. Push to Hugging Face Spaces
git push space main
```
For detailed configuration options and hardware recommendations, see [`docs/HF_SPACES_DEPLOY.md`](docs/HF_SPACES_DEPLOY.md).

---

## Roadmap & Completion Status

All project phases have been completed and verified against the criteria in `docs/PLAN.md`:

| Phase | Description | Deliverables | Status |
|---|---|---|:---:|
| **0** | Setup & Brain | Folder structure, config files, seeds, smoke tests | **DONE** |
| **1** | Data Audit & EDA | `reports/01_eda.md`, `configs/columns.yaml`, missingness heatmaps | **DONE** |
| **2** | Preprocessing & Features | `splits.parquet`, `tabular.parquet`, `X_seq.npy`, `mask.npy`, `y.npy` | **DONE** |
| **3** | LightGBM Baseline | `results/lgbm/` (model, preds, metrics.json) | **DONE** |
| **4a** | Masked GRU Model | `results/gru/` (padding invariance proven, metrics.json) | **DONE** |
| **4b** | StatementTransformer | `results/transformer/` (learned positional encodings, attention pooling) | **DONE** |
| **5** | Evaluation & Calibration | `src/eval/compare.py`, Platt & Isotonic calibration, `reports/05_evaluation.md` | **DONE** |
| **6** | Business Profit Simulation | `src/business/profit.py`, cutoff sweeps, sensitivity grid, `reports/06_business.md` | **DONE** |
| **7** | Explainability Hub | SHAP TreeExplainer, adverse action reasons, attention timeline, `reports/07_explainability.md` | **DONE** |
| **8a** | Institutional Research Report | `reports/FINAL_REPORT.md` (8 sections, embedded figures & tables) | **DONE** |
| **8b** | Interactive Streamlit Demo & Deploy | `app/streamlit_app.py`, `app/sample_data.parquet`, `docs/HF_SPACES_DEPLOY.md` | **DONE** |

---

## License & Responsible Data Handling
- Do not commit or distribute raw/processed customer statements or prediction arrays containing customer IDs.
- For data licensing and competition terms, refer to the source dataset agreement.
