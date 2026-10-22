```markdown
<div align="center">

# RiskTrail

### Sequential Credit Default Prediction, Underwriting Profit Optimization & Lender-Grade Explainability

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-brightgreen?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-136%20Passing-success?style=for-the-badge&logo=pytest&logoColor=white)](tests/)

<p align="center">
  <a href="#project-overview">Overview</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#architecture--tech-stack">Architecture</a> •
  <a href="#use-cases">Use Cases</a> •
  <a href="#getting-started--setup">Getting Started</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#license">License</a>
</p>

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architecture & Tech Stack](#architecture--tech-stack)
  - [System Architecture](#system-architecture)
  - [Technology Breakdown](#technology-breakdown)
- [Use Cases](#use-cases)
- [Getting Started & Setup](#getting-started--setup)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
  - [Running the Pipelines](#running-the-pipelines)
  - [Launching the Web Application](#launching-the-web-application)
- [Project Structure](#project-structure)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [License & Ethical Use](#license--ethical-use)

---

## Project Overview

**RiskTrail** is a production-grade credit risk modeling and underwriting optimization platform engineered to predict consumer default from longitudinal credit card statement sequences. In traditional credit bureaus, customer histories are collapsed into static tabular averages, obscuring critical temporal warning signals such as escalating delinquency trajectories, balance spikes, and deteriorating payment velocity. RiskTrail systematically benchmarks industry-standard gradient boosted decision trees against deep sequential neural architectures while closing the gap between statistical probability estimation and commercial lender profitability.

The platform executes a rigorous dual-track methodology: first engineering a 214-feature tabular dataset across historical aggregations (mean, standard deviation, slope, delta), and second constructing front-padded 3D sequence tensors `[N, 13, 22]` preserving multi-month statement dynamics without future data leakage. Predictions from both tabular (LightGBM) and neural models (Masked GRU and StatementTransformer with learned positional encodings and attention pooling) undergo post-hoc probability calibration (Platt scaling and Isotonic regression) to ensure true expected loss alignment.

Beyond predictive scoring, RiskTrail implements a business decision layer that evaluates bank economics across varying interest margins and charge-off losses ($V_{\text{good}} = \$100$, $L_{\text{default}} = \$500$). By conducting continuous cutoff sweeps and sensitivity grids (2x to 20x loss-to-profit ratios), the system identifies the exact cutoffs that maximize net portfolio revenue. Every adverse credit determination is backed by regulatory-grade adverse-action reason codes derived via TreeExplainer SHAP in strict compliance with the Equal Credit Opportunity Act (ECOA) and Fair Credit Reporting Act (FCRA).

---

## Key Features

- **Dual-Track Machine Learning Benchmark**: Direct empirical comparison between a tuned 214-feature LightGBM tabular benchmark and deep recurrent (`MaskedGRUModel`) and attention-based (`StatementTransformer`) architectures.
- **Zero-Leakage Front-Padding Sequence Engine**: Standardized 13-statement temporal horizon with front-padding invariants guaranteeing that step $T=12$ represents the most recent statement and padded positions carry zero mathematical influence.
- **Post-Hoc Probability Calibration**: Incorporates Isotonic Regression and Platt Scaling to correct for neural prior probability shifts caused by class imbalance reweighting (`pos_weight = 3.09`), restoring Brier score fidelity from $0.036$ down to $\le 0.0001$.
- **Lender Economic & Profit Cutoff Simulation**: Translates raw probabilities into net dollar return curves:
  $$\text{Profit}(c) = V_{\text{good}} \cdot N_{\text{approved, good}}(c) - L_{\text{default}} \cdot N_{\text{approved, default}}(c)$$
  Identifying the profit-maximizing operational cutoff and proving $+\$18,000$ value-added relative to a naive "Approve Everyone" policy.
- **Lender-Grade Explainability Hub**: Automated generation of regulatory adverse-action notices powered by TreeExplainer SHAP log-odds feature attributions, paired with 13-statement temporal attention timelines.
- **Interactive Underwriting Dashboard**: Lightweight Streamlit console providing real-time risk gauges, cutoff threshold adjustment, portfolio profit curves, and applicant trajectory audits.
- **Privacy-Preserving Standalone Deployment**: Ships with an anonymized 50-account synthetic cohort (`app/sample_data.csv`) and portable model weights (`app/model.txt`), preventing real client data leakage.

---

## Architecture & Tech Stack

### System Architecture

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      RAW MONTHLY STATEMENT DATA                        │
 │           (Up to 13 Historical Statements per Customer Account)        │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                   ┌─────────────────┴─────────────────┐
                   ▼                                   ▼
        ┌─────────────────────┐             ┌─────────────────────┐
        │  TABULAR PIPELINE   │             │  SEQUENCE PIPELINE  │
        │ Vectorized Groupby  │             │ Front-Padded Tensor │
        │ 214 Aggregations:   │             │ [N, 13, 22] + Mask  │
        │ last, mean, slope   │             │ Z-Score Scaled      │
        └──────────┬──────────┘             └──────────┬──────────┘
                   │                                   │
                   ▼                                   ▼
        ┌─────────────────────┐             ┌─────────────────────┐
        │  LIGHTGBM BASELINE  │             │ SEQUENTIAL NEURAL   │
        │  Tree Booster Grid  │             │ Masked (Bi)GRU /    │
        │  Comp Metric: 0.583 │             │ StatementTransformer│
        └──────────┬──────────┘             └──────────┬──────────┘
                   │                                   │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │    CALIBRATION & EVALUATION       │
                   │    Isotonic & Platt Scaling       │
                   │    Competition Metric / ROC / PR  │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │    BUSINESS PROFIT ENGINE         │
                   │    Cutoff Optimization Sweep      │
                   │    2x to 20x Sensitivity Analysis │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │   EXPLAINABILITY & REGULATORY     │
                   │   TreeExplainer SHAP Reason Codes │
                   │   Statement Attention Allocation  │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │   STREAMLIT INTERACTIVE APP       │
                   │   Real-Time Underwriting Console  │
                   └───────────────────────────────────┘
```

### Technology Breakdown

| Domain | Technology / Library | Purpose |
|---|---|---|
| **Language** | Python 3.11+ | Core runtime environment |
| **Tabular Modeling** | LightGBM | Gradient boosted decision tree benchmark |
| **Deep Learning** | PyTorch (CPU/CUDA) | Recurrent (`GRU`) and attention (`Transformer`) sequence models |
| **Explainability** | SHAP (TreeExplainer) | Exact local feature attributions and adverse action reason codes |
| **Data Processing** | Pandas, PyArrow, NumPy | High-performance vectorized tabular feature transforms and Parquet I/O |
| **Evaluation** | Scikit-Learn | Calibration curves, Brier score, ROC-AUC, PR-AUC, Gini coefficient |
| **Visualization** | Matplotlib | Publication-standard reporting charts and risk gauges |
| **Interactive UI** | Streamlit | Web-based underwriting, profit simulation, and audit dashboard |
| **Testing** | Pytest | 136-test automated unit, integration, and leakage verification suite |

---

## Use Cases

- **Automated Credit Card Underwriting**: Financial institutions evaluate applicants with variable statement histories (1 to 13 months) to issue instantaneous credit decisions backed by empirical default probabilities.
- **Dynamic Portfolio Profit Maximization**: Chief Risk Officers (CROs) model expected charge-off losses against net interest margins to calibrate underwriting cutoffs dynamically during shifting macroeconomic credit cycles.
- **Regulatory Adverse-Action Disclosures**: Consumer lending compliance officers automatically generate legally compliant adverse-action statements satisfying ECOA and FCRA notification mandates.
- **Early-Warning Account Monitoring**: Card issuers identify currently performing accounts exhibiting subtle deteriorating balance velocity and payment slopes prior to delinquency events.

---

## Getting Started & Setup

### Prerequisites

- **Python**: Version 3.10 or higher (tested on Python 3.11.9)
- **Git**
- **Hardware**: Standard x86_64 CPU (minimum 8 GB RAM recommended)

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/niyati10000/RiskTrail.git
   cd RiskTrail
   ```

2. **Initialize and Activate Virtual Environment**:
   - **Linux / macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *(For CPU-only environments, PyTorch CPU wheels install automatically via standard pip).*

4. **Environment Configuration**:
   ```bash
   cp .env.example .env
   ```
   *(Windows: `Copy-Item .env.example .env`)*

---

### Running the Pipelines

To reproduce the entire modeling, evaluation, and explainability lifecycle from scratch:

```bash
# 1. Feature Engineering (Tabular aggregations and Front-Padded sequence tensors)
python src/features/tabular.py
python src/features/sequence.py

# 2. Model Training (LightGBM, Masked GRU, StatementTransformer)
python src/models/lgbm.py
python src/models/gru.py
python src/models/transformer.py

# 3. Unified Cross-Model Evaluation & Probability Calibration
python src/eval/compare.py

# 4. Business Profit Cutoff Sweep & Sensitivity Analysis
python src/business/profit.py

# 5. Explainability (SHAP TreeExplainer & Temporal Attention)
python src/explain/shap_lgbm.py
python src/explain/seq_importance.py

# 6. Generate Synthetic Cohort for Cloud / Standalone Deployment
python src/data/create_sample.py
```

---

### Launching the Web Application

Launch the interactive underwriting decision dashboard locally:

```bash
streamlit run app/streamlit_app.py
```

Once launched, navigate to `http://localhost:8501` in your browser.

- **Score Tab**: Adjust the **Approval Cutoff slider** to view real-time **APPROVED / DECLINED** determinations and observe the applicant risk gauge.
- **Profit Curve Tab**: Inspect total portfolio revenue curves across cutoff thresholds $c \in [0, 1]$ relative to the "Approve Everyone" baseline.
- **Why This Decision Tab**: Review top-5 SHAP adverse-action drivers and inspect the 13-month historical statement trajectory.

---

## Project Structure

```text
RiskTrail/
├── configs/                   # Declarative YAML parameters
│   ├── business.yaml          # Profit margins ($100), loss assumptions ($500), sweep grid
│   ├── columns.yaml           # Metadata taxonomy: IDs, dates, categoricals, numerics
│   ├── data.yaml              # Paths, sequence lengths (13), padding configuration
│   ├── gru.yaml               # Masked GRU hyperparameters
│   ├── lgbm.yaml              # LightGBM tree hyperparameters
│   └── transformer.yaml       # StatementTransformer layer dimensions & heads
├── data/                      # Data storage (gitignored per privacy standards)
│   ├── raw/                   # Raw customer statement parquet files
│   ├── interim/               # Imputed, cleaned statement series
│   └── processed/             # Tabular.parquet, X_seq.npy, mask.npy, splits.parquet
├── docs/                      # Architectural and governance specifications
│   ├── ARCHITECTURE.md        # Technical component contracts
│   ├── BRAIN.md               # Context anchor & locked-in engineering decisions
│   ├── DATABASE_SCHEMA.md     # Invariant array contracts and schema definitions
│   ├── DESIGN_SYSTEM.md       # Visual palette, typography, and chart standards
│   ├── HF_SPACES_DEPLOY.md    # Hugging Face deployment checklist & security guide
│   ├── PLAN.md                # Multi-phase engineering roadmap
│   └── SECURITY_AND_ACCESS.md # Data governance, secrets hygiene, and fair lending
├── reports/                   # Audit reports and publication assets
│   ├── figures/               # Standard publication figures (PNG, 150/300 DPI)
│   ├── 01_eda.md              # Missingness and customer distribution audit
│   ├── 05_evaluation.md       # Model benchmarking & calibration analysis
│   ├── 06_business.md         # Portfolio cutoff optimization report
│   ├── 07_explainability.md   # SHAP adverse action & attention attribution report
│   └── FINAL_REPORT.md        # Consolidated institutional research paper
├── results/                   # Persisted experiment runs and weights
│   ├── gru/                   # Masked GRU PyTorch weights & validation predictions
│   ├── lgbm/                  # LightGBM booster model (text) & validation predictions
│   └── transformer/           # StatementTransformer weights & permutation importance
├── src/                       # Core modular codebase
│   ├── business/profit.py     # Economic profit simulation & cutoff sweeps
│   ├── data/                  # Loading, cleaning, stratified customer splitting
│   ├── eval/                  # Competition metric, Gini, calibration curves
│   ├── explain/               # SHAP TreeExplainer & sequence attention extractors
│   ├── features/              # Tabular groupby aggregates & front-padded sequence tensors
│   ├── models/                # PyTorch architectures (GRU, Transformer) & LightGBM
│   └── utils/                 # Reproducibility seeds, logging, file I/O
├── tests/                     # Automated testing suite
│   ├── test_app.py            # Streamlit data contracts & plot generation tests
│   ├── test_features.py       # Groupby aggregation & slope verification tests
│   ├── test_metrics.py        # Gini, capture@4%, and competition metric tests
│   ├── test_sequence.py       # Front-padding leakage & tensor invariant tests
│   └── test_split.py          # Customer stratification & isolation tests
├── app/                       # Standalone demo & deployment package
│   ├── app.py                 # Forwarding entrypoint wrapper
│   ├── streamlit_app.py       # Core Streamlit user interface
│   ├── model.txt              # Portable LightGBM booster weights (text format)
│   ├── sample_data.csv        # Anonymized synthetic sample accounts (50 records)
│   ├── sample_statements.csv  # Synthetic 13-month statement histories
│   ├── sample_shap.json       # Precomputed SHAP attribution matrix
│   └── requirements.txt       # Streamlit Cloud & Spaces dependencies
├── requirements.txt           # Master project dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Secrets, binary weights, and customer data ignore rules
└── README.md                  # Project documentation
```

---

## Testing & Quality Assurance

RiskTrail enforces exhaustive regression and unit testing across mathematical formulations, padding leakage invariants, data integrity, and UI components.

To execute the test suite:

```bash
pytest -v
```

### Verified Test Suite Invariants (136 Passing Tests):
- **Split Leakage Isolation**: Verifies with absolute certainty that zero customer IDs appear in more than one split partition (train/valid/test).
- **Mathematical Padding Invariance**: Proves that mutating padded steps in input sequences yields exactly `diff == 0.0` in neural model outputs.
- **Closed-Form Trajectory Validation**: Validates vectorized $\beta$ slope algorithms against single-statement edge cases ($n \le 1 \rightarrow 0.0$) with zero NaNs.
- **Metric Verification**: Validates normalized Gini ($2 \times \text{AUC} - 1$) and default capture rate at 4% against exact analytical solutions.
- **UI Contract Integrity**: Ensures all synthetic data loaders, Matplotlib risk gauges, profit curves, and SHAP charts render without exceptions.

---

## License & Ethical Use

Distributed under the **MIT License**. See `LICENSE` for details.

### Educational & Regulatory Notice
> **DISCLAIMER**: This software repository is designed strictly for educational, academic benchmarking, and research purposes. It is **not** certified for real-world automated credit decisions, credit line assignment, or consumer underwriting. In compliance with the Fair Housing Act and the Equal Credit Opportunity Act (ECOA), production credit underwriting models must undergo rigorous disparate impact assessments, fair lending audits, and model governance reviews before consumer deployment.
```
