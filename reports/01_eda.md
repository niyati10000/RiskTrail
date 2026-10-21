# 01_eda.md — Phase 1: Data Audit & Exploratory Data Analysis

**Execution Date**: 2026-10-21  
**Author**: Agent / Owner A (Data & Feature Pipeline)  
**Dataset**: Credit Statement Sequence Benchmark (`data/raw/train_data.csv`, `data/raw/train_labels.csv`)

---

## 1. Executive Summary

| Dimension | Measured Value | Notes |
| :--- | :--- | :--- |
| **Unique Customers** | **1,000** | One unique row per customer in labels |
| **Total Statements** | **11,877** | Averaging 11.9 statements per customer |
| **Statements / Customer Range** | **[2, 13]** | **Strictly verified: max <= 13** |
| **Full History (13 Statements)** | **80.8%** | 808 customers have full 13 months |
| **Target Class Balance** | **24.4% Default** | 244 default (1) vs 756 non-default (0) |
| **Date Range** | **2017-03-12 to 2018-03-13** | Total duration: 366 days (~13 months) |
| **Total Features** | **26** | 4 Categoricals, 22 Numerics |

---

## 2. Target Class Balance

Credit default prediction exhibits inherent class imbalance:
- **Non-Default (0)**: 756 customers (75.6%)
- **Default (1)**: 244 customers (24.4%)
- **Class Ratio**: 1 : 3.10 (approx ~1:3)

![Target Class Balance](figures/fig_class_balance.png)

> **Modeling Decision**: Accuracy is an invalid metric under this class imbalance. As locked in `docs/BRAIN.md`, evaluation will strictly use the **competition metric** (0.5 * Gini + 0.5 * Capture Rate @ 4%), along with ROC-AUC and PR-AUC.

---

## 3. Sequence History Distribution

The maximum sequence length is strictly confirmed to be **13 statements**, matching the credit cycle design:

| Statement Count | Customer Count | Percentage |
| :---: | :---: | :---: |
| 2 | 14 | 1.4% |
| 3 | 15 | 1.5% |
| 4 | 18 | 1.8% |
| 5 | 16 | 1.6% |
| 6 | 23 | 2.3% |
| 7 | 12 | 1.2% |
| 8 | 16 | 1.6% |
| 9 | 27 | 2.7% |
| 10 | 22 | 2.2% |
| 11 | 13 | 1.3% |
| 12 | 16 | 1.6% |
| 13 | 808 | 80.8% |

![Statements per Customer](figures/fig_statements_per_customer.png)

> **Padding Decision**: Shorter histories (19.2% of customers) will be **front-padded (left-padded)** with a boolean mask `[N, 13]`, ensuring position index 12 is always the most recent statement date.

---

## 4. Column Schema & Categorical Cardinalities

The features span 5 banking credit domains: Delinquency (`D_*`), Spend (`S_*`), Payment (`P_*`), Balance (`B_*`), and Risk (`R_*`).

### Categorical Features (4)
| Column | Cardinality | Distinct Values / Notes |
| :--- | :---: | :--- |
| `D_63` | 5 | Sample: ['CR', 'XZ', 'CL', 'CO', 'XM'] |
| `D_64` | 4 | Sample: ['U', '-1', 'O', 'R'] |
| `B_30` | 3 | Sample: [np.float64(2.0), np.float64(0.0), np.float64(1.0)] |
| `B_38` | 7 | Sample: [np.float64(4.0), np.float64(2.0), np.float64(6.0), np.float64(1.0), np.float64(5.0), np.float64(3.0)] |

### Numerical Features (22)
`D_39, D_41, D_42, D_43, D_44, P_2, P_3, P_4, B_1, B_2, B_3, B_4, B_5, B_9, S_3, S_5, S_6, S_7, S_8, R_1, R_2, R_3`

All numerical features have been cast to `float32` in `data/interim/statements.parquet` to ensure lean memory usage.

---

## 5. Missingness Analysis

### Overall Missingness
| Feature | Missing Rate (%) | Notes |
| :--- | :---: | :--- |
| `D_42` | 59.2% | Feature with informative missingness |
| `D_43` | 40.0% | Feature with informative missingness |
| `S_7` | 30.5% | Feature with informative missingness |
| `S_3` | 20.1% | Feature with informative missingness |
| `P_3` | 15.3% | Feature with informative missingness |

### Temporal Missingness Across Statement Steps
![Missingness by Step](figures/fig_missingness_by_step.png)

> **Key Finding**: Missingness varies across feature groups and statement steps. As mandated in `docs/BRAIN.md`, missing indicators (`<col>__missing`) will be created in Phase 2 because absence of a statement or credit signal is predictive of default risk.

---

## 6. Date Range & Time-Aware Check Feasibility

- **Global Statement Range**: `2017-03-12` to `2018-03-13`
- **Customer Final Statement Range**: `2018-03-07` to `2018-03-13`

![Temporal Distribution](figures/fig_temporal_distribution.png)

### Feasibility Assessment: Time-Aware Holdout
- **Finding**: Most customers have their final observation in the terminal months of the observation window (early 2018).
- **Feasibility Verdict**: **Feasible with Caveats**.
  - A pure out-of-time (OOT) holdout can be constructed by reserving customers whose final statement occurs after a chosen cutoff date (e.g. 2018-03-01).
  - However, because credit performance is measured at a fixed calendar maturity horizon (e.g., performance over subsequent 120 days), strict customer-level stratified splitting (`splits.parquet`) must remain the primary benchmark to avoid introducing artificial macroeconomic distribution shifts between train and test.
  - **Decision**: Primary splits will be **customer-stratified 70/15/15**. An optional `time_holdout` boolean column will be added in `splits.parquet` in Phase 2 for validation.

---

## 7. Exit Criteria Verification (Phase 1)

- [x] Converted raw CSVs to interim Parquet with `float32` and `category` dtypes (`src/data/load.py`).
- [x] Generated standard figures (`fig_class_balance.png`, `fig_statements_per_customer.png`, `fig_missingness_by_step.png`, `fig_temporal_distribution.png`) adhering to `docs/DESIGN_SYSTEM.md`.
- [x] Saved classified column registry to `configs/columns.yaml`.
- [x] Documented time-aware check feasibility and findings in `reports/01_eda.md`.
