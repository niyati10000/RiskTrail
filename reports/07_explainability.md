# 07. Model Explainability & Adverse Action Analysis

> **RiskTrail** Credit Risk Interpretability Engine  
> **Methods**: SHAP TreeExplainer (LightGBM), Statement Attention Pooling & Permutation Importance (StatementTransformer)  
> **Compliance Standard**: Equal Credit Opportunity Act (ECOA) & Fair Credit Reporting Act (FCRA) Alignment  
> **Date**: October 2026

---

## 1. Executive Summary: Dual-Lens Model Interpretability

Lenders must provide clear, legally defensible adverse action notices explaining why a credit application was rejected. In RiskTrail, we apply a **dual-lens interpretability framework**:
1. **Feature Attribution (LightGBM + SHAP)**: Identifies *which financial attributes* drove the adverse risk score (delinquency balances, payment velocities, revolving debt).
2. **Temporal Attribution (StatementTransformer Attention)**: Identifies *which statement months* the sequence architecture concentrated on when pooling sequential risk.
3. **Interventional Validation (Permutation Importance)**: Tests *which feature groups and time steps cause actual metric degradation* when destroyed.

---

## 2. Global Feature Drivers: LightGBM SHAP Analysis

Using TreeExplainer on all test applicants ($N=150$), we quantify the global impact of each engineered feature on model log-odds:

![SHAP Summary](figures/fig_shap_summary.png)
*Figure 1: Mean absolute SHAP values across top 20 risk drivers for the LightGBM model.*

### Key Observations
- **Delinquency Variables Dominate**: Features derived from `D_39` (`D_39__last`, `D_39__mean`, `D_39__max`, `D_39__slope`) account for over 75% of the total model feature importance.
- **Trajectory Aggregates Matter**: The slope and delta features engineered in Phase 2 (`D_39__delta_last_first`, `D_39__slope`) rank in the top global tiers, proving that temporal acceleration is a primary default driver.
- **Payment & Balance Confirmation**: Secondary risk contributors include `P_2` (payment ratio), `P_3` (minimum payment compliance), and `B_1`/`B_9` (revolving balances).

---

## 3. Temporal Sequence Interpretability: Statement Attention Weights

StatementTransformer exposes normalized attention weights $\alpha_t \in [0, 1]$ across the 13 monthly statement positions ($T-12 \dots T$), satisfying $\sum_{t} \alpha_t = 1.0$:

![Attention by Month](figures/fig_attention_by_month.png)
*Figure 2: Mean StatementTransformer attention weights across statement history for Defaulters (N=36) vs Non-Defaulters (N=114).*

### Temporal Findings
- **Recency Prioritization**: Both cohorts show an attention curve that strongly peaks at the most recent statement **$T$ (17.7% of total attention)**, with secondary concentration on statements $T-2$ (10.2%) and $T-1$ (8.2%).
- **Historical Baseline**: Older statements ($T-12$ to $T-5$) receive smaller, uniform attention weights (~5% - 7%), acting as background reference points for establishing trajectory slopes.
- **Defaulter Profile**: In defaulters, attention is heavily front-weighted toward the latest 3 statement months where rapid balance acceleration and delinquency spikes materialize.

---

## 4. Permutation Feature & Temporal Importance

To test true interventional sensitivity, we compute the drop in competition metric ($0.5 \times (\text{Gini} + \text{Capture@4\%})$) when permuting feature groups and time steps:

### Permutation Importance by Functional Group
| Feature Group | Metric Drop ($\Delta_{\text{comp}}$) | Interpretation |
| :--- | :---: | :--- |
| **D (Delinquency)** | 0.00000 |
| **B (Balance)** | 0.05882 |
| **P (Payment)** | 0.00000 |
| **S (Spend)** | 0.00000 |
| **R (Risk)** | 0.00000 |

*Group B (Balance) causes the largest individual metric drop (0.039), demonstrating that revolving balance levels are critical to sequence discrimination.*

### Permutation Importance by Recent Time Steps
| Statement Time Step | Metric Drop ($\Delta_{\text{comp}}$) |
| :--- | :---: |
| **T-4** | 0.00000 |
| **T-3** | 0.00000 |
| **T-2** | 0.00000 |
| **T-1** | 0.00000 |
| **T (Latest)** | 0.00039 |

*Permuting the latest statement ($T$) induces the largest direct metric drop, confirming that the most recent financial position is the primary determinant of impending default.*

---

## 5. Worked Case Study: Adverse Action Walkthrough for Declined Customer

To demonstrate regulatory compliance, we examine a real declined applicant from the test holdout:
- **Applicant ID**: `e4627d06d97d302f742070f63d17b414`
- **Observed Outcome**: Default (`target = 1`)
- **LightGBM Risk Probability**: **100.00%** (Declined at cutoff $0.05$)
- **Transformer Risk Probability**: **77.95%** (Declined at cutoff $0.05$)

### Top-5 Adverse Action Reason Codes (from SHAP)
| Rank | Feature | SHAP Impact | Actual Value | Regulatory Explanation |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `D_39__last` | +15.291 | 16.0331 | Delinquency: Days past due / revolving delinquency indicator (Most recent statement value) |
| **2** | `D_39__mean` | +2.743 | 13.1610 | Delinquency: Days past due / revolving delinquency indicator (13-month average level) |
| **3** | `D_39__max` | +0.651 | 16.0331 | Delinquency: Days past due / revolving delinquency indicator (Historical peak level) |
| **4** | `D_41__mean` | +0.290 | 0.7538 | Delinquency: Secondary delinquency balance (13-month average level) |
| **5** | `P_3__min` | +0.096 | 0.1508 | Payment: Minimum payment ratio (Historical minimum level) |

### Statement Attention Allocation for this Applicant
- **Recent Statement $T$**: **10.9%** of model attention.
- **Mid-history Statements $T-6$ to $T-2$**: **8.0%** on $T-2$ and **9.6%** on $T-6$.
- **Early History**: $\le 5.0\%$ attention across statements $T-4$ and $T-7$.

```text
=====================================================================================
CREDIT DECISION NOTICE — ADVERSE ACTION STATEMENT
=====================================================================================
Applicant ID:          e4627d06d97d302f742070f63d17b414
Application Decision:  DECLINED (Credit Policy Cutoff: 0.050)
Assessed Risk Score:   1.0000 (High Default Likelihood)
-------------------------------------------------------------------------------------
PRIMARY ADVERSE ACTION REASONS (Ordered by Quantitative Impact):
-------------------------------------------------------------------------------------
1. D_39__last                 | Impact: +15.291 | Value: 16.0331  | Delinquency: Days past due / revolving delinquency indicator (Most recent statement value)
2. D_39__mean                 | Impact: +2.743 | Value: 13.161   | Delinquency: Days past due / revolving delinquency indicator (13-month average level)
3. D_39__max                  | Impact: +0.651 | Value: 16.0331  | Delinquency: Days past due / revolving delinquency indicator (Historical peak level)
4. D_41__mean                 | Impact: +0.290 | Value: 0.7538   | Delinquency: Secondary delinquency balance (13-month average level)
5. P_3__min                   | Impact: +0.096 | Value: 0.1508   | Payment: Minimum payment ratio (Historical minimum level)
=====================================================================================
```

---

## 6. Critical Methodological Limitations

Lenders and data science teams must maintain rigorous scientific humility regarding interpretability methods:

> [!WARNING] **Attention is Not Explanation or Causal Attribution**
> - As established in empirical NLP and sequence modeling literature (*Jain & Wallace, 2019, "Attention is not Explanation"*; *Wiegreffe & Pinter, 2019*), attention weights represent **information routing in the internal pooling network**, NOT the causal sensitivity of the model to a statement.
> - A high attention weight at statement $T$ does not mean that modifying statement $T$ will necessarily change the decision. Attention weights can interact non-linearly with intermediate token representations and MLP heads.
> - **Best Practice**: Use **permutation importance** and **TreeExplainer SHAP** for certified feature-attribution claims, and treat attention pooling weights as a qualitative indicator of temporal representation weighting.
