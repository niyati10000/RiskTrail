# DESIGN_SYSTEM.md — Charts, Report & Demo Style

Applies to every figure, report page and the optional Streamlit demo. Consistency makes the report look professional and lets any AI agent produce matching output.

## Principles
- One message per chart; the title states the finding, not the chart type.
- Same model = same color in every figure.
- Never rely on color alone (add line styles/markers).
- Label axes with units; always show the split (valid/test) in the subtitle or caption.

## Palette
| Role | Hex |
|---|---|
| LightGBM | `#2563EB` (blue) |
| GRU/LSTM | `#16A34A` (green) |
| Transformer / attention GRU | `#9333EA` (purple) |
| Approve-everyone baseline | `#6B7280` (gray) |
| Good / non-default | `#0EA5E9` |
| Default / risk | `#DC2626` (red) |
| Highlight / optimal cutoff | `#F59E0B` (amber) |
| Background / grid | `#FFFFFF` / `#E5E7EB` |

## Typography & sizing
- Font: Inter or DejaVu Sans (matplotlib default fallback).
- Figure size 8x5 in for single plots, 12x5 for side-by-side; dpi 150 for report, 300 for print.
- Title 14 pt, axis labels 11 pt, ticks 9 pt.

## Standard figures (names used in the report)
1. `fig_class_balance` — default vs non-default counts.
2. `fig_missingness_by_step` — heatmap: feature groups x statement position.
3. `fig_model_comparison` — grouped bars: competition metric, ROC-AUC, PR-AUC per model.
4. `fig_calibration` — reliability diagram, one line per model + diagonal.
5. `fig_profit_by_cutoff` — profit vs cutoff, one line per model, approve-everyone dashed gray, optimal cutoff marked in amber.
6. `fig_shap_summary` — SHAP beeswarm for LightGBM, top 20 features.
7. `fig_attention_by_month` — mean attention weight per statement position (defaulters vs non-defaulters).
8. `fig_trend_segment` — model performance for deteriorating vs stable customers.

## Tables
- Model comparison table: rows = models, columns = competition metric, ROC-AUC, PR-AUC, Brier; bold the best per column.
- Round to 4 decimals for metrics, 0 decimals for currency.

## Report structure
1. Introduction & business problem
2. Data & preprocessing (A)
3. Tabular baseline: LightGBM (A)
4. Sequence models (B)
5. Evaluation & calibration (shared)
6. Business profit analysis (shared)
7. Explainability (shared)
8. Findings, limitations, future work
Each person writes their own sections; shared sections are written jointly and reviewed by both.

## Streamlit demo (optional)
- Layout: sidebar (cutoff slider, profit/loss inputs), main area with three tabs: **Score**, **Profit curve**, **Why this decision**.
- Score tab: risk probability gauge, approve/decline at the chosen cutoff.
- Why tab: top 5 SHAP reasons as a horizontal bar chart (red = raises risk, blue = lowers).
- Keep the same palette; no custom CSS beyond a max content width.
- Always show a note: "Demo on anonymized/sample data; not for real credit decisions."
