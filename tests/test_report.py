"""Unit tests for Phase 8a: Final Report Skeleton and Artifact Verification.

Validates:
1. Existence and integrity of reports/FINAL_REPORT.md.
2. Presence of all 8 core report sections from docs/DESIGN_SYSTEM.md.
3. Embedded figure references for all 10 standard publication figures.
4. Retention of ownership placeholders [A writes] and [B writes].
5. Limitations section covering data license, assumed profit/loss, and fairness audits.
6. Future work roadmap.
"""

from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_final_report_structure_and_sections():
    """Verify FINAL_REPORT.md exists and contains all 8 required sections."""
    report_file = PROJECT_ROOT / "reports" / "FINAL_REPORT.md"
    assert report_file.exists(), "reports/FINAL_REPORT.md does not exist!"

    content = report_file.read_text(encoding="utf-8")

    expected_sections = [
        "1. Introduction & Business Problem",
        "2. Data & Preprocessing (Owner A)",
        "3. Tabular Baseline: LightGBM (Owner A)",
        "4. Sequence Models: Masked GRU & StatementTransformer (Owner B)",
        "5. Cross-Model Evaluation & Calibration (Shared)",
        "6. Business Profit Simulation & Policy Optimization (Shared)",
        "7. Explainability & Adverse Action Analysis (Shared)",
        "8. Findings, Limitations & Future Work (Shared)",
    ]

    for sec in expected_sections:
        assert sec in content, f"Missing section: {sec}"


def test_final_report_placeholders():
    """Verify ownership placeholders [A writes] and [B writes] are present."""
    report_file = PROJECT_ROOT / "reports" / "FINAL_REPORT.md"
    content = report_file.read_text(encoding="utf-8")

    assert "[A writes" in content, "Missing [A writes] placeholder!"
    assert "[B writes" in content, "Missing [B writes] placeholder!"


def test_final_report_embedded_figures_and_tables():
    """Verify all key standard figures and comparison tables are embedded."""
    report_file = PROJECT_ROOT / "reports" / "FINAL_REPORT.md"
    content = report_file.read_text(encoding="utf-8")

    figures = [
        "figures/fig_class_balance.png",
        "figures/fig_statements_per_customer.png",
        "figures/fig_missingness_by_step.png",
        "figures/fig_temporal_distribution.png",
        "figures/fig_model_comparison.png",
        "figures/fig_calibration.png",
        "figures/fig_trend_segment.png",
        "figures/fig_profit_by_cutoff.png",
        "figures/fig_shap_summary.png",
        "figures/fig_attention_by_month.png",
    ]

    for fig in figures:
        assert fig in content, f"Missing figure link: {fig}"

    assert "Competition Metric" in content
    assert "Optimal Cutoff" in content
    assert "Loss-to-Profit Sensitivity Grid" in content


def test_final_report_limitations_and_future_work():
    """Verify presence of limitations and future work discussion."""
    report_file = PROJECT_ROOT / "reports" / "FINAL_REPORT.md"
    content = report_file.read_text(encoding="utf-8")

    assert "Limitations" in content
    assert "Data License" in content or "Synthetic Benchmark" in content
    assert "Profit and Loss Coefficients" in content or "Assumed" in content
    assert "Fairness" in content or "Reg B" in content or "ECOA" in content
    assert "Future Work" in content
