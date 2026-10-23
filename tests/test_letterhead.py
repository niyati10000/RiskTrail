"""Tests for RiskTrail Official Bank Letterhead Document Replica."""

import pytest
from app.letterhead import generate_bank_letter_html, generate_bank_letter_text

SAMPLE_REASONS = [
    {
        "rank": 1,
        "feature": "D_39__slope",
        "feature_value": 0.8124,
        "shap_impact": 0.4521,
        "adverse_action_reason": "Delinquency: Days past due (Monthly slope)",
    },
    {
        "rank": 2,
        "feature": "B_1__last",
        "feature_value": 0.6543,
        "shap_impact": 0.3211,
        "adverse_action_reason": "Balance: Total revolving credit balance",
    },
]


def test_generate_bank_letter_html_declined():
    """Verify HTML output contains formal bank stationery elements for declined application."""
    html = generate_bank_letter_html(
        customer_id="SYNTH_0045",
        risk_prob=0.8523,
        cutoff=0.05,
        is_approved=False,
        reason_codes=SAMPLE_REASONS,
    )
    assert "RISKTRAIL FINANCIAL SERVICES CORP." in html
    assert "FORMAL DETERMINATION: APPLICATION DECLINED" in html
    assert "ADV-2026-SYNTH_0045" in html
    assert "85.23%" in html
    assert "Equal Credit Opportunity Act" in html
    assert "Fair Credit Reporting Act" in html
    assert "Sarah Jenkins, CFA" in html
    assert "D_39__slope" in html


def test_generate_bank_letter_html_approved():
    """Verify HTML output contains approval notice for approved application."""
    html = generate_bank_letter_html(
        customer_id="SYNTH_0010",
        risk_prob=0.0125,
        cutoff=0.05,
        is_approved=True,
        reason_codes=SAMPLE_REASONS,
    )
    assert "RISKTRAIL FINANCIAL SERVICES CORP." in html
    assert "FORMAL DETERMINATION: APPLICATION APPROVED" in html
    assert "1.25%" in html


def test_generate_bank_letter_text():
    """Verify text output contains complete printable disclosure."""
    text = generate_bank_letter_text(
        customer_id="SYNTH_0045",
        risk_prob=0.8523,
        cutoff=0.05,
        is_approved=False,
        reason_codes=SAMPLE_REASONS,
    )
    assert "RISKTRAIL FINANCIAL SERVICES CORP." in text
    assert "DECLINED" in text
    assert "ADV-2026-SYNTH_0045" in text
    assert "ECOA § 1002.9" in text
    assert "FCRA" in text
