"""Tests for RiskTrail AI Copilot Multi-Provider Engine."""

import pytest
from app.copilot import (
    build_system_prompt,
    query_local_copilot,
    query_gemini_api,
    query_groq_api,
)

SAMPLE_CONTEXT = {
    "customer_id": "SYNTH_0045",
    "risk_prob": 0.8523,
    "cutoff": 0.05,
    "is_approved": False,
    "ground_truth": 1,
    "profit_good": 100.0,
    "loss_default": 500.0,
    "reasons": [
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
    ],
}


def test_build_system_prompt():
    """Verify system prompt contains customer ID, risk score, and SHAP factors."""
    prompt = build_system_prompt(SAMPLE_CONTEXT)
    assert "SYNTH_0045" in prompt
    assert "85.23%" in prompt
    assert "DECLINED" in prompt
    assert "D_39__slope" in prompt
    assert "ECOA" in prompt


def test_query_local_copilot_why_declined():
    """Verify local copilot explains adverse underwriting decisions accurately."""
    resp = query_local_copilot("Why was this applicant declined?", SAMPLE_CONTEXT)
    assert "DECLINED" in resp
    assert "SYNTH_0045" in resp
    assert "85.23%" in resp
    assert "D_39__slope" in resp


def test_query_local_copilot_how_to_improve():
    """Verify local copilot provides actionable credit repair suggestions."""
    resp = query_local_copilot("How can the borrower improve their credit score?", SAMPLE_CONTEXT)
    assert "Actionable Steps" in resp
    assert "Delinquenc" in resp
    assert "SYNTH_0045" in resp


def test_query_local_copilot_profit_economics():
    """Verify local copilot explains profit curve and cutoff threshold mathematics."""
    resp = query_local_copilot("What cutoff maximizes profit?", SAMPLE_CONTEXT)
    assert "Profit" in resp
    assert "$100" in resp
    assert "$500" in resp


def test_query_local_copilot_feature_dictionary():
    """Verify local copilot explains D_39 and slope concepts."""
    resp = query_local_copilot("Explain D_39 slope and B_1", SAMPLE_CONTEXT)
    assert "D_39" in resp
    assert "Delinquency" in resp
    assert "slope" in resp


def test_query_local_copilot_regulatory_compliance():
    """Verify local copilot explains ECOA, FCRA, and CFPB compliance rules."""
    resp = query_local_copilot("What does ECOA and FCRA say about adverse action?", SAMPLE_CONTEXT)
    assert "Equal Credit Opportunity Act" in resp
    assert "Regulation B" in resp
    assert "FCRA" in resp


def test_query_gemini_missing_key():
    """Verify Gemini API handler warns when key is missing."""
    resp = query_gemini_api("", "Why was this customer declined?", SAMPLE_CONTEXT)
    assert "Gemini API Key Missing" in resp


def test_query_groq_missing_key():
    """Verify Groq API handler warns when key is missing."""
    resp = query_groq_api("", "Why was this customer declined?", SAMPLE_CONTEXT)
    assert "Groq API Key Missing" in resp
