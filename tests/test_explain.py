"""Unit tests for Phase 7: Model Explainability, SHAP TreeExplainer, and Sequence Attention.

Validates:
1. Feature description dictionary and aggregation mapping.
2. LightGBM TreeExplainer SHAP values calculation and shape alignment.
3. Adverse action top-5 reason code extraction and compliance notice formatting.
4. StatementTransformer attention weight properties across statement positions.
5. Permutation importance by feature group (D, B, P, S, R) and by time step.
6. Complete worked case study generation for declined applicants.
7. Publication figures and reports/07_explainability.md integrity.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.explain.seq_importance import (
    compute_permutation_importance,
    compute_temporal_attention_weights,
    generate_worked_case_study,
    load_test_sequence_tensors,
    load_trained_transformer,
    run_sequence_importance_pipeline,
)
from src.explain.shap_lgbm import (
    compute_shap_explanations,
    describe_feature,
    format_adverse_action_notice,
    get_customer_reason_codes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_describe_feature():
    """Verify human-readable feature descriptions map base and aggregation components."""
    desc_last = describe_feature("D_39__last")
    assert "Delinquency" in desc_last
    assert "Most recent" in desc_last

    desc_mean = describe_feature("P_2__mean")
    assert "Payment" in desc_mean
    assert "average" in desc_mean

    desc_slope = describe_feature("B_1__slope")
    assert "Balance" in desc_slope
    assert "slope" in desc_slope or "trajectory" in desc_slope


def test_compute_shap_explanations():
    """Verify LightGBM TreeExplainer SHAP values match test split dimensions."""
    shap_vals, X_test, feature_cols = compute_shap_explanations()
    assert isinstance(shap_vals, np.ndarray)
    assert shap_vals.shape == (150, 214)
    assert len(X_test) == 150
    assert len(feature_cols) == 214


def test_get_customer_reason_codes_and_notice():
    """Verify adverse action extraction returns top-5 positive risk drivers and formatted notice."""
    cid = "e4627d06d97d302f742070f63d17b414"
    reasons = get_customer_reason_codes(cid, top_k=5)

    assert len(reasons) == 5
    # Verify ranked descending by SHAP impact
    for i in range(len(reasons) - 1):
        assert reasons[i]["shap_impact"] >= reasons[i + 1]["shap_impact"]
        assert reasons[i]["shap_impact"] > 0.0

    notice = format_adverse_action_notice(
        customer_id=cid,
        risk_score=0.9999,
        cutoff=0.05,
        reason_codes=reasons,
    )
    assert cid in notice
    assert "DECLINED" in notice
    assert reasons[0]["feature"] in notice


def test_compute_temporal_attention_weights():
    """Verify StatementTransformer outputs [150, 13] normalized statement attention weights."""
    x_seq, cat_seq, mask, y_test, test_cids = load_test_sequence_tensors()
    model = load_trained_transformer()

    attn = compute_temporal_attention_weights(model, x_seq, cat_seq, mask)
    assert attn.shape == (150, 13)

    # Weights on padded steps must be strictly 0.0
    for i in range(150):
        pad = ~mask[i].numpy()
        if pad.any():
            assert (attn[i, pad] == 0.0).all()
        # Valid steps sum to 1.0
        assert abs(attn[i, mask[i].numpy()].sum() - 1.0) < 1e-5


def test_permutation_importance_structure():
    """Verify permutation importance computes metric drops across feature groups and steps."""
    x_seq, cat_seq, mask, y_test, test_cids = load_test_sequence_tensors()
    model = load_trained_transformer()

    res = compute_permutation_importance(model, x_seq, cat_seq, mask, y_test, n_repeats=2)
    assert "base_competition_metric" in res
    assert "feature_group_importance" in res
    assert "temporal_step_importance" in res

    g_imp = res["feature_group_importance"]
    for expected_g in ["D (Delinquency)", "B (Balance)", "P (Payment)", "S (Spend)", "R (Risk)"]:
        assert expected_g in g_imp
        assert g_imp[expected_g] >= 0.0

    t_imp = res["temporal_step_importance"]
    assert len(t_imp) == 13


def test_worked_case_study_structure():
    """Verify generate_worked_case_study returns aligned dual-lens explainability record."""
    cid = "e4627d06d97d302f742070f63d17b414"
    study = generate_worked_case_study(customer_id=cid)

    assert study["customer_id"] == cid
    assert study["ground_truth_label"] == 1
    assert study["lightgbm_predicted_risk"] > 0.95
    assert len(study["top_5_shap_reasons"]) == 5
    assert len(study["statement_attention_timeline"]) == 13


def test_explainability_pipeline_and_artifacts():
    """Verify run_sequence_importance_pipeline generates all figures and report."""
    res = run_sequence_importance_pipeline()
    assert res is not None

    fig_shap = PROJECT_ROOT / "reports" / "figures" / "fig_shap_summary.png"
    fig_attn = PROJECT_ROOT / "reports" / "figures" / "fig_attention_by_month.png"
    report_file = PROJECT_ROOT / "reports" / "07_explainability.md"

    assert fig_shap.exists() and fig_shap.stat().st_size > 5000
    assert fig_attn.exists() and fig_attn.stat().st_size > 5000
    assert report_file.exists()

    content = report_file.read_text(encoding="utf-8")
    assert "Model Explainability & Adverse Action Analysis" in content
    assert "TreeExplainer" in content
    assert "Attention Allocation" in content
    assert "Attention is Not Explanation" in content
