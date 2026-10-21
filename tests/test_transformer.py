"""Unit tests for Phase 4b: Transformer Sequence Model and Attention Pooling.

Validates:
1. Forward pass shapes and probability bounds.
2. Padding invariance (corrupting padded positions yields exactly 0.0 difference).
3. Attention pooling weight properties (sums to 1.0 on valid steps, strictly 0.0 on padding).
4. Backbone switching ("transformer" vs "gru" with attention pooling).
5. Pooling modes ("attention", "cls", "mean").
6. Checkpoint persistence and reload parity.
7. Prediction shapes, alignment with customer splits, and metrics contract.
8. Parameter count compactness (< 150k parameters).
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.transformer import (
    AttentionPooling,
    StatementTransformer,
    train_transformer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dummy_sequence_batch():
    """Generate reproducible synthetic sequence batch with front-padding."""
    B, T, F, C = 4, 13, 22, 4
    torch.manual_seed(42)

    # Varied customer statement lengths (1 to 13)
    lengths = torch.tensor([3, 13, 1, 7], dtype=torch.long)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, T - length :] = True

    x_num = torch.randn(B, T, F)
    x_cat = torch.randint(0, 3, (B, T, C))
    return x_num, x_cat, mask, lengths


def test_attention_pooling_standalone():
    """Verify AttentionPooling sums to 1.0 on valid positions and 0.0 on padded positions."""
    B, T, D = 3, 13, 64
    lengths = torch.tensor([4, 13, 1])
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, T - length :] = True

    h = torch.randn(B, T, D)
    pooler = AttentionPooling(d_model=D, attn_dim=32, dropout=0.0)
    pooler.eval()

    pooled, attn_weights = pooler(h, mask)

    assert pooled.shape == (B, D)
    assert attn_weights.shape == (B, T)

    # Weights on padded steps must be strictly 0.0
    for i in range(B):
        padded_steps = ~mask[i]
        if padded_steps.any():
            assert (attn_weights[i, padded_steps] == 0.0).all().item()

        # Valid steps must sum to 1.0
        valid_sum = attn_weights[i, mask[i]].sum().item()
        assert abs(valid_sum - 1.0) < 1e-6


def test_transformer_forward_shape(dummy_sequence_batch):
    """Verify StatementTransformer outputs binary logits [B] and probabilities in [0, 1]."""
    x_num, x_cat, mask, _ = dummy_sequence_batch
    B = x_num.shape[0]

    model = StatementTransformer(
        num_features=22,
        cat_cardinalities=[5, 4, 3, 7],
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        pooling="attention",
    )
    model.eval()

    with torch.no_grad():
        logits = model(x_num, x_cat, mask)
        probs = model.predict_proba(x_num, x_cat, mask)

    assert logits.shape == (B,)
    assert probs.shape == (B,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_transformer_padding_invariance(dummy_sequence_batch):
    """Verify modifying padded positions produces strictly 0.0 difference in predictions."""
    x_num, x_cat, mask, _ = dummy_sequence_batch

    model = StatementTransformer(
        num_features=22,
        cat_cardinalities=[5, 4, 3, 7],
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        pooling="attention",
    )
    model.eval()

    with torch.no_grad():
        preds_clean = model.predict_proba(x_num, x_cat, mask)

    # Corrupt padded positions with extreme numbers and out-of-range categories
    x_num_corrupted = x_num.clone()
    x_cat_corrupted = x_cat.clone()
    x_num_corrupted[~mask] = 99999.0
    x_cat_corrupted[~mask] = 5

    with torch.no_grad():
        preds_corrupted = model.predict_proba(x_num_corrupted, x_cat_corrupted, mask)

    diff = (preds_clean - preds_corrupted).abs().max().item()
    assert diff == 0.0, f"Padded positions leaked into prediction! Max diff: {diff}"


def test_transformer_attention_weights_exposure(dummy_sequence_batch):
    """Verify get_attention_weights exposes [B, 13] normalized weights for Phase 7 explainability."""
    x_num, x_cat, mask, _ = dummy_sequence_batch
    B, T = mask.shape

    model = StatementTransformer(
        num_features=22,
        cat_cardinalities=[5, 4, 3, 7],
        d_model=64,
        pooling="attention",
    )
    model.eval()

    with torch.no_grad():
        logits, attn_weights = model(x_num, x_cat, mask, return_attention=True)
        attn_direct = model.get_attention_weights(x_num, x_cat, mask)

    assert logits.shape == (B,)
    assert attn_weights.shape == (B, T)
    assert torch.allclose(attn_weights, attn_direct)

    # Padded steps have 0 attention
    assert (attn_weights[~mask] == 0.0).all().item()


@pytest.mark.parametrize("backbone", ["transformer", "gru", "lstm"])
def test_backbone_variants_and_padding_invariance(dummy_sequence_batch, backbone):
    """Verify attention pooling and padding invariance across transformer and recurrent backbones."""
    x_num, x_cat, mask, _ = dummy_sequence_batch

    model = StatementTransformer(
        num_features=22,
        cat_cardinalities=[5, 4, 3, 7],
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        backbone=backbone,
        pooling="attention",
        attn_dim=16,
        mlp_dims=[16],
    )
    model.eval()

    with torch.no_grad():
        preds_clean = model.predict_proba(x_num, x_cat, mask)

    # Corrupt padded positions
    x_num_corrupted = x_num.clone()
    x_cat_corrupted = x_cat.clone()
    x_num_corrupted[~mask] = -88888.0
    x_cat_corrupted[~mask] = 4

    with torch.no_grad():
        preds_corrupted = model.predict_proba(x_num_corrupted, x_cat_corrupted, mask)

    diff = (preds_clean - preds_corrupted).abs().max().item()
    assert diff == 0.0, f"Backbone '{backbone}' failed padding invariance test! Diff: {diff}"


@pytest.mark.parametrize("pooling", ["attention", "cls", "mean"])
def test_pooling_modes(dummy_sequence_batch, pooling):
    """Verify different pooling modes execute cleanly."""
    x_num, x_cat, mask, _ = dummy_sequence_batch
    B = x_num.shape[0]

    model = StatementTransformer(
        num_features=22,
        cat_cardinalities=[5, 4, 3, 7],
        d_model=32,
        nhead=2,
        num_layers=1,
        pooling=pooling,
        mlp_dims=[16],
    )
    model.eval()

    with torch.no_grad():
        probs = model.predict_proba(x_num, x_cat, mask)

    assert probs.shape == (B,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_transformer_parameter_count_modest():
    """Verify model parameter count is modest for credit sequences (< 150k params)."""
    model = StatementTransformer(
        num_features=22,
        cat_cardinalities=[5, 4, 3, 7],
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        pooling="attention",
        mlp_dims=[64, 32],
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params < 150_000, f"Parameter count too high: {total_params:,}"
    assert total_params > 10_000, f"Parameter count surprisingly low: {total_params:,}"


def test_transformer_checkpoint_reload(dummy_sequence_batch, tmp_path):
    """Verify saved model state_dict reloads and produces bitwise identical predictions."""
    x_num, x_cat, mask, _ = dummy_sequence_batch

    model = StatementTransformer(
        num_features=22,
        cat_cardinalities=[5, 4, 3, 7],
        d_model=32,
        nhead=2,
        num_layers=1,
    )
    model.eval()

    with torch.no_grad():
        preds_orig = model.predict_proba(x_num, x_cat, mask)

    ckpt_path = tmp_path / "test_transformer.pt"
    torch.save(model.state_dict(), ckpt_path)

    reloaded_model = StatementTransformer(
        num_features=22,
        cat_cardinalities=[5, 4, 3, 7],
        d_model=32,
        nhead=2,
        num_layers=1,
    )
    reloaded_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    reloaded_model.eval()

    with torch.no_grad():
        preds_reloaded = reloaded_model.predict_proba(x_num, x_cat, mask)

    assert torch.equal(preds_orig, preds_reloaded)


def test_transformer_saved_results_and_splits_alignment():
    """Verify persisted results/transformer/ files match shapes, splits, and valid metric contracts."""
    res_dir = PROJECT_ROOT / "results" / "transformer"
    assert (res_dir / "preds_valid.npy").exists()
    assert (res_dir / "preds_test.npy").exists()
    assert (res_dir / "metrics.json").exists()
    assert (res_dir / "config.json").exists()
    assert (res_dir / "model.pt").exists()

    preds_val = np.load(res_dir / "preds_valid.npy")
    preds_test = np.load(res_dir / "preds_test.npy")

    # Splits are 150 validation, 150 test
    assert preds_val.shape == (150,)
    assert preds_test.shape == (150,)
    assert (preds_val >= 0.0).all() and (preds_val <= 1.0).all()
    assert (preds_test >= 0.0).all() and (preds_test <= 1.0).all()

    with open(res_dir / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    for split in ["valid", "test"]:
        assert split in metrics
        m = metrics[split]
        assert "competition_metric" in m
        assert "normalized_gini" in m
        assert "capture_rate_at_4pct" in m
        assert "roc_auc" in m
        assert "pr_auc" in m
        assert "brier" in m
        assert m["competition_metric"] >= 0.5
        assert m["roc_auc"] >= 0.95
