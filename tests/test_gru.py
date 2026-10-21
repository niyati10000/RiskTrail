"""Unit tests for PyTorch recurrent sequence model (src/models/gru.py).

Validates:
1. PADDING INVARIANCE:
   Modifying or corrupting values at padded positions (where mask == False) produces
   strictly ZERO difference in predictions (100% mathematical padding invariance).
2. Saved artifacts in results/gru/:
   - preds_valid.npy [150] float32 in [0, 1]
   - preds_test.npy [150] float32 in [0, 1]
   - metrics.json
   - config.json
   - model.pt
3. Alignment with splits.parquet.
4. Architectural flexibility:
   - GRU and LSTM variants
   - Bidirectional recurrence
   - Masked pooling modes: 'all', 'masked_last', 'masked_mean', 'masked_max'
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.gru import MaskedGRUModel, SequenceDataset, load_sequence_split_data, train_gru

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_GRU_DIR = PROJECT_ROOT / "results" / "gru"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@pytest.fixture
def gru_artifacts():
    """Load persisted GRU model artifacts from results/gru/."""
    preds_val_path = RESULTS_GRU_DIR / "preds_valid.npy"
    preds_test_path = RESULTS_GRU_DIR / "preds_test.npy"
    metrics_path = RESULTS_GRU_DIR / "metrics.json"
    config_path = RESULTS_GRU_DIR / "config.json"
    model_path = RESULTS_GRU_DIR / "model.pt"

    assert preds_val_path.exists(), f"Missing {preds_val_path}"
    assert preds_test_path.exists(), f"Missing {preds_test_path}"
    assert metrics_path.exists(), f"Missing {metrics_path}"
    assert config_path.exists(), f"Missing {config_path}"
    assert model_path.exists(), f"Missing {model_path}"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return {
        "preds_valid": np.load(preds_val_path),
        "preds_test": np.load(preds_test_path),
        "metrics": metrics,
        "config": config,
        "model_path": model_path,
    }


def test_gru_padding_invariance():
    """Verify that corrupting padded positions produces strictly ZERO change in predictions."""
    torch.manual_seed(42)
    B, T, F, C = 8, 13, 22, 4
    cat_cards = [5, 4, 3, 7]

    model = MaskedGRUModel(
        num_features=F,
        cat_cardinalities=cat_cards,
        hidden_dim=32,
        num_layers=2,
        pooling="all",
    )
    model.eval()

    # Generate random sequences with variable statement lengths (1..13)
    lengths = torch.randint(1, 14, (B,))
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, T - length:] = True

    x_num = torch.randn(B, T, F)
    x_cat = torch.randint(0, 3, (B, T, C))

    # Clean predictions
    with torch.no_grad():
        preds_clean = model.predict_proba(x_num, x_cat, mask)

    # Corrupt padded positions with extreme numbers and noise
    x_num_corrupted = x_num.clone()
    x_cat_corrupted = x_cat.clone()
    x_num_corrupted[~mask] = 99999.0
    x_cat_corrupted[~mask] = 5

    with torch.no_grad():
        preds_corrupted = model.predict_proba(x_num_corrupted, x_cat_corrupted, mask)

    diff = (preds_clean - preds_corrupted).abs().max().item()
    assert diff == 0.0, f"Padded positions influenced model output! Max diff: {diff}"


@pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("pooling", ["all", "masked_last", "masked_mean", "masked_max"])
def test_gru_architectural_variants(rnn_type, bidirectional, pooling):
    """Verify forward pass and padding invariance across all recurrent configurations."""
    B, T, F, C = 4, 13, 22, 4
    cat_cards = [5, 4, 3, 7]

    model = MaskedGRUModel(
        num_features=F,
        cat_cardinalities=cat_cards,
        hidden_dim=16,
        num_layers=1,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
        pooling=pooling,
        mlp_dims=[16],
    )
    model.eval()

    lengths = torch.tensor([3, 13, 1, 7])
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, T - length:] = True

    x_num = torch.randn(B, T, F)
    x_cat = torch.randint(0, 3, (B, T, C))

    with torch.no_grad():
        probs = model.predict_proba(x_num, x_cat, mask)

    assert probs.shape == (B,)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
    assert not torch.isnan(probs).any()

    # Verify padding invariance on this configuration
    x_num_corrupt = x_num.clone()
    x_num_corrupt[~mask] = -8888.8
    with torch.no_grad():
        probs_corrupt = model.predict_proba(x_num_corrupt, x_cat, mask)

    assert torch.allclose(probs, probs_corrupt, atol=1e-6)


def test_gru_predictions_shape_and_range(gru_artifacts):
    """Verify saved prediction arrays in results/gru/ match split shapes and bounds."""
    splits_df = pd.read_parquet(PROCESSED_DIR / "splits.parquet")
    n_valid = (splits_df["split"] == "valid").sum()
    n_test = (splits_df["split"] == "test").sum()

    preds_val = gru_artifacts["preds_valid"]
    preds_test = gru_artifacts["preds_test"]

    assert preds_val.shape == (n_valid,) == (150,)
    assert preds_val.dtype == np.float32
    assert np.all(preds_val >= 0.0) and np.all(preds_val <= 1.0)
    assert not np.isnan(preds_val).any()

    assert preds_test.shape == (n_test,) == (150,)
    assert preds_test.dtype == np.float32
    assert np.all(preds_test >= 0.0) and np.all(preds_test <= 1.0)
    assert not np.isnan(preds_test).any()


def test_gru_metrics_sanity(gru_artifacts):
    """Verify that GRU achieves expected benchmark thresholds on test and validation splits."""
    metrics = gru_artifacts["metrics"]

    assert "valid" in metrics and "test" in metrics
    for split_key in ["valid", "test"]:
        m = metrics[split_key]
        assert m["competition_metric"] >= 0.50
        assert m["roc_auc"] >= 0.90
        assert m["normalized_gini"] >= 0.80
        assert m["brier"] <= 0.10


def test_gru_model_checkpoint_loadable(gru_artifacts):
    """Verify that model.pt can be deserialized and executed on a test batch."""
    model_path = gru_artifacts["model_path"]
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    config = gru_artifacts["config"]
    model = MaskedGRUModel(
        num_features=config.get("num_features", 22),
        cat_cardinalities=config.get("cat_cardinalities", [5, 4, 3, 7]),
        hidden_dim=config.get("hidden_dim", 64),
        num_layers=config.get("num_layers", 2),
        bidirectional=config.get("bidirectional", False),
        pooling=config.get("pooling", "all"),
        mlp_dims=config.get("mlp_dims", [64, 32]),
    )
    model.load_state_dict(state_dict)
    model.eval()

    x_num = torch.randn(2, 13, 22)
    x_cat = torch.zeros(2, 13, 4, dtype=torch.long)
    mask = torch.ones(2, 13, dtype=torch.bool)

    with torch.no_grad():
        out = model.predict_proba(x_num, x_cat, mask)
    assert out.shape == (2,)
    assert not torch.isnan(out).any()
