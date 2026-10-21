"""GRU / LSTM sequence model architecture, training, and inference.

Processes variable-length sequence tensors [N, 13, F] with boolean padding masks.
Architecture pipeline:
1. Categorical embeddings + continuous feature projection
2. Configurable recurrent backbone: (Bi)GRU or LSTM
3. Strict mask-aware pooling:
   - Last valid statement state
   - Masked mean pooling across valid steps
   - Masked max pooling across valid steps
   - Guarantees zero leakage from padding steps (modifying padded values produces identical outputs)
4. Multi-layer perceptron (MLP) -> Sigmoid classification head

Training:
- AdamW optimizer with CosineAnnealingLR / OneCycleLR
- Weighted BCE loss (pos_weight computed from train split)
- Gradient norm clipping
- Early stopping on the primary competition metric
- Fixed seeds for full reproducibility
- Outputs saved to results/gru/: preds_valid.npy, preds_test.npy, metrics.json, config.json, model.pt
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from src.eval.metrics import competition_metric, compute_all_metrics
from src.utils.io import load_yaml, save_json
from src.utils.logging import get_logger
from src.utils.seed import seed_everything

logger = get_logger("RiskTrail.Models.GRU")


class MaskedGRUModel(nn.Module):
    """Deep Recurrent Sequence Model with mask-aware pooling for credit default prediction."""

    def __init__(
        self,
        num_features: int = 22,
        cat_cardinalities: Sequence[int] = (5, 4, 3, 7),
        emb_dim: int = 4,
        proj_dim: int = 48,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        rnn_type: str = "gru",
        bidirectional: bool = False,
        pooling: str = "all",
        mlp_dims: Sequence[int] = (64, 32),
    ):
        super().__init__()
        self.num_features = num_features
        self.cat_cardinalities = list(cat_cardinalities)
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.pooling = pooling.lower()

        # Categorical embeddings (cardinality + 1 to safely handle category codes and padding index 0)
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, emb_dim, padding_idx=0)
            for card in self.cat_cardinalities
        ])
        total_emb_dim = len(self.cat_cardinalities) * emb_dim

        # Numeric projection
        self.num_proj = nn.Sequential(
            nn.Linear(num_features, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Recurrent layer: (Bi)GRU or LSTM
        rnn_in_dim = proj_dim + total_emb_dim
        RNNClass = nn.GRU if self.rnn_type == "gru" else nn.LSTM
        self.rnn = RNNClass(
            input_size=rnn_in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Direction multiplier
        d_mult = 2 if bidirectional else 1
        rnn_out_dim = hidden_dim * d_mult

        # Determine pooled feature dimension
        if self.pooling in ["all", "concat", "masked_all"]:
            pooled_dim = rnn_out_dim * 3  # last + mean + max
        else:
            pooled_dim = rnn_out_dim

        # MLP classification head
        layers: List[nn.Module] = []
        curr_dim = pooled_dim
        for h_dim in mlp_dims:
            layers.extend([
                nn.Linear(curr_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def _left_align_valid_steps(
        self, x: torch.Tensor, lengths: torch.Tensor, max_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align valid statement steps to index 0..length-1 for exact recurrence from h_0=0.

        Guarantees that padded positions are completely unread by the recurrent network.
        """
        B, T, F = x.shape
        t_idx = torch.arange(T, device=x.device)
        # In front-padding, valid steps start at (T - lengths)
        gather_indices = (T - lengths).unsqueeze(1) + t_idx.unsqueeze(0)
        gather_indices = torch.clamp(gather_indices, 0, T - 1)

        x_left = torch.gather(x, 1, gather_indices.unsqueeze(-1).expand(-1, -1, F))
        left_mask = t_idx.unsqueeze(0) < lengths.unsqueeze(1)
        x_left = x_left * left_mask.unsqueeze(-1).float()
        return x_left, left_mask

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass computing raw classification logits.

        Args:
            x_num: Continuous features tensor [B, T, F].
            x_cat: Categorical features tensor [B, T, C].
            mask: Boolean padding mask [B, T] (True = real, False = padding).

        Returns:
            torch.Tensor: Binary logits [B].
        """
        B, T, _ = x_num.shape
        lengths = mask.sum(dim=1).clamp(min=1)

        # Defensively sanitize padded inputs so padding can NEVER cause out-of-bounds embedding lookups
        mask_3d = mask.unsqueeze(-1)
        x_num_clean = torch.where(mask_3d, x_num, torch.zeros_like(x_num))
        x_cat_clean = torch.where(mask_3d, x_cat, torch.zeros_like(x_cat))

        # 1. Project numeric features
        num_feat = self.num_proj(x_num_clean)

        # 2. Embed categoricals
        cat_embs = [
            self.embeddings[i](x_cat_clean[:, :, i].clamp(min=0))
            for i in range(len(self.embeddings))
        ]
        cat_feat = torch.cat(cat_embs, dim=-1)

        # 3. Concatenate feature representations
        x_combined = torch.cat([num_feat, cat_feat], dim=-1)

        # 4. Left-align valid steps to completely eliminate padding influence
        x_left, left_mask = self._left_align_valid_steps(x_combined, lengths, T)

        # 5. Pack sequence for recurrent execution
        packed_in = pack_padded_sequence(
            x_left, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed_in)
        out_left, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)

        D = out_left.shape[-1]

        # 6. Mask-aware pooling
        # Last valid state
        last_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, D)
        last_state = out_left.gather(1, last_idx).squeeze(1)

        # Masked mean state
        mean_state = (out_left * left_mask.unsqueeze(-1).float()).sum(dim=1) / lengths.unsqueeze(1).float()

        # Masked max state
        out_masked = out_left.masked_fill(~left_mask.unsqueeze(-1), -1e9)
        max_state = out_masked.max(dim=1)[0]

        if self.pooling in ["all", "concat", "masked_all"]:
            pooled = torch.cat([last_state, mean_state, max_state], dim=-1)
        elif self.pooling == "masked_mean":
            pooled = mean_state
        elif self.pooling == "masked_max":
            pooled = max_state
        else:  # default "masked_last"
            pooled = last_state

        # 7. Classification head
        logits = self.mlp(pooled).squeeze(-1)
        return logits

    def predict_proba(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute default prediction probabilities in [0, 1]."""
        with torch.no_grad():
            logits = self.forward(x_num, x_cat, mask)
            return torch.sigmoid(logits)


class SequenceDataset(Dataset):
    """PyTorch Dataset yielding continuous tensors, categorical indices, masks, and labels."""

    def __init__(
        self,
        x_num: np.ndarray,
        x_cat: np.ndarray,
        mask: np.ndarray,
        y: Optional[np.ndarray] = None,
    ):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self) -> int:
        return len(self.x_num)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.y is not None:
            return self.x_num[idx], self.x_cat[idx], self.mask[idx], self.y[idx]
        return self.x_num[idx], self.x_cat[idx], self.mask[idx]


def load_sequence_split_data(
    data_dir: Path,
) -> Tuple[
    SequenceDataset, SequenceDataset, SequenceDataset,
    np.ndarray, np.ndarray,
    List[str], List[str],
    Dict[str, Any],
]:
    """Load pre-processed sequence arrays and partition by customer splits."""
    x_seq = np.load(data_dir / "X_seq.npy")
    cat_seq = np.load(data_dir / "cat_seq.npy")
    mask = np.load(data_dir / "mask.npy")
    y = np.load(data_dir / "y.npy")
    ids = np.load(data_dir / "ids.npy", allow_pickle=True)

    splits_df = pd.read_parquet(data_dir / "splits.parquet")
    split_id_col = "customer_ID" if "customer_ID" in splits_df.columns else "customer_id"

    # Index mapping for customer order
    cid_to_idx = {cid: idx for idx, cid in enumerate(ids)}

    train_cids = splits_df[splits_df["split"] == "train"][split_id_col].tolist()
    valid_cids = splits_df[splits_df["split"] == "valid"][split_id_col].tolist()
    test_cids = splits_df[splits_df["split"] == "test"][split_id_col].tolist()

    train_idx = [cid_to_idx[cid] for cid in train_cids]
    valid_idx = [cid_to_idx[cid] for cid in valid_cids]
    test_idx = [cid_to_idx[cid] for cid in test_cids]

    train_ds = SequenceDataset(x_seq[train_idx], cat_seq[train_idx], mask[train_idx], y[train_idx])
    valid_ds = SequenceDataset(x_seq[valid_idx], cat_seq[valid_idx], mask[valid_idx], y[valid_idx])
    test_ds = SequenceDataset(x_seq[test_idx], cat_seq[test_idx], mask[test_idx], y[test_idx])

    y_valid = y[valid_idx]
    y_test = y[test_idx]

    # Cardinalities from cat_seq (number of unique category codes)
    cat_cards = [int(cat_seq[:, :, c].max()) + 1 for c in range(cat_seq.shape[-1])]

    metadata = {
        "num_features": x_seq.shape[-1],
        "cat_cardinalities": cat_cards,
        "n_train": len(train_idx),
        "n_valid": len(valid_idx),
        "n_test": len(test_idx),
    }

    return train_ds, valid_ds, test_ds, y_valid, y_test, valid_cids, test_cids, metadata


def train_gru(
    config: Optional[Dict[str, Any]] = None,
    data_dir: Union[str, Path] = PROJECT_ROOT / "data" / "processed",
    output_dir: Union[str, Path] = PROJECT_ROOT / "results" / "gru",
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Train PyTorch MaskedGRUModel with early stopping and log evaluation metrics.

    Args:
        config: Model hyperparameter configuration dictionary or None (loads configs/gru.yaml).
        data_dir: Directory containing X_seq.npy, cat_seq.npy, mask.npy, y.npy, splits.parquet.
        output_dir: Directory to save results/gru/ artifacts.
        seed: Random seed.
        device: 'cpu' or 'cuda'. If None, auto-selects.

    Returns:
        Dict[str, Any]: Validation and test metric scores and run summary.
    """
    seed_everything(seed)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    # Load configuration
    if config is None:
        config_path = PROJECT_ROOT / "configs" / "gru.yaml"
        cfg = load_yaml(config_path) if config_path.exists() else {}
    else:
        cfg = config.copy()

    training_cfg = cfg.get("training", {})
    batch_size = training_cfg.get("batch_size", 64)
    lr = training_cfg.get("learning_rate", 0.001)
    weight_decay = training_cfg.get("weight_decay", 0.01)
    max_epochs = training_cfg.get("max_epochs", 35)
    patience = training_cfg.get("early_stopping_patience", 7)
    grad_clip = training_cfg.get("gradient_clip", 1.0)
    scheduler_type = training_cfg.get("lr_scheduler", "cosine")

    start_time = time.time()
    logger.info("Loading sequence datasets and split partitions...")
    (
        train_ds, valid_ds, test_ds,
        y_valid_true, y_test_true,
        valid_cids, test_cids,
        meta,
    ) = load_sequence_split_data(data_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Calculate class imbalance weighting strictly on train labels
    y_train = train_ds.y
    n_pos = float(y_train.sum().item())
    n_neg = float((len(y_train) - n_pos))
    pos_weight_val = n_neg / n_pos if n_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=torch_device)

    logger.info(
        f"Initialized sequence training: {meta['n_train']} train ({n_pos:.0f} defaults, "
        f"pos_weight={pos_weight_val:.2f}), {meta['n_valid']} valid, {meta['n_test']} test. Device: {device}"
    )

    # Instantiate model
    model = MaskedGRUModel(
        num_features=meta["num_features"],
        cat_cardinalities=meta["cat_cardinalities"],
        emb_dim=cfg.get("emb_dim", 4),
        proj_dim=cfg.get("proj_dim", 48),
        hidden_dim=cfg.get("hidden_dim", 64),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.2),
        rnn_type=cfg.get("rnn_type", "gru"),
        bidirectional=cfg.get("bidirectional", False),
        pooling=cfg.get("pooling", "all"),
        mlp_dims=cfg.get("mlp_dims", [64, 32]),
    ).to(torch_device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=max_epochs * len(train_loader)
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-5
        )

    best_score = -float("inf")
    best_epoch = 0
    best_state_dict: Optional[Dict[str, Any]] = None
    best_preds_val: Optional[np.ndarray] = None
    no_improve_epochs = 0

    logger.info(f"Starting training across up to {max_epochs} epochs with early stopping (patience={patience})...")

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0

        for bx_num, bx_cat, bmask, by in train_loader:
            bx_num = bx_num.to(torch_device)
            bx_cat = bx_cat.to(torch_device)
            bmask = bmask.to(torch_device)
            by = by.to(torch_device)

            optimizer.zero_grad()
            logits = model(bx_num, bx_cat, bmask)
            loss = criterion(logits, by)
            loss.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()
            train_loss += loss.item() * len(by)

            if scheduler_type == "onecycle":
                scheduler.step()

        if scheduler_type != "onecycle":
            scheduler.step()

        train_loss /= len(train_ds)

        # Validation evaluation
        model.eval()
        val_preds_list: List[float] = []
        with torch.no_grad():
            for bx_num, bx_cat, bmask, _ in valid_loader:
                bx_num = bx_num.to(torch_device)
                bx_cat = bx_cat.to(torch_device)
                bmask = bmask.to(torch_device)
                p = torch.sigmoid(model(bx_num, bx_cat, bmask))
                val_preds_list.extend(p.cpu().numpy().tolist())

        val_preds_arr = np.array(val_preds_list, dtype=np.float32)
        val_metric = competition_metric(y_valid_true, val_preds_arr)

        logger.info(
            f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Comp Metric: {val_metric:.5f}"
        )

        if val_metric > best_score:
            best_score = val_metric
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_preds_val = val_preds_arr
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (best epoch {best_epoch}).")
                break

    assert best_state_dict is not None and best_preds_val is not None, "Training failed to produce checkpoints!"

    # Load best weights
    model.load_state_dict(best_state_dict)
    model.eval()

    # Generate aligned predictions for test split
    test_preds_list: List[float] = []
    with torch.no_grad():
        for bx_num, bx_cat, bmask, _ in test_loader:
            bx_num = bx_num.to(torch_device)
            bx_cat = bx_cat.to(torch_device)
            bmask = bmask.to(torch_device)
            p = torch.sigmoid(model(bx_num, bx_cat, bmask))
            test_preds_list.extend(p.cpu().numpy().tolist())

    preds_test = np.array(test_preds_list, dtype=np.float32)
    preds_valid = best_preds_val

    # Compute comprehensive evaluation metrics
    metrics_valid = compute_all_metrics(y_valid_true, preds_valid)
    metrics_test = compute_all_metrics(y_test_true, preds_test)

    total_time = round(time.time() - start_time, 2)

    # Persist artifacts in results/gru/
    preds_valid_path = output_dir / "preds_valid.npy"
    preds_test_path = output_dir / "preds_test.npy"
    metrics_path = output_dir / "metrics.json"
    config_path = output_dir / "config.json"
    model_path = output_dir / "model.pt"

    np.save(preds_valid_path, preds_valid)
    np.save(preds_test_path, preds_test)
    torch.save(best_state_dict, model_path)

    metrics_payload = {
        "valid": metrics_valid,
        "test": metrics_test,
    }
    save_json(metrics_payload, metrics_path)

    config_payload = {
        "model_name": cfg.get("model_name", "gru"),
        "rnn_type": cfg.get("rnn_type", "gru"),
        "num_features": meta["num_features"],
        "cat_cardinalities": meta["cat_cardinalities"],
        "hidden_dim": cfg.get("hidden_dim", 64),
        "num_layers": cfg.get("num_layers", 2),
        "bidirectional": cfg.get("bidirectional", False),
        "pooling": cfg.get("pooling", "all"),
        "mlp_dims": cfg.get("mlp_dims", [64, 32]),
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "pos_weight": round(pos_weight_val, 4),
        "learning_rate": lr,
        "seed": seed,
        "training_time_seconds": total_time,
    }
    save_json(config_payload, config_path)

    logger.info(f"Saved all results to {output_dir}")

    # Load LightGBM baseline metrics for direct side-by-side comparison
    lgbm_metrics = None
    lgbm_metrics_file = PROJECT_ROOT / "results" / "lgbm" / "metrics.json"
    if lgbm_metrics_file.exists():
        with open(lgbm_metrics_file, "r", encoding="utf-8") as f:
            lgbm_metrics = json.load(f)

    # Print formatted comparative metric table
    _print_comparison_table(metrics_valid, metrics_test, lgbm_metrics, best_epoch, total_time)

    return {
        "valid": metrics_valid,
        "test": metrics_test,
        "best_epoch": best_epoch,
        "output_dir": str(output_dir),
    }


def _print_comparison_table(
    val_m: Dict[str, float],
    test_m: Dict[str, float],
    lgbm_m: Optional[Dict[str, Any]],
    best_epoch: int,
    duration: float,
) -> None:
    """Print clean comparison report table placing GRU next to LightGBM benchmark."""
    print("\n" + "=" * 96)
    print("RiskTrail Phase 4a: Masked GRU vs LightGBM Benchmark Evaluation")
    print("=" * 96)
    print(f"GRU Optimal Epoch: {best_epoch} | Training Duration: {duration}s\n")

    header = f"{'Metric':<28} | {'Val (LGBM)':<12} | {'Val (GRU)':<12} | {'Test (LGBM)':<12} | {'Test (GRU)':<12}"
    print(header)
    print("-" * 96)

    def _val(m_dict: Optional[Dict[str, Any]], split: str, key: str) -> str:
        if m_dict and split in m_dict and key in m_dict[split]:
            return f"{m_dict[split][key]:.5f}"
        return "N/A"

    metrics_to_show = [
        ("Competition Metric", "competition_metric"),
        ("Normalized Gini", "normalized_gini"),
        ("Capture Rate @ 4%", "capture_rate_at_4pct"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("Brier Calibration", "brier"),
        ("Official Weighted Metric", "competition_metric_weighted"),
        ("Official Weighted Gini", "normalized_gini_weighted"),
        ("Official Weighted Capture@4%", "capture_rate_at_4pct_weighted"),
    ]

    for label, key in metrics_to_show:
        lgbm_val = _val(lgbm_m, "valid", key)
        gru_val = f"{val_m[key]:.5f}"
        lgbm_test = _val(lgbm_m, "test", key)
        gru_test = f"{test_m[key]:.5f}"
        print(f"{label:<28} | {lgbm_val:<12} | {gru_val:<12} | {lgbm_test:<12} | {gru_test:<12}")

    print("=" * 96 + "\n")


if __name__ == "__main__":
    train_gru()
