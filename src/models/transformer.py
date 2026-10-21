"""Transformer sequence model architecture, training, and inference.

Implements multi-head self-attention with learned positional encodings (13 positions),
key-padding masks, attention pooling with per-statement attention weight extraction,
and an optional GRU+attention-pooling variant behind a config flag.

Architecture pipeline:
1. Categorical embeddings + continuous feature projection -> d_model
2. Learned positional encoding over 13 monthly statement positions
3. Configurable backbone:
   - "transformer": Multi-layer TransformerEncoder with src_key_padding_mask
   - "gru": Recurrent GRU with left-aligned execution and right-aligned statement state restoration
4. Attention Pooling:
   - Parameterized query network weighting valid monthly statements
   - Padded statement steps receive strictly 0.0 attention weight
   - Exposes normalized attention weights [B, 13] for Phase 7 explainability
   - Supports alternative pooling ("cls", "mean") via config
5. MLP classification head -> Sigmoid default probability

Training & Evaluation Contract:
- AdamW optimizer with CosineAnnealingLR or OneCycleLR
- Class-weighted BCE loss (pos_weight derived from train split)
- Gradient norm clipping
- Early stopping on the primary competition metric (0.5 * (Gini + Capture@4%))
- Fixed seed (42) for deterministic reproducibility
- Outputs saved to results/transformer/: preds_valid.npy, preds_test.npy, metrics.json, config.json, model.pt
- 3-way comparative metric report vs LightGBM and GRU
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

logger = get_logger("RiskTrail.Models.Transformer")


class AttentionPooling(nn.Module):
    """Attention pooling layer weighting sequence steps for credit risk aggregation.

    Computes query-based attention scores for each time step, masks out padded statements
    with -1e9 before softmax, and produces a normalized weight vector summing to 1.0
    strictly across valid monthly statements.
    """

    def __init__(self, d_model: int, attn_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(d_model, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, 1, bias=False),
        )

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute pooled representation and normalized statement attention weights.

        Args:
            h: Hidden state representations [B, T, D].
            mask: Boolean padding mask [B, T] (True = valid, False = padding).

        Returns:
            pooled: Attentive weighted sum [B, D].
            attn_weights: Normalized attention weights [B, T] (sum=1 over valid steps, 0 on padding).
        """
        # Raw scores: [B, T, 1] -> [B, T]
        scores = self.attn_net(h).squeeze(-1)

        # Mask out padding positions with large negative value
        scores = scores.masked_fill(~mask, -1e9)

        # Softmax over sequence length
        attn_weights = torch.softmax(scores, dim=-1)

        # Defensively clamp padding weights to exact zero and renormalize
        attn_weights = attn_weights * mask.float()
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Weighted context sum: [B, 1, T] @ [B, T, D] -> [B, 1, D] -> [B, D]
        pooled = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)
        return pooled, attn_weights


class StatementTransformer(nn.Module):
    """Transformer Encoder sequence model with learned positional encodings and attention pooling.

    Supports both 'transformer' and 'gru' backbones, ensuring modular comparison
    while strictly preserving padding invariance.
    """

    def __init__(
        self,
        num_features: int = 22,
        cat_cardinalities: Sequence[int] = (5, 4, 3, 7),
        emb_dim: int = 4,
        proj_dim: int = 48,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        activation: str = "relu",
        backbone: str = "transformer",
        pooling: str = "attention",
        attn_dim: int = 32,
        mlp_dims: Sequence[int] = (64, 32),
        max_seq_len: int = 13,
    ):
        super().__init__()
        self.num_features = num_features
        self.cat_cardinalities = list(cat_cardinalities)
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.backbone = backbone.lower()
        self.pooling = pooling.lower()
        self.max_seq_len = max_seq_len

        # 1. Categorical embeddings (cardinality + 1 for category codes and padding index 0)
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, emb_dim, padding_idx=0)
            for card in self.cat_cardinalities
        ])
        total_emb_dim = len(self.cat_cardinalities) * emb_dim

        # 2. Continuous feature projection
        self.num_proj = nn.Sequential(
            nn.Linear(num_features, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Combined projection into d_model token space
        comb_in_dim = proj_dim + total_emb_dim
        if comb_in_dim != d_model:
            self.input_proj = nn.Sequential(
                nn.Linear(comb_in_dim, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            self.input_proj = nn.Identity()

        # 3. Learned Positional Encoding (13 statement positions)
        if self.pooling == "cls":
            # Extra position for CLS token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.pos_encoder = nn.Embedding(max_seq_len + 1, d_model)
        else:
            self.cls_token = None
            self.pos_encoder = nn.Embedding(max_seq_len, d_model)

        self.pos_dropout = nn.Dropout(dropout)

        # 4. Backbone architecture: Transformer Encoder or GRU
        if self.backbone == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
            self.rnn = None
        elif self.backbone in ["gru", "lstm"]:
            RNNClass = nn.GRU if self.backbone == "gru" else nn.LSTM
            self.rnn = RNNClass(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.transformer_encoder = None
        else:
            raise ValueError(f"Unsupported backbone: '{backbone}'. Choose 'transformer', 'gru', or 'lstm'.")

        # 5. Pooling layer
        if self.pooling == "attention":
            self.pooler = AttentionPooling(d_model=d_model, attn_dim=attn_dim, dropout=dropout)
        else:
            self.pooler = None

        # 6. MLP Classification Head
        mlp_layers: List[nn.Module] = []
        curr_dim = d_model
        for h_dim in mlp_dims:
            mlp_layers.extend([
                nn.Linear(curr_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            curr_dim = h_dim
        mlp_layers.append(nn.Linear(curr_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def _left_align_valid_steps(
        self, x: torch.Tensor, lengths: torch.Tensor, max_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align valid front-padded steps to start at index 0 for recurrent execution."""
        B, T, D = x.shape
        t_idx = torch.arange(T, device=x.device)
        gather_indices = (T - lengths).unsqueeze(1) + t_idx.unsqueeze(0)
        gather_indices = torch.clamp(gather_indices, 0, T - 1)
        x_left = torch.gather(x, 1, gather_indices.unsqueeze(-1).expand(-1, -1, D))
        left_mask = t_idx.unsqueeze(0) < lengths.unsqueeze(1)
        x_left = x_left * left_mask.unsqueeze(-1).float()
        return x_left, left_mask

    def _right_align_steps(
        self, out_left: torch.Tensor, lengths: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """Shift left-aligned RNN output states back to front-padded positions [T - length : T]."""
        B, T, D = out_left.shape
        t_idx = torch.arange(T, device=out_left.device)
        src_idx = t_idx.unsqueeze(0) - (T - lengths).unsqueeze(1)
        src_idx = torch.clamp(src_idx, 0, T - 1)
        out_right = torch.gather(out_left, 1, src_idx.unsqueeze(-1).expand(-1, -1, D))
        valid_mask = t_idx.unsqueeze(0) >= (T - lengths).unsqueeze(1)
        out_right = out_right * valid_mask.unsqueeze(-1).float()
        return out_right

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass computing binary logits and optional statement attention weights.

        Args:
            x_num: Continuous features tensor [B, T, F].
            x_cat: Categorical features tensor [B, T, C].
            mask: Boolean padding mask [B, T] (True = valid statement, False = padding).
            return_attention: If True, returns (logits, attn_weights) where attn_weights is [B, T].

        Returns:
            logits: Binary classification logits [B].
            attn_weights (optional): Normalized attention weights [B, T] across 13 monthly statements.
        """
        B, T, _ = x_num.shape
        lengths = mask.sum(dim=1).clamp(min=1)

        # Defensively sanitize padded inputs so corrupted padding values can never affect activations
        mask_3d = mask.unsqueeze(-1)
        x_num_clean = torch.where(mask_3d, x_num, torch.zeros_like(x_num))
        x_cat_clean = torch.where(mask_3d, x_cat, torch.zeros_like(x_cat))

        # 1. Project continuous features
        num_feat = self.num_proj(x_num_clean)

        # 2. Embed categoricals
        cat_embs = [
            self.embeddings[i](x_cat_clean[:, :, i].clamp(min=0))
            for i in range(len(self.embeddings))
        ]
        cat_feat = torch.cat(cat_embs, dim=-1)

        # 3. Concatenate and project to d_model token space
        x_combined = torch.cat([num_feat, cat_feat], dim=-1)
        x_tokens = self.input_proj(x_combined)

        # 4. Learned Positional Encodings
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x_tokens = torch.cat([cls_tokens, x_tokens], dim=1)  # [B, T+1, d_model]
            pos_indices = torch.arange(T + 1, device=x_num.device).unsqueeze(0).expand(B, -1)
            x_tokens = self.pos_dropout(x_tokens + self.pos_encoder(pos_indices))
            mask_with_cls = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        else:
            pos_indices = torch.arange(T, device=x_num.device).unsqueeze(0).expand(B, -1)
            x_tokens = self.pos_dropout(x_tokens + self.pos_encoder(pos_indices))
            mask_with_cls = mask

        # 5. Backbone execution
        if self.backbone == "transformer":
            # PyTorch src_key_padding_mask requires True for PADDING (ignored keys)
            key_padding_mask = ~mask_with_cls
            h = self.transformer_encoder(x_tokens, src_key_padding_mask=key_padding_mask)
            # Enforce zero representation on padded positions
            h = h * mask_with_cls.unsqueeze(-1).float()
        else:  # "gru" / "lstm"
            x_left, _ = self._left_align_valid_steps(x_tokens, lengths, T)
            packed_in = pack_padded_sequence(
                x_left, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.rnn(packed_in)
            out_left, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)
            h = self._right_align_steps(out_left, lengths, T)
            h = h * mask.unsqueeze(-1).float()

        # 6. Pooling & Attention Extraction
        if self.pooling == "attention":
            pooled, attn_weights = self.pooler(h, mask)
        elif self.pooling == "cls":
            pooled = h[:, 0, :]  # CLS token representation
            # For CLS mode, uniform attention on valid steps as fallback
            attn_weights = mask.float() / lengths.unsqueeze(1).float()
        elif self.pooling == "mean":
            pooled = (h * mask.unsqueeze(-1).float()).sum(dim=1) / lengths.unsqueeze(1).float()
            attn_weights = mask.float() / lengths.unsqueeze(1).float()
        else:  # default last valid state
            last_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, self.d_model)
            pooled = h.gather(1, last_idx).squeeze(1)
            attn_weights = torch.zeros_like(mask, dtype=torch.float32)
            attn_weights.scatter_(1, (lengths - 1).unsqueeze(1), 1.0)

        # 7. Classification head
        logits = self.mlp(pooled).squeeze(-1)

        if return_attention:
            return logits, attn_weights
        return logits

    def predict_proba(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute default prediction probabilities in [0, 1]."""
        with torch.no_grad():
            logits = self.forward(x_num, x_cat, mask, return_attention=False)
            return torch.sigmoid(logits)

    def get_attention_weights(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract normalized attention pooling weights [B, 13] per monthly statement."""
        with torch.no_grad():
            _, attn_weights = self.forward(x_num, x_cat, mask, return_attention=True)
            return attn_weights


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

    cat_cards = [int(cat_seq[:, :, c].max()) + 1 for c in range(cat_seq.shape[-1])]

    metadata = {
        "num_features": x_seq.shape[-1],
        "cat_cardinalities": cat_cards,
        "n_train": len(train_idx),
        "n_valid": len(valid_idx),
        "n_test": len(test_idx),
    }

    return train_ds, valid_ds, test_ds, y_valid, y_test, valid_cids, test_cids, metadata


def train_transformer(
    config: Optional[Dict[str, Any]] = None,
    data_dir: Union[str, Path] = PROJECT_ROOT / "data" / "processed",
    output_dir: Union[str, Path] = PROJECT_ROOT / "results" / "transformer",
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Train PyTorch StatementTransformer with early stopping and log evaluation metrics.

    Args:
        config: Model hyperparameter configuration dictionary or None (loads configs/transformer.yaml).
        data_dir: Directory containing X_seq.npy, cat_seq.npy, mask.npy, y.npy, splits.parquet.
        output_dir: Directory to save results/transformer/ artifacts.
        seed: Random seed.
        device: 'cpu' or 'cuda'. If None, auto-selects.

    Returns:
        Dict[str, Any]: Dictionary with valid and test evaluation metrics, best epoch, and paths.
    """
    start_time = time.time()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config_path = PROJECT_ROOT / "configs" / "transformer.yaml"
        cfg = load_yaml(config_path) if config_path.exists() else {}
    else:
        cfg = config

    train_cfg = cfg.get("training", {})
    seed = train_cfg.get("seed", seed)
    seed_everything(seed)

    if device is None:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)
    logger.info(f"Training StatementTransformer on device: {torch_device}")

    # Load customer-stratified sequence splits
    train_ds, valid_ds, test_ds, y_valid_true, y_test_true, valid_cids, test_cids, meta = (
        load_sequence_split_data(data_dir)
    )

    batch_size = train_cfg.get("batch_size", 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Class-imbalance pos_weight derived strictly from train split
    y_train_arr = train_ds.y.numpy()
    pos_count = float((y_train_arr == 1).sum())
    neg_count = float((y_train_arr == 0).sum())
    pos_weight_val = neg_count / max(pos_count, 1.0)
    pos_weight = torch.tensor([pos_weight_val], device=torch_device)
    logger.info(f"Class distribution: {int(pos_count)} positives, {int(neg_count)} negatives (pos_weight: {pos_weight_val:.4f})")

    # Instantiate StatementTransformer
    model = StatementTransformer(
        num_features=meta["num_features"],
        cat_cardinalities=meta["cat_cardinalities"],
        emb_dim=cfg.get("emb_dim", 4),
        proj_dim=cfg.get("proj_dim", 48),
        d_model=cfg.get("d_model", 64),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 2),
        dim_feedforward=cfg.get("dim_feedforward", 128),
        dropout=cfg.get("dropout", 0.1),
        activation=cfg.get("activation", "relu"),
        backbone=cfg.get("backbone", "transformer"),
        pooling=cfg.get("pooling", "attention"),
        attn_dim=cfg.get("attn_dim", 32),
        mlp_dims=cfg.get("mlp_dims", [64, 32]),
    ).to(torch_device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"StatementTransformer initialized with {total_params:,} trainable parameters (Backbone: {model.backbone}, Pooling: {model.pooling})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    lr = train_cfg.get("learning_rate", 0.0005)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    max_epochs = train_cfg.get("max_epochs", 35)
    patience = train_cfg.get("early_stopping_patience", 7)
    grad_clip = train_cfg.get("gradient_clip", 1.0)

    lr_scheduler_type = train_cfg.get("lr_scheduler", "cosine").lower()
    if lr_scheduler_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=max_epochs
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    best_val_comp_metric = -1.0
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_preds_val: Optional[np.ndarray] = None
    best_epoch = 0
    no_improve_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0

        for bx_num, bx_cat, bmask, by in train_loader:
            bx_num = bx_num.to(torch_device)
            bx_cat = bx_cat.to(torch_device)
            bmask = bmask.to(torch_device)
            by = by.to(torch_device)

            optimizer.zero_grad()
            logits = model(bx_num, bx_cat, bmask, return_attention=False)
            loss = criterion(logits, by)
            loss.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()
            if lr_scheduler_type == "onecycle":
                scheduler.step()

            train_loss += loss.item() * len(by)

        if lr_scheduler_type != "onecycle":
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
                p = torch.sigmoid(model(bx_num, bx_cat, bmask, return_attention=False))
                val_preds_list.extend(p.cpu().numpy().tolist())

        val_preds = np.array(val_preds_list, dtype=np.float32)
        val_comp_metric = competition_metric(y_valid_true, val_preds)

        logger.info(
            f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Comp Metric: {val_comp_metric:.5f}"
        )

        if val_comp_metric > best_val_comp_metric + 1e-6:
            best_val_comp_metric = val_comp_metric
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_preds_val = val_preds.copy()
            best_epoch = epoch
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (best epoch {best_epoch}).")
                break

    assert best_state_dict is not None and best_preds_val is not None, "Training failed to produce checkpoints!"

    # Load best weights for final evaluation
    model.load_state_dict(best_state_dict)
    model.eval()

    # Generate aligned predictions for test split
    test_preds_list: List[float] = []
    with torch.no_grad():
        for bx_num, bx_cat, bmask, _ in test_loader:
            bx_num = bx_num.to(torch_device)
            bx_cat = bx_cat.to(torch_device)
            bmask = bmask.to(torch_device)
            p = torch.sigmoid(model(bx_num, bx_cat, bmask, return_attention=False))
            test_preds_list.extend(p.cpu().numpy().tolist())

    preds_test = np.array(test_preds_list, dtype=np.float32)
    preds_valid = best_preds_val

    # Compute comprehensive evaluation metrics
    metrics_valid = compute_all_metrics(y_valid_true, preds_valid)
    metrics_test = compute_all_metrics(y_test_true, preds_test)

    total_time = round(time.time() - start_time, 2)

    # Persist artifacts in results/transformer/
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
        "model_name": cfg.get("model_name", "transformer"),
        "backbone": cfg.get("backbone", "transformer"),
        "num_features": meta["num_features"],
        "cat_cardinalities": meta["cat_cardinalities"],
        "d_model": cfg.get("d_model", 64),
        "nhead": cfg.get("nhead", 4),
        "num_layers": cfg.get("num_layers", 2),
        "dim_feedforward": cfg.get("dim_feedforward", 128),
        "pooling": cfg.get("pooling", "attention"),
        "attn_dim": cfg.get("attn_dim", 32),
        "mlp_dims": cfg.get("mlp_dims", [64, 32]),
        "total_parameters": total_params,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "pos_weight": round(pos_weight_val, 4),
        "learning_rate": lr,
        "seed": seed,
        "training_time_seconds": total_time,
    }
    save_json(config_payload, config_path)

    logger.info(f"Saved all transformer results to {output_dir}")

    # Load baseline benchmarks (LightGBM & GRU) for 3-way direct comparison
    lgbm_metrics = None
    lgbm_metrics_file = PROJECT_ROOT / "results" / "lgbm" / "metrics.json"
    if lgbm_metrics_file.exists():
        with open(lgbm_metrics_file, "r", encoding="utf-8") as f:
            lgbm_metrics = json.load(f)

    gru_metrics = None
    gru_metrics_file = PROJECT_ROOT / "results" / "gru" / "metrics.json"
    if gru_metrics_file.exists():
        with open(gru_metrics_file, "r", encoding="utf-8") as f:
            gru_metrics = json.load(f)

    # Print formatted 3-way comparative metric report
    _print_three_way_comparison_table(
        val_m=metrics_valid,
        test_m=metrics_test,
        lgbm_m=lgbm_metrics,
        gru_m=gru_metrics,
        best_epoch=best_epoch,
        duration=total_time,
    )

    return {
        "valid": metrics_valid,
        "test": metrics_test,
        "best_epoch": best_epoch,
        "output_dir": str(output_dir),
    }


def _print_three_way_comparison_table(
    val_m: Dict[str, float],
    test_m: Dict[str, float],
    lgbm_m: Optional[Dict[str, Any]],
    gru_m: Optional[Dict[str, Any]],
    best_epoch: int,
    duration: float,
) -> None:
    """Print clean comparative report table placing Transformer next to LightGBM and GRU."""
    print("\n" + "=" * 115)
    print("RiskTrail Phase 4b: Transformer Encoder vs Masked GRU vs LightGBM Benchmark")
    print("=" * 115)
    print(f"Transformer Optimal Epoch: {best_epoch} | Training Duration: {duration}s\n")

    header = (
        f"{'Metric':<25} | "
        f"{'Val (LGBM)':<11} | {'Val (GRU)':<11} | {'Val (Xfrmr)':<11} | "
        f"{'Test (LGBM)':<11} | {'Test (GRU)':<11} | {'Test (Xfrmr)':<11}"
    )
    print(header)
    print("-" * 115)

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
        gru_val = _val(gru_m, "valid", key)
        xfrmr_val = f"{val_m[key]:.5f}"

        lgbm_test = _val(lgbm_m, "test", key)
        gru_test = _val(gru_m, "test", key)
        xfrmr_test = f"{test_m[key]:.5f}"

        print(
            f"{label:<25} | "
            f"{lgbm_val:<11} | {gru_val:<11} | {xfrmr_val:<11} | "
            f"{lgbm_test:<11} | {gru_test:<11} | {xfrmr_test:<11}"
        )

    print("=" * 115 + "\n")


if __name__ == "__main__":
    train_transformer()
