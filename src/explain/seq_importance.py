"""Explainability for sequence models.

Analyzes temporal attention weights per statement month and computes permutation
importance across feature groups and statement time steps.
"""

from typing import Dict
import numpy as np
import torch


def compute_temporal_attention_weights(
    model: torch.nn.Module,
    X_seq: torch.Tensor,
    mask: torch.Tensor,
) -> np.ndarray:
    """Extract mean attention weights across statement positions for defaulters vs non-defaulters."""
    raise NotImplementedError("Phase 7 implementation pending.")
