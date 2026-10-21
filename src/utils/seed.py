"""Random seed management for reproducible runs across Python, NumPy, and PyTorch."""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Fix random seeds for Python, NumPy, and PyTorch for deterministic execution.

    Args:
        seed: The integer seed value to use across all random generators.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
