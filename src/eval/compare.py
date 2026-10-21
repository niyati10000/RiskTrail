"""Cross-model evaluation and comparison orchestrator.

Reads standardized predictions and metrics from results/<model>/ and builds
comparative performance summaries and segment-level analyses (e.g. deteriorating vs stable customers).
"""

from pathlib import Path
from typing import List
import pandas as pd


def compare_models(results_dir: Path, model_names: List[str]) -> pd.DataFrame:
    """Load results for multiple models and compile a standardized comparison table."""
    raise NotImplementedError("Phase 5 implementation pending.")
