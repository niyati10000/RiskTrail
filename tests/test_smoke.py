"""Smoke tests verifying that all required dependencies and internal modules import cleanly."""

import importlib
import pytest


REQUIRED_DEPENDENCIES = [
    "pandas",
    "pyarrow",
    "numpy",
    "sklearn",
    "lightgbm",
    "torch",
    "shap",
    "matplotlib",
    "yaml",
    "pytest",
    "streamlit",
]

INTERNAL_MODULES = [
    "src.utils.seed",
    "src.utils.io",
    "src.utils.logging",
    "src.data.load",
    "src.data.clean",
    "src.data.split",
    "src.features.tabular",
    "src.features.sequence",
    "src.models.lgbm",
    "src.models.gru",
    "src.models.transformer",
    "src.eval.metrics",
    "src.eval.calibration",
    "src.eval.compare",
    "src.business.profit",
    "src.explain.shap_lgbm",
    "src.explain.seq_importance",
]


@pytest.mark.parametrize("pkg_name", REQUIRED_DEPENDENCIES)
def test_import_required_dependencies(pkg_name: str) -> None:
    """Verify each pinned third-party package can be imported successfully."""
    module = importlib.import_module(pkg_name)
    assert module is not None, f"Failed to import {pkg_name}"


@pytest.mark.parametrize("mod_name", INTERNAL_MODULES)
def test_import_internal_modules(mod_name: str) -> None:
    """Verify each internal project module can be imported successfully."""
    module = importlib.import_module(mod_name)
    assert module is not None, f"Failed to import {mod_name}"


def test_seed_utility() -> None:
    """Verify seed_everything runs without exceptions."""
    from src.utils.seed import seed_everything
    seed_everything(42)


def test_io_utilities(tmp_path) -> None:
    """Verify load_yaml, save_yaml, load_json, save_json roundtrip correctly."""
    from src.utils.io import save_yaml, load_yaml, save_json, load_json

    sample_dict = {"project": "RiskTrail", "phase": 0, "active": True}

    yaml_file = tmp_path / "test_config.yaml"
    save_yaml(sample_dict, yaml_file)
    assert yaml_file.exists()
    loaded_yaml = load_yaml(yaml_file)
    assert loaded_yaml == sample_dict

    json_file = tmp_path / "test_metrics.json"
    save_json(sample_dict, json_file)
    assert json_file.exists()
    loaded_json = load_json(json_file)
    assert loaded_json == sample_dict


def test_logging_utility() -> None:
    """Verify get_logger creates a functioning logger."""
    from src.utils.logging import get_logger
    logger = get_logger("SmokeTest")
    assert logger is not None
    logger.info("Smoke test logger operational.")
