"""Input/Output utility functions for YAML configurations and JSON results."""

import json
from pathlib import Path
from typing import Any, Union
import yaml
import numpy as np


class EnhancedJSONEncoder(json.JSONEncoder):
    """JSON Encoder that supports NumPy data types and Path objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def load_yaml(path: Union[str, Path]) -> dict:
    """Load a YAML configuration file into a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        dict: Parsed YAML contents.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"YAML configuration file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(data: dict, path: Union[str, Path]) -> None:
    """Save a dictionary to a YAML file.

    Args:
        data: Dictionary to serialize.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(path: Union[str, Path]) -> dict:
    """Load a JSON file into a dictionary.

    Args:
        path: Path to the JSON file.

    Returns:
        dict: Parsed JSON contents.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Union[str, Path], indent: int = 2) -> None:
    """Save a dictionary to a JSON file.

    Args:
        data: Dictionary to serialize.
        path: Destination file path.
        indent: Indentation level for pretty-printing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, cls=EnhancedJSONEncoder)
