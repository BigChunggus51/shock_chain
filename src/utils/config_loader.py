"""
YAML Configuration Loader.

Loads and merges configuration files from the configs/ directory.
"""

from __future__ import annotations

from pathlib import Path
import yaml


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load environment configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_shock_config(shock_path: str = "configs/shocks.yaml") -> dict:
    """Load shock catalog from YAML."""
    with open(shock_path, "r") as f:
        return yaml.safe_load(f)


def load_all_configs(
    config_path: str = "configs/default.yaml",
    shock_path: str = "configs/shocks.yaml",
) -> dict:
    """
    Load and return all configs in a single dict.

    Returns
    -------
    dict with keys: "environment", "reward", "training", "topology", "shocks"
    """
    config = load_config(config_path)
    shock_config = load_shock_config(shock_path)

    return {
        "environment": config.get("environment", {}),
        "reward": config.get("reward", {}),
        "training": config.get("training", {}),
        "topology": config.get("topology", {}),
        "shocks": shock_config,
    }
