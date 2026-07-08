"""Load and validate the single project configuration file.

Why this matters: every battery spec, market choice, and model threshold in a
governed system must be traceable to one reviewed source of truth. Hard-coded
parameters scattered through code are how silent errors (wrong hub, wrong
efficiency) turn into wrong revenue numbers. Everything tunable lives in
``config/config.yaml``; this module is the only way code reads it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repo root = two levels above this file's package (src/forecast_to_dispatch/).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

_REQUIRED_TOP_LEVEL_KEYS = {
    "market",
    "dates",
    "battery",
    "ancillary_services",
    "forecast",
    "governance",
    "paths",
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read config.yaml and fail loudly if a required section is missing.

    Why this matters: a governed system must not run on a partial or stale
    configuration — a missing battery section silently defaulting to zero
    capacity would produce a meaningless (but plausible-looking) backtest.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    missing = _REQUIRED_TOP_LEVEL_KEYS - set(config)
    if missing:
        raise KeyError(f"Config {config_path} is missing required sections: {sorted(missing)}")
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """Turn a relative path from config['paths'] into an absolute repo path."""
    if key not in config["paths"]:
        raise KeyError(f"Unknown path key '{key}'; available: {sorted(config['paths'])}")
    return REPO_ROOT / config["paths"][key]
