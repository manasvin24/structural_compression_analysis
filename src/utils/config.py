"""Load and validate the project-wide YAML configuration."""

from pathlib import Path
from typing import Any

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "default.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _resolve_paths(cfg)
    return cfg


def _resolve_paths(cfg: dict[str, Any]) -> None:
    """Turn relative paths in the config into absolute ones."""
    paths = cfg.get("paths", {})
    for key in ("artifacts_dir", "results_dir", "calibration_dir", "plots_dir"):
        if key in paths and not Path(paths[key]).is_absolute():
            paths[key] = str(_PROJECT_ROOT / paths[key])
