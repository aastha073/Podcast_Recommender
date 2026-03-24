from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def load_project_config(config_path: str | None = None) -> dict[str, Any]:
    """Load the canonical project config as a plain dictionary."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
