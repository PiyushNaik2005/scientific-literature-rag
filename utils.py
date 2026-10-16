"""
utils.py — Shared helpers: config loading, logging setup, seed fixing.

Design choice: a single canonical place for cross-cutting concerns so every
module just does `from utils import get_config, get_logger` and nothing more.
"""

import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ── Singleton config cache ────────────────────────────────────────────────────
_CONFIG: dict[str, Any] | None = None
_PROJECT_ROOT = Path(__file__).parent


def get_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load and cache the YAML configuration.

    Parameters
    ----------
    path : optional override path; defaults to <project_root>/config.yaml

    Returns
    -------
    dict with the full configuration tree
    """
    global _CONFIG
    if _CONFIG is None:
        cfg_path = Path(path) if path else _PROJECT_ROOT / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger that writes to stdout and to logs/<name>.log.

    Parameters
    ----------
    name  : logger name (typically __name__ of the calling module)
    level : log level (default INFO)
    """
    cfg = get_config()
    log_dir = _PROJECT_ROOT / cfg["paths"]["logs_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_dir / f"{name.replace('.', '_')}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def fix_seed(seed: int | None = None) -> None:
    """
    Fix random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

    Parameters
    ----------
    seed : seed value; reads from config if None
    """
    import torch

    if seed is None:
        seed = get_config().get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    get_logger(__name__).info("Random seed fixed to %d", seed)


def ensure_dirs() -> None:
    """Create all project directories declared in config.paths."""
    cfg = get_config()
    for key, rel_path in cfg["paths"].items():
        full = _PROJECT_ROOT / rel_path
        full.mkdir(parents=True, exist_ok=True)
