"""Configuration loading, typed settings objects and reproducible seeding."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from alphaforge.utils.logging import get_logger

log = get_logger("config")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (base is not mutated)."""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def deep_get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


@dataclass
class Config:
    """Thin typed wrapper around the raw configuration dictionary."""

    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(
        cls, path: str | Path | None = None, overrides: dict[str, Any] | None = None
    ) -> Config:
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        raw = load_yaml(path)
        raw = deep_merge(raw, overrides or {})
        raw = deep_merge(raw, _env_overrides())
        log.debug(f"Loaded config from {path}")
        return cls(raw=raw)

    def get(self, dotted: str, default: Any = None) -> Any:
        return deep_get(self.raw, dotted, default)

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        return dict(value) if isinstance(value, dict) else {}

    def to_dict(self) -> dict[str, Any]:
        import copy

        return copy.deepcopy(self.raw)


def _env_overrides() -> dict[str, Any]:
    """Map a small set of ALPHAFORGE_* env vars onto the config tree."""
    out: dict[str, Any] = {}
    mapping = {
        "ALPHAFORGE_DATA_PROVIDER": "data.provider",
        "ALPHAFORGE_MODEL_TYPE": "model.type",
        "ALPHAFORGE_PORTFOLIO_METHOD": "portfolio.method",
        "ALPHAFORGE_BACKTEST_REBALANCE": "backtest.rebalance",
        "ALPHAFORGE_LOG_LEVEL": "logging.level",
    }
    for env_key, dotted in mapping.items():
        value = os.environ.get(env_key)
        if value is None:
            continue
        target: dict[str, Any] = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = _coerce(value)
    return out


def _coerce(value: str) -> Any:
    low = value.strip().lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def set_global_seed(seed: int = 42) -> None:
    """Seed every RNG used by the platform for reproducible research runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


__all__ = ["Config", "PROJECT_ROOT", "load_yaml", "deep_merge", "deep_get", "set_global_seed"]
