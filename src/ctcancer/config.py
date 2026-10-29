"""Configuration loading (YAML + profile + dotted CLI overrides) and reproducibility helpers."""
from __future__ import annotations

import argparse
import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class Config(dict):
    """dict with attribute access and dotted get: cfg.get_path('training.lr')."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def path(self, key: str) -> Path:
        """Resolve a paths.<key> entry relative to the project root."""
        p = Path(os.path.expanduser(self["paths"][key]))
        return p if p.is_absolute() else PROJECT_ROOT / p


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(value)
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None,
                profile: str | None = None) -> Config:
    """Load default.yaml, merge the profile file, an optional user file and overrides."""
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    path = path or os.getenv("CTC_CONFIG")  # deployment override without code changes
    if path and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with open(path) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    prof = profile or cfg.get("profile", "lite")
    prof_file = PROJECT_ROOT / "configs" / f"profile_{prof}.yaml"
    if prof_file.exists():
        with open(prof_file) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    cfg = apply_overrides(cfg, overrides)
    return Config(cfg)


def cli_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config file")
    p.add_argument("--profile", default=None, choices=["lite", "full"], help="compute profile")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="dotted override, e.g. --set training.epochs=5")
    return p


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (if installed) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def resolve_device(pref: str = "auto"):
    import torch

    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
