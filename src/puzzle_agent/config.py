"""Configuration loading (config.yaml + environment)."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(custom_path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(custom_path) if custom_path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)

    # Generator API key: config file takes priority, then env var
    gen = cfg.get("generator", {})
    if not gen.get("api_key"):
        gen["api_key"] = os.environ.get("DEEPSEEK_API_KEY", "")
    cfg["generator"] = gen

    # Teacher API key (backward compat)
    t = cfg.get("teacher", {})
    key_env = t.get("api_key_env", "DEEPSEEK_API_KEY")
    t["api_key"] = os.environ.get(key_env, "") or gen.get("api_key", "")
    cfg["teacher"] = t

    return cfg


def resolve(cfg: Dict[str, Any], rel: str) -> Path:
    return Path(cfg["_root"]) / rel
