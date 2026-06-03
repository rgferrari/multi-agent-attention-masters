"""YAML config loader with override support."""

from __future__ import annotations

from typing import Any, Dict, Iterable


def _set_nested(cfg: Dict[str, Any], keys: list[str], value: Any) -> None:
    cursor = cfg
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def load_config(path: str, overrides: Iterable[str]) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Missing PyYAML. Install with: pip install pyyaml") from exc

    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}', expected key=value")
        key, raw = override.split("=", 1)
        value = yaml.safe_load(raw)
        _set_nested(cfg, key.split("."), value)

    return cfg
