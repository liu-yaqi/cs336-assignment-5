from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, List, Literal, get_args, get_origin, get_type_hints

import typer
import yaml


def load_dataclass_config_from_yaml(config_path: str, config_cls: type[Any]) -> Any:
    path = Path(config_path)
    if not path.exists():
        raise typer.BadParameter(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise typer.BadParameter("YAML config must be a mapping/dictionary.")

    config_field_names = {f.name for f in fields(config_cls)}
    unknown_keys = sorted(set(raw.keys()) - config_field_names)
    if unknown_keys:
        raise typer.BadParameter(
            f"Unknown config keys in {config_path}: {unknown_keys}"
        )

    return config_cls(**raw)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from '{value}'")


def _coerce_override_value(raw_value: str, field_type: Any) -> Any:
    origin = get_origin(field_type)
    args = get_args(field_type)

    if origin is Literal:
        literal_values = list(args)
        for candidate in literal_values:
            if str(candidate) == raw_value:
                return candidate
        raise ValueError(f"Expected one of {literal_values}, got '{raw_value}'")

    if origin is not None and type(None) in args:
        non_none_types = [arg for arg in args if arg is not type(None)]
        if raw_value.strip().lower() in {"none", "null"}:
            return None
        if len(non_none_types) == 1:
            return _coerce_override_value(raw_value, non_none_types[0])

    if field_type is bool:
        return _parse_bool(raw_value)
    if field_type is int:
        return int(raw_value)
    if field_type is float:
        return float(raw_value)
    if field_type is str:
        return raw_value

    raise ValueError(f"Unsupported override type: {field_type}")


def apply_cli_overrides_to_dataclass(config: Any, set_values: List[str]) -> Any:
    if not set_values:
        return config

    config_cls = type(config)
    config_field_names = {f.name for f in fields(config_cls)}
    type_hints = get_type_hints(config_cls)

    for item in set_values:
        if "=" not in item:
            raise typer.BadParameter(
                f"Invalid --set value '{item}'. Expected KEY=VALUE format."
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        if key not in config_field_names:
            raise typer.BadParameter(
                f"Unknown config key '{key}'. Valid keys: {sorted(config_field_names)}"
            )

        field_type = type_hints.get(key, type(getattr(config, key)))
        try:
            coerced_value = _coerce_override_value(raw_value, field_type)
        except ValueError as exc:
            raise typer.BadParameter(
                f"Failed to parse --set {key}={raw_value}: {exc}"
            ) from exc

        setattr(config, key, coerced_value)

    return config
