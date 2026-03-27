from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


_GLOBAL_PREPROCESSING_CONFIG: Dict[str, Any] | None = None


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as file_handle:
        data = yaml.safe_load(file_handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {path}")

    return data


def _set_by_dotted_key(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current: Dict[str, Any] = config

    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value


def _apply_overrides(
    config: Dict[str, Any], overrides: Iterable[str]
) -> Dict[str, Any]:
    merged = deepcopy(config)

    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"Invalid override '{item}'. Expected KEY=VALUE format for --set."
            )

        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{item}'. Key cannot be empty.")

        value = yaml.safe_load(raw_value)
        _set_by_dotted_key(merged, key, value)

    return merged


def add_preprocessing_config_arguments(
    parser: argparse.ArgumentParser,
    default_config: str | None = "config.yaml",
    config_required: bool = False,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None if config_required else default_config,
        required=config_required,
        help="Path to preprocessing config YAML",
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        help=(
            "Override config values (repeatable). "
            "Use KEY=VALUE and dotted keys, e.g. --set output_dir=/tmp/out "
            "--set workflow.process_reference=false"
        ),
    )
    return parser


def resolve_preprocessing_runtime_arguments(
    default_config: Path | str | None = None,
    argv: Iterable[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    add_preprocessing_config_arguments(
        parser,
        default_config=(str(default_config) if default_config is not None else None),
    )
    args, _ = parser.parse_known_args(argv)
    return args


def set_preprocessing_global_config(config: Dict[str, Any]) -> Dict[str, Any]:
    global _GLOBAL_PREPROCESSING_CONFIG
    _GLOBAL_PREPROCESSING_CONFIG = deepcopy(config)
    return deepcopy(_GLOBAL_PREPROCESSING_CONFIG)


def load_preprocessing_config(
    config_path: Path | str | None,
    overrides: Iterable[str] | None = None,
    default_config: Path | str | None = None,
    argv: Iterable[str] | None = None,
) -> Dict[str, Any]:
    runtime_args = None
    if config_path is None or overrides is None:
        runtime_args = resolve_preprocessing_runtime_arguments(
            default_config=default_config,
            argv=argv,
        )

    if config_path is None:
        config_path = (
            runtime_args.config if runtime_args is not None else default_config
        )

    if config_path is None:
        raise ValueError(
            "No config path provided and no default config could be resolved."
        )

    if overrides is None and runtime_args is not None:
        overrides = runtime_args.config_overrides

    cfg = _read_yaml(Path(config_path))
    if overrides:
        cfg = _apply_overrides(cfg, overrides)

    set_preprocessing_global_config(cfg)
    return deepcopy(cfg)


def get_preprocessing_global_config(
    config_path: Path | str | None = None,
    overrides: Iterable[str] | None = None,
    default_config: Path | str | None = None,
    argv: Iterable[str] | None = None,
    force_reload: bool = False,
) -> Dict[str, Any]:
    global _GLOBAL_PREPROCESSING_CONFIG

    if _GLOBAL_PREPROCESSING_CONFIG is None or force_reload:
        cfg = load_preprocessing_config(
            config_path=config_path,
            overrides=overrides,
            default_config=default_config,
            argv=argv,
        )
        _GLOBAL_PREPROCESSING_CONFIG = deepcopy(cfg)

    return deepcopy(_GLOBAL_PREPROCESSING_CONFIG)
