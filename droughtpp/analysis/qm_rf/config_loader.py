from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Set

import yaml


_IMPORT_KEYS = ("import_config", "imports", "extends")
_INTERPOLATION_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_\.]*)\}")
_INTERPOLATION_FULLMATCH_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_\.]*)\}$")
_GLOBAL_QM_RF_CONFIG: Dict[str, Any] | None = None


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {path}")

    return data


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged


def _resolve_ref(context: Dict[str, Any], ref: str) -> Any:
    if ref in context:
        return context[ref]

    current: Any = context
    for part in ref.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Unknown config variable in interpolation: {ref}")
        current = current[part]

    return current


def _expand_node(node: Any, context: Dict[str, Any]):
    changed = False

    if isinstance(node, str):
        fullmatch = _INTERPOLATION_FULLMATCH_PATTERN.fullmatch(node)
        if fullmatch:
            resolved = _resolve_ref(context, fullmatch.group(1))
            return deepcopy(resolved), True

        def repl(match: re.Match[str]) -> str:
            resolved = _resolve_ref(context, match.group(1))
            return str(resolved)

        expanded = _INTERPOLATION_PATTERN.sub(repl, node)
        if expanded != node:
            changed = True
        return expanded, changed

    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            out_value, value_changed = _expand_node(value, context)
            out[key] = out_value
            changed = changed or value_changed
        return out, changed

    if isinstance(node, list):
        out_list = []
        for value in node:
            out_value, value_changed = _expand_node(value, context)
            out_list.append(out_value)
            changed = changed or value_changed
        return out_list, changed

    return node, False


def _interpolate_config(config: Dict[str, Any], max_passes: int = 20) -> Dict[str, Any]:
    interpolated = deepcopy(config)

    for _ in range(max_passes):
        interpolated, changed = _expand_node(interpolated, interpolated)
        if not changed:
            return interpolated

    raise ValueError(
        "Config interpolation did not converge. Possible circular variable reference."
    )


def _normalize_imports(raw_imports: Any) -> list[str]:
    if raw_imports is None:
        return []
    if isinstance(raw_imports, str):
        return [raw_imports]
    if isinstance(raw_imports, list):
        if not all(isinstance(item, str) for item in raw_imports):
            raise ValueError(
                "All entries in imports/import_config/extends must be paths"
            )
        return raw_imports
    raise ValueError(
        "import_config/imports/extends must be a string or list of strings"
    )


def _load_with_imports(config_path: Path, seen: Set[Path]) -> Dict[str, Any]:
    resolved_path = config_path.resolve()
    if resolved_path in seen:
        raise ValueError(f"Circular config import detected at: {resolved_path}")

    seen.add(resolved_path)
    cfg = _read_yaml(resolved_path)

    imports = []
    for key in _IMPORT_KEYS:
        if key in cfg:
            imports.extend(_normalize_imports(cfg.pop(key)))

    merged: Dict[str, Any] = {}
    for import_item in imports:
        import_path = Path(import_item)
        if not import_path.is_absolute():
            import_path = resolved_path.parent / import_path
        imported_cfg = _load_with_imports(import_path, seen)
        merged = _deep_merge(merged, imported_cfg)

    merged = _deep_merge(merged, cfg)
    seen.remove(resolved_path)
    return merged


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


def resolve_qm_rf_runtime_arguments(
    default_config: Path | str | None = None,
    argv: Iterable[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    add_qm_rf_config_arguments(
        parser,
        default_config=(str(default_config) if default_config is not None else None),
    )
    args, _ = parser.parse_known_args(argv)
    return args


def set_qm_rf_global_config(config: Dict[str, Any]) -> Dict[str, Any]:
    global _GLOBAL_QM_RF_CONFIG
    _GLOBAL_QM_RF_CONFIG = deepcopy(config)
    return deepcopy(_GLOBAL_QM_RF_CONFIG)


def get_qm_rf_global_config(
    config_path: Path | str | None = None,
    overrides: Iterable[str] | None = None,
    default_config: Path | str | None = None,
    argv: Iterable[str] | None = None,
    force_reload: bool = False,
) -> Dict[str, Any]:
    global _GLOBAL_QM_RF_CONFIG

    if _GLOBAL_QM_RF_CONFIG is None or force_reload:
        cfg = load_qm_rf_config(
            config_path=config_path,
            overrides=overrides,
            default_config=default_config,
            argv=argv,
        )
        _GLOBAL_QM_RF_CONFIG = deepcopy(cfg)

    return deepcopy(_GLOBAL_QM_RF_CONFIG)


def load_qm_rf_config(
    config_path: Path | str | None,
    overrides: Iterable[str] | None = None,
    default_config: Path | str | None = None,
    argv: Iterable[str] | None = None,
) -> Dict[str, Any]:
    runtime_args = None
    if config_path is None or overrides is None:
        runtime_args = resolve_qm_rf_runtime_arguments(
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

    config_path = Path(config_path)
    cfg = _load_with_imports(config_path, seen=set())

    if overrides:
        cfg = _apply_overrides(cfg, overrides)

    interpolated_cfg = _interpolate_config(cfg)
    set_qm_rf_global_config(interpolated_cfg)
    return interpolated_cfg


def add_qm_rf_config_arguments(
    parser: argparse.ArgumentParser,
    default_config: str | None = "train_config.yaml",
    config_required: bool = False,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None if config_required else default_config,
        required=config_required,
        help="Path to config YAML",
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        help=(
            "Override config values (repeatable). "
            "Use KEY=VALUE, supports dotted keys, e.g. --set output_dir=/tmp/out "
            "--set leave_out_years='[2015,2016]'."
        ),
    )
    return parser
