#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def load_yaml(path: Path) -> dict:
    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def normalize_grid(grid: Dict[str, List]) -> Dict[str, List]:
    out: Dict[str, List] = {}
    for key, values in grid.items():
        if isinstance(values, list):
            out[key] = values
        else:
            out[key] = [values]
    return out


def deduplicate_combos(combos: List[Dict[str, object]]) -> List[Dict[str, object]]:
    seen = set()
    unique: List[Dict[str, object]] = []
    for combo in combos:
        if str(combo.get("ml_arguments.model_type", "")).lower() != "xgboost":
            combo = {
                k: v for k, v in combo.items() if not k.startswith("ml_arguments.xgb_")
            }
        key = tuple(sorted(combo.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(combo)
    return unique


def build_combinations(grid: Dict[str, List]) -> List[Dict[str, object]]:
    keys = list(grid.keys())
    values_product = itertools.product(*(grid[k] for k in keys))
    combos = [dict(zip(keys, vals)) for vals in values_product]
    return deduplicate_combos(combos)


def build_job_name(prefix: str, combo: Dict[str, object], idx: int) -> str:
    model_type = str(combo.get("ml_arguments.model_type", "model"))
    n_est = combo.get("ml_arguments.n_estimators", "na")
    rs = combo.get("ml_arguments.random_state", "na")
    return f"{prefix}-{idx:03d}-{model_type}-n{n_est}-rs{rs}"


def build_command(
    project_root: Path,
    config_path: str,
    workflow_overrides: List[str],
    combo: Dict[str, object],
) -> str:
    parts: List[str] = [
        "python",
        "-m",
        "droughtpp.analysis.qm_rf.main_qm_rf",
        "--config",
        config_path,
    ]
    for item in workflow_overrides:
        parts.extend(["--set", item])
    for key, value in combo.items():
        parts.extend(["--set", f"{key}={value}"])
    return " ".join(parts)


def submit_job(
    sbatch_cfg: dict,
    project_root: Path,
    job_name: str,
    wrapped_cmd: str,
    dry_run: bool,
) -> Tuple[bool, str]:
    sbatch_cmd = ["sbatch"]

    partition = sbatch_cfg.get("partition")
    account = sbatch_cfg.get("account")
    time = sbatch_cfg.get("time")
    mem = sbatch_cfg.get("mem")
    constraint = sbatch_cfg.get("constraint")
    output = sbatch_cfg.get("output")
    exclusive = bool(sbatch_cfg.get("exclusive", False))

    if partition:
        sbatch_cmd.extend(["--partition", str(partition)])
    if account:
        sbatch_cmd.extend(["--account", str(account)])
    if time:
        sbatch_cmd.extend(["--time", str(time)])
    if mem:
        sbatch_cmd.extend(["--mem", str(mem)])
    if constraint:
        sbatch_cmd.extend(["--constraint", str(constraint)])
    if output:
        sbatch_cmd.extend(["--output", str(output)])
    if exclusive:
        sbatch_cmd.append("--exclusive")

    sbatch_cmd.extend(["--job-name", job_name])
    sbatch_cmd.extend(["--chdir", str(project_root)])
    sbatch_cmd.extend(["--wrap", wrapped_cmd])

    res = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    return True, res.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    default_grid = Path(__file__).resolve().parent / "rf-sweep-grid.yaml"
    parser.add_argument(
        "--grid-file",
        default=str(default_grid),
        help="Path to YAML file with sbatch config and hyperparameter grid",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting",
    )
    args = parser.parse_args()

    grid_path = Path(args.grid_file).resolve()
    cfg = load_yaml(grid_path)

    project_root = Path(cfg["project_root"]).resolve()
    config_path = str(cfg["config_path"])
    sbatch_cfg = dict(cfg.get("sbatch", {}))
    workflow_overrides = list(cfg.get("workflow_overrides", []))
    grid = normalize_grid(dict(cfg.get("grid", {})))

    combos = build_combinations(grid)
    prefix = str(sbatch_cfg.get("job_name_prefix", "qm-rf-grid"))

    print(f"Submitting {len(combos)} jobs from grid: {grid_path}")
    for idx, combo in enumerate(combos):
        job_name = build_job_name(prefix, combo, idx)
        wrapped_cmd = build_command(
            project_root, config_path, workflow_overrides, combo
        )
        ok, msg = submit_job(
            sbatch_cfg=sbatch_cfg,
            project_root=project_root,
            job_name=job_name,
            wrapped_cmd=wrapped_cmd,
            dry_run=args.dry_run,
        )
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {job_name}: {msg}")


if __name__ == "__main__":
    main()
