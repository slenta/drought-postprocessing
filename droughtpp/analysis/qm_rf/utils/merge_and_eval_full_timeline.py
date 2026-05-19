#!/usr/bin/env python3
"""Merge per-year RF outputs into full-timeline member files and run evaluation.

Usage:
    python -m droughtpp.analysis.qm_rf.utils.merge_and_eval_full_timeline --config droughtpp/analysis/qm_rf/config.yaml

This script DOES NOT wait for jobs; it assumes you run it after per-year jobs finished.
It concatenates per-year outputs, writes combined JSONs, and runs full-timeline evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List
from IPython import embed
import numpy as np

import xarray as xr

from droughtpp.analysis.qm_rf.config_loader import get_qm_rf_global_config
from droughtpp.analysis.qm_rf.utils.cwb_evaluation import evaluate_cwb
from droughtpp.analysis.qm_rf.utils.visualization import plot_leadmonth_skill_timeseries


def load_json(path: Path) -> List[str]:
    with open(path, "r") as fh:
        return json.load(fh)


def write_json(path: Path, data: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def concat_member_files(paths: List[str], var_name: str, out_path: Path) -> Path:
    das = []
    for p in paths:
        with xr.open_dataset(p) as ds:
            das.append(ds[var_name].load())

    combined = xr.concat(das, dim="time").sortby("time")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out = combined.to_dataset(name=var_name)
    ds_out.to_netcdf(str(out_path))
    ds_out.close()
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("droughtpp/analysis/qm_rf/config.yaml"))
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    args = parser.parse_args()

    cfg = get_qm_rf_global_config(default_config=args.config)
    n_features = cfg["n_features"]
    output_dir = Path(cfg["output_dir"])
    var_name = cfg["var_name"]
    training_target = str(cfg.get("training_target", "residual")).lower()
    rf_results_tag_base = str(cfg["rf_results_tag"])
    results_tag = f"{rf_results_tag_base}_nf{n_features}"

    target_root = Path(var_name) / Path(training_target) / Path(results_tag)

    start_year, end_year = args.start_year, args.end_year

    years = list(range(start_year, end_year + 1))

    leadmonths = cfg["leadmonth"]
    per_lead_combined = {}
    leadmonth_skill_summary = {}

    for leadmonth in leadmonths:
        print(f"Processing leadmonth {leadmonth}...")
        per_year_corrected = []
        per_year_residuals = []
        per_year_spei = []
        for y in years:
            paths_dir = Path(output_dir) / Path("paths") / Path(var_name) / Path(training_target) / Path(results_tag) / Path(str(y)) / Path("rf_eval") / Path(f"lm{leadmonth}")
            corrected_json = paths_dir / "corrected_cwb_paths.json"
            residual_json = paths_dir / "corrected_residuals_paths.json"
            spei_json = paths_dir / "corrected_spei_paths.json"
            per_year_corrected.append(corrected_json)
            if residual_json.exists():
                per_year_residuals.append(residual_json)
            if spei_json.exists():
                per_year_spei.append(spei_json)

        per_year_lists = [load_json(p) for p in per_year_corrected]
        n_members = len(per_year_lists[0])

        full_timeline_root = target_root / Path("full_timeline") / Path("rf_eval") / Path(f"lm{leadmonth}")
        out_base = output_dir / Path("data") / full_timeline_root
        out_base.mkdir(parents=True, exist_ok=True)
        combined_json = output_dir / Path("paths") / full_timeline_root / Path("corrected_cwb_paths.json")
        combined_res_json = output_dir / Path("paths") / full_timeline_root / Path("corrected_residuals_paths.json")

        if cfg["workflow"].get("run_rf_evaluate") == True:
            combined_paths = []
            for member_idx in range(n_members):
                member_paths = [per_year_lists[yr_idx][member_idx] for yr_idx in range(len(years))]
                last_stem = Path(member_paths[-1]).stem
                out_path = out_base / Path(f"{last_stem}_fulltimeline_lm{leadmonth}.nc")
                print(f"Concatenating member {member_idx+1}/{n_members} -> {out_path}")
                concat_member_files(member_paths, var_name, out_path)
                combined_paths.append(str(out_path))

            write_json(combined_json, combined_paths)
            print(f"Wrote combined corrected_cwb_paths.json: {combined_json}")
        else:
            combined_paths = load_json(combined_json)

        per_lead_combined[int(leadmonth)] = combined_paths

        if cfg["workflow"].get("run_rf_evaluate") == True:
            combined_res_paths = []
            per_year_res_lists = [load_json(p) for p in per_year_residuals]
            res_out_base = out_base
            for member_idx in range(n_members):
                member_res_paths = [per_year_res_lists[yr_idx][member_idx] for yr_idx in range(len(years))]
                last_stem = Path(member_res_paths[-1]).stem
                out_path = res_out_base / f"{last_stem}_residuals_fulltimeline_lm{leadmonth}.nc"
                print(f"Concatenating residual member {member_idx+1}/{n_members} -> {out_path}")
                concat_member_files(member_res_paths, var_name, out_path)
                combined_res_paths.append(str(out_path))
            write_json(combined_res_json, combined_res_paths)
            print(f"Wrote combined corrected_residuals_paths.json: {combined_res_json}")
        else:
            combined_res_paths = load_json(combined_res_json)


        full_paths_dir = output_dir / Path("paths") / full_timeline_root
        corrected_json = full_paths_dir / Path("corrected_cwb_paths.json")
        residuals_json = full_paths_dir / Path("corrected_residuals_paths.json") if (full_paths_dir / Path("corrected_residuals_paths.json")).exists() else None
        spei_json = full_paths_dir / Path("corrected_spei_paths.json") if (full_paths_dir / Path("corrected_spei_paths.json")).exists() else None

        out_data_dir = output_dir / Path("data") / full_timeline_root
        plot_dir = Path(cfg["plot_dir"]) / Path(var_name) / Path(training_target) / Path(results_tag) / Path("full_timeline") / Path("rf_eval") / Path(f"lm{leadmonth}")

        if cfg["workflow"].get("evaluate_cwb") == True:
            eval_years = years
            print(f"Running full-timeline evaluation for lm{leadmonth} on years {eval_years}...")
            leadmonth_skill_summary[int(leadmonth)] = evaluate_cwb(
                corrected_cwb_json=corrected_json,
                hindcasts_json=Path(cfg["hindcasts_json"]),
                reference_data=Path(cfg["reference_data"]),
                out_dir=out_data_dir,
                cwb_var=cfg["var_name"],
                eval_years=eval_years,
                std_multiplier=float(cfg.get("spei_std_multiplier", 1.0)),
                plot_dir=str(plot_dir),
                qm_hindcasts_json=Path(cfg["output_dir"]) / Path("paths") / Path(cfg["var_name"]) / Path(f"nq{cfg['qm_arguments'].get('n_quantiles')}") / Path(f"lm{leadmonth}") / Path("qm_hindcast_paths.json"),
                qm_residuals_json=(Path(cfg["output_dir"]) / Path("paths") / Path(cfg["var_name"]) / Path(f"nq{cfg['qm_arguments'].get('n_quantiles')}") / Path(f"lm{leadmonth}") / Path("qm_residuals_paths.json")),
                ml_residuals_json=residuals_json,
                land_mask_path=Path(cfg.get("land_mask_path")) if cfg.get("land_mask_path") else None,
                stipple_significant_correlation=bool(cfg["workflow"].get("stipple_significant_correlation", True)),
                correlation_significance_level=float(cfg["workflow"].get("correlation_significance_level", 0.05)),
            )
            print(f"Completed full-timeline evaluation for lm{leadmonth}.")

    print("All leadmonths processed.")

    # Create combined analysis across all leadmonths (lm_all)
    all_lead_jsons = []
    all_lead_qm_jsons = []
    for leadmonth in leadmonths:
        jm = output_dir / Path("paths") / target_root / Path("full_timeline") / Path("rf_eval") / Path(f"lm{leadmonth}") / Path("corrected_cwb_paths.json")
        if jm.exists():
            all_lead_jsons.append(jm)
        qm_jm = (
            Path(cfg["output_dir"])
            / Path("paths")
            / Path(cfg["var_name"])
            / Path(f"nq{cfg['qm_arguments'].get('n_quantiles')}")
            / Path(f"lm{leadmonth}")
            / Path("qm_hindcast_paths.json")
        )
        if qm_jm.exists():
            all_lead_qm_jsons.append(qm_jm)

    if all_lead_jsons:
        print("Creating combined full-timeline across all leadmonths (lm_all) ...")
        # load per-lead lists
        per_lead_lists = [load_json(p) for p in all_lead_jsons]
        n_members_all = len(per_lead_lists[0])
        per_lead_qm_lists = [load_json(p) for p in all_lead_qm_jsons] if all_lead_qm_jsons else []

        full_timeline_all_root = target_root / Path("full_timeline") / Path("rf_eval") / Path("lm_all")
        out_base_all = output_dir / Path("data") / full_timeline_all_root
        out_base_all.mkdir(parents=True, exist_ok=True)
        combined_all_json = output_dir / Path("paths") / full_timeline_all_root / Path("corrected_cwb_paths.json")
        combined_all_qm_json = output_dir / Path("paths") / full_timeline_all_root / Path("qm_hindcast_paths.json")

        if cfg["workflow"].get("run_rf_evaluate") == True:
            combined_all_paths = []
            for member_idx in range(n_members_all):
                member_files = [per_lead_lists[i][member_idx] for i in range(len(per_lead_lists))]
                last_stem = Path(member_files[-1]).stem
                base_stem = last_stem.split("_lm", 1)[0]
                out_path = out_base_all / Path(f"{base_stem}_ml_fulltimeline_lm_all.nc")
                print(f"Concatenating across leadmonths member {member_idx+1}/{n_members_all} -> {out_path}")
                concat_member_files(member_files, var_name, out_path)
                combined_all_paths.append(str(out_path))

            write_json(combined_all_json, combined_all_paths)
            print(f"Wrote combined corrected_cwb_paths.json for lm_all: {combined_all_json}")
        else:
            combined_all_paths = load_json(combined_all_json)


        if cfg["workflow"].get("run_rf_evaluate") == True:
            combined_all_qm_paths = []
            n_qm_members_all = len(per_lead_qm_lists[0])
            if n_qm_members_all == n_members_all:
                for member_idx in range(n_qm_members_all):
                    member_files = [per_lead_qm_lists[i][member_idx] for i in range(len(per_lead_qm_lists))]
                    last_stem = Path(member_files[-1]).stem
                    base_stem = last_stem.split("_lm", 1)[0]
                    out_path = out_base_all / Path(f"{base_stem}_qm_fulltimeline_lm_all.nc")
                    print(f"Concatenating QM across leadmonths member {member_idx+1}/{n_qm_members_all} -> {out_path}")
                    concat_member_files(member_files, var_name, out_path)
                    combined_all_qm_paths.append(str(out_path))

                write_json(combined_all_qm_json, combined_all_qm_paths)
                print(f"Wrote combined qm_hindcast_paths.json for lm_all: {combined_all_qm_json}")
        else:
            combined_all_qm_paths = load_json(combined_all_qm_json)


        # Run a single evaluation across all leadmonths combined
        out_data_dir = output_dir / Path("data") / full_timeline_all_root
        plot_dir = Path(cfg["plot_dir"]) / Path(var_name) / Path(training_target) / Path(results_tag) / Path("full_timeline") / Path("rf_eval") / Path("lm_all")
        if cfg["workflow"].get("evaluate_cwb") == True:
            print("Running full-timeline evaluation for lm_all (all leadmonths combined)...")
            evaluate_cwb(
                corrected_cwb_json=combined_all_json,
                hindcasts_json=Path(cfg["hindcasts_json"]),
                reference_data=Path(cfg["reference_data"]),
                out_dir=out_data_dir,
                cwb_var=cfg["var_name"],
                eval_years=years,
                std_multiplier=float(cfg.get("spei_std_multiplier", 1.0)),
                plot_dir=str(plot_dir),
                qm_hindcasts_json=combined_all_qm_json if combined_all_qm_paths else None,
                qm_residuals_json=None,
                ml_residuals_json=None,
                land_mask_path=Path(cfg.get("land_mask_path")) if cfg.get("land_mask_path") else None,
                compute_monthly_anomalies=True,
                stipple_significant_correlation=bool(cfg["workflow"].get("stipple_significant_correlation", True)),
                correlation_significance_level=float(cfg["workflow"].get("correlation_significance_level", 0.05)),
            )
            print("Completed full-timeline evaluation for lm_all.")

        if leadmonth_skill_summary:
            leadmonth_keys = sorted(leadmonth_skill_summary)
            original_acc = [leadmonth_skill_summary[k]["original_acc"] for k in leadmonth_keys]
            ml_acc = [leadmonth_skill_summary[k]["ml_acc"] for k in leadmonth_keys]
            qm_acc = [leadmonth_skill_summary[k].get("qm_acc", np.nan) for k in leadmonth_keys]
            print(leadmonth_skill_summary)

            skill_plot_dir = Path(cfg["plot_dir"]) / Path(var_name) / Path(training_target) / Path(results_tag) / Path("full_timeline") / Path("rf_eval") / Path("lm_all")
            plot_leadmonth_skill_timeseries(
                leadmonths=leadmonth_keys,
                original_acc=original_acc,
                ml_acc=ml_acc,
                qm_acc=qm_acc if any(np.isfinite(qm_acc)) else None,
                output_path=skill_plot_dir / "hindcast_skill_per_leadmonth.png",
                title="Hindcast skill per leadmonth",
            )


if __name__ == "__main__":
    main()
