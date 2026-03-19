import json
from pathlib import Path

import numpy as np
import xarray as xr

from droughtpp.evaluation.evaluation import (
    brier_skill_score_between_ensembles,
    mae_per_member_grid,
    rmse_per_member_grid,
)
from droughtpp.analysis.qm_rf.utils.visualization import (
    plot_mae_skill_metrics,
    plot_bss_skill_metrics,
)
from droughtpp.analysis.qm_rf.utils.evaluation import (
    load_paths_from_json,
    load_eval_data,
    select_eval_years,
    intersect_time_coordinates,
)


def _infer_lat_lon_names(da):
    lat_name = "latitude" if "latitude" in da.dims else "lat"
    lon_name = "longitude" if "longitude" in da.dims else "lon"

    if lat_name not in da.dims or lon_name not in da.dims:
        raise ValueError(f"Could not infer latitude/longitude dims from {da.dims}")

    return lat_name, lon_name


def _load_var(path, var_name):
    with xr.open_dataset(path, decode_times=False) as ds:
        if var_name not in ds:
            raise KeyError(f"{var_name} not found in {path}")
        return ds[var_name].load()


def _to_time_member_lat_lon(da, member_dim):
    lat_name, lon_name = _infer_lat_lon_names(da)

    if "time" not in da.dims:
        raise ValueError("Expected a time dimension in SPEI data.")

    if member_dim not in da.dims:
        da = da.expand_dims({member_dim: [0]})

    return da.transpose("time", member_dim, lat_name, lon_name)


def _to_time_lat_lon(da):
    lat_name, lon_name = _infer_lat_lon_names(da)

    if "time" not in da.dims:
        raise ValueError("Expected a time dimension in reference SPEI data.")

    return da.transpose("time", lat_name, lon_name)


def evaluate_spei(
    corrected_spei_paths,
    hindcasts_spei_json: Path,
    reference_spei_json: Path,
    out_dir: Path,
    spei_var: str = "spei",
    eval_years=None,
    std_multiplier: float = 1.0,
    plot_dir: Path | None = None,
):
    if eval_years is None:
        eval_years = []

    corrected_spei_paths = [Path(path) for path in corrected_spei_paths]

    baseline_spei_paths = load_paths_from_json(hindcasts_spei_json)
    reference_spei_paths = load_paths_from_json(reference_spei_json)

    if len(reference_spei_paths) == 0:
        raise ValueError("reference_spei_json does not contain any SPEI path.")

    if len(baseline_spei_paths) != len(corrected_spei_paths):
        raise ValueError(
            "Mismatch between number of baseline SPEI files and corrected SPEI files."
        )

    reference_spei = load_eval_data(reference_spei_paths[0], spei_var)
    reference_spei = select_eval_years(reference_spei, eval_years)

    corrected_members = []
    baseline_members = []
    reference_members = []

    for baseline_spei_path, corrected_spei_path in zip(
        baseline_spei_paths, corrected_spei_paths
    ):
        corrected_spei = load_eval_data(corrected_spei_path, spei_var)
        baseline_spei = load_eval_data(baseline_spei_path, spei_var)

        corrected_spei = select_eval_years(corrected_spei, eval_years)
        baseline_spei = select_eval_years(baseline_spei, eval_years)

        corrected_spei, baseline_spei, reference_member = xr.align(
            corrected_spei,
            baseline_spei,
            reference_spei,
            join="inner",
        )

        corrected_members.append(corrected_spei)
        baseline_members.append(baseline_spei)
        reference_members.append(reference_member)

    common_time = intersect_time_coordinates(
        corrected_members + baseline_members + reference_members
    )

    corrected_members = [member.sel(time=common_time) for member in corrected_members]
    baseline_members = [member.sel(time=common_time) for member in baseline_members]
    reference_members = [member.sel(time=common_time) for member in reference_members]

    corrected_ensemble = xr.concat(corrected_members, dim="member").transpose(
        "time", "member", "latitude", "longitude"
    )
    baseline_ensemble = xr.concat(baseline_members, dim="member").transpose(
        "time", "member", "latitude", "longitude"
    )
    reference = reference_members[0].transpose("time", "latitude", "longitude")

    corrected_np = corrected_ensemble.values
    baseline_np = baseline_ensemble.values
    reference_np = reference.values

    mae_corrected_member = mae_per_member_grid(corrected_np, reference_np)
    mae_baseline_member = mae_per_member_grid(baseline_np, reference_np)

    rmse_corrected_member = rmse_per_member_grid(corrected_np, reference_np)
    rmse_baseline_member = rmse_per_member_grid(baseline_np, reference_np)

    mae_corrected = np.nanmean(mae_corrected_member, axis=0)
    mae_baseline = np.nanmean(mae_baseline_member, axis=0)
    mae_diff = mae_corrected - mae_baseline

    rmse_corrected = np.nanmean(rmse_corrected_member, axis=0)
    rmse_baseline = np.nanmean(rmse_baseline_member, axis=0)
    rmse_diff = rmse_corrected - rmse_baseline

    bss_upper, bs_corrected_upper, bs_baseline_upper = (
        brier_skill_score_between_ensembles(
            corrected_np,
            baseline_np,
            reference_np,
            std_multiplier=std_multiplier,
            extreme_type="upper",
        )
    )
    bss_lower, bs_corrected_lower, bs_baseline_lower = (
        brier_skill_score_between_ensembles(
            corrected_np,
            baseline_np,
            reference_np,
            std_multiplier=-1.0,
            extreme_type="lower",
        )
    )

    if plot_dir is not None:
        plot_dir = Path(plot_dir) / "rf_spei_results"
        plot_dir.mkdir(parents=True, exist_ok=True)

        plot_mae_skill_metrics(
            mae_baseline_member,
            mae_corrected_member,
            out_dir=str(plot_dir),
            n_members_display=3,
            metric_name="MAE",
            file_prefix="mae",
        )

        plot_mae_skill_metrics(
            rmse_baseline_member,
            rmse_corrected_member,
            out_dir=str(plot_dir),
            n_members_display=3,
            metric_name="RMSE",
            file_prefix="rmse",
        )

        plot_bss_skill_metrics(
            bss_upper,
            bss_lower,
            out_dir=str(plot_dir),
            upper_threshold_label="mean + 1σ",
            lower_threshold_label="mean - 1σ",
        )

    return {
        "spei_mae_corrected_mean": float(np.nanmean(mae_corrected)),
        "spei_mae_baseline_mean": float(np.nanmean(mae_baseline)),
        "spei_mae_diff_mean": float(np.nanmean(mae_diff)),
        "spei_rmse_corrected_mean": float(np.nanmean(rmse_corrected)),
        "spei_rmse_baseline_mean": float(np.nanmean(rmse_baseline)),
        "spei_rmse_diff_mean": float(np.nanmean(rmse_diff)),
        "spei_bs_corrected_upper_mean": float(np.nanmean(bs_corrected_upper)),
        "spei_bs_baseline_upper_mean": float(np.nanmean(bs_baseline_upper)),
        "spei_bss_upper_mean": float(np.nanmean(bss_upper)),
        "spei_bs_corrected_lower_mean": float(np.nanmean(bs_corrected_lower)),
        "spei_bs_baseline_lower_mean": float(np.nanmean(bs_baseline_lower)),
        "spei_bss_lower_mean": float(np.nanmean(bss_lower)),
        "spei_n_members": int(corrected_ensemble.sizes["member"]),
        "spei_n_time": int(corrected_ensemble.sizes["time"]),
    }


def evaluate_spei_pair(
    output_spei_path,
    gt_spei_path,
    reference_spei_path,
    output_var="spei",
    gt_var="spei",
    reference_var="spei",
    member_dim="member",
    std_multiplier=1.0,
    plot_dir=None,
):
    """
    Generic SPEI evaluation: compare two SPEI outputs against a reference.
    Returns xr.Dataset with detailed metrics.
    """
    output_da = _load_var(output_spei_path, output_var)
    gt_da = _load_var(gt_spei_path, gt_var)
    reference_da = _load_var(reference_spei_path, reference_var)

    output_da, gt_da, reference_da = xr.align(
        output_da, gt_da, reference_da, join="inner"
    )

    output_da = _to_time_member_lat_lon(output_da, member_dim)
    gt_da = _to_time_member_lat_lon(gt_da, member_dim)
    reference_da = _to_time_lat_lon(reference_da)

    lat_name, lon_name = _infer_lat_lon_names(reference_da)

    output_np = output_da.values
    gt_np = gt_da.values
    reference_np = reference_da.values

    mae_output_member = mae_per_member_grid(output_np, reference_np)
    mae_gt_member = mae_per_member_grid(gt_np, reference_np)

    rmse_output_member = rmse_per_member_grid(output_np, reference_np)
    rmse_gt_member = rmse_per_member_grid(gt_np, reference_np)

    mae_output = np.nanmean(mae_output_member, axis=0)
    mae_gt = np.nanmean(mae_gt_member, axis=0)
    mae_diff = mae_output - mae_gt

    rmse_output = np.nanmean(rmse_output_member, axis=0)
    rmse_gt = np.nanmean(rmse_gt_member, axis=0)
    rmse_diff = rmse_output - rmse_gt

    bss_upper, bs_output_upper, bs_gt_upper = brier_skill_score_between_ensembles(
        output_np,
        gt_np,
        reference_np,
        std_multiplier=std_multiplier,
        extreme_type="upper",
    )
    bss_lower, bs_output_lower, bs_gt_lower = brier_skill_score_between_ensembles(
        output_np,
        gt_np,
        reference_np,
        std_multiplier=-1.0,
        extreme_type="lower",
    )

    if plot_dir is not None:
        plot_dir = Path(plot_dir) / "rf_spei_results"
        plot_dir.mkdir(parents=True, exist_ok=True)

        plot_mae_skill_metrics(
            mae_gt_member,
            mae_output_member,
            out_dir=str(plot_dir),
            n_members_display=3,
            metric_name="MAE",
            file_prefix="mae",
        )

        plot_mae_skill_metrics(
            rmse_gt_member,
            rmse_output_member,
            out_dir=str(plot_dir),
            n_members_display=3,
            metric_name="RMSE",
            file_prefix="rmse",
        )

        plot_bss_skill_metrics(
            bss_upper,
            bss_lower,
            out_dir=str(plot_dir),
            upper_threshold_label="mean + 1σ",
            lower_threshold_label="mean - 1σ",
        )

    return {
        "mae_output_mean": float(np.nanmean(mae_output)),
        "mae_gt_mean": float(np.nanmean(mae_gt)),
        "mae_diff_mean": float(np.nanmean(mae_diff)),
        "rmse_output_mean": float(np.nanmean(rmse_output)),
        "rmse_gt_mean": float(np.nanmean(rmse_gt)),
        "rmse_diff_mean": float(np.nanmean(rmse_diff)),
        "bss_upper_mean": float(np.nanmean(bss_upper)),
        "bss_lower_mean": float(np.nanmean(bss_lower)),
    }
