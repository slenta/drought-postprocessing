import json
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps
import xarray as xr

from droughtpp.evaluation.evaluation import (
    brier_skill_score_between_ensembles,
    mae_per_member_grid,
)
from droughtpp.evaluation.spei_calculation.spei_calc import compute_spei_from_surplus


def _ensure_spatial_dims(data_array: xr.DataArray) -> xr.DataArray:
    rename_map = {}
    if "lat" in data_array.dims:
        rename_map["lat"] = "latitude"
    if "lon" in data_array.dims:
        rename_map["lon"] = "longitude"
    if rename_map:
        data_array = data_array.rename(rename_map)
    return data_array


def _select_eval_years(data_array: xr.DataArray, eval_years) -> xr.DataArray:
    if not eval_years:
        return data_array

    years = pd.to_datetime(data_array["time"].values).year
    return data_array.isel(time=np.isin(years, eval_years))


def _load_spei_data(path: Path, var_name: str) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        data_array = ds[var_name].load()

    data_array = _ensure_spatial_dims(data_array).squeeze()
    if "time" not in data_array.dims:
        raise ValueError(f"Expected time dimension in {path}")

    return data_array.dropna("time", how="all")


def _compute_spei_from_path(
    path: Path,
    var_name: str,
    month_range,
    eval_years,
) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        surplus = ds[var_name].load()

    surplus = _ensure_spatial_dims(surplus).squeeze()
    surplus = _select_eval_years(surplus, eval_years)
    return compute_spei_from_surplus(
        surplus,
        month_range=month_range,
        var_names=[var_name, "spei"],
        dist=sps.fisk,
    )


def _intersect_time_coordinates(data_arrays):
    common_time = reduce(
        np.intersect1d,
        [np.asarray(data_array["time"].values) for data_array in data_arrays],
    )
    if len(common_time) == 0:
        raise ValueError("No common SPEI time steps found across evaluation inputs.")
    return common_time


def evaluate_spei(
    corrected_spei_paths,
    hindcasts_json: Path,
    reference_data: Path,
    out_dir: Path,
    surplus_var: str = "CWB",
    spei_var: str = "spei",
    month_range=(1, 3),
    eval_years=None,
    std_multiplier: float = 1.0,
):
    if eval_years is None:
        eval_years = []

    corrected_spei_paths = [Path(path) for path in corrected_spei_paths]
    if not corrected_spei_paths:
        raise ValueError("No corrected SPEI files were provided for evaluation.")

    with open(hindcasts_json, "r") as fh:
        hindcast_paths = [Path(path) for path in json.load(fh)]

    if len(hindcast_paths) != len(corrected_spei_paths):
        raise ValueError(
            "Mismatch between number of hindcast files and corrected SPEI files."
        )

    reference_spei = _compute_spei_from_path(
        Path(reference_data),
        var_name=surplus_var,
        month_range=month_range,
        eval_years=eval_years,
    )

    corrected_members = []
    baseline_members = []
    reference_members = []

    for hindcast_path, corrected_spei_path in zip(hindcast_paths, corrected_spei_paths):
        corrected_spei = _load_spei_data(corrected_spei_path, spei_var)
        baseline_spei = _compute_spei_from_path(
            hindcast_path,
            var_name=surplus_var,
            month_range=month_range,
            eval_years=eval_years,
        )

        corrected_spei, baseline_spei, reference_member = xr.align(
            corrected_spei,
            baseline_spei,
            reference_spei,
            join="inner",
        )

        corrected_members.append(corrected_spei)
        baseline_members.append(baseline_spei)
        reference_members.append(reference_member)

    common_time = _intersect_time_coordinates(
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

    mae_corrected = np.nanmean(mae_corrected_member, axis=0)
    mae_baseline = np.nanmean(mae_baseline_member, axis=0)
    mae_diff = mae_corrected - mae_baseline

    bss, bs_corrected, bs_baseline = brier_skill_score_between_ensembles(
        corrected_np,
        baseline_np,
        reference_np,
        std_multiplier=std_multiplier,
    )

    spei_results = xr.Dataset(
        data_vars={
            "mae_corrected_member": (
                ["member", "latitude", "longitude"],
                mae_corrected_member,
            ),
            "mae_baseline_member": (
                ["member", "latitude", "longitude"],
                mae_baseline_member,
            ),
            "mae_corrected": (["latitude", "longitude"], mae_corrected),
            "mae_baseline": (["latitude", "longitude"], mae_baseline),
            "mae_diff": (["latitude", "longitude"], mae_diff),
            "bs_corrected": (["latitude", "longitude"], bs_corrected),
            "bs_baseline": (["latitude", "longitude"], bs_baseline),
            "bss": (["latitude", "longitude"], bss),
        },
        coords={
            "member": np.arange(corrected_ensemble.sizes["member"]),
            "latitude": corrected_ensemble["latitude"].values,
            "longitude": corrected_ensemble["longitude"].values,
        },
        attrs={
            "month_range": str(tuple(month_range)),
            "surplus_var": surplus_var,
            "spei_var": spei_var,
            "std_multiplier": float(std_multiplier),
            "n_members": int(corrected_ensemble.sizes["member"]),
            "n_years": int(corrected_ensemble.sizes["time"]),
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    spei_results_path = out_dir / "spei_evaluation.nc"
    spei_results.to_netcdf(spei_results_path)

    metrics = {
        "spei_results_path": str(spei_results_path),
        "spei_mae_corrected_mean": float(np.nanmean(mae_corrected)),
        "spei_mae_baseline_mean": float(np.nanmean(mae_baseline)),
        "spei_mae_diff_mean": float(np.nanmean(mae_diff)),
        "spei_bs_corrected_mean": float(np.nanmean(bs_corrected)),
        "spei_bs_baseline_mean": float(np.nanmean(bs_baseline)),
        "spei_bss_mean": float(np.nanmean(bss)),
        "spei_n_members": int(corrected_ensemble.sizes["member"]),
        "spei_n_years": int(corrected_ensemble.sizes["time"]),
    }

    return metrics