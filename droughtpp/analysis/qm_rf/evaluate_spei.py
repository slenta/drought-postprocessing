from pathlib import Path

import numpy as np
import xarray as xr

from droughtpp.evaluation.evaluation import (
    mae_per_member_grid,
    brier_skill_score_between_ensembles,
)


def _load_var(path, var_name):
    with xr.open_dataset(path, decode_times=False) as ds:
        if var_name not in ds:
            raise KeyError(f"{var_name} not found in {path}")
        return ds[var_name].load()


def _infer_lat_lon_names(da):
    lat_name = "latitude" if "latitude" in da.dims else "lat"
    lon_name = "longitude" if "longitude" in da.dims else "lon"

    if lat_name not in da.dims or lon_name not in da.dims:
        raise ValueError(f"Could not infer latitude/longitude dims from {da.dims}")

    return lat_name, lon_name


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
    output_spei_path,
    gt_spei_path,
    reference_spei_path,
    output_var="spei",
    gt_var="spei",
    reference_var="spei",
    member_dim="member",
    std_multiplier=1.0,
    save_path=None,
):
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

    mae_output = np.nanmean(mae_output_member, axis=0)
    mae_gt = np.nanmean(mae_gt_member, axis=0)
    mae_diff = mae_output - mae_gt

    bss, bs_output, bs_gt = brier_skill_score_between_ensembles(
        output_np,
        gt_np,
        reference_np,
        std_multiplier=std_multiplier,
    )

    results = xr.Dataset(
        data_vars={
            "mae_output_member": (
                [member_dim, lat_name, lon_name],
                mae_output_member,
            ),
            "mae_gt_member": (
                [member_dim, lat_name, lon_name],
                mae_gt_member,
            ),
            "mae_output": ([lat_name, lon_name], mae_output),
            "mae_gt": ([lat_name, lon_name], mae_gt),
            "mae_diff": ([lat_name, lon_name], mae_diff),
            "bs_output": ([lat_name, lon_name], bs_output),
            "bs_gt": ([lat_name, lon_name], bs_gt),
            "bss": ([lat_name, lon_name], bss),
        },
        coords={
            member_dim: output_da.coords[member_dim],
            lat_name: reference_da.coords[lat_name],
            lon_name: reference_da.coords[lon_name],
        },
        attrs={
            "output_spei_path": str(output_spei_path),
            "gt_spei_path": str(gt_spei_path),
            "reference_spei_path": str(reference_spei_path),
        },
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_netcdf(save_path)

    return results
