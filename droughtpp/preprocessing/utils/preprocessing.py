from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from IPython import embed
import pandas as pd

import xarray as xr

from droughtpp.preprocessing.utils.cwb_calc import compute_cwb_global
from droughtpp.preprocessing.utils.cwb_leadmonth_merge import (
    combine_cwb_leadmonths_by_month_range,
)


def resolve_path(path_value: str | Path, config_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def load_path_list_json(path: Path) -> list[str]:
    with open(path, "r") as file_handle:
        data = json.load(file_handle)

    if isinstance(data, list):
        return [str(item) for item in data]

    if isinstance(data, dict) and "paths" in data and isinstance(data["paths"], list):
        return [str(item) for item in data["paths"]]


def load_leadmonth_mapping_json(path: Path) -> dict[int, str]:
    with open(path, "r") as file_handle:
        data = json.load(file_handle)

    parsed: dict[int, str] = {}
    for key, value in data.items():
        parsed[int(key)] = str(value)

    return parsed


def load_hindcast_pair_mapping_json(path: Path) -> dict[int, dict[str, str]]:
    with open(path, "r") as file_handle:
        data = json.load(file_handle)

    parsed: dict[int, dict[str, str]] = {}
    for key, value in data.items():

        precip_path = value.get("precip")
        temp_path = value.get("temp")

        parsed[int(key)] = {
            "precip": str(precip_path),
            "temp": str(temp_path),
        }

    return parsed


def write_paths_json(paths: list[str], out_path: Path) -> None:
    with open(out_path, "w") as file_handle:
        json.dump(paths, file_handle, indent=2)


def infer_member_dim(
    precip_da: xr.DataArray,
    temperature_da: xr.DataArray,
    dataset_cfg: dict[str, Any],
) -> str | None:
    if not bool(dataset_cfg.get("member", True)):
        return None

    for candidate in ("number", "member"):
        if candidate in precip_da.dims and candidate in temperature_da.dims:
            return candidate

    return None


def shift_dataset_time_by_leadmonth(result_ds: xr.Dataset, leadmonth: int) -> xr.Dataset:
    month_shift = max(int(leadmonth) - 1, 0)
    if month_shift == 0 or "time" not in result_ds.coords:
        return result_ds

    shifted_time = pd.to_datetime(result_ds["time"].values) + pd.DateOffset(
        months=month_shift
    )
    return result_ds.assign_coords(time=shifted_time)


def process_dataset_pair(
    precip_path: str,
    temperature_path: str,
    dataset_cfg: dict[str, Any],
    out_path: Path,
    output_var_name: str,
    config_dir: Path,
) -> str:
    precip_ds = xr.open_dataset(resolve_path(precip_path, config_dir))
    temperature_ds = xr.open_dataset(resolve_path(temperature_path, config_dir))

    result_ds = compute_cwb_global(
        precip=precip_ds,
        temp=temperature_ds,
        var_names=[
            dataset_cfg["temp_var"],
            dataset_cfg["precip_var"],
            output_var_name,
            dataset_cfg["lat_name"],
            dataset_cfg["lon_name"],
        ],
        temp_unit=dataset_cfg.get("temp_unit", "K"),
        member=bool(dataset_cfg.get("member")),
    )
    precip_ds.close()
    temperature_ds.close()

    result_ds.to_netcdf(
        out_path,
        encoding={
            "time": {
                "units": "days since 1900-01-01",
                "calendar": "standard",
                "dtype": "f8",
            }
        },
    )
    result_ds.close()
    return str(out_path)


def process_group(
    precip_paths: list[str],
    temperature_paths: list[str],
    dataset_cfg: dict[str, Any],
    output_dir: Path,
    output_var_name: str,
    group_prefix: str,
    config_dir: Path,
) -> list[str]:

    output_paths: list[str] = []
    for index, (precip_path, temperature_path) in enumerate(
        zip(precip_paths, temperature_paths)
    ):
        output_filename = f"{group_prefix}_cwb_{index:03d}.nc"
        out_path = output_dir / output_filename
        output_paths.append(
            process_dataset_pair(
                precip_path=precip_path,
                temperature_path=temperature_path,
                dataset_cfg=dataset_cfg,
                out_path=out_path,
                output_var_name=output_var_name,
                config_dir=config_dir,
            )
        )

    return output_paths


def process_hindcast_leadmonth(
    precip_path: str,
    temperature_path: str,
    dataset_cfg: dict[str, Any],
    output_dir: Path,
    output_var_name: str,
    leadmonth: int,
    config_dir: Path,
) -> list[str]:
    precip_ds = xr.open_dataset(resolve_path(precip_path, config_dir))
    temperature_ds = xr.open_dataset(resolve_path(temperature_path, config_dir))

    member_dim = infer_member_dim(
        precip_ds[dataset_cfg["precip_var"]],
        temperature_ds[dataset_cfg["temp_var"]],
        dataset_cfg,
    )

    if member_dim is None:
        out_path = output_dir / f"hindcast_lm{leadmonth}_cwb_000.nc"
        return [
            process_dataset_pair(
                precip_path=precip_path,
                temperature_path=temperature_path,
                dataset_cfg=dataset_cfg,
                out_path=out_path,
                output_var_name=output_var_name,
                config_dir=config_dir,
            )
        ]

    total_members = int(precip_ds.sizes[member_dim])
    configured_n_members = dataset_cfg.get("n_members", None)
    if configured_n_members is None:
        n_members = total_members
    else:
        n_members = min(total_members, int(configured_n_members))

    output_paths: list[str] = []
    for member_index in range(n_members):
        precip_member = precip_ds.isel({member_dim: member_index})
        temperature_member = temperature_ds.isel({member_dim: member_index})

        result_ds = compute_cwb_global(
            precip=precip_member,
            temp=temperature_member,
            var_names=[
                dataset_cfg["temp_var"],
                dataset_cfg["precip_var"],
                output_var_name,
                dataset_cfg["lat_name"],
                dataset_cfg["lon_name"],
            ],
            temp_unit=dataset_cfg.get("temp_unit", "K"),
            member=False,
        )
        result_ds = shift_dataset_time_by_leadmonth(result_ds, leadmonth)
        out_path = (
            output_dir / f"hindcast_lm{leadmonth}_cwb_member_{member_index:03d}.nc"
        )
        result_ds.to_netcdf(
            out_path,
            encoding={
                "time": {
                    "units": "days since 1900-01-01",
                    "calendar": "standard",
                    "dtype": "f8",
                }
            },
        )
        result_ds.close()
        output_paths.append(str(out_path))

    precip_ds.close()
    temperature_ds.close()

    return output_paths


def combine_hindcast_month_window(
    by_leadmonth_outputs: dict[int, list[str]],
    leadmonths: list[int],
    start_month: int,
    end_month: int,
    output_dir: Path,
    output_paths_dir: Path,
) -> tuple[list[str], str]:
    first_leadmonth = int(min(leadmonths))
    n_files = len(by_leadmonth_outputs[first_leadmonth])

    merged_outputs: list[str] = []
    for file_index in range(n_files):
        leadmonth_file_map = {
            int(leadmonth): by_leadmonth_outputs[int(leadmonth)][file_index]
            for leadmonth in leadmonths
        }
        out_path = output_dir / (
            f"hindcast_cwb_month_window_{start_month:02d}-{end_month:02d}_{file_index:03d}.nc"
        )
        combine_cwb_leadmonths_by_month_range(
            leadmonth_to_file=leadmonth_file_map,
            start_month=start_month,
            end_month=end_month,
            out_path=out_path,
        )
        merged_outputs.append(str(out_path))

    merged_paths_json = (
        output_paths_dir
        / f"hindcast_cwb_month_window_paths_{start_month:02d}-{end_month:02d}.json"
    )
    write_paths_json(merged_outputs, merged_paths_json)
    return merged_outputs, str(merged_paths_json)


def compute_ensemble_monthly_thresholds(
    cwb_paths: list[str],
    lower_threshold_percentile: float | None = None,
    input_var_name: str = "CWB",
) -> dict[int, xr.DataArray]:

    lat_dim = "latitude"
    lon_dim = "longitude"

    datasets: list[xr.Dataset] = []
    for cwb_path in cwb_paths:
        datasets.append(xr.open_dataset(cwb_path))

    combined = xr.concat([ds[input_var_name] for ds in datasets], dim="time")
    for ds in datasets:
        ds.close()

    monthly_lower_thresholds: dict[int, xr.DataArray] = {}

    for month, month_data in combined.groupby("time.month"):
        month = int(month)
        reduce_dims = [dim for dim in month_data.dims if dim not in (lat_dim, lon_dim)]
        if lower_threshold_percentile is not None:
            lower_threshold_da = month_data.quantile(
                q=lower_threshold_percentile / 100.0,
                dim=reduce_dims,
                skipna=True,
            ).squeeze(drop=True)
            monthly_lower_thresholds[month] = lower_threshold_da

    return monthly_lower_thresholds


def compute_threshold_exceedance_intensity(
    cwb_path: str,
    monthly_lower_thresholds: dict[int, xr.DataArray] | None,
    output_data_dir: Path,
    input_var_name: str = "CWB",
    output_var_name: str = "CWB_INTENSITY",
) -> str:

    output_data_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(cwb_path)
    with xr.open_dataset(source_path) as source_ds:
        cwb_da = source_ds[input_var_name]
        lat_dim = "latitude"
        lon_dim = "longitude"

        intensity_by_month: list[xr.DataArray] = []
        for month, month_data in cwb_da.groupby("time.month"):
            month = int(month)
            month_intensity = xr.zeros_like(month_data)

            if monthly_lower_thresholds and month in monthly_lower_thresholds:
                lower_threshold_da = monthly_lower_thresholds[month]
                month_intensity = month_intensity + xr.where(
                    month_data < lower_threshold_da,
                    lower_threshold_da - month_data,
                    0,
                )

            intensity_by_month.append(month_intensity)

        intensity_da = xr.concat(intensity_by_month, dim="time").sortby("time")

        intensity_da.name = output_var_name
        intensity_da.attrs = dict(cwb_da.attrs)
        intensity_da.attrs["description"] = (
            f"{input_var_name} lower-threshold exceedance intensity"
        )

        out_path = output_data_dir / f"{source_path.stem}_intensity.nc"
        intensity_da.to_dataset(name=output_var_name).to_netcdf(out_path)

    return str(out_path)
