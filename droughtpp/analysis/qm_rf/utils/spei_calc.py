"""
Compute global Standardized Precipitation Evapotranspiration Index (SPEI)
from pre-calculated surplus (P - PET) using the Beguería et al. (2010) method
(log-logistic distribution) and the modern 'spei' Python package (Vonk, 2025).

References:
- Vicente-Serrano, Beguería & López-Moreno (2010), J. Climate 23(7):1696–1718.
- Vonk, M.A. (2025). "SPEI: A Python package for calculating and visualizing drought indices."

Requires:
    pip install numpy pandas xarray scipy spei tqdm

Usage:
    python spei_calc.py --input surplus_file.nc --output spei_output.nc --timescale 3
"""

# Set thread limits BEFORE importing numpy/scipy
import os
import json
import argparse
from pathlib import Path
from IPython import embed

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as sps
import spei as si
from tqdm import tqdm

from droughtpp.analysis.qm_rf.config_loader import (
    add_qm_rf_config_arguments,
    load_qm_rf_config,
)


def _normalize_month_ranges(month_range) -> list[tuple[int, int]]:

    first_item = month_range[0]
    if isinstance(first_item, (tuple, list)):
        month_ranges = [tuple(rng) for rng in month_range]
    else:
        month_ranges = [tuple(month_range)]


    return month_ranges


def _compute_spei_for_month_range(
    surplus: xr.DataArray,
    month_range: tuple[int, int],
    var_names: list = ["CWB", "spei"],
    dist=sps.fisk,
) -> xr.DataArray:
    """Compute SPEI for one inclusive calendar month window.

    The month_range tuple is interpreted as calendar months, not positions
    within the first available year. For example, (1, 3) means January
    through March in every year.
    """

    time = surplus["time"]
    lats = surplus["latitude"].values
    lons = surplus["longitude"].values

    time_index = pd.to_datetime(time.values)
    years = np.array([t.year for t in time_index])
    start_idx, end_idx = month_range

    selected_months = list(range(start_idx, end_idx + 1))

    unique_years = np.unique(years)
    out_times = []
    for y in unique_years:
        masks = [(t.year == y and t.month in selected_months) for t in time_index]
        ts = time_index[masks]
        out_times.append(ts.max())

    n_years = len(unique_years)
    nlat, nlon = len(lats), len(lons)

    spei_out = xr.DataArray(
        data=np.full((n_years, nlat, nlon), np.nan, dtype=np.float32),
        coords={
            "time": np.array(out_times, dtype="datetime64[ns]"),
            "latitude": lats,
            "longitude": lons,
        },
        dims=["time", "latitude", "longitude"],
        name=var_names[1],
    )

    for i in tqdm(range(nlat), desc=f"Computing SPEI for calendar month_range {month_range}"):
        for j in range(nlon):
            surplus_grid = surplus[:, i, j].values

            surplus_clean = np.nan_to_num(surplus_grid, nan=0.0)
            surplus_series = pd.Series(surplus_clean, index=time_index)

            annual_vals = []
            for y in unique_years:
                sel = surplus_series[
                    (surplus_series.index.year == int(y))
                    & (surplus_series.index.month.isin(selected_months))
                ]
                annual_vals.append(sel.sum())

            annual_series = pd.Series(annual_vals, index=pd.to_datetime(out_times))
            spei_series = si.spei(series=annual_series, dist=dist, timescale=1)
            spei_out[:, i, j] = spei_series.values

    spei_out.attrs.update(
        {
            "description": f"SPEI computed on annual aggregates over months {month_range}",
            "standardization": "log-logistic (Fisk)",
            "month_range": str(month_range),
            "note": "NaN values converted to 0 for aggregation",
        }
    )
    spei_out["time"].encoding.update(
        {
            "units": "days since 1900-01-01",
            "calendar": "standard",
            "dtype": "float64",
        }
    )
    spei_out["time"].attrs = surplus["time"].attrs

    return spei_out


def compute_spei_from_surplus(
    surplus: xr.DataArray,
    month_range: tuple | list[tuple[int, int]] = (1, 3),
    var_names: list = ["CWB", "spei"],
    dist=sps.fisk,
) -> xr.DataArray:
    """
    Compute SPEI globally at each grid cell from pre-calculated surplus (P - PET).

    Parameters
    ----------
    surplus : xr.DataArray
        Monthly surplus (precipitation minus PET) [mm].
        Must have dims ('time', 'latitude', 'longitude').
    month_range : tuple or list of tuple, optional
        Calendar month range or ranges to sum over before computing SPEI.
        Each tuple is inclusive and uses calendar months, not positional
        indices. Examples: (1, 3) for January-March, or
        [(1, 3), (2, 4), (3, 5)] to compute and stack several windows.
    var_names : str, optional
        Output variable name in the resulting xarray. Default = 'SPEI'.
    dist : scipy.stats distribution, optional
        Distribution for SPEI standardization. Default = sps.fisk (log-logistic).

    Returns
    -------
    xr.DataArray
        Global SPEI with shape (time, latitude, longitude) for a single
        range, or (window, time, latitude, longitude) when multiple ranges
        are provided. Multi-range outputs use labels like '1-3' on the
        window dimension and are standardized (mean=0, std=1).

    Notes
    -----
    - Standardization via log-logistic (Fisk) distribution.
    - Missing or constant data per grid cell are skipped gracefully.
    - NaN values converted to 0.
    - Multiple ranges are stacked along a new window dimension.
    """

    month_ranges = _normalize_month_ranges(month_range)

    spei_fields = []
    for range in month_ranges:
        spei_field = _compute_spei_for_month_range(
                surplus=surplus,
                month_range=range,
                var_names=var_names,
                dist=dist,
        )
        spei_fields.append(spei_field)


    if len(spei_fields) == 1:
        result = spei_fields[0]
        result.attrs["month_range"] = month_ranges[0]
        return result

    flattened = xr.concat(spei_fields, dim="time").sortby("time")
    flattened["time"].encoding.update(
        {
            "units": "days since 1900-01-01",
            "calendar": "standard",
            "dtype": "float64",
        }
    )
    flattened.attrs.update(
        {
            "description": f"SPEI computed on annual aggregates over months {month_ranges}",
            "standardization": "log-logistic (Fisk)",
            "month_range": str(month_ranges),
            "note": "NaN values converted to 0 for aggregation",
        }
    )
    return flattened


def _ensure_spatial_dims(data_array: xr.DataArray) -> xr.DataArray:
    rename_map = {}
    if "lat" in data_array.dims:
        rename_map["lat"] = "latitude"
    if "lon" in data_array.dims:
        rename_map["lon"] = "longitude"
    if rename_map:
        data_array = data_array.rename(rename_map)
    return data_array


def _compute_and_save_spei(
    input_path: Path,
    output_path: Path,
    input_var: str,
    output_var: str,
    month_range: tuple | list[tuple[int, int]],
    dist,
) -> str:
    with xr.open_dataset(input_path) as ds:
        surplus = ds[input_var].load().squeeze()

    surplus = _ensure_spatial_dims(surplus)

    spei = compute_spei_from_surplus(
        surplus,
        month_range=month_range,
        var_names=[input_var, output_var],
        dist=dist,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    spei.to_netcdf(output_path)
    return str(output_path)


def _load_paths_from_json(json_path: Path) -> list[Path]:
    with open(json_path, "r") as fh:
        return [Path(path) for path in json.load(fh)]


def _slice_paths_for_slurm_array(input_paths: list[Path]) -> list[Path]:
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")

    if task_id is None or task_count is None:
        return input_paths

    task_id_int = int(task_id)
    task_count_int = int(task_count)
    if task_count_int <= 1:
        return input_paths

    return input_paths[task_id_int::task_count_int]


def _compute_spei_for_path_list(
    input_paths: list[Path],
    output_dir: Path,
    input_var: str,
    output_var: str,
    month_range: tuple | list[tuple[int, int]],
    dist,
) -> list[str]:
    input_paths = _slice_paths_for_slurm_array(input_paths)
    written_paths = []
    for input_path in input_paths:
        out_path = output_dir / f"{input_path.stem}_spei.nc"
        written_path = _compute_and_save_spei(
            input_path=input_path,
            output_path=out_path,
            input_var=input_var,
            output_var=output_var,
            month_range=month_range,
            dist=dist,
        )
        written_paths.append(written_path)
    return written_paths


def compute_spei_for_path_list(
    input_paths: list[Path],
    output_dir: Path,
    input_var: str,
    output_var: str,
    month_range: tuple | list[tuple[int, int]],
    dist,
) -> list[str]:
    return _compute_spei_for_path_list(
        input_paths=input_paths,
        output_dir=output_dir,
        input_var=input_var,
        output_var=output_var,
        month_range=month_range,
        dist=dist,
    )


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def _task_specific_json_path(path: Path) -> Path:
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")

    if task_id is None or task_count is None:
        return path

    task_count_int = int(task_count)
    if task_count_int <= 1:
        return path

    return path.with_name(
        f"{path.stem}.task{int(task_id):05d}_of_{task_count_int:05d}{path.suffix}"
    )


if __name__ == "__main__":
    default_cfg_path = Path(
        "/work/bk1318/k202208/crai/hindcast-pp/Ml-Drought-Postprocessing/droughtpp/analysis/qm_rf/config.yaml"
    )
    parser = argparse.ArgumentParser()
    add_qm_rf_config_arguments(parser, default_config=str(default_cfg_path))
    args = parser.parse_args()
    cfg = load_qm_rf_config(
        config_path=args.config,
        overrides=args.config_overrides,
        default_config=default_cfg_path,
    )

    input_var = cfg.get("var_name", "CWB")
    output_var = "spei"
    month_range = cfg.get("spei_month_range")
    spei_job_stage = str(cfg.get("workflow", {}).get("spei_job_stage", "all")).strip().lower()
    training_target = str(cfg.get("training_target", "residual")).lower()
    rf_results_tag_base = str(cfg["rf_results_tag"])
    n_features = cfg["n_features"]
    results_tag = f"{rf_results_tag_base}_nf{n_features}"
    target_root = Path(input_var) / Path(training_target) / Path(results_tag)

    reference_path = Path(cfg["reference_data"])
    hindcasts_json_path = Path(cfg["hindcasts_json"])

    out_dir = Path(cfg["output_dir"])
    spei_root = out_dir / "data" / "spei"
    reference_out_dir = spei_root / "reference"
    original_hindcasts_out_dir = spei_root / "original_hindcasts"
    qm_hindcasts_out_dir = spei_root / "qm_hindcasts"
    full_timeline_all_root = target_root / Path("full_timeline") / Path("rf_eval") / Path("lm_all")
    full_timeline_spei_out_dir = out_dir / Path("data") / full_timeline_all_root / Path("spei")
    full_timeline_cwb_json = out_dir / Path("paths") / full_timeline_all_root / Path("corrected_cwb_paths.json")
    full_timeline_spei_json = out_dir / Path("paths") / full_timeline_all_root / Path("corrected_spei_paths.json")


    if spei_job_stage in {"all", "reference"}:
        reference_output = reference_out_dir / f"{reference_path.stem}_spei.nc"
        reference_spei_path = _compute_and_save_spei(
            input_path=reference_path,
            output_path=reference_output,
            input_var=input_var,
            output_var=output_var,
            month_range=month_range,
            dist=sps.fisk,
        )
        _write_json(spei_root / "reference_spei_paths.json", [reference_spei_path])

    if spei_job_stage in {"all", "original_hindcast"}:
        hindcast_paths = _load_paths_from_json(hindcasts_json_path)
        hindcast_spei_paths = _compute_spei_for_path_list(
            input_paths=hindcast_paths,
            output_dir=original_hindcasts_out_dir,
            input_var=input_var,
            output_var=output_var,
            month_range=month_range,
            dist=sps.fisk,
        )
        _write_json(
            _task_specific_json_path(spei_root / "original_hindcasts_spei_paths.json"),
            hindcast_spei_paths,
        )

    if spei_job_stage in {"all", "qm"}:
        qm_fulltimeline_spei_paths = []
        n_quantiles = cfg.get("qm_arguments", {}).get("n_quantiles")
        qm_fulltimeline_json_path = (
            out_dir
            / "paths"
            / "qm"
            / f"nq{n_quantiles}"
            / "full_timeline"
            / "qm_hindcast_paths.json"
        )
        qm_fulltimeline_paths = _load_paths_from_json(qm_fulltimeline_json_path)
        qm_fulltimeline_out_dir = spei_root / "qm_hindcasts_fulltimeline"
        qm_fulltimeline_spei_paths = _compute_spei_for_path_list(
            input_paths=qm_fulltimeline_paths,
            output_dir=qm_fulltimeline_out_dir,
            input_var=input_var,
            output_var=output_var,
            month_range=month_range,
            dist=sps.fisk,
        )
        _write_json(
            _task_specific_json_path(spei_root / "qm_hindcasts_fulltimeline_spei_paths.json"),
            qm_fulltimeline_spei_paths,
        )

    if spei_job_stage in {"all", "ml"}:
        full_timeline_cwb_paths = _load_paths_from_json(full_timeline_cwb_json)
        full_timeline_spei_paths = _compute_spei_for_path_list(
            input_paths=full_timeline_cwb_paths,
            output_dir=full_timeline_spei_out_dir,
            input_var=input_var,
            output_var=output_var,
            month_range=month_range,
            dist=sps.fisk,
        )
        _write_json(_task_specific_json_path(full_timeline_spei_json), full_timeline_spei_paths)

    print(f"Saved {spei_job_stage} SPEI outputs under {spei_root}")
