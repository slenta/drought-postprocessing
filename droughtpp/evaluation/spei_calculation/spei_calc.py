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
from pathlib import Path

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

from droughtpp.analysis.qm_rf.config_loader import load_qm_rf_config


def compute_spei_from_surplus(
    surplus: xr.DataArray,
    month_range: tuple = (1, 3),
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
    month_range : tuple, optional
        1-based inclusive index range of months within each year over which
        to sum the surplus before computing SPEI. Example: (1,3) sums the
        first three months in each year. Files are expected to contain the
        same set of months each year (e.g., 6 months per year).
    var_names : str, optional
        Output variable name in the resulting xarray. Default = 'SPEI'.
    dist : scipy.stats distribution, optional
        Distribution for SPEI standardization. Default = sps.fisk (log-logistic).

    Returns
    -------
    xr.DataArray
        Global SPEI with shape (time, latitude, longitude),
        standardized (mean=0, std=1).

    Notes
    -----
    - Standardization via log-logistic (Fisk) distribution.
    - Missing or constant data per grid cell are skipped gracefully.
    - NaN values converted to 0.
    """

    time = surplus["time"]
    lats = surplus["latitude"].values
    lons = surplus["longitude"].values
    n_time = len(time)

    # We'll compute one SPEI value per year by summing surplus over the
    # requested month indices within each year (1-based positions). The
    # output time coordinate will be the timestamp of the last month in the
    # selected window for each year.

    # Prepare time index and year/month mapping
    time_index = pd.to_datetime(time.values)
    years = np.array([t.year for t in time_index])

    # Determine the month ordering within a year from the first available year
    first_year = years[0]
    mask_first_year = years == first_year
    months_first_year = list(time_index[mask_first_year].month)
    if not months_first_year:
        raise ValueError(
            "No monthly records found for the first year in `time` coordinate."
        )

    # Map 1-based positions to actual calendar months
    n_months_per_year = len(months_first_year)
    start_idx, end_idx = month_range
    if not (1 <= start_idx <= n_months_per_year and 1 <= end_idx <= n_months_per_year):
        raise ValueError(
            f"month_range {month_range} out of bounds for {n_months_per_year} months/year"
        )
    if end_idx < start_idx:
        raise ValueError("month_range end must be >= start")

    selected_months = months_first_year[start_idx - 1 : end_idx]

    # Build list of unique years and corresponding timestamp for last month in window
    unique_years = np.unique(years)
    out_times = []
    for y in unique_years:
        # find the timestamp of the last selected month in that year
        masks = [(t.year == y and t.month in selected_months) for t in time_index]
        ts = time_index[masks]
        if len(ts) == 0:
            out_times.append(pd.NaT)
        else:
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

    for i in tqdm(range(nlat), desc="Computing SPEI (per latitude)"):
        for j in range(nlon):
            surplus_grid = surplus[:, i, j].values

            # Skip if all zeros or all NaNs
            if np.all(surplus_grid == 0) or np.all(np.isnan(surplus_grid)):
                continue

            # Replace NaNs with 0 for aggregation (mirrors previous behaviour)
            surplus_clean = np.nan_to_num(surplus_grid, nan=0.0)
            surplus_series = pd.Series(surplus_clean, index=time_index)

            # Aggregate per year by summing over the selected months
            annual_vals = []
            for y in unique_years:
                sel = surplus_series[
                    (surplus_series.index.year == int(y))
                    & (surplus_series.index.month.isin(selected_months))
                ]
                if len(sel) == 0:
                    annual_vals.append(np.nan)
                else:
                    annual_vals.append(sel.sum())

            annual_series = pd.Series(annual_vals, index=pd.to_datetime(out_times))

            spei_series = si.spei(series=annual_series, dist=dist, timescale=1)

            # Store results
            spei_out[:, i, j] = spei_series.values

    spei_out.attrs.update(
        {
            "description": f"SPEI computed on annual aggregates over months {month_range}",
            "standardization": "log-logistic (Fisk)",
            "creator": "compute_spei_from_surplus",
            "month_range": month_range,
            "note": "NaN values converted to 0 for aggregation",
        }
    )

    return spei_out


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
    month_range: tuple,
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


def _compute_spei_for_path_list(
    input_paths: list[Path],
    output_dir: Path,
    input_var: str,
    output_var: str,
    month_range: tuple,
    dist,
) -> list[str]:
    written_paths = []
    for input_path in input_paths:
        out_path = output_dir / f"{input_path.stem}_spei.nc"
        written_path = _compute_and_save_spei(
            input_path=input_path,
            out_path=out_path,
            input_var=input_var,
            output_var=output_var,
            month_range=month_range,
            dist=dist,
        )
        written_paths.append(written_path)
    return written_paths


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    default_cfg_path = (
        Path(__file__).resolve().parents[2] / "analysis" / "qm_rf" / "config.yaml"
    )
    cfg = load_qm_rf_config(None, default_config=default_cfg_path)

    input_var = cfg.get("var_name", "CWB")
    output_var = "spei"
    month_range = tuple(cfg.get("spei_month_range", [1, 3]))

    reference_path = Path(cfg["reference_data"])
    hindcasts_json_path = Path(cfg["hindcasts_json"])
    qm_json_path = Path(cfg["qm_json_path"])

    out_dir = Path(cfg["output_dir"])
    spei_root = out_dir / "data" / "spei"
    reference_out_dir = spei_root / "reference"
    original_hindcasts_out_dir = spei_root / "original_hindcasts"
    qm_hindcasts_out_dir = spei_root / "qm_hindcasts"

    reference_output = reference_out_dir / f"{reference_path.stem}_spei.nc"
    reference_spei_path = _compute_and_save_spei(
        input_path=reference_path,
        output_path=reference_output,
        input_var=input_var,
        output_var=output_var,
        month_range=month_range,
        dist=sps.fisk,
    )

    hindcast_paths = _load_paths_from_json(hindcasts_json_path)
    hindcast_spei_paths = _compute_spei_for_path_list(
        input_paths=hindcast_paths,
        output_dir=original_hindcasts_out_dir,
        input_var=input_var,
        output_var=output_var,
        month_range=month_range,
        dist=sps.fisk,
    )

    qm_paths = _load_paths_from_json(qm_json_path)
    qm_spei_paths = _compute_spei_for_path_list(
        input_paths=qm_paths,
        output_dir=qm_hindcasts_out_dir,
        input_var=input_var,
        output_var=output_var,
        month_range=month_range,
        dist=sps.fisk,
    )

    _write_json(spei_root / "reference_spei_paths.json", [reference_spei_path])
    _write_json(spei_root / "original_hindcasts_spei_paths.json", hindcast_spei_paths)
    _write_json(spei_root / "qm_hindcasts_spei_paths.json", qm_spei_paths)

    print(f"Saved SPEI outputs under {spei_root}")
