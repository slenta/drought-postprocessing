import os
import json
from pathlib import Path
from functools import reduce

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


def compute_qm_distributions(
    json_list_path,
    ref_path,
    var_name,
    qm_json_path=None,
    qm_out_dir=None,
    align_time=True,
):
    """
    For each hindcast file listed in json_list_path collect flattened numpy arrays:
      - original hindcast values ('orig')
      - quantile-mapped values ('qm') (file with suffix '_qm')
      - reference observation values ('obs')  (from ref_path)

    Returns a dict keyed by file-stem -> dict with keys: orig, qm, obs, file, qm_file
    """

    with open(json_list_path, "r") as fh:
        files = json.load(fh)
    with open(qm_json_path, "r") as fh:
        qm_files = json.load(fh)

    ds_obs = xr.open_dataset(ref_path)
    obs_da = ds_obs[var_name]

    out = {}
    for fp, qm_fp in tqdm(
        list(zip(files, qm_files)),
        desc="Collecting distributions",
    ):
        p = Path(fp)
        base = p.stem
        qm_path = Path(qm_fp)

        ds = xr.open_dataset(fp)
        ds_qm = xr.open_dataset(qm_path)

        sim_da = ds[var_name]
        qm_da = ds_qm[var_name]

        obs_vals = obs_da.values.ravel()
        orig_vals = sim_da.values.ravel()
        qm_vals = qm_da.values.ravel()

        # filter finite only
        out[base] = {
            "orig": orig_vals,
            "qm": qm_vals,
            "obs": obs_vals,
            "file": str(fp),
            "qm_file": str(qm_path),
        }

        ds.close()
        ds_qm.close()

    ds_obs.close()
    return out


def load_paths_from_json(json_path: Path) -> list[Path]:
    with open(json_path, "r") as fh:
        return [Path(path) for path in json.load(fh)]


def ensure_spatial_dims(data_array: xr.DataArray) -> xr.DataArray:
    rename_map = {}
    if "lat" in data_array.dims:
        rename_map["lat"] = "latitude"
    if "lon" in data_array.dims:
        rename_map["lon"] = "longitude"
    if rename_map:
        data_array = data_array.rename(rename_map)
    return data_array


def select_eval_years(data_array: xr.DataArray, eval_years) -> xr.DataArray:
    if not eval_years:
        return data_array

    years = pd.to_datetime(data_array["time"].values).year
    return data_array.isel(time=np.isin(years, eval_years))


def load_eval_data(path: Path, var_name: str) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        data_array = ds[var_name].load()

    data_array = ensure_spatial_dims(data_array).squeeze()
    if "time" not in data_array.dims:
        raise ValueError(f"Expected time dimension in {path}")

    return data_array.dropna("time", how="all")


def count_extreme_drought_events(ensemble, reference, threshold_lower):
    """
    Count reference drought events, correctly hit droughts, and wrong drought predictions.

    Args:
        ensemble: np.ndarray (time, member, lat, lon)
        reference: np.ndarray (time, lat, lon)
        threshold_lower: np.ndarray (lat, lon)

    Returns:
        tuple[int, int, int]: (reference_count, hit_count, wrong_count)
    """
    ensemble = np.asarray(ensemble)
    reference = np.asarray(reference)
    threshold_lower = np.asarray(threshold_lower)

    reference_drought = reference < threshold_lower[None, :, :]
    predicted_drought = ensemble < threshold_lower[None, None, :, :]

    valid = (
        np.isfinite(ensemble)
        & np.isfinite(reference)[:, None, :, :]
        & np.isfinite(threshold_lower)[None, None, :, :]
    )
    reference_drought_expanded = reference_drought[:, None, :, :]

    reference_count = int(np.count_nonzero(reference_drought_expanded & valid))
    hit_count = int(
        np.count_nonzero(predicted_drought & reference_drought_expanded & valid)
    )
    wrong_count = int(
        np.count_nonzero(predicted_drought & (~reference_drought_expanded) & valid)
    )

    return reference_count, hit_count, wrong_count


def compute_gridcell_drought_hit_rate_percent(ensemble, reference, threshold_lower):
    """
    Compute per-gridcell percentage of correctly predicted drought events.

    Args:
        ensemble: np.ndarray (time, member, lat, lon)
        reference: np.ndarray (time, lat, lon)
        threshold_lower: np.ndarray (lat, lon)

    Returns:
        np.ndarray (lat, lon): hit-rate percentage in [0, 100], NaN where no
        reference drought events are available.
    """
    ensemble = np.asarray(ensemble)
    reference = np.asarray(reference)
    threshold_lower = np.asarray(threshold_lower)

    reference_drought = reference < threshold_lower[None, :, :]
    predicted_drought = ensemble < threshold_lower[None, None, :, :]

    valid = (
        np.isfinite(ensemble)
        & np.isfinite(reference)[:, None, :, :]
        & np.isfinite(threshold_lower)[None, None, :, :]
    )
    reference_drought_expanded = reference_drought[:, None, :, :]

    hit_count = np.sum(
        predicted_drought & reference_drought_expanded & valid, axis=(0, 1)
    )
    reference_count = np.sum(reference_drought_expanded & valid, axis=(0, 1))

    hit_rate = np.full(reference_count.shape, np.nan, dtype=float)
    has_events = reference_count > 0
    hit_rate[has_events] = (hit_count[has_events] / reference_count[has_events]) * 100.0

    return hit_rate


def intersect_time_coordinates(data_arrays) -> np.ndarray:
    common_time = reduce(
        np.intersect1d,
        [np.asarray(data_array["time"].values) for data_array in data_arrays],
    )
    if len(common_time) == 0:
        raise ValueError("No common time steps found across evaluation inputs.")
    return common_time
