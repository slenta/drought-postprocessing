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


def intersect_time_coordinates(data_arrays) -> np.ndarray:
    common_time = reduce(
        np.intersect1d,
        [np.asarray(data_array["time"].values) for data_array in data_arrays],
    )
    if len(common_time) == 0:
        raise ValueError("No common time steps found across evaluation inputs.")
    return common_time
