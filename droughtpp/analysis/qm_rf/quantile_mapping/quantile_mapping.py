import os
import json
import xarray as xr
import numpy as np
from tqdm import tqdm
from cmethods import adjust


def quantile_map_json(
    json_list_path, ref_path, var_name, out_dir="evaluation", n_quantiles=250, kind="+"
):
    """
    Apply quantile mapping (using cmethods.adjust) for var_name for each NetCDF
    file listed in json_list_path, using ref_path as the observation reference.
    Saves outputs to out_dir with suffix _qm before the extension.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(json_list_path, "r") as f:
        files = json.load(f)

    ds_obs = xr.open_dataset(ref_path)
    obs_da = ds_obs[var_name]

    for fp in tqdm(files, desc="Quantile mapping"):
        ds = xr.open_dataset(fp)

        # align observation and simulation on time coordinate if present
        sim_da = ds[var_name]
        print(obs_da.shape, sim_da.shape, fp, obs_da.shape)
        common = np.intersect1d(obs_da["time"].values, sim_da["time"].values)
        obs_al = obs_da.sel(time=common)
        sim_al = sim_da.sel(time=common)
        print(obs_al.shape, sim_al.shape)

        adjusted = adjust(
            method="quantile_mapping",
            obs=obs_al,
            simh=sim_al,
            simp=sim_al,
            n_quantiles=n_quantiles,
            kind=kind,
        )

        ds_out = ds.copy()
        ds_out[var_name] = adjusted[var_name]
        base, ext = os.path.splitext(os.path.basename(fp))
        out_path = os.path.join(out_dir, f"{base}_qm{ext}")
        ds_out.to_netcdf(out_path)
        # compute residuals (original minus adjusted) and save separately
        residual = ds[var_name] - adjusted[var_name]
        ds_res = ds.copy()
        ds_res[var_name] = residual
        out_path_res = os.path.join(out_dir, f"{base}_qm_residual{ext}")
        ds_res.to_netcdf(out_path_res)
        ds_res.close()

        ds.close()
        ds_out.close()

    ds_obs.close()
