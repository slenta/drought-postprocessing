import os
import json
from pathlib import Path
import xarray as xr
import numpy as np
from tqdm import tqdm
from cmethods import adjust

from ..config_loader import get_qm_rf_global_config


def quantile_map_json(
    kind="+",
    config_path=None,
    config_overrides=None,
):
    """
    Apply quantile mapping using globally configured paths/variables.
    Reads required inputs from qm_rf config and writes QM hindcast and
    residual outputs plus their JSON path lists.
    """
    default_cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = get_qm_rf_global_config(
        config_path=config_path,
        overrides=config_overrides,
        default_config=default_cfg_path,
    )

    os.makedirs(cfg["output_dir"], exist_ok=True)
    data_out_dir = os.path.join(cfg["output_dir"], "data")
    qm_hindcast_out_dir = os.path.join(data_out_dir, "qm_hindcast")
    qm_residuals_out_dir = os.path.join(data_out_dir, "qm_residuals")
    os.makedirs(data_out_dir, exist_ok=True)
    os.makedirs(qm_hindcast_out_dir, exist_ok=True)
    os.makedirs(qm_residuals_out_dir, exist_ok=True)

    qm_hindcast_paths = []
    qm_residual_paths = []
    with open(cfg["hindcasts_json"], "r") as f:
        files = json.load(f)

    ds_obs = xr.open_dataset(cfg["reference_data"])
    obs_da = ds_obs[cfg["var_name"]]
    print(cfg["n_quantiles"])

    for fp in tqdm(files, desc="Quantile mapping"):
        ds = xr.open_dataset(fp)

        # align observation and simulation on time coordinate if present
        sim_da = ds[cfg["var_name"]]
        common = np.intersect1d(obs_da["time"].values, sim_da["time"].values)
        obs_al = obs_da.sel(time=common).squeeze()
        sim_al = sim_da.sel(time=common).squeeze()

        adjusted = adjust(
            method="quantile_mapping",
            obs=obs_al,
            simh=sim_al,
            simp=sim_al,
            n_quantiles=cfg["n_quantiles"],
            kind=kind,
        )

        ds_out = ds.copy()
        ds_out[cfg["var_name"]] = adjusted[cfg["var_name"]]
        base, ext = os.path.splitext(os.path.basename(fp))
        out_path = os.path.join(qm_hindcast_out_dir, f"{base}_qm{ext}")
        ds_out.to_netcdf(out_path)
        qm_hindcast_paths.append(out_path)

        # compute residuals (reference minus QM-adjusted) and save separately
        residual = obs_al - adjusted[cfg["var_name"]]
        ds_res = ds.copy()
        ds_res[cfg["var_name"]] = residual
        out_path_res = os.path.join(qm_residuals_out_dir, f"{base}_qm_residual{ext}")
        ds_res.to_netcdf(out_path_res)
        qm_residual_paths.append(out_path_res)
        ds_res.close()

        ds.close()
        ds_out.close()

    qm_hindcast_paths_json = str(cfg["qm_json_path"])
    os.makedirs(os.path.dirname(qm_hindcast_paths_json), exist_ok=True)

    with open(qm_hindcast_paths_json, "w") as fh:
        json.dump(qm_hindcast_paths, fh, indent=2)

    residuals_paths_json = str(cfg["residuals_json"])
    os.makedirs(os.path.dirname(residuals_paths_json), exist_ok=True)

    with open(residuals_paths_json, "w") as fh:
        json.dump(qm_residual_paths, fh, indent=2)

    ds_obs.close()
    return qm_hindcast_paths_json
