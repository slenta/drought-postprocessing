import os
import json
from pathlib import Path
import xarray as xr
import numpy as np
from tqdm import tqdm
from cmethods import adjust
from IPython import embed

from ..config_loader import get_qm_rf_global_config
from ..utils.preprocessing import compute_leadmonth_mask


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
    quantile_tag = f"nq{int(cfg['qm_arguments']['n_quantiles'])}"

    with open(cfg["hindcasts_json"], "r") as f:
        hind_files = json.load(f)

    ref_ds = xr.open_dataset(cfg["reference_data"])
    ref_da = ref_ds[cfg["var_name"]]
    var_name = cfg["var_name"]

    leadmonths = cfg["leadmonth"]

    for leadmonth in leadmonths:
        leadmonth_data_dir = (
            Path(cfg["output_dir"])
            / "data"
            / var_name
            / quantile_tag
            / f"lm{leadmonth}"
        )
        qm_hindcast_out_dir = leadmonth_data_dir / "qm_hindcast"
        qm_residuals_out_dir = leadmonth_data_dir / "qm_residuals"
        qm_hindcast_out_dir.mkdir(parents=True, exist_ok=True)
        qm_residuals_out_dir.mkdir(parents=True, exist_ok=True)

        qm_hindcast_paths = []
        qm_residual_paths = []

        for hind_fp in tqdm(hind_files, desc=f"Quantile mapping lm{leadmonth}"):
            hind_ds = xr.open_dataset(hind_fp)

            hind_da = hind_ds[cfg["var_name"]]
            common_time = np.intersect1d(ref_da["time"].values, hind_da["time"].values)
            hind_ds_aligned = hind_ds.sel(time=common_time)
            ref_aligned = ref_da.sel(time=common_time).squeeze()
            hind_aligned = hind_da.sel(time=common_time).squeeze()

            month_mask = compute_leadmonth_mask(hind_aligned["time"].values, leadmonth)
            hind_ds_selected = hind_ds_aligned.isel(time=month_mask)

            ref_selected = ref_aligned.isel(time=month_mask)
            hind_selected = hind_aligned.isel(time=month_mask)

            adjusted = adjust(
                method="quantile_mapping",
                obs=ref_selected,
                simh=hind_selected,
                simp=hind_selected,
                n_quantiles=cfg["qm_arguments"]["n_quantiles"],
                kind=kind,
            )

            # Create QM output: subset original dataset to selected time, then replace variable
            qm_ds_out = hind_ds_selected.copy()
            qm_ds_out[cfg["var_name"]] = adjusted[cfg["var_name"]]
            base, ext = os.path.splitext(os.path.basename(hind_fp))
            qm_out_path = qm_hindcast_out_dir / f"{base}_qm_lm{leadmonth}{ext}"
            qm_ds_out.to_netcdf(str(qm_out_path))
            qm_hindcast_paths.append(str(qm_out_path))
            qm_ds_out.close()

            # Create residual output: subset original dataset to selected time, then replace variable
            residual = ref_selected - adjusted[cfg["var_name"]]
            residual_ds_out = hind_ds_selected.copy()
            residual_ds_out[cfg["var_name"]] = residual
            residual_out_path = (
                qm_residuals_out_dir / f"{base}_qm_residual_lm{leadmonth}{ext}"
            )
            residual_ds_out.to_netcdf(str(residual_out_path))
            qm_residual_paths.append(str(residual_out_path))
            residual_ds_out.close()

            hind_ds.close()

        qm_hindcast_paths_json = (
            Path(cfg["output_dir"])
            / "paths"
            / var_name
            / quantile_tag
            / f"lm{leadmonth}"
            / "qm_hindcast_paths.json"
        )
        qm_hindcast_paths_json.parent.mkdir(parents=True, exist_ok=True)
        with open(str(qm_hindcast_paths_json), "w") as fh:
            json.dump(qm_hindcast_paths, fh, indent=2)

        residuals_paths_json = (
            Path(cfg["output_dir"])
            / "paths"
            / var_name
            / quantile_tag
            / f"lm{leadmonth}"
            / "qm_residuals_paths.json"
        )
        residuals_paths_json.parent.mkdir(parents=True, exist_ok=True)
        with open(str(residuals_paths_json), "w") as fh:
            json.dump(qm_residual_paths, fh, indent=2)

    ref_ds.close()
    return str(
        Path(cfg["output_dir"])
        / "paths"
        / var_name
        / quantile_tag
        / f"lm{leadmonths[0]}"
        / "qm_hindcast_paths.json"
    )
