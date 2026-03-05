import os
import json
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm


def compute_qm_distributions(
    json_list_path, ref_path, var_name, qm_out_dir=None, align_time=True
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

    ds_obs = xr.open_dataset(ref_path)
    obs_da = ds_obs[var_name]

    out = {}
    for fp in tqdm(files, desc="Collecting distributions"):
        p = Path(fp)
        base, ext = p.stem, p.suffix

        # find QM file (prefer qm_out_dir if given)
        qm_path = Path(qm_out_dir) / f"{base}_qm{ext}"

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
