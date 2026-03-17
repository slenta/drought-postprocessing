import json
from pathlib import Path
from typing import List

import yaml
import numpy as np
import pandas as pd
import xarray as xr
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import scipy.stats as sps

from .model_rf import RandomForestBiasCorrector
from ..config_loader import load_qm_rf_config
from droughtpp.analysis.qm_rf.utils.spei_evaluation import evaluate_spei
from droughtpp.analysis.qm_rf.utils.spei_from_rf import (
    combine_qm_and_residuals,
    compute_spei_from_rf_corrected,
)


def collect_eval_set(
    residuals_json: Path,
    hindcasts_json: Path,
    var_name: str,
    predictor_vars: List[str],
    eval_years: List[int],
):
    with open(residuals_json, "r") as fh:
        residuals = json.load(fh)
    with open(hindcasts_json, "r") as fh:
        hindcasts = json.load(fh)

    X_parts = []
    y_parts = []

    RF = RandomForestBiasCorrector

    for resid_path, hindcast_path in zip(residuals, hindcasts):
        resid_path = str(resid_path)
        hindcast_path = str(hindcast_path)

        res_da = xr.open_dataset(resid_path)[var_name]
        pred_da = xr.open_dataset(hindcast_path)[var_name]

        res_s = RF._stack_by_samples(res_da)
        pred_s = RF._stack_by_samples(pred_da)

        years = pd.to_datetime(res_s.coords["time"].values).year

        X_full = pd.DataFrame(
            {
                "predictor": pred_s.values,  # stacked predictor
                "lat": res_s.coords["latitude"].values,
                "lon": res_s.coords["longitude"].values,
            }
        )
        y_full = pd.Series(res_s.values, name="residual")

        finite_mask = np.isfinite(y_full.values) & np.all(
            np.isfinite(X_full.values), axis=1
        )

        if years is not None and len(eval_years) > 0:
            eval_mask_year = np.isin(years, eval_years)
        else:
            eval_mask_year = np.zeros_like(finite_mask, dtype=bool)

        eval_mask = finite_mask & (eval_mask_year)

        if eval_mask.any():
            X_parts.append(X_full.loc[eval_mask].reset_index(drop=True))
            y_parts.append(y_full.loc[eval_mask].reset_index(drop=True))

    if len(X_parts) == 0:
        return pd.DataFrame(columns=predictor_vars), pd.Series(dtype=float)

    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    return X, y


def save_predicted_residuals(
    model,
    hindcasts_json: Path,
    var_name: str,
    predictor_vars: List[str],
    eval_years: List[int] = [],
    out_dir: Path = None,
    spei_paths_json: Path = None,
    suffix: str = "_qm_rf_residuals",
):
    if out_dir is None:
        raise ValueError("out_dir must be provided to save_predicted_residuals().")

    with open(hindcasts_json, "r") as fh:
        hindcasts = json.load(fh)

    if spei_paths_json is None:
        spei_paths_json = Path(out_dir) / "corrected_spei_paths.json"

    spei_paths = []
    spei_paths_json.parent.mkdir(parents=True, exist_ok=True)

    for fp in tqdm(hindcasts, desc="Saving RF residuals"):
        fp = str(fp)
        ds = xr.open_dataset(fp)
        hind_da = ds[var_name]

        RF = RandomForestBiasCorrector
        pred_s = RF._stack_by_samples(hind_da)

        years = pd.to_datetime(pred_s.coords["time"].values).year

        X_pred = pd.DataFrame(
            {
                "predictor": pred_s.values,  # stacked predictor
                "lat": pred_s.coords["latitude"].values,
                "lon": pred_s.coords["longitude"].values,
            }
        )

        eval_mask_year = np.isin(years, eval_years)
        preds_array = np.full(X_pred.shape[0], np.nan, dtype=float)
        preds_array[eval_mask_year] = model.predict(X_pred.values[eval_mask_year])

        pred_da_stacked = xr.DataArray(
            preds_array, coords=(pred_s.coords["sample"],), dims=("sample",)
        )
        pred_da = pred_da_stacked.unstack("sample")

        # attach predicted residual to a copy of the original dataset to preserve structure
        ds_out = ds.copy()
        ds_out[var_name] = pred_da
        valid_times = np.unique(pred_s.coords["time"].values[eval_mask_year])
        orig_time_sel = ds["time"].sel(time=pd.to_datetime(valid_times))

        # subset ds_out to the same time index
        ds_out = ds_out.sel(time=pd.to_datetime(valid_times))
        ds_out = ds_out.assign_coords(time=orig_time_sel)
        ds_out["time"].attrs = ds["time"].attrs

        p = Path(fp)
        out_base = Path(f"{out_dir}/rf_residuals/")
        out_base.mkdir(parents=True, exist_ok=True)
        spei_base = Path(f"{out_dir}/data/spei_corrected/")
        spei_base.mkdir(parents=True, exist_ok=True)

        out_path = f"{out_base}/{p.stem}{suffix}.nc"
        spei_out_path = f"{spei_base}/{p.stem}_qm_rf_spei.nc"

        # calculate spei and save to NetCDF
        corrected_cwb = combine_qm_and_residuals(
            hind_da, pred_da, var_name="CWB"
        ).squeeze()
        spei = compute_spei_from_rf_corrected(
            corrected_cwb, month_range=(1, 3), var_names=["CWB", "spei"], dist=sps.fisk
        )

        ds_spei = ds_out.copy()
        ds_spei["spei"] = spei

        ds_spei.to_netcdf(str(spei_out_path))
        ds_out.to_netcdf(str(out_path))
        spei_paths.append(str(Path(spei_out_path)))
        ds.close()
        ds_out.close()

    with open(spei_paths_json, "w") as fh:
        json.dump(spei_paths, fh, indent=2)

    return spei_paths_json


def evaluate(config_path: Path | None = None, config_overrides=None):
    default_cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = load_qm_rf_config(
        config_path,
        overrides=config_overrides,
        default_config=default_cfg_path,
    )

    residuals_json = Path(cfg["residuals_json"])
    hindcasts_json = Path(cfg["hindcasts_json"])

    out_dir = Path(cfg.get("output_dir", "./rf_out"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(cfg.get("model_path", str(out_dir / "rf_model.joblib")))
    var_name = cfg.get("var_name")
    predictor_vars = cfg.get("predictor_vars", [var_name])
    eval_years = cfg.get("leave_out_years", [])
    evaluate_spei_flag = cfg.get("evaluate_spei", True)
    generate_corrected_spei = cfg.get("generate_corrected_spei", True)

    X_eval, y_eval = collect_eval_set(
        residuals_json, hindcasts_json, var_name, predictor_vars, eval_years
    )

    model = load(model_path)

    preds = model.predict(X_eval.values)
    rmse = float(np.sqrt(mean_squared_error(y_eval.values, preds)))
    r2 = float(r2_score(y_eval.values, preds))

    metrics = {"rmse": rmse, "r2": r2, "n_eval_samples": int(len(y_eval))}

    # save predicted residuals per-file (same structure as input hindcasts)

    spei_paths_json = Path(
        cfg.get("corrected_spei_paths_json", str(out_dir / "corrected_spei_paths.json"))
    )

    if generate_corrected_spei:
        spei_paths_json = save_predicted_residuals(
            model,
            hindcasts_json,
            var_name,
            predictor_vars,
            eval_years,
            out_dir=out_dir,
            spei_paths_json=spei_paths_json,
        )

    if evaluate_spei_flag:

        with open(spei_paths_json, "r") as fh:
            corrected_spei_paths = json.load(fh)

        plot_dir = cfg.get("plot_dir", str(out_dir / "qm_plots"))
        evaluate_spei(
            corrected_spei_paths=corrected_spei_paths,
            hindcasts_spei_json=Path(cfg["hindcasts_spei_json"]),
            reference_spei_json=Path(cfg["reference_spei_json"]),
            out_dir=out_dir,
            eval_years=eval_years,
            std_multiplier=float(cfg.get("spei_std_multiplier", 1.0)),
            plot_dir=plot_dir,
        )


if __name__ == "__main__":
    evaluate()
