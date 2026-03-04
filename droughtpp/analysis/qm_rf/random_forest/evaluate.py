import json
import argparse
from pathlib import Path
from typing import List
import os

import yaml
import numpy as np
import pandas as pd
import xarray as xr
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from IPython import embed

from .rf_net import RandomForestBiasCorrector


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
    suffix: str = "_qm_rf_residuals",
):
    """Predict residuals per hindcast file and save as NetCDF with added variable <var_name>_qm_rf_residual."""
    with open(hindcasts_json, "r") as fh:
        hindcasts = json.load(fh)

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
        out_base = Path(out_dir)
        out_base.mkdir(parents=True, exist_ok=True)
        out_path = out_base / f"{p.stem}{suffix}{p.suffix or '.nc'}"
        embed()

        ds_out.to_netcdf(str(out_path))
        ds.close()
        ds_out.close()


def evaluate(config_path: Path):
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    residuals_json = Path(cfg["residuals_json"])
    hindcasts_json = Path(cfg["hindcasts_json"])

    out_dir = Path(cfg.get("output_dir", "./rf_out"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(cfg.get("model_path", str(out_dir / "rf_model.joblib")))
    var_name = cfg.get("var_name", "tas")
    predictor_vars = cfg.get("predictor_vars", [var_name])
    eval_years = cfg.get("leave_out_years", [])

    X_eval, y_eval = collect_eval_set(
        residuals_json, hindcasts_json, var_name, predictor_vars, eval_years
    )
    print(X_eval.shape, y_eval.shape)

    model = load(model_path)

    preds = model.predict(X_eval.values)
    rmse = float(np.sqrt(mean_squared_error(y_eval.values, preds)))
    r2 = float(r2_score(y_eval.values, preds))

    metrics = {"rmse": rmse, "r2": r2, "n_eval_samples": int(len(y_eval))}

    # save predicted residuals per-file (same structure as input hindcasts)
    save_predicted_residuals(
        model, hindcasts_json, var_name, predictor_vars, eval_years, out_dir=out_dir
    )

    metrics_path = out_dir / "evaluation_metrics.yaml"
    with open(metrics_path, "w") as fh:
        yaml.safe_dump(metrics, fh)

    print(f"Evaluation complete. Metrics saved to: {metrics_path}")
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained RF on QM residuals.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="train_config.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()
    evaluate(Path(args.config))
