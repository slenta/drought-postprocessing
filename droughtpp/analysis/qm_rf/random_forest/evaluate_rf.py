import json
from pathlib import Path
from typing import List

import yaml
import numpy as np
import pandas as pd
import xarray as xr
from joblib import load
from tqdm import tqdm
import scipy.stats as sps

from .model_rf import RandomForestBiasCorrector
from .features import RFFeatureBuilder
from ..config_loader import get_qm_rf_global_config
from droughtpp.analysis.qm_rf.utils.spei_evaluation import evaluate_spei
from droughtpp.analysis.qm_rf.utils.cwb_evaluation import evaluate_cwb
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
    feature_builder: RFFeatureBuilder,
    leadmonth: int,
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
        X_full, years = feature_builder.build(pred_da)
        y_full = pd.Series(res_s.values, name="residual")

        finite_mask = np.isfinite(y_full.values) & np.all(
            np.isfinite(X_full.values), axis=1
        )

        leadmonth_mask = np.isclose(
            X_full["lead_month"].values.astype(float),
            float(leadmonth),
        )

        if years is not None and len(eval_years) > 0:
            eval_mask_year = np.isin(years, eval_years)
        else:
            eval_mask_year = np.zeros_like(finite_mask, dtype=bool)

        eval_mask = finite_mask & eval_mask_year & leadmonth_mask

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
    qm_hindcasts_json: Path,
    var_name: str,
    predictor_vars: List[str],
    eval_years: List[int],
    out_dir: Path,
    spei_paths_json: Path,
    residual_paths_json: Path,
    corrected_cwb_paths_json: Path,
    feature_builder: RFFeatureBuilder,
    leadmonth: int,
    save_spei: bool = True,
    suffix: str = "_qm_rf_residuals",
):

    with open(hindcasts_json, "r") as fh:
        hindcasts = json.load(fh)
    with open(qm_hindcasts_json, "r") as fh:
        qm_hindcasts = json.load(fh)

    if len(hindcasts) != len(qm_hindcasts):
        raise ValueError(
            "Mismatch between number of original hindcast files and QM hindcast files."
        )

    spei_paths = []
    residual_paths = []
    corrected_cwb_paths = []
    spei_paths_json.parent.mkdir(parents=True, exist_ok=True)
    residual_paths_json.parent.mkdir(parents=True, exist_ok=True)
    corrected_cwb_paths_json.parent.mkdir(parents=True, exist_ok=True)

    for fp, qm_fp in tqdm(
        zip(hindcasts, qm_hindcasts),
        total=len(hindcasts),
        desc="Saving RF residuals",
    ):
        fp = str(fp)
        qm_fp = str(qm_fp)
        ds = xr.open_dataset(fp)
        ds_qm = xr.open_dataset(qm_fp)
        hind_da = ds[var_name]
        qm_da = ds_qm[var_name]

        RF = RandomForestBiasCorrector
        pred_s = RF._stack_by_samples(hind_da)
        X_pred, years = feature_builder.build(hind_da)

        eval_mask_year = np.isin(years, eval_years)
        leadmonth_mask = np.isclose(
            X_pred["lead_month"].values.astype(float),
            float(leadmonth),
        )
        eval_mask = eval_mask_year & leadmonth_mask

        preds_array = np.full(X_pred.shape[0], np.nan, dtype=float)
        if eval_mask.any():
            preds_array[eval_mask] = model.predict(X_pred.values[eval_mask])

        pred_da_stacked = xr.DataArray(
            preds_array, coords=(pred_s.coords["sample"],), dims=("sample",)
        )
        pred_da = pred_da_stacked.unstack("sample")

        # attach predicted residual to a copy of the original dataset to preserve structure
        ds_out = ds.copy()
        ds_out[var_name] = pred_da
        valid_times = np.unique(pred_s.coords["time"].values[eval_mask])
        if len(valid_times) == 0:
            ds.close()
            ds_qm.close()
            ds_out.close()
            continue
        orig_time_sel = ds["time"].sel(time=pd.to_datetime(valid_times))

        # subset ds_out to the same time index
        ds_out = ds_out.sel(time=pd.to_datetime(valid_times))
        ds_out = ds_out.assign_coords(time=orig_time_sel)
        ds_out["time"].attrs = ds["time"].attrs

        p = Path(fp)
        out_base = Path(f"{out_dir}/rf_residuals/")
        out_base.mkdir(parents=True, exist_ok=True)
        cwb_base = Path(f"{out_dir}/cwb_corrected/")
        cwb_base.mkdir(parents=True, exist_ok=True)

        out_path = f"{out_base}/{p.stem}{suffix}.nc"
        ds_out.to_netcdf(str(out_path))
        residual_paths.append(str(Path(out_path)))

        corrected_cwb = combine_qm_and_residuals(
            qm_da, pred_da, var_name=var_name
        ).squeeze()
        corrected_cwb_path = cwb_base / f"{p.stem}_qm_rf_cwb.nc"
        ds_cwb = ds.copy()
        ds_cwb[var_name] = corrected_cwb
        ds_cwb = ds_cwb.sel(time=pd.to_datetime(valid_times))
        ds_cwb = ds_cwb.assign_coords(time=orig_time_sel)
        ds_cwb["time"].attrs = ds["time"].attrs
        ds_cwb.to_netcdf(str(corrected_cwb_path))
        ds_cwb.close()
        corrected_cwb_paths.append(str(corrected_cwb_path))

        if save_spei:
            spei_base = Path(f"{out_dir}/spei_corrected/")
            spei_base.mkdir(parents=True, exist_ok=True)
            spei_out_path = f"{spei_base}/{p.stem}_qm_rf_spei.nc"

            spei = compute_spei_from_rf_corrected(
                corrected_cwb,
                month_range=(1, 3),
                var_names=[var_name, "spei"],
                dist=sps.fisk,
            )

            ds_spei = ds_out.copy()
            ds_spei["spei"] = spei
            ds_spei.to_netcdf(str(spei_out_path))
            ds_spei.close()
            spei_paths.append(str(Path(spei_out_path)))

        ds.close()
        ds_qm.close()
        ds_out.close()

    if save_spei:
        with open(spei_paths_json, "w") as fh:
            json.dump(spei_paths, fh, indent=2)

    with open(residual_paths_json, "w") as fh:
        json.dump(residual_paths, fh, indent=2)

    with open(corrected_cwb_paths_json, "w") as fh:
        json.dump(corrected_cwb_paths, fh, indent=2)

    return spei_paths_json, residual_paths_json, corrected_cwb_paths_json


def evaluate(config_path: Path | None = None, config_overrides=None):
    default_cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = get_qm_rf_global_config(
        config_path=config_path,
        overrides=config_overrides,
        default_config=default_cfg_path,
    )

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    workflow = cfg["workflow"]
    run_rf_evaluate = bool(workflow.get("run_rf_evaluate", False))
    evaluate_cwb_flag = bool(workflow.get("evaluate_cwb", False))
    evaluate_spei_flag = bool(workflow.get("evaluate_spei", False))

    feature_builder = RFFeatureBuilder(
        Path(cfg["reference_data"]),
        cfg["var_name"],
        cfg.get("additional_reference_features", []),
    )

    leadmonths = cfg["leadmonth"]

    for leadmonth in leadmonths:
        leadmonth_label = f"lm{leadmonth}"
        month_data_dir = out_dir / "data" / leadmonth_label
        month_data_dir.mkdir(parents=True, exist_ok=True)

        metrics = {}
        model = None

        if run_rf_evaluate:
            residuals_json = (
                Path(cfg["output_dir"])
                / "paths"
                / f"lm{leadmonth}"
                / "qm_residuals_paths.json"
            )
            X_eval, y_eval = collect_eval_set(
                residuals_json,
                Path(cfg["hindcasts_json"]),
                cfg["var_name"],
                cfg["predictor_vars"],
                cfg["leave_out_years"],
                feature_builder,
                leadmonth=leadmonth,
            )
            model_path = Path(cfg["model_dir"]) / f"lm{leadmonth}" / cfg["model_name"]
            model = load(model_path)
            metrics["n_eval_samples"] = int(len(y_eval))
            print(
                f"Evaluating RF predictions for leadmonth={leadmonth} on held-out years {cfg['leave_out_years']} with model from {model_path}"
            )

        spei_paths_json = (
            out_dir / "paths" / leadmonth_label / "corrected_spei_paths.json"
        )
        residual_paths_json = (
            out_dir / "paths" / leadmonth_label / "corrected_residuals_paths.json"
        )
        corrected_cwb_paths_json = (
            out_dir / "paths" / leadmonth_label / "corrected_cwb_paths.json"
        )

        if run_rf_evaluate:
            qm_hindcasts_json = (
                Path(cfg["output_dir"])
                / "paths"
                / f"lm{leadmonth}"
                / "qm_hindcast_paths.json"
            )
            save_predicted_residuals(
                model,
                Path(cfg["hindcasts_json"]),
                qm_hindcasts_json,
                cfg["var_name"],
                cfg["predictor_vars"],
                cfg["leave_out_years"],
                out_dir=month_data_dir,
                spei_paths_json=spei_paths_json,
                residual_paths_json=residual_paths_json,
                corrected_cwb_paths_json=corrected_cwb_paths_json,
                feature_builder=feature_builder,
                leadmonth=leadmonth,
                save_spei=cfg["workflow"]["evaluate_spei"],
            )

        metrics_path = None

        if evaluate_cwb_flag:
            plot_dir = f"{cfg['plot_dir']}/cwb_rf_eval/{leadmonth_label}"
            evaluate_cwb(
                corrected_cwb_json=Path(corrected_cwb_paths_json),
                hindcasts_json=Path(cfg["hindcasts_json"]),
                reference_data=Path(cfg["reference_data"]),
                out_dir=month_data_dir,
                cwb_var=cfg["var_name"],
                eval_years=cfg["leave_out_years"],
                std_multiplier=float(cfg["spei_std_multiplier"]),
                plot_dir=plot_dir,
                qm_hindcasts_json=(
                    Path(cfg["output_dir"])
                    / "paths"
                    / f"lm{leadmonth}"
                    / "qm_hindcast_paths.json"
                ),
            )

        if evaluate_spei_flag:

            with open(spei_paths_json, "r") as fh:
                corrected_spei_paths = json.load(fh)

            plot_dir = f"{cfg['plot_dir']}/spei_rf_eval/{leadmonth_label}"
            spei_metrics = evaluate_spei(
                corrected_spei_paths=corrected_spei_paths,
                hindcasts_spei_json=Path(cfg["hindcasts_spei_json"]),
                reference_spei_json=Path(cfg["reference_spei_json"]),
                out_dir=month_data_dir,
                eval_years=cfg["leave_out_years"],
                std_multiplier=float(cfg["spei_std_multiplier"]),
                plot_dir=plot_dir,
            )
            metrics.update(spei_metrics)
            metrics_path = (
                out_dir / "metrics" / f"spei_eval_metrics_{leadmonth_label}.yaml"
            )
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w") as fh:
                yaml.safe_dump(metrics, fh)


if __name__ == "__main__":
    evaluate()
