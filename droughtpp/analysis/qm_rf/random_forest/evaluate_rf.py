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

from .model_rf import (
    RandomForestBiasCorrector,
    MixedEffectsTreeRegressor,
    analyze_merf_contributions,
    save_merf_diagnostics_plot,
)
from .knn_random_effects import LocalKNNRandomEffectsRegressor
from .features import RFFeatureBuilder
from ..config_loader import get_qm_rf_global_config
from droughtpp.analysis.qm_rf.utils.spei_evaluation import evaluate_spei
from droughtpp.analysis.qm_rf.utils.cwb_evaluation import evaluate_cwb
from droughtpp.analysis.qm_rf.utils.intensity_evaluation import evaluate_intensity
from droughtpp.analysis.qm_rf.utils.visualization import plot_feature_importance_table
from droughtpp.analysis.qm_rf.utils.spei_from_rf import (
    combine_qm_and_residuals,
    compute_spei_from_rf_corrected,
)


def _build_group_labels(stacked_da: xr.DataArray, years: np.ndarray, group_by: str):
    group_by = str(group_by).lower()

    if group_by == "year":
        return years.astype(str)

    if group_by == "calendar_month":
        if "time" not in stacked_da.coords:
            raise ValueError(
                "group_by=calendar_month requested but time coord is missing"
            )
        months = pd.to_datetime(stacked_da.coords["time"].values).month
        return months.astype(str)

    if group_by == "member":
        if "member" not in stacked_da.coords:
            raise ValueError("group_by=member requested but member coord is missing")
        return np.asarray(stacked_da.coords["member"].values).astype(str)

    if "latitude" not in stacked_da.coords or "longitude" not in stacked_da.coords:
        raise ValueError(
            f"group_by={group_by} requires latitude/longitude coords in stacked samples"
        )

    lat = np.asarray(stacked_da.coords["latitude"].values)
    lon = np.asarray(stacked_da.coords["longitude"].values)

    if group_by == "grid_id":
        return np.char.add(
            np.round(lat, 6).astype(str), np.char.add("_", np.round(lon, 6).astype(str))
        )

    if group_by == "grid_year":
        grid = np.char.add(
            np.round(lat, 6).astype(str), np.char.add("_", np.round(lon, 6).astype(str))
        )
        return np.char.add(grid, np.char.add("_", years.astype(str)))

    raise ValueError(f"Unsupported mixed group_by value: {group_by}")


def _build_knn_neighbor_features(
    stacked_da: xr.DataArray,
    X_full: pd.DataFrame,
    group_by: str,
) -> np.ndarray | None:
    mode = str(group_by).lower()
    if mode == "knn_spatial":
        if "latitude" not in stacked_da.coords or "longitude" not in stacked_da.coords:
            raise ValueError("knn_spatial requires latitude/longitude coordinates")
        lat = np.asarray(stacked_da.coords["latitude"].values, dtype=float)
        lon = np.asarray(stacked_da.coords["longitude"].values, dtype=float)
        return np.column_stack([lat, lon])
    if mode == "knn_feature":
        return np.asarray(X_full.values, dtype=float)
    return None


def collect_eval_set(
    residuals_json: Path,
    qm_hindcasts_json: Path,
    var_name: str,
    predictor_vars: List[str],
    eval_years: List[int],
    feature_builder: RFFeatureBuilder,
    group_by: str = "grid_id",
):
    with open(residuals_json, "r") as fh:
        residuals = json.load(fh)
    with open(qm_hindcasts_json, "r") as fh:
        qm_hindcasts = json.load(fh)

    X_parts = []
    y_parts = []
    group_parts = []

    RF = RandomForestBiasCorrector

    for resid_path, qm_hindcast_path in zip(residuals, qm_hindcasts):
        resid_path = str(resid_path)
        qm_hindcast_path = str(qm_hindcast_path)

        res_da = xr.open_dataset(resid_path)[var_name]
        pred_da = xr.open_dataset(qm_hindcast_path)[var_name]

        res_s = RF._stack_by_samples(res_da)
        X_full, years = feature_builder.build(pred_da)
        y_full = pd.Series(res_s.values, name="residual")
        if str(group_by).lower() in {"knn_spatial", "knn_feature"}:
            groups_full = np.repeat("knn", len(y_full)).astype(str)
        else:
            groups_full = _build_group_labels(res_s, years, group_by)

        finite_mask = np.isfinite(y_full.values) & np.all(
            np.isfinite(X_full.values), axis=1
        )

        if years is not None and len(eval_years) > 0:
            eval_mask_year = np.isin(years, eval_years)
        else:
            eval_mask_year = np.zeros_like(finite_mask, dtype=bool)

        eval_mask = finite_mask & eval_mask_year

        if eval_mask.any():
            X_parts.append(X_full.loc[eval_mask].reset_index(drop=True))
            y_parts.append(y_full.loc[eval_mask].reset_index(drop=True))
            group_parts.append(pd.Series(groups_full[eval_mask]).reset_index(drop=True))

    if len(X_parts) == 0:
        return (
            pd.DataFrame(columns=predictor_vars),
            pd.Series(dtype=float),
            np.array([], dtype=str),
        )

    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    groups = pd.concat(group_parts, ignore_index=True).astype(str).values
    return X, y, groups


def save_predicted_residuals(
    model,
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
    model_type: str = "rf",
    mixed_group_by: str = "grid_id",
    save_spei: bool = True,
    suffix: str = "_qm_rf_residuals",
):
    with open(qm_hindcasts_json, "r") as fh:
        qm_hindcasts = json.load(fh)

    spei_paths = []
    residual_paths = []
    corrected_cwb_paths = []
    spei_paths_json.parent.mkdir(parents=True, exist_ok=True)
    residual_paths_json.parent.mkdir(parents=True, exist_ok=True)
    corrected_cwb_paths_json.parent.mkdir(parents=True, exist_ok=True)

    for qm_fp in tqdm(
        qm_hindcasts, total=len(qm_hindcasts), desc="Saving RF residuals"
    ):
        qm_fp = str(qm_fp)
        ds_qm = xr.open_dataset(qm_fp)
        qm_da = ds_qm[var_name]

        RF = RandomForestBiasCorrector
        pred_s = RF._stack_by_samples(qm_da)
        X_pred, years = feature_builder.build(qm_da)
        group_mode = str(mixed_group_by).lower()
        is_knn_mode = group_mode in {"knn_spatial", "knn_feature"}
        if is_knn_mode:
            groups_pred = np.repeat("knn", X_pred.shape[0]).astype(str)
            knn_pred = _build_knn_neighbor_features(pred_s, X_pred, mixed_group_by)
        else:
            groups_pred = _build_group_labels(pred_s, years, mixed_group_by)
            knn_pred = None

        eval_mask_year = np.isin(years, eval_years)
        eval_mask = eval_mask_year

        preds_array = np.full(X_pred.shape[0], np.nan, dtype=float)
        if eval_mask.any():
            if str(model_type).lower() in {"mixed_rf", "mixed"}:
                if isinstance(model, LocalKNNRandomEffectsRegressor):
                    preds_array[eval_mask] = model.predict(
                        X_pred.values[eval_mask],
                        neighbor_features=knn_pred[eval_mask],
                    )
                else:
                    preds_array[eval_mask] = model.predict(
                        X_pred.values[eval_mask],
                        groups=groups_pred[eval_mask],
                    )
            else:
                preds_array[eval_mask] = model.predict(X_pred.values[eval_mask])

        pred_da_stacked = xr.DataArray(
            preds_array, coords=(pred_s.coords["sample"],), dims=("sample",)
        )
        pred_da = pred_da_stacked.unstack("sample")

        # attach predicted residual to a copy of the QM dataset to preserve structure
        ds_out = ds_qm.copy()
        ds_out[var_name] = pred_da
        valid_times = np.unique(pred_s.coords["time"].values[eval_mask])
        orig_time_sel = ds_qm["time"].sel(time=pd.to_datetime(valid_times))

        # subset ds_out to the same time index
        ds_out = ds_out.sel(time=pd.to_datetime(valid_times))
        ds_out = ds_out.assign_coords(time=orig_time_sel)
        ds_out["time"].attrs = ds_qm["time"].attrs

        p = Path(qm_fp)
        out_base = Path(f"{out_dir}/rf_residuals/")
        out_base.mkdir(parents=True, exist_ok=True)
        cwb_base = Path(f"{out_dir}/cwb_corrected/")
        cwb_base.mkdir(parents=True, exist_ok=True)

        out_path = f"{out_base}/{p.stem}{suffix}_lm{leadmonth}.nc"
        ds_out.to_netcdf(str(out_path))
        residual_paths.append(str(Path(out_path)))

        corrected_cwb = combine_qm_and_residuals(
            qm_da, pred_da, var_name=var_name
        ).squeeze()
        corrected_cwb_path = cwb_base / f"{p.stem}_qm_rf_cwb_lm{leadmonth}.nc"
        ds_cwb = ds_qm.copy()
        ds_cwb[var_name] = corrected_cwb
        ds_cwb = ds_cwb.sel(time=pd.to_datetime(valid_times))
        ds_cwb = ds_cwb.assign_coords(time=orig_time_sel)
        ds_cwb["time"].attrs = ds_qm["time"].attrs
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
    var_name = cfg["var_name"]
    workflow = cfg["workflow"]
    model_type = str(cfg.get("ml_arguments", {}).get("model_type", "rf")).lower()
    mixed_group_by = cfg.get("ml_arguments", {}).get("mixed_group_by", "grid_id")
    run_rf_evaluate = bool(workflow.get("run_rf_evaluate", False))
    evaluate_cwb_flag = bool(workflow.get("evaluate_cwb", False))
    evaluate_intensity_flag = bool(workflow.get("evaluate_intensity", False))
    evaluate_spei_flag = bool(workflow.get("evaluate_spei", False))
    rf_results_tag = cfg["rf_results_tag"]
    train_leadmonth_specific = bool(
        cfg.get("ml_arguments", {}).get("train_leadmonth_specific", True)
    )

    feature_builder = RFFeatureBuilder(
        Path(cfg["reference_data"]),
        cfg["var_name"],
        cfg.get("additional_reference_features", []),
        cfg.get("include_obs_prev_lags", True),
        cfg.get("feature_flags", {}),
    )

    leadmonths = cfg["leadmonth"]

    for leadmonth in leadmonths:
        rf_results_tag_lm = rf_results_tag
        leadmonth_label = f"lm{leadmonth}"

        metrics = {}
        model = None

        needs_rf_eval_paths = (
            run_rf_evaluate
            or evaluate_cwb_flag
            or evaluate_intensity_flag
            or evaluate_spei_flag
        )

        if needs_rf_eval_paths:
            residuals_json = (
                Path(cfg["output_dir"])
                / var_name
                / "paths"
                / f"lm{leadmonth}"
                / "qm_residuals_paths.json"
            )
            qm_hindcasts_json = (
                Path(cfg["output_dir"])
                / var_name
                / "paths"
                / f"lm{leadmonth}"
                / "qm_hindcast_paths.json"
            )
            X_eval, y_eval, groups_eval = collect_eval_set(
                residuals_json,
                qm_hindcasts_json,
                cfg["var_name"],
                cfg["predictor_vars"],
                cfg["leave_out_years"],
                feature_builder,
                group_by=mixed_group_by,
            )
            feature_count = int(X_eval.shape[1])
            rf_results_tag_lm = f"{rf_results_tag}_nf{feature_count}"

        if run_rf_evaluate:
            model_suffix = f"lm{leadmonth}" if train_leadmonth_specific else "alllm"
            model_path = (
                Path(cfg["model_dir"])
                / var_name
                / rf_results_tag_lm
                / f"{cfg['model_name']}_{model_suffix}.joblib"
            )
            model = load(model_path)
            metrics["n_eval_samples"] = int(len(y_eval))
            print(
                f"Evaluating RF predictions for leadmonth={leadmonth} on held-out years {cfg['leave_out_years']} with model from {model_path}"
            )

            # Print MERF diagnostic if this is a mixed model
            if isinstance(model, MixedEffectsTreeRegressor):
                diagnostics = analyze_merf_contributions(
                    model,
                    X_eval.values,
                    groups_eval,
                    y_eval.values,
                )

                # Save diagnostics plot
                leadmonth_dir = (
                    leadmonth_label if train_leadmonth_specific else "all_leadmonths"
                )
                diag_plot_path = (
                    out_dir
                    / "plots"
                    / var_name
                    / "cwb_rf_eval"
                    / rf_results_tag_lm
                    / leadmonth_dir
                    / f"{cfg['model_name']}_merf_eval_diagnostics.png"
                )
                save_merf_diagnostics_plot(diagnostics, diag_plot_path)

        month_data_dir = (
            out_dir
            / var_name
            / "data"
            / "rf_eval"
            / rf_results_tag_lm
            / leadmonth_label
        )
        month_data_dir.mkdir(parents=True, exist_ok=True)

        spei_paths_json = (
            out_dir
            / var_name
            / "paths"
            / "rf_eval"
            / rf_results_tag_lm
            / leadmonth_label
            / "corrected_spei_paths.json"
        )
        residual_paths_json = (
            out_dir
            / var_name
            / "paths"
            / "rf_eval"
            / rf_results_tag_lm
            / leadmonth_label
            / "corrected_residuals_paths.json"
        )
        corrected_cwb_paths_json = (
            out_dir
            / var_name
            / "paths"
            / "rf_eval"
            / rf_results_tag_lm
            / leadmonth_label
            / "corrected_cwb_paths.json"
        )

        if run_rf_evaluate:
            save_predicted_residuals(
                model,
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
                model_type=model_type,
                mixed_group_by=mixed_group_by,
                save_spei=cfg["workflow"]["evaluate_spei"],
            )

        metrics_path = None

        if evaluate_cwb_flag:
            plot_dir = f"{cfg['plot_dir']}/{var_name}/cwb_rf_eval/{rf_results_tag_lm}/{leadmonth_label}"
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
                    / var_name
                    / "paths"
                    / f"lm{leadmonth}"
                    / "qm_hindcast_paths.json"
                ),
                qm_residuals_json=(
                    Path(cfg["output_dir"])
                    / var_name
                    / "paths"
                    / f"lm{leadmonth}"
                    / "qm_residuals_paths.json"
                ),
                ml_residuals_json=Path(residual_paths_json),
                land_mask_path=cfg.get("land_mask_path"),
            )

            if run_rf_evaluate and hasattr(model, "feature_importances_"):
                importances = np.asarray(model.feature_importances_, dtype=float)
                feature_names = list(X_eval.columns)
                model_label = type(model).__name__
                plot_feature_importance_table(
                    feature_names=feature_names,
                    importances=importances,
                    out_dir=plot_dir,
                    model_label=model_label,
                    top_n=30,
                    file_prefix="ml_feature_importance",
                )

        if evaluate_intensity_flag:
            plot_dir = f"{cfg['plot_dir']}/{var_name}/intensity_rf_eval/{rf_results_tag_lm}/{leadmonth_label}"
            evaluate_intensity(
                corrected_intensity_json=Path(corrected_cwb_paths_json),
                hindcasts_json=Path(cfg["hindcasts_json"]),
                reference_data=Path(cfg["reference_data"]),
                out_dir=month_data_dir,
                intensity_var=cfg["var_name"],
                eval_years=cfg["leave_out_years"],
                std_multiplier=float(cfg["spei_std_multiplier"]),
                plot_dir=plot_dir,
                qm_hindcasts_json=(
                    Path(cfg["output_dir"])
                    / var_name
                    / "paths"
                    / f"lm{leadmonth}"
                    / "qm_hindcast_paths.json"
                ),
                land_mask_path=cfg.get("land_mask_path"),
            )

        if evaluate_spei_flag:

            with open(spei_paths_json, "r") as fh:
                corrected_spei_paths = json.load(fh)

            plot_dir = f"{cfg['plot_dir']}/{var_name}/spei_rf_eval/{rf_results_tag_lm}/{leadmonth_label}"
            spei_metrics = evaluate_spei(
                corrected_spei_paths=corrected_spei_paths,
                hindcasts_spei_json=Path(cfg["hindcasts_spei_json"]),
                reference_spei_json=Path(cfg["reference_spei_json"]),
                out_dir=month_data_dir,
                eval_years=cfg["leave_out_years"],
                std_multiplier=float(cfg["spei_std_multiplier"]),
                plot_dir=plot_dir,
                land_mask_path=cfg.get("land_mask_path"),
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
