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
from cmethods import adjust

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
from droughtpp.analysis.qm_rf.utils.evaluation import (
    build_knn_neighbor_features,
    build_mixed_group_labels,
)


def _append_eval_block(
    res_da: xr.DataArray,
    pred_da: xr.DataArray,
    feature_builder: RFFeatureBuilder,
    group_by: str,
    eval_years: List[int],
    knn_feature_columns: List[str] | None,
    X_parts: list,
    y_parts: list,
    group_parts: list,
    knn_parts: list,
    quantile_group_predictor: str | None,
    quantile_group_n: int | None,
):
    res_s = RandomForestBiasCorrector._stack_by_samples(res_da)
    X_full, years = feature_builder.build(pred_da)
    y_full = pd.Series(res_s.values, name="residual")
    knn_full = build_knn_neighbor_features(
        res_s,
        X_full,
        group_by,
        knn_feature_columns=knn_feature_columns,
    )
    if str(group_by).lower() in {"knn_spatial", "knn_feature", "knn_proximity"}:
        groups_full = np.repeat("knn", len(y_full)).astype(str)
    else:
        groups_full = build_mixed_group_labels(
            res_s,
            years,
            group_by,
            X_full=X_full,
            quantile_group_predictor=quantile_group_predictor,
            quantile_group_n=quantile_group_n,
        )

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
        if knn_full is not None:
            knn_parts.append(np.asarray(knn_full[eval_mask], dtype=float))


def collect_eval_set(
    residuals_json: Path,
    qm_hindcasts_json: Path,
    var_name: str,
    predictor_vars: List[str],
    eval_years: List[int],
    feature_builder: RFFeatureBuilder,
    group_by: str = "grid_id",
    correction_target: str = "member",
    knn_feature_columns: List[str] | None = None,
    quantile_group_predictor: str | None = None,
    quantile_group_n: int | None = None,
):
    with open(residuals_json, "r") as fh:
        residuals = json.load(fh)
    with open(qm_hindcasts_json, "r") as fh:
        qm_hindcasts = json.load(fh)

    X_parts = []
    y_parts = []
    group_parts = []
    knn_parts = []

    correction_mode = str(correction_target).lower()
    if correction_mode in {"ensemble_mean", "mean", "ens_mean"}:
        residual_members = []
        predictor_members = []
        for resid_path, qm_hindcast_path in zip(residuals, qm_hindcasts):
            with xr.open_dataset(str(resid_path)) as ds_res, xr.open_dataset(
                str(qm_hindcast_path)
            ) as ds_qm:
                residual_members.append(ds_res[var_name].load())
                predictor_members.append(ds_qm[var_name].load())

        res_da = xr.concat(residual_members, dim="member").mean("member", skipna=True)
        pred_da = xr.concat(predictor_members, dim="member").mean("member", skipna=True)
        _append_eval_block(
            res_da,
            pred_da,
            feature_builder,
            group_by,
            eval_years,
            knn_feature_columns,
            X_parts,
            y_parts,
            group_parts,
            knn_parts,
            quantile_group_predictor,
            quantile_group_n,
        )
    else:
        for resid_path, qm_hindcast_path in zip(residuals, qm_hindcasts):
            resid_path = str(resid_path)
            qm_hindcast_path = str(qm_hindcast_path)

            with xr.open_dataset(resid_path) as ds_res, xr.open_dataset(
                qm_hindcast_path
            ) as ds_qm:
                res_da = ds_res[var_name].load()
                pred_da = ds_qm[var_name].load()

            _append_eval_block(
                res_da,
                pred_da,
                feature_builder,
                group_by,
                eval_years,
                knn_feature_columns,
                X_parts,
                y_parts,
                group_parts,
                knn_parts,
                quantile_group_predictor,
                quantile_group_n,
            )

    if len(X_parts) == 0:
        return (
            pd.DataFrame(columns=predictor_vars),
            pd.Series(dtype=float),
            np.array([], dtype=str),
            None,
        )

    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    groups = pd.concat(group_parts, ignore_index=True).astype(str).values
    knn_features = np.vstack(knn_parts) if len(knn_parts) > 0 else None
    return X, y, groups, knn_features


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
    knn_feature_columns: List[str] | None = None,
    quantile_group_predictor: str | None = None,
    quantile_group_n: int | None = None,
    save_spei: bool = True,
    suffix: str = "_qm_rf_residuals",
    correction_target: str = "member",
    spread_reconstruction: str = "preserve_qm",
    baseline_hindcasts_json: Path | None = None,
):
    with open(qm_hindcasts_json, "r") as fh:
        qm_hindcasts = json.load(fh)

    spei_paths = []
    residual_paths = []
    corrected_cwb_paths = []
    spei_paths_json.parent.mkdir(parents=True, exist_ok=True)
    residual_paths_json.parent.mkdir(parents=True, exist_ok=True)
    corrected_cwb_paths_json.parent.mkdir(parents=True, exist_ok=True)

    out_base = Path(f"{out_dir}/rf_residuals/")
    out_base.mkdir(parents=True, exist_ok=True)
    cwb_base = Path(f"{out_dir}/cwb_corrected/")
    cwb_base.mkdir(parents=True, exist_ok=True)

    correction_mode = str(correction_target).lower()

    if correction_mode in {"ensemble_mean", "mean", "ens_mean"}:
        qm_member_das = []
        for qm_fp in qm_hindcasts:
            with xr.open_dataset(str(qm_fp)) as ds_qm:
                qm_member_das.append(ds_qm[var_name].load())

        qm_ensemble = xr.concat(qm_member_das, dim="member")
        qm_mean = qm_ensemble.mean("member", skipna=True)

        RF = RandomForestBiasCorrector
        pred_s = RF._stack_by_samples(qm_mean)
        X_pred, years = feature_builder.build(qm_mean)
        group_mode = str(mixed_group_by).lower()
        is_knn_mode = group_mode in {"knn_spatial", "knn_feature", "knn_proximity"}
        if is_knn_mode:
            groups_pred = np.repeat("knn", X_pred.shape[0]).astype(str)
            knn_pred = build_knn_neighbor_features(
                pred_s,
                X_pred,
                mixed_group_by,
                knn_feature_columns=knn_feature_columns,
            )
        else:
            groups_pred = build_mixed_group_labels(
                pred_s,
                years,
                mixed_group_by,
                X_full=X_pred,
                quantile_group_predictor=quantile_group_predictor,
                quantile_group_n=quantile_group_n,
            )
            knn_pred = None

        eval_mask_year = np.isin(years, eval_years)
        finite_x_mask = np.all(np.isfinite(X_pred.values), axis=1)
        eval_mask = eval_mask_year & finite_x_mask

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

        pred_da = xr.DataArray(
            preds_array, coords=(pred_s.coords["sample"],), dims=("sample",)
        ).unstack("sample")

        valid_times = np.unique(pred_s.coords["time"].values[eval_mask_year])
        qm_mean_valid = qm_mean.sel(time=pd.to_datetime(valid_times))
        pred_da_valid = pred_da.sel(time=pd.to_datetime(valid_times))
        corrected_mean = qm_mean_valid + pred_da_valid

        spread_mode = str(spread_reconstruction).lower()
        spread_scale = xr.ones_like(corrected_mean)
        if (
            spread_mode in {"baseline_std", "match_baseline_std"}
            and baseline_hindcasts_json is not None
        ):
            with open(baseline_hindcasts_json, "r") as fh:
                baseline_hindcasts = json.load(fh)
            baseline_members = []
            for baseline_fp in baseline_hindcasts[: len(qm_hindcasts)]:
                with xr.open_dataset(str(baseline_fp)) as ds_base:
                    baseline_members.append(ds_base[var_name].load())
            baseline_ensemble = xr.concat(baseline_members, dim="member")
            baseline_ensemble = baseline_ensemble.sel(time=pd.to_datetime(valid_times))
            qm_ensemble_valid = qm_ensemble.sel(time=pd.to_datetime(valid_times))
            sigma_baseline = baseline_ensemble.std("member", skipna=True)
            sigma_qm = qm_ensemble_valid.std("member", skipna=True)
            spread_scale = xr.where(
                np.abs(sigma_qm) > 0.0, sigma_baseline / sigma_qm, 1.0
            )
            spread_scale = spread_scale.fillna(1.0)

        for i, qm_fp in enumerate(
            tqdm(qm_hindcasts, total=len(qm_hindcasts), desc="Saving RF residuals")
        ):
            qm_fp = str(qm_fp)
            with xr.open_dataset(qm_fp) as ds_qm:
                qm_member = ds_qm[var_name].sel(time=pd.to_datetime(valid_times)).load()
                qm_member_anom = qm_member - qm_mean_valid

                corrected_cwb = corrected_mean + spread_scale * qm_member_anom
                pred_member = corrected_cwb - qm_member

                p = Path(qm_fp)
                out_path = f"{out_base}/{p.stem}{suffix}_lm{leadmonth}.nc"
                corrected_cwb_path = cwb_base / f"{p.stem}_qm_rf_cwb_lm{leadmonth}.nc"

                orig_time_sel = ds_qm["time"].sel(time=pd.to_datetime(valid_times))

                ds_out = ds_qm.sel(time=pd.to_datetime(valid_times)).copy()
                ds_out[var_name] = pred_member
                ds_out = ds_out.assign_coords(time=orig_time_sel)
                ds_out["time"].attrs = ds_qm["time"].attrs
                ds_out.to_netcdf(str(out_path))
                residual_paths.append(str(Path(out_path)))

                ds_cwb = ds_qm.sel(time=pd.to_datetime(valid_times)).copy()
                ds_cwb[var_name] = corrected_cwb
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

                ds_out.close()
    else:
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
            is_knn_mode = group_mode in {"knn_spatial", "knn_feature", "knn_proximity"}
            if is_knn_mode:
                groups_pred = np.repeat("knn", X_pred.shape[0]).astype(str)
                knn_pred = build_knn_neighbor_features(
                    pred_s,
                    X_pred,
                    mixed_group_by,
                    knn_feature_columns=knn_feature_columns,
                )
            else:
                groups_pred = build_mixed_group_labels(
                    pred_s,
                    years,
                    mixed_group_by,
                    X_full=X_pred,
                    quantile_group_predictor=quantile_group_predictor,
                    quantile_group_n=quantile_group_n,
                )
                knn_pred = None

            eval_mask_year = np.isin(years, eval_years)
            finite_x_mask = np.all(np.isfinite(X_pred.values), axis=1)
            eval_mask = eval_mask_year & finite_x_mask

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

            ds_out = ds_qm.copy()
            ds_out[var_name] = pred_da
            valid_times = np.unique(pred_s.coords["time"].values[eval_mask_year])
            orig_time_sel = ds_qm["time"].sel(time=pd.to_datetime(valid_times))

            ds_out = ds_out.sel(time=pd.to_datetime(valid_times))
            ds_out = ds_out.assign_coords(time=orig_time_sel)
            ds_out["time"].attrs = ds_qm["time"].attrs

            p = Path(qm_fp)
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


def save_predicted_event_probabilities(
    model,
    qm_hindcasts_json: Path,
    var_name: str,
    eval_years: List[int],
    out_dir: Path,
    event_probability_paths_json: Path,
    feature_builder: RFFeatureBuilder,
    leadmonth: int,
    model_type: str = "rf",
    mixed_group_by: str = "grid_id",
    knn_feature_columns: List[str] | None = None,
    quantile_group_predictor: str | None = None,
    quantile_group_n: int | None = None,
    correction_target: str = "member",
    event_probability_var: str = "event_probability",
):
    with open(qm_hindcasts_json, "r") as fh:
        qm_hindcasts = json.load(fh)

    event_probability_paths = []
    event_probability_paths_json.parent.mkdir(parents=True, exist_ok=True)
    event_base = Path(f"{out_dir}/event_probability/")
    event_base.mkdir(parents=True, exist_ok=True)

    correction_mode = str(correction_target).lower()

    if correction_mode in {"ensemble_mean", "mean", "ens_mean"}:
        qm_member_das = []
        for qm_fp in qm_hindcasts:
            with xr.open_dataset(str(qm_fp)) as ds_qm:
                qm_member_das.append(ds_qm[var_name].load())

        qm_ensemble = xr.concat(qm_member_das, dim="member")
        qm_mean = qm_ensemble.mean("member", skipna=True)

        pred_s = RandomForestBiasCorrector._stack_by_samples(qm_mean)
        X_pred, years = feature_builder.build(qm_mean)
        group_mode = str(mixed_group_by).lower()
        is_knn_mode = group_mode in {"knn_spatial", "knn_feature", "knn_proximity"}

        if is_knn_mode:
            groups_pred = np.repeat("knn", X_pred.shape[0]).astype(str)
            knn_pred = build_knn_neighbor_features(
                pred_s,
                X_pred,
                mixed_group_by,
                knn_feature_columns=knn_feature_columns,
            )
        else:
            groups_pred = build_mixed_group_labels(
                pred_s,
                years,
                mixed_group_by,
                X_full=X_pred,
                quantile_group_predictor=quantile_group_predictor,
                quantile_group_n=quantile_group_n,
            )
            knn_pred = None

        eval_mask_year = np.isin(years, eval_years)
        finite_x_mask = np.all(np.isfinite(X_pred.values), axis=1)
        eval_mask = eval_mask_year & finite_x_mask

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

        pred_da = xr.DataArray(
            preds_array, coords=(pred_s.coords["sample"],), dims=("sample",)
        ).unstack("sample")
        pred_da = pred_da.clip(min=0.0, max=1.0)

        valid_times = np.unique(pred_s.coords["time"].values[eval_mask_year])
        pred_da_valid = pred_da.sel(time=pd.to_datetime(valid_times))

        for qm_fp in qm_hindcasts:
            qm_fp = str(qm_fp)
            with xr.open_dataset(qm_fp) as ds_qm:
                p = Path(qm_fp)
                out_path = event_base / f"{p.stem}_qm_rf_eventprob_lm{leadmonth}.nc"

                orig_time_sel = ds_qm["time"].sel(time=pd.to_datetime(valid_times))
                ds_out = ds_qm.sel(time=pd.to_datetime(valid_times)).copy()
                ds_out[event_probability_var] = pred_da_valid
                ds_out = ds_out.assign_coords(time=orig_time_sel)
                ds_out["time"].attrs = ds_qm["time"].attrs
                ds_out = ds_out[[event_probability_var]]
                ds_out.to_netcdf(str(out_path))
                ds_out.close()
                event_probability_paths.append(str(out_path))
    else:
        for qm_fp in tqdm(
            qm_hindcasts,
            total=len(qm_hindcasts),
            desc="Saving RF event probabilities",
        ):
            qm_fp = str(qm_fp)
            ds_qm = xr.open_dataset(qm_fp)
            qm_da = ds_qm[var_name]

            pred_s = RandomForestBiasCorrector._stack_by_samples(qm_da)
            X_pred, years = feature_builder.build(qm_da)
            group_mode = str(mixed_group_by).lower()
            is_knn_mode = group_mode in {"knn_spatial", "knn_feature", "knn_proximity"}

            if is_knn_mode:
                groups_pred = np.repeat("knn", X_pred.shape[0]).astype(str)
                knn_pred = build_knn_neighbor_features(
                    pred_s,
                    X_pred,
                    mixed_group_by,
                    knn_feature_columns=knn_feature_columns,
                )
            else:
                groups_pred = build_mixed_group_labels(
                    pred_s,
                    years,
                    mixed_group_by,
                    X_full=X_pred,
                    quantile_group_predictor=quantile_group_predictor,
                    quantile_group_n=quantile_group_n,
                )
                knn_pred = None

            eval_mask_year = np.isin(years, eval_years)
            finite_x_mask = np.all(np.isfinite(X_pred.values), axis=1)
            eval_mask = eval_mask_year & finite_x_mask

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
            pred_da = pred_da_stacked.unstack("sample").clip(min=0.0, max=1.0)

            valid_times = np.unique(pred_s.coords["time"].values[eval_mask_year])
            orig_time_sel = ds_qm["time"].sel(time=pd.to_datetime(valid_times))

            ds_out = ds_qm.sel(time=pd.to_datetime(valid_times)).copy()
            ds_out[event_probability_var] = pred_da.sel(
                time=pd.to_datetime(valid_times)
            )
            ds_out = ds_out.assign_coords(time=orig_time_sel)
            ds_out["time"].attrs = ds_qm["time"].attrs
            ds_out = ds_out[[event_probability_var]]

            p = Path(qm_fp)
            out_path = event_base / f"{p.stem}_qm_rf_eventprob_lm{leadmonth}.nc"
            ds_out.to_netcdf(str(out_path))
            event_probability_paths.append(str(out_path))

            ds_qm.close()
            ds_out.close()

    with open(event_probability_paths_json, "w") as fh:
        json.dump(event_probability_paths, fh, indent=2)

    return event_probability_paths_json


def apply_post_qm_from_val_years(
    corrected_cwb_test_json: Path,
    corrected_cwb_val_json: Path,
    reference_data: Path,
    var_name: str,
    out_dir: Path,
    n_quantiles: int,
    kind: str,
    make_spei: bool,
):
    with open(corrected_cwb_test_json, "r") as fh:
        test_paths = [Path(p) for p in json.load(fh)]
    with open(corrected_cwb_val_json, "r") as fh:
        val_paths = [Path(p) for p in json.load(fh)]

    cwb_out_dir = out_dir / "cwb_corrected_postqm"
    cwb_out_dir.mkdir(parents=True, exist_ok=True)
    spei_out_dir = out_dir / "spei_corrected_postqm"
    if make_spei:
        spei_out_dir.mkdir(parents=True, exist_ok=True)

    post_cwb_paths = []
    post_spei_paths = []

    with xr.open_dataset(reference_data) as ds_ref:
        ref_da = ds_ref[var_name].load()

    for test_path, val_path in zip(test_paths, val_paths):
        with xr.open_dataset(test_path) as ds_test, xr.open_dataset(val_path) as ds_val:
            test_da = ds_test[var_name].squeeze()
            val_da = ds_val[var_name].squeeze()

            common_val_time = np.intersect1d(
                ref_da["time"].values,
                val_da["time"].values,
            )
            if common_val_time.size == 0:
                raise ValueError(
                    f"No overlapping val-year timestamps between reference and {val_path}"
                )

            ref_cal = ref_da.sel(time=common_val_time)
            simh_cal = val_da.sel(time=common_val_time)
            simp_eval = test_da

            adjusted = adjust(
                method="quantile_mapping",
                obs=ref_cal,
                simh=simh_cal,
                simp=simp_eval,
                n_quantiles=int(n_quantiles),
                kind=str(kind),
            )

            ds_cwb = ds_test.copy()
            ds_cwb[var_name] = adjusted[var_name]
            cwb_out_path = cwb_out_dir / f"{test_path.stem}_postqm.nc"
            ds_cwb.to_netcdf(str(cwb_out_path))
            post_cwb_paths.append(str(cwb_out_path))

            if make_spei:
                spei = compute_spei_from_rf_corrected(
                    adjusted[var_name].squeeze(),
                    month_range=(1, 3),
                    var_names=[var_name, "spei"],
                    dist=sps.fisk,
                )
                ds_spei = ds_test.copy()
                ds_spei["spei"] = spei
                spei_out_path = spei_out_dir / f"{test_path.stem}_postqm_spei.nc"
                ds_spei.to_netcdf(str(spei_out_path))
                ds_spei.close()
                post_spei_paths.append(str(spei_out_path))

            ds_cwb.close()

    post_cwb_json = out_dir / "paths" / "post_qm_corrected_cwb_paths.json"
    post_cwb_json.parent.mkdir(parents=True, exist_ok=True)
    with open(post_cwb_json, "w") as fh:
        json.dump(post_cwb_paths, fh, indent=2)

    post_spei_json = None
    if make_spei:
        post_spei_json = out_dir / "paths" / "post_qm_corrected_spei_paths.json"
        post_spei_json.parent.mkdir(parents=True, exist_ok=True)
        with open(post_spei_json, "w") as fh:
            json.dump(post_spei_paths, fh, indent=2)

    return post_cwb_json, post_spei_json


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
    training_target = str(cfg.get("training_target", "residual")).lower()
    is_event_target = training_target in {"event", "event_p90"}
    event_cfg = cfg.get("event_model", {}) or {}
    event_percentile = float(event_cfg.get("percentile", 90.0))
    event_probability_threshold = float(event_cfg.get("probability_threshold", 0.5))
    target_scope = f"{var_name}/{training_target}"
    feature_var = cfg["var_name"]
    feature_reference_path = Path(cfg["reference_data"])
    workflow = cfg["workflow"]
    model_type = str(cfg.get("ml_arguments", {}).get("model_type", "rf")).lower()
    mixed_group_by = cfg.get("ml_arguments", {}).get("mixed_group_by", "grid_id")
    split_years_cfg = cfg.get("ml_arguments", {}).get("split_years", {}) or {}
    val_years = split_years_cfg.get("val", [])
    eval_years = split_years_cfg.get("test", cfg.get("leave_out_years", []))
    correction_target = cfg.get("ml_arguments", {}).get("correction_target", "member")
    knn_feature_columns = cfg.get("ml_arguments", {}).get("knn_feature_columns")
    quantile_group_predictor = cfg.get("ml_arguments", {}).get(
        "quantile_group_predictor"
    )
    quantile_group_n = cfg.get("ml_arguments", {}).get("quantile_group_n", 10)
    spread_reconstruction = cfg.get("ml_arguments", {}).get(
        "spread_reconstruction", "preserve_qm"
    )
    run_rf_evaluate = bool(workflow.get("run_rf_evaluate", False))
    evaluate_cwb_flag = bool(workflow.get("evaluate_cwb", False))
    evaluate_intensity_flag = bool(workflow.get("evaluate_intensity", False))
    evaluate_spei_flag = bool(workflow.get("evaluate_spei", False))
    rf_results_tag = str(cfg["rf_results_tag"])
    if is_event_target:
        rf_results_tag = f"{rf_results_tag}_eventp{int(event_percentile)}"
    train_leadmonth_specific = bool(
        cfg.get("ml_arguments", {}).get("train_leadmonth_specific", True)
    )

    post_qm_cfg = cfg.get("ml_post_qm", {}) or {}
    post_qm_enabled = bool(post_qm_cfg.get("enabled", False))
    post_qm_n_quantiles = int(post_qm_cfg.get("n_quantiles", 20))
    post_qm_kind = str(post_qm_cfg.get("kind", "+"))

    feature_builder = RFFeatureBuilder(
        feature_reference_path,
        feature_var,
        cfg.get("additional_reference_features", []),
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
                / feature_var
                / "paths"
                / f"lm{leadmonth}"
                / "qm_residuals_paths.json"
            )
            qm_hindcasts_json = (
                Path(cfg["output_dir"])
                / feature_var
                / "paths"
                / f"lm{leadmonth}"
                / "qm_hindcast_paths.json"
            )
            X_eval, y_eval, groups_eval, knn_eval = collect_eval_set(
                residuals_json,
                qm_hindcasts_json,
                feature_var,
                cfg["predictor_vars"],
                eval_years,
                feature_builder,
                group_by=mixed_group_by,
                correction_target=correction_target,
                knn_feature_columns=knn_feature_columns,
                quantile_group_predictor=quantile_group_predictor,
                quantile_group_n=quantile_group_n,
            )
            feature_count = int(X_eval.shape[1])
            rf_results_tag_lm = f"{rf_results_tag}_nf{feature_count}"

        eval_plot_root = (
            Path(cfg["plot_dir"])
            / var_name
            / training_target
            / "rf_eval"
            / rf_results_tag_lm
            / leadmonth_label
        )

        if run_rf_evaluate:
            model_suffix = f"lm{leadmonth}" if train_leadmonth_specific else "alllm"
            model_path = (
                Path(cfg["model_dir"])
                / target_scope
                / rf_results_tag_lm
                / f"{cfg['model_name']}_{model_suffix}.joblib"
            )
            model = load(model_path)
            metrics["n_eval_samples"] = int(len(y_eval))
            print(
                f"Evaluating RF predictions for leadmonth={leadmonth} on held-out years {eval_years} with model from {model_path}"
            )

            # Print mixed-model diagnostics for MERF and KNN mixed variants.
            if isinstance(
                model, (MixedEffectsTreeRegressor, LocalKNNRandomEffectsRegressor)
            ):
                diag_groups = (
                    knn_eval
                    if isinstance(model, LocalKNNRandomEffectsRegressor)
                    else groups_eval
                )
                diagnostics = analyze_merf_contributions(
                    model,
                    X_eval.values,
                    diag_groups,
                    y_eval.values,
                )

                diag_plot_path = (
                    eval_plot_root / f"{cfg['model_name']}_merf_eval_diagnostics.png"
                )

                save_merf_diagnostics_plot(diagnostics, diag_plot_path)

            raw_importances = getattr(model, "feature_importances_", None)
            if raw_importances is None and hasattr(model, "base_model"):
                raw_importances = getattr(
                    model.base_model, "feature_importances_", None
                )

            plot_feature_importance_table(
                feature_names=X_eval.columns.tolist(),
                importances=np.asarray(raw_importances, dtype=float),
                out_dir=eval_plot_root,
                model_label=type(model).__name__,
                top_n=30,
                file_prefix=f"{cfg['model_name']}_{leadmonth_label}_feature_importance",
            )

        month_data_dir = (
            out_dir
            / target_scope
            / "data"
            / "rf_eval"
            / rf_results_tag_lm
            / leadmonth_label
        )
        month_data_dir.mkdir(parents=True, exist_ok=True)

        spei_paths_json = (
            out_dir
            / target_scope
            / "paths"
            / "rf_eval"
            / rf_results_tag_lm
            / leadmonth_label
            / "corrected_spei_paths.json"
        )
        residual_paths_json = (
            out_dir
            / target_scope
            / "paths"
            / "rf_eval"
            / rf_results_tag_lm
            / leadmonth_label
            / "corrected_residuals_paths.json"
        )
        corrected_cwb_paths_json = (
            out_dir
            / target_scope
            / "paths"
            / "rf_eval"
            / rf_results_tag_lm
            / leadmonth_label
            / "corrected_cwb_paths.json"
        )
        event_probability_paths_json = (
            out_dir
            / target_scope
            / "paths"
            / "rf_eval"
            / rf_results_tag_lm
            / leadmonth_label
            / "event_probability_paths.json"
        )

        if run_rf_evaluate:
            if is_event_target:
                save_predicted_event_probabilities(
                    model,
                    qm_hindcasts_json,
                    feature_var,
                    eval_years,
                    out_dir=month_data_dir,
                    event_probability_paths_json=event_probability_paths_json,
                    feature_builder=feature_builder,
                    leadmonth=leadmonth,
                    model_type=model_type,
                    mixed_group_by=mixed_group_by,
                    knn_feature_columns=knn_feature_columns,
                    quantile_group_predictor=quantile_group_predictor,
                    quantile_group_n=quantile_group_n,
                    correction_target=correction_target,
                )
            else:
                save_predicted_residuals(
                    model,
                    qm_hindcasts_json,
                    cfg["var_name"],
                    cfg["predictor_vars"],
                    eval_years,
                    out_dir=month_data_dir,
                    spei_paths_json=spei_paths_json,
                    residual_paths_json=residual_paths_json,
                    corrected_cwb_paths_json=corrected_cwb_paths_json,
                    feature_builder=feature_builder,
                    leadmonth=leadmonth,
                    model_type=model_type,
                    mixed_group_by=mixed_group_by,
                    knn_feature_columns=knn_feature_columns,
                    quantile_group_predictor=quantile_group_predictor,
                    quantile_group_n=quantile_group_n,
                    save_spei=cfg["workflow"]["evaluate_spei"],
                    correction_target=correction_target,
                    spread_reconstruction=spread_reconstruction,
                    baseline_hindcasts_json=Path(cfg["hindcasts_json"]),
                )

                if post_qm_enabled:

                    val_data_dir = month_data_dir / "post_qm_val_calibration"
                    val_spei_paths_json = (
                        out_dir
                        / target_scope
                        / "paths"
                        / "rf_eval"
                        / rf_results_tag_lm
                        / leadmonth_label
                        / "val_corrected_spei_paths.json"
                    )
                    val_residual_paths_json = (
                        out_dir
                        / target_scope
                        / "paths"
                        / "rf_eval"
                        / rf_results_tag_lm
                        / leadmonth_label
                        / "val_corrected_residuals_paths.json"
                    )
                    val_corrected_cwb_paths_json = (
                        out_dir
                        / target_scope
                        / "paths"
                        / "rf_eval"
                        / rf_results_tag_lm
                        / leadmonth_label
                        / "val_corrected_cwb_paths.json"
                    )

                    save_predicted_residuals(
                        model,
                        qm_hindcasts_json,
                        cfg["var_name"],
                        cfg["predictor_vars"],
                        val_years,
                        out_dir=val_data_dir,
                        spei_paths_json=val_spei_paths_json,
                        residual_paths_json=val_residual_paths_json,
                        corrected_cwb_paths_json=val_corrected_cwb_paths_json,
                        feature_builder=feature_builder,
                        leadmonth=leadmonth,
                        model_type=model_type,
                        mixed_group_by=mixed_group_by,
                        knn_feature_columns=knn_feature_columns,
                        quantile_group_predictor=quantile_group_predictor,
                        quantile_group_n=quantile_group_n,
                        save_spei=False,
                        correction_target=correction_target,
                        spread_reconstruction=spread_reconstruction,
                        baseline_hindcasts_json=Path(cfg["hindcasts_json"]),
                    )

                    post_cwb_json, post_spei_json = apply_post_qm_from_val_years(
                        corrected_cwb_test_json=Path(corrected_cwb_paths_json),
                        corrected_cwb_val_json=Path(val_corrected_cwb_paths_json),
                        reference_data=Path(cfg["reference_data"]),
                        var_name=cfg["var_name"],
                        out_dir=month_data_dir,
                        n_quantiles=post_qm_n_quantiles,
                        kind=post_qm_kind,
                        make_spei=bool(cfg["workflow"]["evaluate_spei"]),
                    )
                    corrected_cwb_paths_json = post_cwb_json
                    if post_spei_json is not None:
                        spei_paths_json = post_spei_json

        metrics_path = None

        if evaluate_cwb_flag:
            plot_dir = str(eval_plot_root / "cwb")
            evaluate_cwb(
                corrected_cwb_json=(
                    None if is_event_target else Path(corrected_cwb_paths_json)
                ),
                ml_event_probability_json=(
                    Path(event_probability_paths_json) if is_event_target else None
                ),
                hindcasts_json=Path(cfg["hindcasts_json"]),
                reference_data=Path(cfg["reference_data"]),
                out_dir=month_data_dir,
                cwb_var=cfg["var_name"],
                eval_years=eval_years,
                std_multiplier=float(cfg["spei_std_multiplier"]),
                event_percentile=event_percentile,
                probability_threshold=event_probability_threshold,
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
                ml_residuals_json=(
                    None if is_event_target else Path(residual_paths_json)
                ),
                land_mask_path=cfg.get("land_mask_path"),
            )

        if evaluate_intensity_flag:
            plot_dir = str(eval_plot_root / "intensity")
            evaluate_intensity(
                corrected_intensity_json=Path(corrected_cwb_paths_json),
                hindcasts_json=Path(cfg["hindcasts_json"]),
                reference_data=Path(cfg["reference_data"]),
                out_dir=month_data_dir,
                intensity_var=cfg["var_name"],
                eval_years=eval_years,
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

            plot_dir = str(eval_plot_root / "spei")
            spei_metrics = evaluate_spei(
                corrected_spei_paths=corrected_spei_paths,
                hindcasts_spei_json=Path(cfg["hindcasts_spei_json"]),
                reference_spei_json=Path(cfg["reference_spei_json"]),
                out_dir=month_data_dir,
                eval_years=eval_years,
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
