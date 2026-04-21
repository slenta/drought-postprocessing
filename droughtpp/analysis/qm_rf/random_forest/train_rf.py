import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from joblib import dump
from IPython import embed
from sklearn.model_selection import train_test_split

from .model_rf import (
    RandomForestBiasCorrector,
    MixedEffectsTreeRegressor,
    analyze_merf_contributions,
    save_merf_diagnostics_plot,
)
from .knn_random_effects import LocalKNNRandomEffectsRegressor
from .features import RFFeatureBuilder
from ..config_loader import get_qm_rf_global_config
from droughtpp.analysis.qm_rf.utils.evaluation import (
    build_knn_neighbor_features,
    build_mixed_group_labels,
    compute_threshold_grid_from_reference,
)
from droughtpp.analysis.qm_rf.utils.visualization import plot_feature_importance_table


def plot_training_curve(training_history: dict, out_path: Path):
    if not training_history:
        return

    x_vals = training_history.get("x", [])
    train_vals = training_history.get("train", [])
    val_vals = training_history.get("val", None)
    metric = str(training_history.get("metric", "metric")).upper()
    model_type = str(training_history.get("model_type", "model"))

    if len(x_vals) == 0 or len(train_vals) == 0:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, train_vals, label="train", linewidth=2)
    if val_vals is not None and len(val_vals) == len(x_vals):
        plt.plot(x_vals, val_vals, label="validation", linewidth=2)

    x_label = "n_trees" if model_type == "rf" else "boosting_round"
    plt.xlabel(x_label)
    plt.ylabel(metric)
    plt.title(f"{model_type.upper()} train/validation {metric}")
    plt.grid(alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _append_train_block(
    res_da: xr.DataArray,
    pred_da: xr.DataArray,
    land_mask: np.ndarray,
    feature_builder: RFFeatureBuilder,
    group_by: str,
    eval_years: List[int],
    knn_feature_columns: List[str] | None,
    X_parts: list,
    y_parts: list,
    group_parts: list,
    knn_parts: list,
    year_parts: list,
    quantile_group_predictor: str | None,
    quantile_group_n: int | None,
):
    res_da = res_da.where(land_mask)
    pred_da = pred_da.where(land_mask)

    res_s = RandomForestBiasCorrector._stack_by_samples(res_da)
    X_full, years = feature_builder.build(pred_da)
    y_full = pd.Series(res_s.values, name="residual")
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
    knn_full = build_knn_neighbor_features(
        res_s,
        X_full,
        group_by,
        knn_feature_columns=knn_feature_columns,
    )

    finite_mask = np.isfinite(y_full.values) & np.all(
        np.isfinite(X_full.values), axis=1
    )
    eval_mask_year = np.isin(years, eval_years)
    train_mask = finite_mask & (~eval_mask_year)

    X_parts.append(X_full.loc[train_mask].reset_index(drop=True))
    y_parts.append(y_full.loc[train_mask].reset_index(drop=True))
    group_parts.append(pd.Series(groups_full[train_mask]).reset_index(drop=True))
    year_parts.append(np.asarray(years[train_mask], dtype=int))
    if knn_full is not None:
        knn_parts.append(np.asarray(knn_full[train_mask], dtype=float))


def _build_land_mask(feature_builder: RFFeatureBuilder) -> np.ndarray:
    land_mask = (
        np.isfinite(feature_builder.reference_data_array.values[0])
        if feature_builder.reference_data_array.ndim == 3
        else np.isfinite(feature_builder.reference_data_array.values)
    )

    for feature in feature_builder.additional_reference_features:
        arr = (
            feature.get("monthly_values")
            if feature.get("mode", "lagged") == "monthly_climatology"
            else feature.get("values")
        )
        arr_mask = np.isfinite(arr[0]) if arr.ndim == 3 else np.isfinite(arr)
        land_mask = land_mask & arr_mask

    return land_mask


def _sample_reference_values(
    feature_builder: RFFeatureBuilder,
    years: np.ndarray,
    months: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    lat_indices = feature_builder._lookup_coordinate_indices(
        latitudes,
        feature_builder.reference_lat_index,
    )
    lon_indices = feature_builder._lookup_coordinate_indices(
        longitudes,
        feature_builder.reference_lon_index,
    )

    obs_values = np.full(len(years), np.nan, dtype=float)
    valid_spatial = (lat_indices >= 0) & (lon_indices >= 0)
    valid_indices = np.where(valid_spatial)[0]

    for idx in valid_indices:
        time_index = feature_builder.reference_time_index.get(
            (int(years[idx]), int(months[idx]))
        )
        if time_index is None:
            continue
        obs_values[idx] = feature_builder.reference_values[
            time_index,
            lat_indices[idx],
            lon_indices[idx],
        ]

    return obs_values


def _append_event_train_block(
    pred_da: xr.DataArray,
    land_mask: np.ndarray,
    feature_builder: RFFeatureBuilder,
    group_by: str,
    eval_years: List[int],
    knn_feature_columns: List[str] | None,
    event_percentile: float,
    ref_threshold_grid: np.ndarray,
    X_parts: list,
    y_parts: list,
    group_parts: list,
    knn_parts: list,
    year_parts: list,
    quantile_group_predictor: str | None,
    quantile_group_n: int | None,
):
    pred_da = pred_da.where(land_mask)
    pred_s = RandomForestBiasCorrector._stack_by_samples(pred_da)

    X_full, years = feature_builder.build(pred_da)
    timestamps = pd.to_datetime(pred_s.coords["time"].values)
    months = np.asarray(timestamps.month, dtype=int)
    latitudes = np.asarray(pred_s.coords["latitude"].values)
    longitudes = np.asarray(pred_s.coords["longitude"].values)

    pred_years = pd.to_datetime(pred_da["time"].values).year
    pred_train_mask = ~np.isin(pred_years, np.asarray(eval_years, dtype=int))
    if np.any(pred_train_mask):
        pred_train = pred_da.isel(time=pred_train_mask).values
    else:
        pred_train = pred_da.values
    pred_threshold_grid = np.nanpercentile(pred_train, event_percentile, axis=0)

    pred_lat_values = np.asarray(pred_da["latitude"].values)
    pred_lon_values = np.asarray(pred_da["longitude"].values)
    pred_lat_index = {
        round(float(value), 6): index for index, value in enumerate(pred_lat_values)
    }
    pred_lon_index = {
        round(float(value), 6): index for index, value in enumerate(pred_lon_values)
    }
    pred_lat_indices = feature_builder._lookup_coordinate_indices(
        latitudes, pred_lat_index
    )
    pred_lon_indices = feature_builder._lookup_coordinate_indices(
        longitudes, pred_lon_index
    )
    valid_pred_spatial = (pred_lat_indices >= 0) & (pred_lon_indices >= 0)

    qm_values = np.asarray(pred_s.values, dtype=float)
    qm_event_p90 = np.full(len(qm_values), np.nan, dtype=float)
    valid_pred_indices = np.where(valid_pred_spatial)[0]
    qm_threshold_flat = pred_threshold_grid[
        pred_lat_indices[valid_pred_indices], pred_lon_indices[valid_pred_indices]
    ]
    qm_event_p90[valid_pred_indices] = (
        qm_values[valid_pred_indices] > qm_threshold_flat
    ).astype(float)
    X_full["qm_event_p90"] = qm_event_p90

    obs_values = _sample_reference_values(
        feature_builder,
        years,
        months,
        latitudes,
        longitudes,
    )

    ref_lat_indices = feature_builder._lookup_coordinate_indices(
        latitudes,
        feature_builder.reference_lat_index,
    )
    ref_lon_indices = feature_builder._lookup_coordinate_indices(
        longitudes,
        feature_builder.reference_lon_index,
    )
    valid_ref_spatial = (ref_lat_indices >= 0) & (ref_lon_indices >= 0)

    ref_event_p90 = np.full(len(obs_values), np.nan, dtype=float)
    valid_ref_indices = np.where(valid_ref_spatial)[0]
    ref_threshold_flat = ref_threshold_grid[
        ref_lat_indices[valid_ref_indices],
        ref_lon_indices[valid_ref_indices],
    ]
    ref_event_p90[valid_ref_indices] = (
        obs_values[valid_ref_indices] > ref_threshold_flat
    ).astype(float)

    y_full = pd.Series(ref_event_p90, name="obs_event_p90")
    if str(group_by).lower() in {"knn_spatial", "knn_feature", "knn_proximity"}:
        groups_full = np.repeat("knn", len(y_full)).astype(str)
    else:
        groups_full = build_mixed_group_labels(
            pred_s,
            years,
            group_by,
            X_full=X_full,
            quantile_group_predictor=quantile_group_predictor,
            quantile_group_n=quantile_group_n,
        )
    knn_full = build_knn_neighbor_features(
        pred_s,
        X_full,
        group_by,
        knn_feature_columns=knn_feature_columns,
    )

    finite_mask = np.isfinite(y_full.values) & np.all(
        np.isfinite(X_full.values), axis=1
    )
    eval_mask_year = np.isin(years, np.asarray(eval_years, dtype=int))
    train_mask = finite_mask & (~eval_mask_year)

    X_parts.append(X_full.loc[train_mask].reset_index(drop=True))
    y_parts.append(y_full.loc[train_mask].reset_index(drop=True))
    group_parts.append(pd.Series(groups_full[train_mask]).reset_index(drop=True))
    year_parts.append(np.asarray(years[train_mask], dtype=int))
    if knn_full is not None:
        knn_parts.append(np.asarray(knn_full[train_mask], dtype=float))


def _split_train_and_val(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: np.ndarray,
    years_train: np.ndarray,
    ml_args: dict,
    is_knn_mixed: bool,
    knn_features_train: np.ndarray | None,
):
    split_cfg = ml_args.get("split_years", {}) or {}
    val_years = split_cfg.get("val", []) if isinstance(split_cfg, dict) else []
    train_years = split_cfg.get("train", None) if isinstance(split_cfg, dict) else None

    years_arr = np.asarray(years_train, dtype=int)
    has_year_split = len(val_years) > 0 or train_years is not None

    if has_year_split:
        val_mask = np.isin(years_arr, np.asarray(val_years, dtype=int))
        if train_years is not None:
            train_mask = np.isin(years_arr, np.asarray(train_years, dtype=int))
            train_mask = train_mask & (~val_mask)
        else:
            train_mask = ~val_mask

        if not np.any(train_mask):
            raise ValueError("No training samples remain after year-based split.")

        X_fit = X_train.loc[train_mask].reset_index(drop=True)
        y_fit = y_train.loc[train_mask].reset_index(drop=True)
        groups_fit = np.asarray(groups_train)[train_mask]

        if np.any(val_mask):
            X_val = X_train.loc[val_mask].reset_index(drop=True)
            y_val = y_train.loc[val_mask].reset_index(drop=True)
            groups_val = np.asarray(groups_train)[val_mask]
        else:
            X_val, y_val, groups_val = None, None, None

        if is_knn_mixed:
            if knn_features_train is None:
                raise ValueError(
                    "knn_features_train is required for KNN mixed training"
                )
            knn_fit = np.asarray(knn_features_train)[train_mask]
            knn_val = (
                np.asarray(knn_features_train)[val_mask] if np.any(val_mask) else None
            )
        else:
            knn_fit, knn_val = None, None

        return X_fit, X_val, y_fit, y_val, groups_fit, groups_val, knn_fit, knn_val

    val_fraction = float(ml_args.get("val_fraction", 0.2))
    if 0.0 < val_fraction < 1.0 and len(y_train) > 10:
        if is_knn_mixed:
            if knn_features_train is None:
                raise ValueError(
                    "knn_features_train is required for KNN mixed training"
                )
            (
                X_fit,
                X_val,
                y_fit,
                y_val,
                groups_fit,
                groups_val,
                knn_fit,
                knn_val,
            ) = train_test_split(
                X_train,
                y_train,
                groups_train,
                knn_features_train,
                test_size=val_fraction,
                random_state=ml_args.get("random_state", 0),
                shuffle=True,
            )
        else:
            X_fit, X_val, y_fit, y_val, groups_fit, groups_val = train_test_split(
                X_train,
                y_train,
                groups_train,
                test_size=val_fraction,
                random_state=ml_args.get("random_state", 0),
                shuffle=True,
            )
            knn_fit, knn_val = None, None
    else:
        X_fit, y_fit = X_train, y_train
        X_val, y_val = None, None
        groups_fit, groups_val = groups_train, None
        knn_fit, knn_val = knn_features_train, None

    return X_fit, X_val, y_fit, y_val, groups_fit, groups_val, knn_fit, knn_val


def collect_train_sets(
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
    """
    Collect training sets from pre-aligned residuals and QM hindcasts.
    Both are already filtered to the same leadmonth, so no further filtering needed.
    """
    with open(residuals_json, "r") as fh:
        residuals = json.load(fh)
    with open(qm_hindcasts_json, "r") as fh:
        qm_hindcasts = json.load(fh)

    X_parts = []
    y_parts = []
    group_parts = []
    knn_parts = []
    year_parts = []

    land_mask = _build_land_mask(feature_builder)

    correction_mode = str(correction_target).lower()
    if correction_mode in {"ensemble_mean", "mean", "ens_mean"}:
        residual_members = []
        predictor_members = []
        for resid_path, qm_hind_path in zip(residuals, qm_hindcasts):
            with xr.open_dataset(str(resid_path)) as ds_res, xr.open_dataset(
                str(qm_hind_path)
            ) as ds_qm:
                residual_members.append(ds_res[var_name].load())
                predictor_members.append(ds_qm[var_name].load())

        res_da = xr.concat(residual_members, dim="member").mean("member", skipna=True)
        pred_da = xr.concat(predictor_members, dim="member").mean("member", skipna=True)
        _append_train_block(
            res_da,
            pred_da,
            land_mask,
            feature_builder,
            group_by,
            eval_years,
            knn_feature_columns,
            X_parts,
            y_parts,
            group_parts,
            knn_parts,
            year_parts,
            quantile_group_predictor,
            quantile_group_n,
        )
    else:
        for resid_path, qm_hind_path in zip(residuals, qm_hindcasts):
            resid_path = str(resid_path)
            qm_hind_path = str(qm_hind_path)

            with xr.open_dataset(resid_path) as ds_res, xr.open_dataset(
                qm_hind_path
            ) as ds_qm:
                res_da = ds_res[var_name].load()
                pred_da = ds_qm[var_name].load()

            _append_train_block(
                res_da,
                pred_da,
                land_mask,
                feature_builder,
                group_by,
                eval_years,
                knn_feature_columns,
                X_parts,
                y_parts,
                group_parts,
                knn_parts,
                year_parts,
                quantile_group_predictor,
                quantile_group_n,
            )

    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    groups = pd.concat(group_parts, ignore_index=True).astype(str).values
    knn_features = np.vstack(knn_parts) if len(knn_parts) > 0 else None
    years = (
        np.concatenate(year_parts) if len(year_parts) > 0 else np.array([], dtype=int)
    )
    return X, y, groups, knn_features, years


def collect_event_train_sets(
    qm_hindcasts_json: Path,
    predictor_var: str,
    eval_years: List[int],
    feature_builder: RFFeatureBuilder,
    event_percentile: float,
    group_by: str = "grid_id",
    correction_target: str = "member",
    knn_feature_columns: List[str] | None = None,
    quantile_group_predictor: str | None = None,
    quantile_group_n: int | None = None,
):
    with open(qm_hindcasts_json, "r") as fh:
        qm_hindcasts = json.load(fh)

    X_parts = []
    y_parts = []
    group_parts = []
    knn_parts = []
    year_parts = []

    land_mask = _build_land_mask(feature_builder)
    ref_threshold_grid = compute_threshold_grid_from_reference(
        feature_builder.reference_data_array,
        eval_years,
        event_percentile,
    )

    correction_mode = str(correction_target).lower()
    if correction_mode in {"ensemble_mean", "mean", "ens_mean"}:
        predictor_members = []
        for qm_hind_path in qm_hindcasts:
            with xr.open_dataset(str(qm_hind_path)) as ds_qm:
                predictor_members.append(ds_qm[predictor_var].load())

        pred_da = xr.concat(predictor_members, dim="member").mean("member", skipna=True)
        _append_event_train_block(
            pred_da,
            land_mask,
            feature_builder,
            group_by,
            eval_years,
            knn_feature_columns,
            event_percentile,
            ref_threshold_grid,
            X_parts,
            y_parts,
            group_parts,
            knn_parts,
            year_parts,
            quantile_group_predictor,
            quantile_group_n,
        )
    else:
        for qm_hind_path in qm_hindcasts:
            with xr.open_dataset(str(qm_hind_path)) as ds_qm:
                pred_da = ds_qm[predictor_var].load()

            _append_event_train_block(
                pred_da,
                land_mask,
                feature_builder,
                group_by,
                eval_years,
                knn_feature_columns,
                event_percentile,
                ref_threshold_grid,
                X_parts,
                y_parts,
                group_parts,
                knn_parts,
                year_parts,
                quantile_group_predictor,
                quantile_group_n,
            )

    if len(X_parts) == 0:
        return (
            pd.DataFrame(),
            pd.Series(dtype=float),
            np.array([], dtype=str),
            None,
            np.array([], dtype=int),
        )

    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    groups = pd.concat(group_parts, ignore_index=True).astype(str).values
    knn_features = np.vstack(knn_parts) if len(knn_parts) > 0 else None
    years = (
        np.concatenate(year_parts) if len(year_parts) > 0 else np.array([], dtype=int)
    )
    return X, y, groups, knn_features, years


def get_leadmonth_qm_paths(
    cfg: dict,
    leadmonth: int,
    var_name: str | None = None,
) -> tuple[Path, Path]:
    if var_name is None:
        var_name = cfg["var_name"]
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
    return residuals_json, qm_hindcasts_json


def collect_train_sets_all_leadmonths(
    cfg: dict,
    leadmonths: List[int],
    feature_builder: RFFeatureBuilder,
    test_years: List[int],
    group_by: str = "grid_id",
    correction_target: str = "member",
    knn_feature_columns: List[str] | None = None,
    quantile_group_predictor: str | None = None,
    quantile_group_n: int | None = None,
):
    X_parts = []
    y_parts = []
    group_parts = []
    knn_parts = []
    year_parts = []

    for leadmonth in leadmonths:
        residuals_json, qm_hindcasts_json = get_leadmonth_qm_paths(cfg, leadmonth)
        X_train, y_train, groups_train, knn_train, years_train = collect_train_sets(
            residuals_json,
            qm_hindcasts_json,
            cfg["var_name"],
            cfg["predictor_vars"],
            test_years,
            feature_builder,
            group_by=group_by,
            correction_target=correction_target,
            knn_feature_columns=knn_feature_columns,
            quantile_group_predictor=quantile_group_predictor,
            quantile_group_n=quantile_group_n,
        )
        if len(y_train) > 0:
            X_parts.append(X_train)
            y_parts.append(y_train)
            group_parts.append(pd.Series(groups_train))
            year_parts.append(np.asarray(years_train, dtype=int))
            if knn_train is not None:
                knn_parts.append(np.asarray(knn_train, dtype=float))

    if not y_parts:
        return (
            pd.DataFrame(),
            pd.Series(dtype=float),
            np.array([], dtype=str),
            None,
            np.array([], dtype=int),
        )

    knn_features = np.vstack(knn_parts) if len(knn_parts) > 0 else None
    return (
        pd.concat(X_parts, ignore_index=True),
        pd.concat(y_parts, ignore_index=True),
        pd.concat(group_parts, ignore_index=True).astype(str).values,
        knn_features,
        np.concatenate(year_parts) if len(year_parts) > 0 else np.array([], dtype=int),
    )


def collect_event_train_sets_all_leadmonths(
    cfg: dict,
    leadmonths: List[int],
    feature_builder: RFFeatureBuilder,
    test_years: List[int],
    event_percentile: float,
    predictor_var_name: str,
    group_by: str = "grid_id",
    correction_target: str = "member",
    knn_feature_columns: List[str] | None = None,
    quantile_group_predictor: str | None = None,
    quantile_group_n: int | None = None,
):
    X_parts = []
    y_parts = []
    group_parts = []
    knn_parts = []
    year_parts = []

    for leadmonth in leadmonths:
        _, qm_hindcasts_json = get_leadmonth_qm_paths(
            cfg,
            leadmonth,
            var_name=predictor_var_name,
        )
        X_train, y_train, groups_train, knn_train, years_train = (
            collect_event_train_sets(
                qm_hindcasts_json=qm_hindcasts_json,
                predictor_var=predictor_var_name,
                eval_years=test_years,
                feature_builder=feature_builder,
                event_percentile=event_percentile,
                group_by=group_by,
                correction_target=correction_target,
                knn_feature_columns=knn_feature_columns,
                quantile_group_predictor=quantile_group_predictor,
                quantile_group_n=quantile_group_n,
            )
        )
        if len(y_train) > 0:
            X_parts.append(X_train)
            y_parts.append(y_train)
            group_parts.append(pd.Series(groups_train))
            year_parts.append(np.asarray(years_train, dtype=int))
            if knn_train is not None:
                knn_parts.append(np.asarray(knn_train, dtype=float))

    if not y_parts:
        return (
            pd.DataFrame(),
            pd.Series(dtype=float),
            np.array([], dtype=str),
            None,
            np.array([], dtype=int),
        )

    knn_features = np.vstack(knn_parts) if len(knn_parts) > 0 else None
    return (
        pd.concat(X_parts, ignore_index=True),
        pd.concat(y_parts, ignore_index=True),
        pd.concat(group_parts, ignore_index=True).astype(str).values,
        knn_features,
        np.concatenate(year_parts) if len(year_parts) > 0 else np.array([], dtype=int),
    )


def dump_training_inputs_csv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    out_dir: Path,
    var_name: str,
    ml_tag: str,
    file_name: str,
):
    target_name = str(y_train.name) if y_train.name else "target"
    csv_df = X_train.copy()
    csv_df[target_name] = y_train.values

    csv_dir = out_dir / var_name / "metrics" / "input_csv" / ml_tag
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(csv_dir / file_name, index=False)


def train_and_save_model(
    cfg: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: np.ndarray,
    years_train: np.ndarray,
    var_name: str,
    rf_results_tag_lm: str,
    model_suffix: str,
    curve_dir_suffix: str,
    curve_file_suffix: str,
    knn_features_train: np.ndarray | None = None,
):
    rf = RandomForestBiasCorrector()
    print(cfg["ml_arguments"]["n_estimators"], X_train.shape, y_train.shape)

    model_type = str(cfg["ml_arguments"].get("model_type", "rf")).lower()
    is_mixed = model_type in {"mixed_rf", "mixed"}
    mixed_group_by = str(cfg["ml_arguments"].get("mixed_group_by", "grid_id")).lower()
    is_knn_mixed = is_mixed and mixed_group_by in {"knn_spatial", "knn_feature", "knn_proximity"}

    (
        X_fit,
        X_val,
        y_fit,
        y_val,
        groups_fit,
        groups_val,
        knn_fit,
        knn_val,
    ) = _split_train_and_val(
        X_train,
        y_train,
        groups_train,
        years_train,
        cfg["ml_arguments"],
        is_knn_mixed,
        knn_features_train,
    )

    model = rf.train_with_eta(
        X=X_fit,
        y=y_fit,
        X_val=X_val,
        y_val=y_val,
        groups=groups_fit if is_mixed else None,
        groups_val=groups_val if is_mixed else None,
        model_type=model_type,
        mixed_base_model=cfg["ml_arguments"].get("mixed_base_model", "rf"),
        mixed_group_by=cfg["ml_arguments"].get("mixed_group_by", "grid_id"),
        n_estimators=cfg["ml_arguments"]["n_estimators"],
        chunk_size=cfg["ml_arguments"]["chunk_size"],
        random_state=cfg["ml_arguments"].get("random_state", 0),
        xgb_learning_rate=cfg["ml_arguments"].get("xgb_learning_rate", 0.1),
        xgb_max_depth=cfg["ml_arguments"].get("xgb_max_depth", 6),
        xgb_subsample=cfg["ml_arguments"].get("xgb_subsample", 0.8),
        xgb_colsample_bytree=cfg["ml_arguments"].get("xgb_colsample_bytree", 0.8),
        xgb_min_child_weight=cfg["ml_arguments"].get("xgb_min_child_weight", 1),
        xgb_eval_metric=cfg["ml_arguments"].get("xgb_eval_metric", "rmse"),
        rf_max_depth=cfg["ml_arguments"].get("rf_max_depth", None),
        rf_min_samples_leaf=cfg["ml_arguments"].get("rf_min_samples_leaf", 1),
        rf_max_features=cfg["ml_arguments"].get("rf_max_features", "sqrt"),
        knn_neighbor_features=knn_fit if is_knn_mixed else None,
        knn_neighbor_features_val=knn_val if is_knn_mixed else None,
        knn_k=cfg["ml_arguments"].get("knn_k", 15),
        knn_metric=cfg["ml_arguments"].get("knn_metric", "euclidean"),
        knn_eps=cfg["ml_arguments"].get("knn_eps", 1e-8),
        knn_weighting=cfg["ml_arguments"].get("knn_weighting", "inverse_distance"),
        knn_gaussian_sigma=cfg["ml_arguments"].get("knn_gaussian_sigma", None),
        re_shrinkage_grid=cfg["ml_arguments"].get("re_shrinkage_grid", [1.0]),
        tune_knn_k_values=cfg["ml_arguments"].get("tune_knn_k_values", None),
        tune_knn_eps_values=cfg["ml_arguments"].get("tune_knn_eps_values", None),
        merf_n_iter=cfg["ml_arguments"].get("merf_n_iter", 1),
        merf_tol=cfg["ml_arguments"].get("merf_tol", 1e-6),
        xgb_early_stopping_rounds=cfg["ml_arguments"].get(
            "xgb_early_stopping_rounds", None
        ),
        rf_early_stopping_patience=cfg["ml_arguments"].get(
            "rf_early_stopping_patience", 0
        ),
        rf_early_stopping_min_delta=cfg["ml_arguments"].get(
            "rf_early_stopping_min_delta", 0.0
        ),
    )

    leadmonth_dir = (
        curve_dir_suffix if str(curve_dir_suffix).startswith("lm") else "all_leadmonths"
    )
    train_plot_dir = (
        Path(cfg["plot_dir"])
        / var_name
        / "rf_train"
        / rf_results_tag_lm
        / leadmonth_dir
    )

    # Print mixed-model diagnostics for MERF and KNN mixed variants.
    if is_mixed and isinstance(
        model, (MixedEffectsTreeRegressor, LocalKNNRandomEffectsRegressor)
    ):
        diag_groups = (
            knn_fit if isinstance(model, LocalKNNRandomEffectsRegressor) else groups_fit
        )
        diagnostics = analyze_merf_contributions(
            model,
            X_fit.values,
            diag_groups,
            y_fit.values,
        )

        # Save diagnostics plot
        diag_plot_path = train_plot_dir / f"{cfg['model_name']}_merf_diagnostics.png"
        save_merf_diagnostics_plot(diagnostics, diag_plot_path)

        # Save one diagnostics table per MERF iteration, if available.
        iteration_diagnostics = (
            rf.training_history.get("merf_iteration_diagnostics", [])
            if isinstance(rf.training_history, dict)
            else []
        )
        for iter_diag in iteration_diagnostics:
            iter_idx = int(iter_diag.get("iteration", 0))
            iter_plot_path = (
                train_plot_dir
                / f"{cfg['model_name']}_merf_diagnostics_iter{iter_idx:02d}.png"
            )
            save_merf_diagnostics_plot(iter_diag, iter_plot_path)

    curve_path = (
        train_plot_dir / f"{cfg['model_name']}_{curve_file_suffix}_train_val_curve.png"
    )

    plot_training_curve(rf.training_history, curve_path)

    raw_importances = getattr(model, "feature_importances_", None)
    if raw_importances is None and hasattr(model, "base_model"):
        raw_importances = getattr(model.base_model, "feature_importances_", None)

    plot_feature_importance_table(
        feature_names=X_train.columns.tolist(),
        importances=np.asarray(raw_importances, dtype=float),
        out_dir=train_plot_dir,
        model_label=type(model).__name__,
        top_n=30,
        file_prefix=f"{cfg['model_name']}_{curve_file_suffix}_feature_importance",
    )

    model_path = (
        Path(cfg["model_dir"])
        / var_name
        / rf_results_tag_lm
        / f"{cfg['model_name']}_{model_suffix}.joblib"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)
    return model_path


def train(config_path: Path | None = None, config_overrides=None):
    default_cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = get_qm_rf_global_config(
        config_path=config_path,
        overrides=config_overrides,
        default_config=default_cfg_path,
    )

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    training_target = str(cfg.get("training_target", "residual")).lower()
    is_event_target = training_target in {"event", "event_p90"}
    event_cfg = cfg.get("event_model", {}) or {}
    event_percentile = float(event_cfg.get("percentile", 90.0))
    training_var_name = cfg["var_name"]
    training_reference_path = Path(cfg["reference_data"])
    path_scope_name = f"{training_var_name}/{training_target}"

    feature_builder = RFFeatureBuilder(
        training_reference_path,
        training_var_name,
        cfg.get("additional_reference_features", []),
        cfg.get("feature_flags", {}),
    )
    leadmonths = cfg["leadmonth"]
    mixed_group_by = cfg.get("ml_arguments", {}).get("mixed_group_by", "grid_id")
    correction_target = cfg.get("ml_arguments", {}).get("correction_target", "member")
    knn_feature_columns = cfg.get("ml_arguments", {}).get("knn_feature_columns")
    quantile_group_predictor = cfg.get("ml_arguments", {}).get(
        "quantile_group_predictor"
    )
    quantile_group_n = cfg.get("ml_arguments", {}).get("quantile_group_n", 10)
    split_years_cfg = cfg.get("ml_arguments", {}).get("split_years", {}) or {}
    test_years = split_years_cfg.get("test", cfg.get("leave_out_years", []))
    train_leadmonth_specific = bool(
        cfg.get("ml_arguments", {}).get("train_leadmonth_specific", True)
    )
    rf_results_tag_base = str(cfg["rf_results_tag"])
    if is_event_target:
        rf_results_tag_base = f"{rf_results_tag_base}_eventp{int(event_percentile)}"

    if not train_leadmonth_specific:
        if is_event_target:
            X_train, y_train, groups_train, knn_features_train, years_train = (
                collect_event_train_sets_all_leadmonths(
                    cfg=cfg,
                    leadmonths=leadmonths,
                    feature_builder=feature_builder,
                    test_years=test_years,
                    event_percentile=event_percentile,
                    predictor_var_name=training_var_name,
                    group_by=mixed_group_by,
                    correction_target=correction_target,
                    knn_feature_columns=knn_feature_columns,
                    quantile_group_predictor=quantile_group_predictor,
                    quantile_group_n=quantile_group_n,
                )
            )
        else:
            X_train, y_train, groups_train, knn_features_train, years_train = (
                collect_train_sets_all_leadmonths(
                    cfg=cfg,
                    leadmonths=leadmonths,
                    feature_builder=feature_builder,
                    test_years=test_years,
                    group_by=mixed_group_by,
                    correction_target=correction_target,
                    knn_feature_columns=knn_feature_columns,
                    quantile_group_predictor=quantile_group_predictor,
                    quantile_group_n=quantile_group_n,
                )
            )

        if len(y_train) == 0:
            print("No training samples found across all leadmonths; skipping.")
            return

        feature_count = int(X_train.shape[1])
        rf_results_tag_lm = f"{rf_results_tag_base}_nf{feature_count}"
        dump_training_inputs_csv(
            X_train=X_train,
            y_train=y_train,
            out_dir=Path(cfg["output_dir"]),
            var_name=path_scope_name,
            ml_tag=rf_results_tag_lm,
            file_name="predictors_target_alllm.csv",
        )
        model_path = train_and_save_model(
            cfg=cfg,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            years_train=years_train,
            var_name=path_scope_name,
            rf_results_tag_lm=rf_results_tag_lm,
            model_suffix="alllm",
            curve_dir_suffix="all_leadmonths",
            curve_file_suffix="alllm",
            knn_features_train=knn_features_train,
        )

        print(f"Training complete for all leadmonths. Model saved to: {model_path}")
        return

    for leadmonth in leadmonths:
        if is_event_target:
            _, qm_hindcasts_json = get_leadmonth_qm_paths(
                cfg,
                leadmonth,
                var_name=training_var_name,
            )
            X_train, y_train, groups_train, knn_features_train, years_train = (
                collect_event_train_sets(
                    qm_hindcasts_json=qm_hindcasts_json,
                    predictor_var=training_var_name,
                    eval_years=test_years,
                    feature_builder=feature_builder,
                    event_percentile=event_percentile,
                    group_by=mixed_group_by,
                    correction_target=correction_target,
                    knn_feature_columns=knn_feature_columns,
                    quantile_group_predictor=quantile_group_predictor,
                    quantile_group_n=quantile_group_n,
                )
            )
        else:
            residuals_json, qm_hindcasts_json = get_leadmonth_qm_paths(
                cfg,
                leadmonth,
            )
            X_train, y_train, groups_train, knn_features_train, years_train = (
                collect_train_sets(
                    residuals_json,
                    qm_hindcasts_json,
                    cfg["var_name"],
                    cfg["predictor_vars"],
                    test_years,
                    feature_builder,
                    group_by=mixed_group_by,
                    correction_target=correction_target,
                    knn_feature_columns=knn_feature_columns,
                    quantile_group_predictor=quantile_group_predictor,
                    quantile_group_n=quantile_group_n,
                )
            )

        feature_count = int(X_train.shape[1])
        rf_results_tag_lm = f"{rf_results_tag_base}_nf{feature_count}"
        dump_training_inputs_csv(
            X_train=X_train,
            y_train=y_train,
            out_dir=Path(cfg["output_dir"]),
            var_name=path_scope_name,
            ml_tag=rf_results_tag_lm,
            file_name=f"predictors_target_lm{leadmonth}.csv",
        )
        model_path = train_and_save_model(
            cfg=cfg,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            years_train=years_train,
            var_name=path_scope_name,
            rf_results_tag_lm=rf_results_tag_lm,
            model_suffix=f"lm{leadmonth}",
            curve_dir_suffix=f"lm{leadmonth}",
            curve_file_suffix=f"lm{leadmonth}",
            knn_features_train=knn_features_train,
        )

        print(
            f"Training complete for leadmonth={leadmonth}. Model saved to: {model_path}"
        )


if __name__ == "__main__":
    train()
