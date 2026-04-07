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


def _build_group_labels(stacked_da: xr.DataArray, years: np.ndarray, group_by: str):
    group_by = str(group_by).lower()

    if group_by == "year":
        return years.astype(str)

    if group_by == "calendar_month":
        months = pd.to_datetime(stacked_da.coords["time"].values).month
        return months.astype(str)

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


def _select_knn_feature_frame(
    X_full: pd.DataFrame,
    knn_feature_columns: List[str] | None,
) -> pd.DataFrame:
    if not knn_feature_columns:
        return X_full

    requested = [str(col) for col in knn_feature_columns]
    missing = [col for col in requested if col not in X_full.columns]
    if missing:
        available = ", ".join(X_full.columns.astype(str).tolist())
        raise ValueError(
            "ml_arguments.knn_feature_columns contains unknown feature(s): "
            f"{missing}. Available features: [{available}]"
        )

    return X_full.loc[:, requested]


def _build_knn_neighbor_features(
    stacked_da: xr.DataArray,
    X_full: pd.DataFrame,
    group_by: str,
    knn_feature_columns: List[str] | None = None,
) -> np.ndarray | None:
    mode = str(group_by).lower()
    if mode == "knn_spatial":
        if "latitude" not in stacked_da.coords or "longitude" not in stacked_da.coords:
            raise ValueError("knn_spatial requires latitude/longitude coordinates")
        lat = np.asarray(stacked_da.coords["latitude"].values, dtype=float)
        lon = np.asarray(stacked_da.coords["longitude"].values, dtype=float)
        return np.column_stack([lat, lon])
    if mode == "knn_feature":
        X_knn = _select_knn_feature_frame(X_full, knn_feature_columns)
        return np.asarray(X_knn.values, dtype=float)
    return None


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
):
    res_da = res_da.where(land_mask)
    pred_da = pred_da.where(land_mask)

    res_s = RandomForestBiasCorrector._stack_by_samples(res_da)
    X_full, years = feature_builder.build(pred_da)
    y_full = pd.Series(res_s.values, name="residual")
    if str(group_by).lower() in {"knn_spatial", "knn_feature"}:
        groups_full = np.repeat("knn", len(y_full)).astype(str)
    else:
        groups_full = _build_group_labels(res_s, years, group_by)
    knn_full = _build_knn_neighbor_features(
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
    if knn_full is not None:
        knn_parts.append(np.asarray(knn_full[train_mask], dtype=float))


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

    # Build static landmask from reference field (finite values)
    land_mask = (
        np.isfinite(feature_builder.reference_data_array.values[0])
        if feature_builder.reference_data_array.ndim == 3
        else np.isfinite(feature_builder.reference_data_array.values)
    )

    # Also mask all additional_reference_features
    for feature in feature_builder.additional_reference_features:
        arr = (
            feature.get("monthly_values")
            if feature.get("mode", "lagged") == "monthly_climatology"
            else feature.get("values")
        )
        # arr shape: (time, lat, lon) or (month, lat, lon)
        arr_mask = np.isfinite(arr[0]) if arr.ndim == 3 else np.isfinite(arr)
        # Update mask: only keep points that are land in all features
        land_mask = land_mask & arr_mask

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
            )

    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    groups = pd.concat(group_parts, ignore_index=True).astype(str).values
    knn_features = np.vstack(knn_parts) if len(knn_parts) > 0 else None
    return X, y, groups, knn_features


def get_leadmonth_qm_paths(cfg: dict, leadmonth: int) -> tuple[Path, Path]:
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
    group_by: str = "grid_id",
    correction_target: str = "member",
    knn_feature_columns: List[str] | None = None,
):
    X_parts = []
    y_parts = []
    group_parts = []
    knn_parts = []

    for leadmonth in leadmonths:
        residuals_json, qm_hindcasts_json = get_leadmonth_qm_paths(cfg, leadmonth)
        X_train, y_train, groups_train, knn_train = collect_train_sets(
            residuals_json,
            qm_hindcasts_json,
            cfg["var_name"],
            cfg["predictor_vars"],
            cfg["leave_out_years"],
            feature_builder,
            group_by=group_by,
            correction_target=correction_target,
            knn_feature_columns=knn_feature_columns,
        )
        if len(y_train) > 0:
            X_parts.append(X_train)
            y_parts.append(y_train)
            group_parts.append(pd.Series(groups_train))
            if knn_train is not None:
                knn_parts.append(np.asarray(knn_train, dtype=float))

    if not y_parts:
        return pd.DataFrame(), pd.Series(dtype=float), np.array([], dtype=str), None

    knn_features = np.vstack(knn_parts) if len(knn_parts) > 0 else None
    return (
        pd.concat(X_parts, ignore_index=True),
        pd.concat(y_parts, ignore_index=True),
        pd.concat(group_parts, ignore_index=True).astype(str).values,
        knn_features,
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
    is_knn_mixed = is_mixed and mixed_group_by in {"knn_spatial", "knn_feature"}

    val_fraction = float(cfg["ml_arguments"].get("val_fraction", 0.2))
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
                random_state=cfg["ml_arguments"].get("random_state", 0),
                shuffle=True,
            )
        else:
            X_fit, X_val, y_fit, y_val, groups_fit, groups_val = train_test_split(
                X_train,
                y_train,
                groups_train,
                test_size=val_fraction,
                random_state=cfg["ml_arguments"].get("random_state", 0),
                shuffle=True,
            )
            knn_fit, knn_val = None, None
    else:
        X_fit, y_fit = X_train, y_train
        X_val, y_val = None, None
        groups_fit, groups_val = groups_train, None
        knn_fit, knn_val = knn_features_train, None

    model = rf.train_with_eta(
        X=X_fit,
        y=y_fit,
        X_val=X_val,
        y_val=y_val,
        groups=groups_fit if is_mixed else None,
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
        knn_k=cfg["ml_arguments"].get("knn_k", 15),
        knn_metric=cfg["ml_arguments"].get("knn_metric", "euclidean"),
        knn_eps=cfg["ml_arguments"].get("knn_eps", 1e-8),
        knn_weighting=cfg["ml_arguments"].get("knn_weighting", "inverse_distance"),
        knn_gaussian_sigma=cfg["ml_arguments"].get("knn_gaussian_sigma", None),
        merf_n_iter=cfg["ml_arguments"].get("merf_n_iter", 1),
        merf_tol=cfg["ml_arguments"].get("merf_tol", 1e-6),
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
        leadmonth_dir = (
            curve_dir_suffix
            if str(curve_dir_suffix).startswith("lm")
            else "all_leadmonths"
        )
        diag_plot_path = (
            Path(cfg["plot_dir"])
            / var_name
            / "rf_train"
            / rf_results_tag_lm
            / leadmonth_dir
            / f"{cfg['model_name']}_merf_diagnostics.png"
        )
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
                Path(cfg["plot_dir"])
                / var_name
                / "rf_train"
                / rf_results_tag_lm
                / leadmonth_dir
                / f"{cfg['model_name']}_merf_diagnostics_iter{iter_idx:02d}.png"
            )
            save_merf_diagnostics_plot(iter_diag, iter_plot_path)

    curve_path = (
        Path(cfg["plot_dir"])
        / var_name
        / "rf_train"
        / rf_results_tag_lm
        / curve_dir_suffix
        / f"{cfg['model_name']}_{curve_file_suffix}_train_val_curve.png"
    )
    plot_training_curve(rf.training_history, curve_path)

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

    feature_builder = RFFeatureBuilder(
        Path(cfg["reference_data"]),
        cfg["var_name"],
        cfg.get("additional_reference_features", []),
        cfg.get("include_obs_prev_lags", True),
        cfg.get("feature_flags", {}),
    )
    leadmonths = cfg["leadmonth"]
    mixed_group_by = cfg.get("ml_arguments", {}).get("mixed_group_by", "grid_id")
    correction_target = cfg.get("ml_arguments", {}).get("correction_target", "member")
    knn_feature_columns = cfg.get("ml_arguments", {}).get("knn_feature_columns")
    train_leadmonth_specific = bool(
        cfg.get("ml_arguments", {}).get("train_leadmonth_specific", True)
    )

    if not train_leadmonth_specific:
        X_train, y_train, groups_train, knn_features_train = (
            collect_train_sets_all_leadmonths(
                cfg=cfg,
                leadmonths=leadmonths,
                feature_builder=feature_builder,
                group_by=mixed_group_by,
                correction_target=correction_target,
                knn_feature_columns=knn_feature_columns,
            )
        )

        if len(y_train) == 0:
            print("No training samples found across all leadmonths; skipping.")
            return

        feature_count = int(X_train.shape[1])
        rf_results_tag_lm = f"{cfg['rf_results_tag']}_nf{feature_count}"
        dump_training_inputs_csv(
            X_train=X_train,
            y_train=y_train,
            out_dir=Path(cfg["output_dir"]),
            var_name=cfg["var_name"],
            ml_tag=rf_results_tag_lm,
            file_name="predictors_target_alllm.csv",
        )
        model_path = train_and_save_model(
            cfg=cfg,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            var_name=cfg["var_name"],
            rf_results_tag_lm=rf_results_tag_lm,
            model_suffix="alllm",
            curve_dir_suffix="all_leadmonths",
            curve_file_suffix="alllm",
            knn_features_train=knn_features_train,
        )

        print(f"Training complete for all leadmonths. Model saved to: {model_path}")
        return

    for leadmonth in leadmonths:
        residuals_json, qm_hindcasts_json = get_leadmonth_qm_paths(cfg, leadmonth)
        X_train, y_train, groups_train, knn_features_train = collect_train_sets(
            residuals_json,
            qm_hindcasts_json,
            cfg["var_name"],
            cfg["predictor_vars"],
            cfg["leave_out_years"],
            feature_builder,
            group_by=mixed_group_by,
            correction_target=correction_target,
            knn_feature_columns=knn_feature_columns,
        )

        feature_count = int(X_train.shape[1])
        rf_results_tag_lm = f"{cfg['rf_results_tag']}_nf{feature_count}"
        dump_training_inputs_csv(
            X_train=X_train,
            y_train=y_train,
            out_dir=Path(cfg["output_dir"]),
            var_name=cfg["var_name"],
            ml_tag=rf_results_tag_lm,
            file_name=f"predictors_target_lm{leadmonth}.csv",
        )
        model_path = train_and_save_model(
            cfg=cfg,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            var_name=cfg["var_name"],
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
