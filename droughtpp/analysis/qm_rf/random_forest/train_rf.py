import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from .model_rf import RandomForestBiasCorrector
from .features import RFFeatureBuilder
from ..config_loader import get_qm_rf_global_config


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


def collect_train_sets(
    residuals_json: Path,
    qm_hindcasts_json: Path,
    var_name: str,
    predictor_vars: List[str],
    eval_years: List[int],
    feature_builder: RFFeatureBuilder,
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

    RF = RandomForestBiasCorrector

    for resid_path, qm_hind_path in zip(residuals, qm_hindcasts):
        resid_path = str(resid_path)
        qm_hind_path = str(qm_hind_path)

        res_da = xr.open_dataset(resid_path)[var_name]
        pred_da = xr.open_dataset(qm_hind_path)[var_name]

        res_s = RF._stack_by_samples(res_da)
        X_full, years = feature_builder.build(pred_da)
        y_full = pd.Series(res_s.values, name="residual")

        finite_mask = np.isfinite(y_full.values) & np.all(
            np.isfinite(X_full.values), axis=1
        )

        # keep only training samples (exclude eval_years)
        if years is not None and len(eval_years) > 0:
            eval_mask_year = np.isin(years, eval_years)
        else:
            eval_mask_year = np.zeros_like(finite_mask, dtype=bool)

        train_mask = finite_mask & (~eval_mask_year)

        X_parts.append(X_full.loc[train_mask].reset_index(drop=True))
        y_parts.append(y_full.loc[train_mask].reset_index(drop=True))

    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    return X, y


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
    )
    leadmonths = cfg["leadmonth"]

    for leadmonth in leadmonths:
        residuals_json = (
            Path(cfg["output_dir"])
            / "paths"
            / f"lm{leadmonth}"
            / "qm_residuals_paths.json"
        )
        qm_hindcasts_json = (
            Path(cfg["output_dir"])
            / "paths"
            / f"lm{leadmonth}"
            / "qm_hindcast_paths.json"
        )
        X_train, y_train = collect_train_sets(
            residuals_json,
            qm_hindcasts_json,
            cfg["var_name"],
            cfg["predictor_vars"],
            cfg["leave_out_years"],
            feature_builder,
        )

        if len(y_train) == 0:
            print(f"No training samples found for leadmonth={leadmonth}; skipping.")
            continue

        rf = RandomForestBiasCorrector()
        print(cfg["ml_arguments"]["n_estimators"], X_train.shape, y_train.shape)

        val_fraction = float(cfg["ml_arguments"].get("val_fraction", 0.2))
        if 0.0 < val_fraction < 1.0 and len(y_train) > 10:
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_train,
                y_train,
                test_size=val_fraction,
                random_state=cfg["ml_arguments"].get("random_state", 0),
                shuffle=True,
            )
        else:
            X_fit, y_fit = X_train, y_train
            X_val, y_val = None, None

        model = rf.train_with_eta(
            X=X_fit,
            y=y_fit,
            X_val=X_val,
            y_val=y_val,
            model_type=cfg["ml_arguments"].get("model_type", "rf"),
            n_estimators=cfg["ml_arguments"]["n_estimators"],
            chunk_size=cfg["ml_arguments"]["chunk_size"],
            random_state=cfg["ml_arguments"].get("random_state", 0),
            xgb_learning_rate=cfg["ml_arguments"].get("xgb_learning_rate", 0.1),
            xgb_max_depth=cfg["ml_arguments"].get("xgb_max_depth", 6),
            xgb_subsample=cfg["ml_arguments"].get("xgb_subsample", 0.8),
            xgb_colsample_bytree=cfg["ml_arguments"].get("xgb_colsample_bytree", 0.8),
            xgb_min_child_weight=cfg["ml_arguments"].get("xgb_min_child_weight", 1),
            xgb_eval_metric=cfg["ml_arguments"].get("xgb_eval_metric", "rmse"),
        )

        curve_path = (
            Path(cfg["plot_dir"])
            / "rf_train_curves"
            / cfg["rf_results_tag"]
            / f"lm{leadmonth}"
            / f"{cfg['model_name']}_lm{leadmonth}_train_val_curve.png"
        )
        plot_training_curve(rf.training_history, curve_path)

        model_path = (
            Path(cfg["model_dir"])
            / cfg["rf_results_tag"]
            / f"{cfg['model_name']}_lm{leadmonth}.joblib"
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        dump(model, model_path)

        print(
            f"Training complete for leadmonth={leadmonth}. Model saved to: {model_path}"
        )


if __name__ == "__main__":
    train()
