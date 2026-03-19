import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import xarray as xr
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from IPython import embed

from .model_rf import RandomForestBiasCorrector
from .features import RFFeatureBuilder
from ..config_loader import get_qm_rf_global_config


def collect_train_sets(
    residuals_json: Path,
    hindcasts_json: Path,
    var_name: str,
    predictor_vars: List[str],
    eval_years: List[int],
    feature_builder: RFFeatureBuilder,
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

    X_train, y_train = collect_train_sets(
        Path(cfg["residuals_json"]),
        Path(cfg["hindcasts_json"]),
        cfg["var_name"],
        cfg["predictor_vars"],
        cfg["leave_out_years"],
        RFFeatureBuilder(Path(cfg["reference_data"]), cfg["var_name"]),
    )

    rf = RandomForestBiasCorrector()
    print(cfg["n_estimators"], X_train.shape, y_train.shape)
    model = rf.train_with_eta(
        X=X_train,
        y=y_train,
        n_estimators=cfg["n_estimators"],
        chunk_size=cfg["chunk_size"],
    )

    model_path = Path(cfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)

    print(f"Training complete. Model saved to: {model_path}")


if __name__ == "__main__":
    train()
