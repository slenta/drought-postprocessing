import json
import argparse
from pathlib import Path
from typing import List

import yaml
import numpy as np
import pandas as pd
import xarray as xr
from joblib import dump
from sklearn.ensemble import RandomForestRegressor

from .rf_net import RandomForestBiasCorrector


def collect_train_sets(
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

    if len(residuals) != len(hindcasts):
        raise RuntimeError(
            "Residuals and hindcasts JSON must contain same number of entries (matched by index)"
        )

    X_parts = []
    y_parts = []

    RF = RandomForestBiasCorrector

    for resid_path, hindcast_path in zip(residuals, hindcasts):
        resid_path = str(resid_path)
        hindcast_path = str(hindcast_path)

        res_da = RF.load_residual_da(resid_path, var_name)
        pred_ds = xr.open_dataset(hindcast_path)

        aligned = xr.align(res_da, *[pred_ds[v] for v in predictor_vars], join="exact")
        res_aligned = aligned[0]
        preds_aligned = aligned[1:]

        res_s = RF._stack_by_samples(res_aligned)
        pred_s_list = [RF._stack_by_samples(p) for p in preds_aligned]

        years = None
        if "time" in res_s.coords:
            try:
                years = pd.to_datetime(res_s.coords["time"].values).year
            except Exception:
                years = None

        X_full = pd.DataFrame(
            {name: p.values for name, p in zip(predictor_vars, pred_s_list)}
        )
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

        if train_mask.any():
            X_parts.append(X_full.loc[train_mask].reset_index(drop=True))
            y_parts.append(y_full.loc[train_mask].reset_index(drop=True))

        pred_ds.close()

    if len(X_parts) == 0:
        raise RuntimeError("No training samples collected (check eval years / data).")

    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    return X, y


def main(config_path: Path):
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    residuals_json = Path(cfg["residuals_json"])
    hindcasts_json = Path(cfg["hindcasts_json"])

    out_dir = Path(cfg.get("output_dir", "./rf_out"))
    out_dir.mkdir(parents=True, exist_ok=True)

    var_name = cfg.get("var_name", "tas")
    predictor_vars = cfg.get("predictor_vars", [var_name])
    eval_years = cfg.get("leave_out_years", [])

    X_train, y_train = collect_train_sets(
        residuals_json, hindcasts_json, var_name, predictor_vars, eval_years
    )

    rf = RandomForestBiasCorrector()
    model = rf.train(
        X_train,
        y_train,
        n_estimators=cfg.get("n_estimators", 100),
        random_state=cfg.get("random_state", 0),
    )

    model_path = out_dir / "rf_model.joblib"
    dump(model, model_path)

    print(f"Training complete. Model saved to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RF bias-corrector on QM residuals."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="train_config.yaml",
        help="Path to training config YAML",
    )
    args = parser.parse_args()
    main(Path(args.config))
