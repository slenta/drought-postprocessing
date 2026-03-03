# ...existing code...
import json
import argparse
from pathlib import Path
from typing import List

import yaml
import numpy as np
import pandas as pd
import xarray as xr
from joblib import dump
from sklearn.metrics import mean_squared_error, r2_score

from .rf_net import RandomForestBiasCorrector


def _residual_path_from_hindcast(
    hindcast_path: str, residual_suffix: str = "_qm_residual"
) -> Path:
    p = Path(hindcast_path)
    stem = p.stem
    parent = p.parent
    suffix = p.suffix or ".nc"
    return parent / f"{stem}{residual_suffix}{suffix}"


def collect_train_eval_sets(
    json_path: Path, var_name: str, predictor_vars: List[str], eval_years: List[int]
):
    with open(json_path, "r") as fh:
        hindcasts = json.load(fh)

    X_train_parts = []
    y_train_parts = []
    X_eval_parts = []
    y_eval_parts = []

    RF = RandomForestBiasCorrector

    for hindcast in hindcasts:
        hindcast = str(hindcast)
        residual_path = _residual_path_from_hindcast(hindcast)

        # load
        res_da = RF.load_residual_da(str(residual_path), var_name)
        pred_ds = xr.open_dataset(hindcast)

        # align and stack
        aligned = xr.align(res_da, *[pred_ds[v] for v in predictor_vars], join="exact")
        res_aligned = aligned[0]
        preds_aligned = aligned[1:]

        res_s = RF._stack_by_samples(res_aligned)
        pred_s_list = [RF._stack_by_samples(p) for p in preds_aligned]

        years = pd.to_datetime(res_s.coords["time"].values).year

        # build full X,y arrays
        X_full = pd.DataFrame(
            {name: p.values for name, p in zip(predictor_vars, pred_s_list)}
        )
        y_full = pd.Series(res_s.values, name="residual")

        finite_mask = np.isfinite(y_full.values) & np.all(
            np.isfinite(X_full.values), axis=1
        )

        # create train/eval masks
        if len(eval_years) > 0:
            eval_mask_year = np.isin(years, eval_years)
        else:
            eval_mask_year = np.zeros_like(finite_mask, dtype=bool)

        train_mask = finite_mask & (~eval_mask_year)
        eval_mask = finite_mask & (eval_mask_year)

        # train on train_mask, eval on eval_mask
        X_train_parts.append(X_full.loc[train_mask].reset_index(drop=True))
        y_train_parts.append(y_full.loc[train_mask].reset_index(drop=True))

        X_eval_parts.append(X_full.loc[eval_mask].reset_index(drop=True))
        y_eval_parts.append(y_full.loc[eval_mask].reset_index(drop=True))

    if len(X_train_parts) == 0:
        raise RuntimeError("No training samples collected (check eval years / data).")

    X_train = pd.concat(X_train_parts, ignore_index=True)
    y_train = pd.concat(y_train_parts, ignore_index=True)

    X_eval = pd.concat(X_eval_parts, ignore_index=True)
    y_eval = pd.concat(y_eval_parts, ignore_index=True)
    X_eval = pd.DataFrame(columns=predictor_vars)
    y_eval = pd.Series(dtype=float)

    return X_train, y_train, X_eval, y_eval


def main(config_path: Path):
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    json_path = Path(cfg["residuals_json"])
    out_dir = Path(cfg.get("output_dir", "./rf_out"))
    out_dir.mkdir(parents=True, exist_ok=True)

    var_name = cfg.get("var_name", "tas")
    predictor_vars = cfg.get("predictor_vars", [var_name])
    eval_years = cfg.get("leave_out_years", [])

    X_train, y_train, X_eval, y_eval = collect_train_eval_sets(
        json_path, var_name, predictor_vars, eval_years
    )

    rf = RandomForestBiasCorrector()
    model = rf.train(
        X_train,
        y_train,
        n_estimators=cfg.get("n_estimators", 100),
        random_state=cfg.get("random_state", 0),
    )

    # save model
    model_path = out_dir / "rf_model.joblib"
    dump(model, model_path)

    # evaluation
    metrics = {}

    preds = model.predict(X_eval.values)
    rmse = float(np.sqrt(mean_squared_error(y_eval.values, preds)))
    r2 = float(r2_score(y_eval.values, preds))
    metrics = {"rmse": rmse, "r2": r2, "n_eval_samples": int(len(y_eval))}

    # save metrics
    metrics_path = out_dir / "evaluation_metrics.yaml"
    with open(metrics_path, "w") as fh:
        yaml.safe_dump(metrics, fh)

    print(f"Model saved to: {model_path}")
    print(f"Evaluation metrics saved to: {metrics_path}")
    print(f"Evaluation metrics: {metrics}")


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
