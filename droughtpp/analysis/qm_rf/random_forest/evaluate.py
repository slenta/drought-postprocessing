import json
import argparse
from pathlib import Path
from typing import List

import yaml
import numpy as np
import pandas as pd
import xarray as xr
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score

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

        if years is not None and len(eval_years) > 0:
            eval_mask_year = np.isin(years, eval_years)
        else:
            eval_mask_year = np.zeros_like(finite_mask, dtype=bool)

        eval_mask = finite_mask & (eval_mask_year)

        if eval_mask.any():
            X_parts.append(X_full.loc[eval_mask].reset_index(drop=True))
            y_parts.append(y_full.loc[eval_mask].reset_index(drop=True))

        pred_ds.close()

    if len(X_parts) == 0:
        return pd.DataFrame(columns=predictor_vars), pd.Series(dtype=float)

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

    model_path = Path(cfg.get("model_path", str(out_dir / "rf_model.joblib")))
    var_name = cfg.get("var_name", "tas")
    predictor_vars = cfg.get("predictor_vars", [var_name])
    eval_years = cfg.get("leave_out_years", [])

    X_eval, y_eval = collect_eval_set(
        residuals_json, hindcasts_json, var_name, predictor_vars, eval_years
    )

    if len(X_eval) == 0:
        print("No evaluation samples collected.")
        return

    model = load(model_path)

    preds = model.predict(X_eval.values)
    rmse = float(np.sqrt(mean_squared_error(y_eval.values, preds)))
    r2 = float(r2_score(y_eval.values, preds))

    metrics = {"rmse": rmse, "r2": r2, "n_eval_samples": int(len(y_eval))}

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
    main(Path(args.config))
