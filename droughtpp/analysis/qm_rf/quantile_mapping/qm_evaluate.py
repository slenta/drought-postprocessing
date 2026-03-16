from pathlib import Path
import json

import numpy as np
import xarray as xr
from tqdm import tqdm

from ..evaluation import compute_qm_distributions
from ..visualization import (
    plot_qm_distributions,
    plot_mae_skill_metrics,
    plot_bss_skill_metrics,
)
from ..config_loader import load_qm_rf_config
from droughtpp.evaluation.evaluation import (
    brier_skill_score_between_ensembles,
    mae_per_member_grid,
    rmse_per_member_grid,
    mean_bias_per_member_grid,
)


def run_qm_evaluation(
    json_list_path,
    ref_path,
    var_name="tas",
    qm_out_dir=None,
    plot_dir="qm_plots",
    n_quantiles=250,
):
    """
    High-level helper: compute distributions, plot, and compare QM vs. original hindcasts.
    """
    dists = compute_qm_distributions(
        json_list_path, ref_path, var_name, qm_out_dir=qm_out_dir
    )
    plot_qm_distributions(dists, var_name, out_dir=plot_dir, n_quantiles=n_quantiles)

    # Compute BSS and MAE for QM vs. original hindcasts
    compute_qm_skill_metrics(
        json_list_path, ref_path, var_name, qm_out_dir=qm_out_dir, plot_dir=plot_dir
    )


def compute_qm_skill_metrics(
    json_list_path, ref_path, var_name="tas", qm_out_dir=None, plot_dir="qm_plots"
):
    """
    Compute skill metrics (BSS, MAE, RMSE, mean bias) comparing
    quantile-mapped hindcasts to original hindcasts.
    Generates visualization plots of the results.
    """
    with open(json_list_path, "r") as fh:
        files = json.load(fh)

    def _normalize_time_to_date(da):
        return da.assign_coords(time=da["time"].values.astype("datetime64[D]"))

    ds_obs = xr.open_dataset(ref_path)
    obs_da = _normalize_time_to_date(ds_obs[var_name])
    common_time = obs_da["time"].values

    # Load ensemble members for original and QM hindcasts
    original_ensemble = []
    qm_ensemble = []

    for fp in tqdm(files, desc="Loading hindcasts for skill metrics"):
        p = Path(fp)
        base = p.stem

        # Load original hindcast
        ds_orig = xr.open_dataset(fp)
        orig_da = _normalize_time_to_date(ds_orig[var_name])

        # Load QM hindcast
        qm_path = Path(qm_out_dir) / f"{base}_qm{p.suffix}"
        ds_qm = xr.open_dataset(qm_path)
        qm_da = _normalize_time_to_date(ds_qm[var_name])

        common_time = np.intersect1d(common_time, orig_da["time"].values)
        common_time = np.intersect1d(common_time, qm_da["time"].values)

        original_ensemble.append(orig_da)
        qm_ensemble.append(qm_da)

        ds_orig.close()
        ds_qm.close()

    obs_da = obs_da.sel(time=common_time)
    original_ensemble = [da.sel(time=common_time).values for da in original_ensemble]
    qm_ensemble = [da.sel(time=common_time).values for da in qm_ensemble]

    reference = obs_da.values.squeeze()  # (time, lat, lon)
    ds_obs.close()

    # Stack to (time, ensemble, lat, lon)
    # Assuming each file has shape (time, lat, lon)
    original_ensemble = np.squeeze(
        np.stack(original_ensemble, axis=1)
    )  # (time, ensemble, lat, lon)
    qm_ensemble = np.squeeze(
        np.stack(qm_ensemble, axis=1)
    )  # (time, ensemble, lat, lon)

    # Compute MAE
    print(original_ensemble.shape, qm_ensemble.shape, reference.shape)
    mae_orig = mae_per_member_grid(original_ensemble, reference)
    mae_qm = mae_per_member_grid(qm_ensemble, reference)

    # Compute RMSE
    rmse_orig = rmse_per_member_grid(original_ensemble, reference)
    rmse_qm = rmse_per_member_grid(qm_ensemble, reference)

    # Compute mean bias
    mean_bias_orig = mean_bias_per_member_grid(original_ensemble, reference)
    mean_bias_qm = mean_bias_per_member_grid(qm_ensemble, reference)

    # Compute BSS for upper tail extremes
    bss_upper, _, _ = brier_skill_score_between_ensembles(
        original_ensemble,
        qm_ensemble,
        reference,
        std_multiplier=1.0,
        extreme_type="upper",
    )

    # Compute BSS for lower tail extremes
    bss_lower, _, _ = brier_skill_score_between_ensembles(
        original_ensemble,
        qm_ensemble,
        reference,
        std_multiplier=-1.0,
        extreme_type="lower",
    )

    # Plot MAE skill metrics
    plot_mae_skill_metrics(
        mae_orig,
        mae_qm,
        out_dir=plot_dir,
        n_members_display=3,
        metric_name="MAE",
        file_prefix="mae",
    )

    # Plot RMSE skill metrics
    plot_mae_skill_metrics(
        rmse_orig,
        rmse_qm,
        out_dir=plot_dir,
        n_members_display=3,
        metric_name="RMSE",
        file_prefix="rmse",
    )

    # Plot mean bias skill metrics
    plot_mae_skill_metrics(
        mean_bias_orig,
        mean_bias_qm,
        out_dir=plot_dir,
        n_members_display=3,
        metric_name="Mean Bias",
        file_prefix="mean_bias",
    )

    # Plot BSS skill metrics
    plot_bss_skill_metrics(
        bss_upper,
        bss_lower,
        out_dir=plot_dir,
        upper_threshold_label="mean + 1σ",
        lower_threshold_label="mean - 1σ",
    )


if __name__ == "__main__":

    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = load_qm_rf_config(cfg_path)

    json_list = cfg["hindcasts_json"]
    reference = cfg.get("reference", cfg.get("reference_data"))
    var = cfg.get("var_name", "tas")
    qm_out = cfg.get("qm_output_dir")
    plots_out = cfg.get("plot_dir")

    run_qm_evaluation(
        json_list,
        reference,
        var_name=var,
        qm_out_dir=qm_out,
        plot_dir=plots_out,
    )
