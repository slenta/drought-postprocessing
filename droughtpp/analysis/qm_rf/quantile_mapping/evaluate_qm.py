from pathlib import Path
import json

import numpy as np
import xarray as xr
from tqdm import tqdm

from ..utils.evaluation import compute_qm_distributions
from ..utils.visualization import (
    plot_qm_distributions,
    plot_example_time_means,
    plot_mae_skill_metrics,
    plot_bss_skill_metrics,
    plot_custom_timeseries,
)
from ..config_loader import get_qm_rf_global_config
from ..utils.evaluation import (
    brier_skill_score_between_ensembles,
    mae_per_member_grid,
    rmse_per_member_grid,
    mean_bias_per_member_grid,
)


def run_qm_evaluation(
    config_path=None,
    config_overrides=None,
):
    """
    High-level helper: compute distributions, plot, and compare QM vs. original hindcasts.
    Loads all data once, aligns to common time, then passes pre-aligned datasets to evaluation functions.
    """
    default_cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = get_qm_rf_global_config(
        config_path=config_path,
        overrides=config_overrides,
        default_config=default_cfg_path,
    )
    quantile_tag = f"nq{int(cfg['qm_arguments']['n_quantiles'])}"

    # Load reference data once
    ref_ds = xr.open_dataset(cfg["reference_data"])
    ref_da = ref_ds[cfg["var_name"]]
    var_name = cfg["var_name"]

    # Load all hindcast files
    with open(cfg["hindcasts_json"], "r") as fh:
        hind_files = json.load(fh)

    hind_das = []
    for fp in tqdm(hind_files, desc="Loading hindcasts"):
        ds = xr.open_dataset(fp)
        hind_das.append(ds[cfg["var_name"]])
        ds.close()

    n_members = len(hind_das)
    hind_subsets_by_member = [[] for _ in range(n_members)]
    qm_subsets_by_member = [[] for _ in range(n_members)]
    ref_subsets = []

    for leadmonth in cfg["leadmonth"]:
        qm_json_path = (
            Path(cfg["output_dir"])
            / "paths"
            / var_name
            / quantile_tag
            / f"lm{leadmonth}"
            / "qm_hindcast_paths.json"
        )

        # Load QM files for this leadmonth
        with open(qm_json_path, "r") as fh:
            qm_files = json.load(fh)

        qm_das = []
        for fp in qm_files:
            ds = xr.open_dataset(fp)
            qm_das.append(ds[cfg["var_name"]])
            ds.close()

        # Compute common time across ref, hind, and qm for this leadmonth
        common_time = ref_da["time"].values
        for da in hind_das:
            common_time = np.intersect1d(common_time, da["time"].values)
        for da in qm_das:
            common_time = np.intersect1d(common_time, da["time"].values)

        # Subset all to common time for this leadmonth
        ref_subset = ref_da.sel(time=common_time)
        hind_subsets = [da.sel(time=common_time) for da in hind_das]
        qm_subsets = [da.sel(time=common_time) for da in qm_das]

        # Plot for this specific leadmonth
        leadmonth_plot_dir = (
            Path(cfg["plot_dir"]) / "qm" / var_name / quantile_tag / f"lm{leadmonth}"
        )
        leadmonth_plot_dir.mkdir(parents=True, exist_ok=True)

        dists = compute_qm_distributions(
            hind_subsets=hind_subsets,
            qm_subsets=qm_subsets,
            ref_subset=ref_subset,
        )
        plot_qm_distributions(
            dists,
            cfg["var_name"],
            out_dir=str(leadmonth_plot_dir),
            n_quantiles=cfg["qm_arguments"]["n_quantiles"],
        )

        compute_qm_skill_metrics(
            hind_subsets=hind_subsets,
            qm_subsets=qm_subsets,
            ref_subset=ref_subset,
            var_name=cfg["var_name"],
            plot_dir=str(leadmonth_plot_dir),
            land_mask_path=cfg.get("land_mask_path"),
        )

        # Append for global all-leadmonths evaluation
        ref_subsets.append(ref_subset)
        for member_idx in range(n_members):
            hind_subsets_by_member[member_idx].append(hind_subsets[member_idx])
            qm_subsets_by_member[member_idx].append(qm_subsets[member_idx])

    # Concatenate all leadmonth subsets and run one combined evaluation
    qm_plot_dir = (
        Path(cfg["plot_dir"]) / "qm" / var_name / quantile_tag / "all_leadmonths"
    )
    qm_plot_dir.mkdir(parents=True, exist_ok=True)

    ref_subset = xr.concat(ref_subsets, dim="time").sortby("time")
    hind_subsets = [
        xr.concat(member_subsets, dim="time").sortby("time")
        for member_subsets in hind_subsets_by_member
    ]
    qm_subsets = [
        xr.concat(member_subsets, dim="time").sortby("time")
        for member_subsets in qm_subsets_by_member
    ]

    dists = compute_qm_distributions(
        hind_subsets=hind_subsets,
        qm_subsets=qm_subsets,
        ref_subset=ref_subset,
    )
    plot_qm_distributions(
        dists,
        cfg["var_name"],
        out_dir=str(qm_plot_dir),
        n_quantiles=cfg["qm_arguments"]["n_quantiles"],
    )

    compute_qm_skill_metrics(
        hind_subsets=hind_subsets,
        qm_subsets=qm_subsets,
        ref_subset=ref_subset,
        var_name=cfg["var_name"],
        plot_dir=str(qm_plot_dir),
        land_mask_path=cfg.get("land_mask_path"),
    )

    ref_ds.close()


def compute_qm_skill_metrics(
    hind_subsets,
    qm_subsets,
    ref_subset,
    var_name="tas",
    plot_dir="qm_plots",
    land_mask_path=None,
):
    """
    Compute skill metrics (BSS, MAE, RMSE, mean bias) comparing
    quantile-mapped hindcasts to original hindcasts.

    Args:
        hind_subsets: list of pre-aligned, pre-subset xarray DataArrays (hindcasts)
        qm_subsets: list of pre-aligned, pre-subset xarray DataArrays (QM)
        ref_subset: pre-aligned, pre-subset xarray DataArray (reference)
        var_name: variable name
        plot_dir: output directory for plots
    """

    def _normalize_time_to_date(da):
        return da.assign_coords(time=da["time"].values.astype("datetime64[D]"))

    ref_subset = _normalize_time_to_date(ref_subset)

    # Stack ensemble members from the list of DataArrays
    original_ensemble = []
    qm_ensemble = []

    for hind_da, qm_da in tqdm(
        list(zip(hind_subsets, qm_subsets)),
        desc="Preparing hindcasts for skill metrics",
    ):
        hind_da = _normalize_time_to_date(hind_da)
        qm_da = _normalize_time_to_date(qm_da)

        original_ensemble.append(hind_da)
        qm_ensemble.append(qm_da)

    reference = ref_subset.values.squeeze()  # (time, lat, lon)

    # Stack to (time, ensemble, lat, lon)
    original_ensemble = np.squeeze(
        np.stack([da.values for da in original_ensemble], axis=1)
    )
    qm_ensemble = np.squeeze(np.stack([da.values for da in qm_ensemble], axis=1))

    time_coord = ref_subset["time"].values
    member_coord = np.arange(original_ensemble.shape[1])
    lat_coord = np.arange(original_ensemble.shape[2])
    lon_coord = np.arange(original_ensemble.shape[3])

    original_da = xr.DataArray(
        original_ensemble,
        dims=("time", "member", "lat", "lon"),
        coords={
            "time": time_coord,
            "member": member_coord,
            "lat": lat_coord,
            "lon": lon_coord,
        },
    )
    qm_da = xr.DataArray(
        qm_ensemble,
        dims=("time", "member", "lat", "lon"),
        coords={
            "time": time_coord,
            "member": member_coord,
            "lat": lat_coord,
            "lon": lon_coord,
        },
    )
    reference_da = xr.DataArray(
        reference,
        dims=("time", "lat", "lon"),
        coords={"time": time_coord, "lat": lat_coord, "lon": lon_coord},
    )

    original_mean_da = original_da.mean(dim="member")
    qm_mean_da = qm_da.mean(dim="member")

    plot_custom_timeseries(
        series_specs=[
            {
                "data": original_da,
                "label": "Original",
                "color": "C0",
                "kind": "ensemble",
            },
            {
                "data": qm_da,
                "label": "QM",
                "color": "C2",
                "kind": "ensemble",
            },
            {
                "data": reference_da,
                "label": "Reference spatial mean",
                "color": "k",
                "kind": "line",
            },
        ],
        out_dir=str(plot_dir),
        variable_name=var_name,
        file_prefix="timeseries_qm_original_reference",
        title="QM, Original, and Reference Spatial Mean",
        land_mask_path=land_mask_path,
    )

    plot_example_time_means(
        baseline_ensemble=original_mean_da,
        qm_ensemble=qm_mean_da,
        corrected_ensemble=None,
        reference=reference_da,
        out_dir=str(plot_dir),
        n_members_display=1,
        n_timesteps=3,
        variable_name=var_name,
        land_mask_path=land_mask_path,
    )

    # Compute MAE
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
        qm_ensemble,
        original_ensemble,
        reference,
        std_multiplier=1.0,
        extreme_type="upper",
    )

    # Compute BSS for lower tail extremes
    bss_lower, _, _ = brier_skill_score_between_ensembles(
        qm_ensemble,
        original_ensemble,
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
        land_mask_path=land_mask_path,
    )

    # Plot RMSE skill metrics
    plot_mae_skill_metrics(
        rmse_orig,
        rmse_qm,
        out_dir=plot_dir,
        n_members_display=3,
        metric_name="RMSE",
        file_prefix="rmse",
        land_mask_path=land_mask_path,
    )

    # Plot mean bias skill metrics
    plot_mae_skill_metrics(
        mean_bias_orig,
        mean_bias_qm,
        out_dir=plot_dir,
        n_members_display=3,
        metric_name="Mean Bias",
        file_prefix="mean_bias",
        land_mask_path=land_mask_path,
    )

    # Plot BSS skill metrics
    plot_bss_skill_metrics(
        bss_upper,
        bss_lower,
        out_dir=plot_dir,
        upper_threshold_label="mean + 1σ",
        lower_threshold_label="mean - 1σ",
        title_prefix="QM vs Original",
        land_mask_path=land_mask_path,
    )


if __name__ == "__main__":
    run_qm_evaluation()
