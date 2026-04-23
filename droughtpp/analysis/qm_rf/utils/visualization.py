import os
import typing as _t
from pathlib import Path
from IPython import embed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def _load_land_mask(land_mask_path):

    with xr.open_dataset(land_mask_path) as ds:
        land_mask = ds.landmask.values.squeeze()[0, :, :]

    return land_mask


def plot_qm_distributions(
    distributions,
    var_name,
    out_dir="qm_plots",
    n_quantiles=200,
    figsize=(8, 6),
    bins=250,
    series_keys=("orig", "qm", "obs"),
    series_labels=("original", "quantile-mapped", "reference"),
    title_tag="QM vs obs",
    file_prefix="ensemble_mean_qm_distribution",
):
    """
    distributions: dict produced by compute_qm_distributions()
    Creates one PNG per entry comparing histograms of original, qm and obs.
    Returns list of written file paths.
    """
    out_paths = []
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    series_values = {k: [] for k in series_keys}
    for _, d in distributions.items():
        for series_key in series_keys:
            if series_key in d:
                series_values[series_key].append(d[series_key])

    missing = [k for k in series_keys if len(series_values[k]) == 0]
    if missing:
        raise ValueError(
            f"Missing distribution series for keys: {missing}. "
            f"Available keys: {list(next(iter(distributions.values())).keys()) if distributions else []}"
        )

    series_means = {key: np.mean(series_values[key], axis=0) for key in series_keys}
    all_means = np.concatenate([arr.ravel() for arr in series_means.values()])
    x_min = float(np.nanmin(all_means))
    x_max = float(np.nanmax(all_means))
    bins = np.linspace(x_min, x_max, bins)

    # for key, d in distributions.items():
    #     orig_vals = d["orig"]
    #     qm_vals = d["qm"]
    #     obs_vals = d["obs"]

    #     plt.figure(figsize=figsize)
    #     plt.hist(orig_vals, bins=bins, density=True, alpha=0.45, label="original")
    #     plt.hist(qm_vals, bins=bins, density=True, alpha=0.45, label="quantile-mapped")
    #     plt.hist(obs_vals, bins=bins, density=True, alpha=0.45, label="reference")
    #     plt.legend()
    #     plt.title(f"{key} — {var_name} distribution (QM vs obs)")
    #     plt.xlabel(var_name)
    #     plt.xlim(x_min, x_max)
    #     plt.ylabel("density")

    #     out_png = Path(out_dir) / f"{key}_qm_distribution_n{n_quantiles}.png"
    #     plt.savefig(str(out_png), bbox_inches="tight")
    #     out_paths.append(str(out_png))
    #     plt.close()

    plt.figure(figsize=figsize)
    for series_key, series_label in zip(series_keys, series_labels):
        plt.hist(
            series_means[series_key],
            bins=bins,
            density=True,
            alpha=0.45,
            label=series_label,
        )
    plt.legend()
    plt.title(f"Ensemble Mean — {var_name} distribution ({title_tag})")
    plt.xlabel(var_name)
    plt.xlim(x_min, x_max)
    plt.ylabel("density")

    out_png = Path(out_dir) / f"{file_prefix}_n{n_quantiles}.png"
    plt.savefig(str(out_png), bbox_inches="tight")
    out_paths.append(str(out_png))
    plt.close()

    return out_paths


def plot_mae_skill_metrics(
    mae_orig,
    mae_corrected,
    mae_qm=None,
    out_dir="qm_plots",
    n_members_display=3,
    metric_name="MAE",
    file_prefix=None,
    land_mask_path=None,
    original_label="Original",
    corrected_label="RF-corrected",
    qm_label="QM",
    reference_label="Reference",
    show_qm_before_corrected=False,
):
    """
    Plot skill metrics comparing original, corrected (RF), and optionally QM hindcasts.

    Args:
        mae_orig: np.ndarray shape (ensemble, lat, lon) - metric values for original hindcasts
        mae_corrected: np.ndarray shape (ensemble, lat, lon) - metric values for corrected hindcasts
        mae_qm: np.ndarray shape (ensemble, lat, lon) - metric values for QM hindcasts (optional)
        out_dir: output directory for plots
        n_members_display: number of ensemble members to display (default 3)
        metric_name: label shown in titles/colorbars (e.g., "MAE" or "RMSE")
        file_prefix: output filename prefix; defaults to lowercase metric_name
        land_mask_path: path to land mask file (optional)

    Returns:
        list of written file paths
    """
    land_mask_path = land_mask_path

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_paths = []
    if file_prefix is None:
        file_prefix = str(metric_name).lower().replace(" ", "_")

    n_members = mae_orig.shape[0]
    members_to_plot = min(n_members_display, n_members)

    land_mask = _load_land_mask(land_mask_path)
    mae_orig = mae_orig * land_mask
    mae_corrected = mae_corrected * land_mask
    if mae_qm is not None:
        mae_qm = mae_qm * land_mask

    # Compute ensemble means
    mae_orig_mean = np.nanmean(mae_orig, axis=0)
    mae_corrected_mean = np.nanmean(mae_corrected, axis=0)
    mae_qm_mean = np.nanmean(mae_qm, axis=0) if mae_qm is not None else None

    # Determine number of columns: 3 for orig/corrected/diff, or 4 if QM available
    n_cols = 4 if mae_qm is not None else 3
    n_rows = members_to_plot + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15 * n_cols / 3, 5 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    title_suffix = " + QM" if mae_qm is not None else ""
    fig.suptitle(
        f"{metric_name}: Original / RF-corrected / QM{title_suffix} / Difference",
        fontsize=12,
        fontweight="bold",
    )

    # Compute global vmin/vmax for viridis (absolute metrics)
    viridis_values = np.concatenate(
        [
            mae_orig.ravel(),
            mae_corrected.ravel(),
            mae_orig_mean.ravel(),
            mae_corrected_mean.ravel(),
        ]
    )
    if mae_qm is not None:
        viridis_values = np.concatenate(
            [viridis_values, mae_qm.ravel(), mae_qm_mean.ravel()]
        )

    finite_viridis = viridis_values[np.isfinite(viridis_values)]
    vmin_viridis = np.nanmin(finite_viridis)
    vmax_viridis = np.nanmax(finite_viridis)

    # Compute global vmin/vmax for differences (RdBu_r)
    diff_values = []
    for i in range(members_to_plot):
        mae_diff = mae_corrected[i] - mae_orig[i]
        diff_values.append(mae_diff.ravel())
    mae_diff_mean = mae_corrected_mean - mae_orig_mean
    diff_values.append(mae_diff_mean.ravel())

    diff_values = np.concatenate(diff_values)
    finite_diff = diff_values[np.isfinite(diff_values)]

    max_abs_diff = np.nanmax(np.abs(finite_diff))
    vmin_diff = -max_abs_diff
    vmax_diff = max_abs_diff

    # Plot first N members
    im_viridis = None
    im_diff = None

    for i in range(members_to_plot):
        row_label = f"Member {i}"

        # Original
        ax = axes[i, 0]
        im_viridis = ax.imshow(
            mae_orig[i],
            cmap="viridis",
            origin="lower",
            vmin=vmin_viridis,
            vmax=vmax_viridis,
        )
        ax.set_title(
            f"Original {row_label} (mean={np.nanmean(mae_orig[i]):.3f})",
            fontsize=9,
        )
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

        # Corrected
        ax = axes[i, 1]
        im_viridis = ax.imshow(
            mae_corrected[i],
            cmap="viridis",
            origin="lower",
            vmin=vmin_viridis,
            vmax=vmax_viridis,
        )
        ax.set_title(
            f"RF-corrected {row_label} (mean={np.nanmean(mae_corrected[i]):.3f})",
            fontsize=9,
        )
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

        # QM (if available)
        if mae_qm is not None:
            ax = axes[i, 2]
            im_viridis = ax.imshow(
                mae_qm[i],
                cmap="viridis",
                origin="lower",
                vmin=vmin_viridis,
                vmax=vmax_viridis,
            )
            ax.set_title(
                f"QM {row_label} (mean={np.nanmean(mae_qm[i]):.3f})",
                fontsize=9,
            )
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")

            # Difference RF vs orig
            ax = axes[i, 3]
            mae_diff = mae_corrected[i] - mae_orig[i]
            im_diff = ax.imshow(
                mae_diff, cmap="RdBu_r", vmin=vmin_diff, vmax=vmax_diff, origin="lower"
            )
            ax.set_title(
                f"Δ{metric_name} (RF-orig) {row_label} (mean={np.nanmean(mae_diff):.3f})",
                fontsize=9,
            )
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
        else:
            # Difference RF vs orig
            ax = axes[i, 2]
            mae_diff = mae_corrected[i] - mae_orig[i]
            im_diff = ax.imshow(
                mae_diff, cmap="RdBu_r", vmin=vmin_diff, vmax=vmax_diff, origin="lower"
            )
            ax.set_title(
                f"Δ{metric_name} (RF-orig) {row_label} (mean={np.nanmean(mae_diff):.3f})",
                fontsize=9,
            )
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")

    # Plot ensemble mean row
    mean_row = members_to_plot
    ax = axes[mean_row, 0]
    im_viridis = ax.imshow(
        mae_orig_mean,
        cmap="viridis",
        origin="lower",
        vmin=vmin_viridis,
        vmax=vmax_viridis,
    )
    ax.set_title(
        f"Original Ens. Mean (mean={np.nanmean(mae_orig_mean):.3f})",
        fontsize=9,
    )
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")

    ax = axes[mean_row, 1]
    im_viridis = ax.imshow(
        mae_corrected_mean,
        cmap="viridis",
        origin="lower",
        vmin=vmin_viridis,
        vmax=vmax_viridis,
    )
    ax.set_title(
        f"RF-corrected Ens. Mean (mean={np.nanmean(mae_corrected_mean):.3f})",
        fontsize=9,
    )
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")

    if mae_qm is not None:
        ax = axes[mean_row, 2]
        im_viridis = ax.imshow(
            mae_qm_mean,
            cmap="viridis",
            origin="lower",
            vmin=vmin_viridis,
            vmax=vmax_viridis,
        )
        ax.set_title(
            f"QM Ens. Mean (mean={np.nanmean(mae_qm_mean):.3f})",
            fontsize=9,
        )
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

        # Ensemble mean difference
        ax = axes[mean_row, 3]
        mae_diff_mean = mae_corrected_mean - mae_orig_mean
        im_diff = ax.imshow(
            mae_diff_mean,
            cmap="RdBu_r",
            vmin=vmin_diff,
            vmax=vmax_diff,
            origin="lower",
        )
        ax.set_title(
            f"Δ{metric_name} (RF-orig) Ens. Mean (mean={np.nanmean(mae_diff_mean):.3f})",
            fontsize=9,
        )
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
    else:
        # Ensemble mean difference
        ax = axes[mean_row, 2]
        mae_diff_mean = mae_corrected_mean - mae_orig_mean
        im_diff = ax.imshow(
            mae_diff_mean,
            cmap="RdBu_r",
            vmin=vmin_diff,
            vmax=vmax_diff,
            origin="lower",
        )
        ax.set_title(
            f"Δ{metric_name} (RF-orig) Ens. Mean (mean={np.nanmean(mae_diff_mean):.3f})",
            fontsize=9,
        )
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

    # Add shared colorbars
    if im_viridis is not None:
        cbar_ax1 = fig.add_axes([0.92, 0.55, 0.02, 0.4])
        fig.colorbar(im_viridis, cax=cbar_ax1, label=str(metric_name))

    if im_diff is not None:
        cbar_ax2 = fig.add_axes([0.92, 0.1, 0.02, 0.4])
        fig.colorbar(im_diff, cax=cbar_ax2, label=f"Δ{metric_name}")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    out_png = (
        Path(out_dir)
        / f"{file_prefix}_skill_first{members_to_plot}_members_and_mean.png"
    )
    plt.savefig(str(out_png), bbox_inches="tight", dpi=150)
    out_paths.append(str(out_png))
    plt.close()

    return out_paths


def plot_bss_skill_metrics(
    bss_upper,
    bss_lower,
    out_dir="qm_plots",
    figsize=(10, 4),
    upper_threshold_label="mean + 1σ",
    lower_threshold_label="mean - 1σ",
    file_prefix="bss",
    title_prefix="QM vs. Original",
    land_mask_path=None,
):
    """
    Plot BSS maps for upper and lower tail extremes.

    Args:
        bss_upper, bss_lower: np.ndarray shape (lat, lon) - BSS for upper/lower tails
        out_dir: output directory for plots
        figsize: figure size
        upper_threshold_label: label describing the upper-extreme threshold
        lower_threshold_label: label describing the lower-extreme threshold
        file_prefix: prefix for output filename
        title_prefix: prefix for plot title

    Returns:
        list of written file paths
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_paths = []

    land_mask = _load_land_mask(land_mask_path)
    bss_upper = bss_upper * land_mask
    bss_lower = bss_lower * land_mask

    # Create figure: Upper and lower tail BSS spatial maps
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(
        (
            f"Brier Skill Score: {title_prefix} "
            f"(Upper threshold: {upper_threshold_label}, "
            f"Lower threshold: {lower_threshold_label})"
        ),
        fontsize=12,
        fontweight="bold",
    )

    # Upper tail BSS
    ax = axes[0]
    im = ax.imshow(bss_upper, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    ax.set_title(
        f"BSS Upper [{upper_threshold_label}] (mean={np.nanmean(bss_upper):.3f})",
        fontsize=9,
    )
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    plt.colorbar(im, ax=ax, label="BSS")

    # Lower tail BSS
    ax = axes[1]
    im = ax.imshow(bss_lower, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    ax.set_title(
        f"BSS Lower [{lower_threshold_label}] (mean={np.nanmean(bss_lower):.3f})",
        fontsize=9,
    )
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    plt.colorbar(im, ax=ax, label="BSS")

    plt.tight_layout()
    out_png = Path(out_dir) / f"{file_prefix}_skill_metrics.png"
    plt.savefig(str(out_png), bbox_inches="tight", dpi=150)
    out_paths.append(str(out_png))
    plt.close()

    return out_paths


def plot_event_bss_comparison_maps(
    bss_qm_vs_orig,
    bss_ml_vs_orig,
    bss_ml_vs_qm,
    out_dir="qm_plots",
    figsize=(15, 4),
    threshold_label="P90 event",
    file_prefix="event_bss_comparison",
    land_mask_path=None,
):
    """
    Plot event-based BSS comparison maps in one 3-panel figure.

    Args:
        bss_qm_vs_orig: np.ndarray (lat, lon), BSS of QM against Original
        bss_ml_vs_orig: np.ndarray (lat, lon), BSS of ML against Original
        bss_ml_vs_qm: np.ndarray (lat, lon), BSS of ML against QM
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_paths = []

    land_mask = _load_land_mask(land_mask_path)
    bss_qm_vs_orig = bss_qm_vs_orig * land_mask
    bss_ml_vs_orig = bss_ml_vs_orig * land_mask
    bss_ml_vs_qm = bss_ml_vs_qm * land_mask

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(
        f"Event BSS Comparison ({threshold_label})",
        fontsize=12,
        fontweight="bold",
    )

    panels = [
        ("QM vs Original", bss_qm_vs_orig),
        ("ML vs Original", bss_ml_vs_orig),
        ("ML vs QM", bss_ml_vs_qm),
    ]

    for ax, (title, data) in zip(axes, panels):
        im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
        ax.set_title(f"{title} (mean={np.nanmean(data):.3f})", fontsize=9)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        plt.colorbar(im, ax=ax, label="BSS")

    plt.tight_layout()
    out_png = Path(out_dir) / f"{file_prefix}.png"
    plt.savefig(str(out_png), bbox_inches="tight", dpi=150)
    out_paths.append(str(out_png))
    plt.close()

    return out_paths


def plot_drought_hit_rate_maps(
    baseline_hit_rate,
    qm_hit_rate,
    corrected_hit_rate,
    out_dir="qm_plots",
    figsize=(15, 4),
    lower_threshold_label="P10",
    file_prefix="drought_hit_rate_p10",
):
    """
    Plot per-gridcell drought hit-rate percentages (0-100).

    Args:
        baseline_hit_rate: np.ndarray (lat, lon), Original hindcast hit-rate in %
        qm_hit_rate: np.ndarray (lat, lon), QM hindcast hit-rate in %
        corrected_hit_rate: np.ndarray (lat, lon), RF-corrected hindcast hit-rate in %
        out_dir: output directory for plots
        figsize: figure size
        lower_threshold_label: label for lower threshold (e.g. P10)
        file_prefix: output filename prefix

    Returns:
        list[str]: written file paths
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_paths = []

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(
        f"Correctly Predicted Droughts per Gridcell (< {lower_threshold_label})",
        fontsize=12,
        fontweight="bold",
    )

    panels = [
        ("Original", baseline_hit_rate),
        ("QM", qm_hit_rate),
        ("RF-corrected", corrected_hit_rate),
    ]

    im_last = None
    for ax, (name, data) in zip(axes, panels):
        im_last = ax.imshow(data, cmap="viridis", vmin=0, vmax=100, origin="lower")
        ax.set_title(f"{name} hit-rate (mean={np.nanmean(data):.1f}%)", fontsize=9)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

    # Add single shared colorbar
    if im_last is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im_last, cax=cbar_ax, label="Correct drought predictions [%]")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    out_png = Path(out_dir) / f"{file_prefix}.png"
    plt.savefig(str(out_png), bbox_inches="tight", dpi=150)
    out_paths.append(str(out_png))
    plt.close()

    return out_paths


def plot_extreme_drought_hit_histogram(
    system_counts,
    out_dir="qm_plots",
    lower_percentile=10.0,
    threshold_label=None,
    file_prefix="extreme_drought_p10_histogram",
):
    """
    Plot extreme drought event counts (< lower percentile of reference).

        For each hindcast type, show:
      - Number of reference drought events
      - Number of correctly hit drought events
      - Number of wrongly predicted drought events

    Args:
                system_counts: list of tuples
                        (system_name, reference_count, hit_count, wrong_count)
        out_dir: output directory for plots
        lower_percentile: lower percentile threshold (default 10)
        threshold_label: custom threshold label for panel titles (e.g. "> 0")
        file_prefix: output filename prefix

    Returns:
        list[str]: written file paths
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_paths = []

    n_systems = len(system_counts)
    fig, axes = plt.subplots(1, n_systems, figsize=(5 * n_systems, 4), squeeze=False)
    axes = axes[0]

    bar_labels = ["Reference droughts", "Correct hits", "Wrong predictions"]
    bar_colors = ["tab:gray", "tab:green", "tab:red"]

    for idx, (system_name, ref_count, hit_count, wrong_count) in enumerate(
        system_counts
    ):
        ax = axes[idx]
        values = [ref_count, hit_count, wrong_count]
        ax.bar(np.arange(3), values, color=bar_colors)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(bar_labels, rotation=20, ha="right")
        ax.set_ylabel("Count")
        title_threshold = (
            threshold_label
            if threshold_label is not None
            else f"<P{int(lower_percentile)}"
        )
        ax.set_title(f"{system_name} ({title_threshold})")

    fig.suptitle(
        "Extreme drought events histogram: reference vs hits vs wrong predictions",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    out_png = Path(out_dir) / f"{file_prefix}.png"
    plt.savefig(str(out_png), bbox_inches="tight", dpi=150)
    out_paths.append(str(out_png))
    plt.close()

    return out_paths


def plot_example_time_means(
    baseline_ensemble,
    qm_ensemble=None,
    corrected_ensemble=None,
    reference=None,
    out_dir="plots",
    n_members_display=3,
    n_timesteps=3,
    variable_name="CWB",
    corrected_label="RF-corrected",
    difference_mode=False,
    land_mask_path=None,
):
    """
    Plot timestep maps for the first n_timesteps and first n_members_display members,
    plus one final row with the ensemble time mean over all time steps.

    Args:
        baseline_ensemble: xr.DataArray shape (time, member, lat, lon) - original hindcasts
        qm_ensemble: xr.DataArray shape (time, member, lat, lon) - QM hindcasts (optional)
        corrected_ensemble: xr.DataArray shape (time, member, lat, lon) - corrected hindcasts (optional)
        reference: xr.DataArray shape (time, lat, lon) - reference data
        out_dir: output directory for plots
        n_members_display: number of members to display
        n_timesteps: number of timesteps per member to display
        variable_name: name of variable for plot titles

    Returns:
        list of written file paths
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_paths = []

    if "member" not in baseline_ensemble.dims:
        baseline_ensemble = baseline_ensemble.expand_dims(member=[0])
    if qm_ensemble is not None and "member" not in qm_ensemble.dims:
        qm_ensemble = qm_ensemble.expand_dims(member=[0])
    if corrected_ensemble is not None and "member" not in corrected_ensemble.dims:
        corrected_ensemble = corrected_ensemble.expand_dims(member=[0])

    n_members = baseline_ensemble.sizes["member"]
    members_to_plot = min(n_members_display, n_members)
    n_time = baseline_ensemble.sizes["time"]
    timesteps_to_plot = min(max(1, int(n_timesteps)), n_time)
    timestep_indices = np.arange(timesteps_to_plot)

    reference_time_mean = reference.mean(dim="time")
    land_mask = _load_land_mask(land_mask_path)

    baseline_ensemble = baseline_ensemble * land_mask
    qm_ensemble = qm_ensemble * land_mask if qm_ensemble is not None else None
    corrected_ensemble = (
        corrected_ensemble * land_mask if corrected_ensemble is not None else None
    )
    reference = reference * land_mask
    reference_time_mean = reference_time_mean * land_mask

    system_columns = [("Original", baseline_ensemble)]
    if corrected_ensemble is not None:
        system_columns.append((corrected_label, corrected_ensemble))
    if qm_ensemble is not None:
        system_columns.append(("QM", qm_ensemble))

    if difference_mode:
        diff_columns = []
        for col_name, col_da in system_columns:
            diff_da = col_da - reference
            if "member" in diff_da.coords and "member" not in diff_da.dims:
                diff_da = diff_da.reset_coords("member", drop=True)
            if "member" in diff_da.dims and diff_da.sizes.get("member", 0) == 0:
                diff_da = diff_da.drop_dims("member").expand_dims(member=[0])
            diff_columns.append((f"{col_name} - Ref", diff_da))
        system_columns = diff_columns

    n_cols = len(system_columns) + 1  # + reference column
    n_rows = (timesteps_to_plot * members_to_plot) + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.8 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    title_parts = [name for name, _ in system_columns] + ["Reference"]
    title_mode = "Differences" if difference_mode else "Values"
    fig.suptitle(
        (
            f"Example maps ({timesteps_to_plot} timesteps/member) — {variable_name} {title_mode}: "
            + " / ".join(title_parts)
        ),
        fontsize=12,
        fontweight="bold",
    )

    # Compute limits for system columns
    system_values = []
    for col_name, col_da in system_columns:
        col_values = []
        for member_idx in range(members_to_plot):
            for timestep_idx in timestep_indices:
                col_values.append(
                    col_da.isel(member=member_idx, time=timestep_idx).values
                )
        col_values.append(col_da.mean(dim="member").mean(dim="time").values)
        system_values.extend(np.asarray(col_values).ravel().tolist())

    system_arr = np.asarray(system_values)
    finite_system = system_arr[np.isfinite(system_arr)]
    if finite_system.size > 0:
        if difference_mode:
            max_abs = np.nanpercentile(np.abs(finite_system), 98)
            if not np.isfinite(max_abs) or max_abs == 0:
                max_abs = 1e-12
            system_vmin, system_vmax = -max_abs, max_abs
        else:
            system_vmin, system_vmax = np.nanpercentile(finite_system, [2, 98])
    else:
        system_vmin, system_vmax = None, None

    # Compute separate limits for reference column
    ref_values = []
    for timestep_idx in timestep_indices:
        ref_values.append(reference.isel(time=timestep_idx).values)
    ref_values.append(reference_time_mean.values)
    ref_arr = np.asarray(ref_values).ravel()
    finite_ref = ref_arr[np.isfinite(ref_arr)]
    if finite_ref.size > 0:
        ref_vmin, ref_vmax = np.nanpercentile(finite_ref, [2, 98])
    else:
        ref_vmin, ref_vmax = None, None

    system_cmap = "RdBu_r" if difference_mode else "coolwarm"

    row_idx = 0
    im_sys_last = None
    im_ref_last = None
    for member_idx in range(members_to_plot):
        for timestep_idx in timestep_indices:
            for col_idx, (col_name, col_da) in enumerate(system_columns):
                ax = axes[row_idx, col_idx]
                im_sys_last = ax.imshow(
                    col_da.isel(member=member_idx, time=timestep_idx),
                    cmap=system_cmap,
                    origin="lower",
                    vmin=system_vmin,
                    vmax=system_vmax,
                )
                ax.set_title(f"{col_name} m={member_idx} t={int(timestep_idx)}")
                ax.set_xlabel("lon")
                ax.set_ylabel("lat")

            ax = axes[row_idx, len(system_columns)]
            im_ref_last = ax.imshow(
                reference.isel(time=timestep_idx),
                cmap="coolwarm",
                origin="lower",
                vmin=ref_vmin,
                vmax=ref_vmax,
            )
            ax.set_title(f"Reference t={int(timestep_idx)}")
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
            row_idx += 1

    mean_row = n_rows - 1
    for col_idx, (col_name, col_da) in enumerate(system_columns):
        ax = axes[mean_row, col_idx]
        im_sys_last = ax.imshow(
            col_da.mean(dim="member").mean(dim="time"),
            cmap=system_cmap,
            origin="lower",
            vmin=system_vmin,
            vmax=system_vmax,
        )
        ax.set_title(f"{col_name} ensemble time mean")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

    ax = axes[mean_row, len(system_columns)]
    im_ref_last = ax.imshow(
        reference_time_mean,
        cmap="coolwarm",
        origin="lower",
        vmin=ref_vmin,
        vmax=ref_vmax,
    )
    ax.set_title("Reference time mean")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")

    # Add colorbars
    if difference_mode:
        if im_sys_last is not None:
            cbar_sys_ax = fig.add_axes([0.90, 0.1, 0.015, 0.8])
            fig.colorbar(
                im_sys_last,
                cax=cbar_sys_ax,
                label=f"{variable_name} difference",
            )
        if im_ref_last is not None:
            cbar_ref_ax = fig.add_axes([0.93, 0.1, 0.015, 0.8])
            fig.colorbar(im_ref_last, cax=cbar_ref_ax, label=variable_name)
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        out_png = (
            Path(out_dir) / f"example_maps_diff_to_ref_{timesteps_to_plot}timesteps.png"
        )
    else:
        if im_ref_last is not None:
            cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
            fig.colorbar(im_ref_last, cax=cbar_ax, label=variable_name)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        out_png = Path(out_dir) / f"example_maps_{timesteps_to_plot}timesteps.png"

    plt.savefig(str(out_png), bbox_inches="tight", dpi=150)
    out_paths.append(str(out_png))
    plt.close()

    return out_paths


def plot_ensemble_timeseries(
    baseline_ensemble,
    corrected_ensemble,
    reference,
    out_dir="qm_plots",
    variable_name="CWB",
    qm_ensemble=None,
    land_mask_path=None,
):
    """
    Plot timeseries with ensemble mean and spread (min/max as shaded region).

    Args:
        baseline_ensemble: np.ndarray, shape (time, member, lat, lon)
        corrected_ensemble: np.ndarray, shape (time, member, lat, lon)
        reference: np.ndarray, shape (time, lat, lon)
        out_dir: output directory for plots
        variable_name: variable label for y-axis
        qm_ensemble: np.ndarray, shape (time, member, lat, lon) (optional)

    Returns:
        list of written file paths
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_paths = []

    baseline_np = baseline_ensemble
    corrected_np = corrected_ensemble
    reference_np = reference
    qm_np = qm_ensemble
    time_coords = np.arange(baseline_np.shape[0])

    land_mask = _load_land_mask(land_mask_path)

    baseline_np = baseline_np * land_mask
    corrected_np = corrected_np * land_mask
    reference_np = reference_np * land_mask
    if qm_np is not None:
        qm_np = qm_np * land_mask

    # Compute spatial mean (time, member)
    baseline_spatial_mean = np.nanmean(baseline_np, axis=(2, 3))
    corrected_spatial_mean = np.nanmean(corrected_np, axis=(2, 3))
    reference_spatial_mean = np.nanmean(reference_np, axis=(1, 2))

    if qm_np is not None:
        qm_spatial_mean = np.nanmean(qm_np, axis=(2, 3))

    # Compute ensemble mean and spread
    baseline_mean = np.nanmean(baseline_spatial_mean, axis=1)
    baseline_min = np.nanmin(baseline_spatial_mean, axis=1)
    baseline_max = np.nanmax(baseline_spatial_mean, axis=1)

    corrected_mean = np.nanmean(corrected_spatial_mean, axis=1)
    corrected_min = np.nanmin(corrected_spatial_mean, axis=1)
    corrected_max = np.nanmax(corrected_spatial_mean, axis=1)

    if qm_np is not None:
        qm_mean = np.nanmean(qm_spatial_mean, axis=1)
        qm_min = np.nanmin(qm_spatial_mean, axis=1)
        qm_max = np.nanmax(qm_spatial_mean, axis=1)

    baseline_corr = None
    corrected_corr = None
    qm_corr = None
    baseline_valid = ~(np.isnan(baseline_mean) | np.isnan(reference_spatial_mean))
    corrected_valid = ~(np.isnan(corrected_mean) | np.isnan(reference_spatial_mean))
    baseline_corr = np.corrcoef(
        baseline_mean[baseline_valid], reference_spatial_mean[baseline_valid]
    )[0, 1]
    corrected_corr = np.corrcoef(
        corrected_mean[corrected_valid], reference_spatial_mean[corrected_valid]
    )[0, 1]
    if qm_np is not None:
        qm_valid = ~(np.isnan(qm_mean) | np.isnan(reference_spatial_mean))
        if qm_valid.sum() > 0:
            qm_corr = np.corrcoef(qm_mean[qm_valid], reference_spatial_mean[qm_valid])[
                0, 1
            ]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot reference as black line
    ax.plot(
        time_coords,
        reference_spatial_mean,
        "k-",
        linewidth=2.5,
        label="Reference",
        zorder=10,
    )

    # Plot Original with spread
    ax.fill_between(
        time_coords,
        baseline_min,
        baseline_max,
        alpha=0.2,
        color="C0",
        label="Original (min/max)",
    )
    baseline_label = "Original mean"
    if baseline_corr is not None:
        baseline_label = f"Original mean (r={baseline_corr:.2f})"
    ax.plot(time_coords, baseline_mean, color="C0", linewidth=2, label=baseline_label)

    # Plot RF-corrected with spread
    ax.fill_between(
        time_coords,
        corrected_min,
        corrected_max,
        alpha=0.2,
        color="C1",
        label="RF-corrected (min/max)",
    )
    ax.plot(
        time_coords,
        corrected_mean,
        color="C1",
        linewidth=2,
        label=(
            f"ML mean (r={corrected_corr:.2f})"
            if corrected_corr is not None
            else "ML mean"
        ),
    )

    # Plot QM with spread if available
    if qm_np is not None:
        ax.fill_between(
            time_coords, qm_min, qm_max, alpha=0.2, color="C2", label="QM (min/max)"
        )
        qm_label = "QM mean"
        if qm_corr is not None:
            qm_label = f"QM mean (r={qm_corr:.2f})"
        ax.plot(time_coords, qm_mean, color="C2", linewidth=2, label=qm_label)

    ax.set_xlabel("Time")
    ax.set_ylabel(variable_name)
    ax.set_title(f"Ensemble Timeseries with Spread (min/max)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = Path(out_dir) / "ensemble_timeseries.png"
    plt.savefig(str(out_png), bbox_inches="tight", dpi=150)
    out_paths.append(str(out_png))
    plt.close()

    return out_paths


def plot_residual_comparison_maps(
    qm_residual_ensemble,
    ml_residual_ensemble,
    out_dir="qm_plots",
    variable_name="CWB",
    file_prefix="residual_comparison",
    land_mask_path=None,
):
    """
    Plot ensemble-time mean residual maps for QM residuals, ML-predicted residuals,
    and their difference.

    Args:
        qm_residual_ensemble: np.ndarray shape (time, member, lat, lon)
        ml_residual_ensemble: np.ndarray shape (time, member, lat, lon)
        out_dir: output directory for plots
        variable_name: variable label for colorbars
        file_prefix: output filename prefix

    Returns:
        list[str]: written file paths
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_paths = []

    qm_res = np.asarray(qm_residual_ensemble)
    ml_res = np.asarray(ml_residual_ensemble)

    qm_mean = np.nanmean(qm_res, axis=(0, 1))
    ml_mean = np.nanmean(ml_res, axis=(0, 1))
    diff_mean = ml_mean - qm_mean

    land_mask = _load_land_mask(land_mask_path)

    qm_mean = qm_mean * land_mask
    ml_mean = ml_mean * land_mask
    diff_mean = diff_mean * land_mask

    res_vals = np.concatenate([qm_mean.ravel(), ml_mean.ravel()])
    res_vals = res_vals[np.isfinite(res_vals)]
    res_abs = float(np.nanpercentile(np.abs(res_vals), 98))
    if not np.isfinite(res_abs) or np.isclose(res_abs, 0.0):
        res_abs = 1e-12

    diff_vals = diff_mean.ravel()
    diff_vals = diff_vals[np.isfinite(diff_vals)]
    if diff_vals.size > 0:
        diff_abs = float(np.nanpercentile(np.abs(diff_vals), 98))
        if not np.isfinite(diff_abs) or np.isclose(diff_abs, 0.0):
            diff_abs = 1e-12
    else:
        diff_abs = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Residual comparison maps ({variable_name})",
        fontsize=12,
        fontweight="bold",
    )

    im_res = axes[0].imshow(
        qm_mean,
        cmap="RdBu_r",
        origin="lower",
        vmin=-res_abs,
        vmax=res_abs,
    )
    axes[0].set_title(f"QM residuals mean ({np.nanmean(qm_mean):.3f})")
    axes[0].set_xlabel("lon")
    axes[0].set_ylabel("lat")

    im_res = axes[1].imshow(
        ml_mean,
        cmap="RdBu_r",
        origin="lower",
        vmin=-res_abs,
        vmax=res_abs,
    )
    axes[1].set_title(f"ML residuals mean ({np.nanmean(ml_mean):.3f})")
    axes[1].set_xlabel("lon")
    axes[1].set_ylabel("lat")

    im_diff = axes[2].imshow(
        diff_mean,
        cmap="RdBu_r",
        origin="lower",
        vmin=-diff_abs,
        vmax=diff_abs,
    )
    axes[2].set_title(f"ML - QM residual mean ({np.nanmean(diff_mean):.3f})")
    axes[2].set_xlabel("lon")
    axes[2].set_ylabel("lat")

    cbar_res_ax = fig.add_axes([0.91, 0.57, 0.015, 0.33])
    fig.colorbar(im_res, cax=cbar_res_ax, label=f"{variable_name} residual")

    cbar_diff_ax = fig.add_axes([0.91, 0.12, 0.015, 0.33])
    fig.colorbar(im_diff, cax=cbar_diff_ax, label=f"{variable_name} residual Δ")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    out_png = Path(out_dir) / f"{file_prefix}_maps.png"
    plt.savefig(str(out_png), bbox_inches="tight", dpi=150)
    out_paths.append(str(out_png))
    plt.close()

    return out_paths


def plot_feature_importance_table(
    feature_names,
    importances,
    out_dir="qm_plots",
    model_label="ML",
    top_n=None,
    file_prefix="feature_importance",
):
    """
    Render feature importances as a table saved to a PNG.

    Args:
        feature_names: sequence of feature names
        importances: sequence of importance values
        out_dir: output directory for plots
        model_label: model name shown in title
        top_n: optional cap on number of displayed features
        file_prefix: output filename prefix

    Returns:
        list[str]: written file paths
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_paths = []

    names = list(feature_names)
    vals = np.asarray(importances, dtype=float)
    if vals.ndim != 1:
        vals = vals.ravel()

    if len(names) != len(vals):
        names = [f"feature_{i}" for i in range(len(vals))]

    order = np.argsort(vals)[::-1]
    if top_n is not None:
        top_n = max(1, int(top_n))
        order = order[:top_n]

    sorted_names = [names[i] for i in order]
    sorted_vals = vals[order]

    total = float(np.nansum(sorted_vals))
    if np.isfinite(total) and total > 0:
        sorted_pct = (sorted_vals / total) * 100.0
    else:
        sorted_pct = np.full_like(sorted_vals, np.nan)

    rows = []
    for rank, (name, val, pct) in enumerate(
        zip(sorted_names, sorted_vals, sorted_pct), start=1
    ):
        pct_str = f"{pct:.2f}" if np.isfinite(pct) else "nan"
        rows.append([str(rank), str(name), f"{float(val):.6f}", pct_str])

    n_rows = max(1, len(rows))
    fig_h = min(0.45 * n_rows + 1.8, 18)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    ax.set_title(
        f"{model_label} feature importance",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    table = ax.table(
        cellText=rows,
        colLabels=["Rank", "Feature", "Importance", "Share [%]"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.08, 0.57, 0.17, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    out_png = Path(out_dir) / f"{file_prefix}_table.png"
    plt.savefig(str(out_png), bbox_inches="tight", dpi=170)
    out_paths.append(str(out_png))
    plt.close()

    return out_paths


def plot_ml_eval_summary_maps(
    reference_all,
    reference_eval,
    corrected_ensemble,
    baseline_ensemble,
    qm_ensemble=None,
    out_dir="qm_plots",
    variable_name="CWB",
    land_mask_path=None,
):
    """
        Plot a 5x3 summary map panel for ML evaluation.

    Rows:
      1) reference std/mean over all timesteps
      2) reference std/mean over eval years
      3) ML ensemble-mean std/mean over eval years
      4) baseline ensemble-mean std/mean over eval years
      5) QM ensemble-mean std/mean over eval years

        Columns:
            1) std
            2) mean
            3) anomaly relative to reference climatology
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_paths = []
    land_mask = _load_land_mask(land_mask_path)

    ref_all = np.asarray(reference_all) * land_mask
    ref_eval = np.asarray(reference_eval) * land_mask
    corr = np.asarray(corrected_ensemble) * land_mask
    base = np.asarray(baseline_ensemble) * land_mask
    qm = np.asarray(qm_ensemble) * land_mask if qm_ensemble is not None else None

    # Time statistics for reference fields
    ref_all_std = np.nanstd(ref_all, axis=0)
    ref_all_mean = np.nanmean(ref_all, axis=0)
    ref_eval_std = np.nanstd(ref_eval, axis=0)
    ref_eval_mean = np.nanmean(ref_eval, axis=0)

    # Ensemble mean timeseries, then time statistics
    corr_ensmean_t = np.nanmean(corr, axis=1)
    corr_std = np.nanstd(corr_ensmean_t, axis=0)
    corr_mean = np.nanmean(corr_ensmean_t, axis=0)

    base_ensmean_t = np.nanmean(base, axis=1)
    base_std = np.nanstd(base_ensmean_t, axis=0)
    base_mean = np.nanmean(base_ensmean_t, axis=0)

    qm_ensmean_t = np.nanmean(qm, axis=1)
    qm_std = np.nanstd(qm_ensmean_t, axis=0)
    qm_mean = np.nanmean(qm_ensmean_t, axis=0)

    clim_mean = ref_all_mean

    std_maps = [ref_all_std, ref_eval_std, corr_std, base_std, qm_std]
    mean_maps = [ref_all_mean, ref_eval_mean, corr_mean, base_mean, qm_mean]
    anomaly_maps = [m - clim_mean for m in mean_maps]

    std_vals = np.concatenate([m.ravel() for m in std_maps])
    mean_vals = np.concatenate([m.ravel() for m in mean_maps])
    anomaly_vals = np.concatenate([m.ravel() for m in anomaly_maps])

    std_vals = std_vals[np.isfinite(std_vals)]
    mean_vals = mean_vals[np.isfinite(mean_vals)]
    anomaly_vals = anomaly_vals[np.isfinite(anomaly_vals)]

    std_vmin = float(np.nanmin(std_vals)) if std_vals.size else 0.0
    std_vmax = float(np.nanmax(std_vals)) if std_vals.size else 1.0
    mean_vmin = float(np.nanmin(mean_vals)) if mean_vals.size else 0.0
    mean_vmax = float(np.nanmax(mean_vals)) if mean_vals.size else 1.0
    anomaly_abs = float(np.nanpercentile(np.abs(anomaly_vals), 98))
    anomaly_vmin, anomaly_vmax = -anomaly_abs, anomaly_abs

    fig, axes = plt.subplots(5, 3, figsize=(16, 18))
    fig.suptitle(
        f"ML Evaluation Summary Maps ({variable_name})",
        fontsize=14,
        fontweight="bold",
    )

    row_titles = [
        "Reference (all timesteps)",
        "Reference (eval years)",
        "ML ensemble mean (eval years)",
        "Baseline ensemble mean (eval years)",
        "QM ensemble mean (eval years)",
    ]

    im_std = None
    im_mean = None
    im_anomaly = None
    for row_idx, row_title in enumerate(row_titles):
        ax_std = axes[row_idx, 0]
        im_std = ax_std.imshow(
            std_maps[row_idx],
            cmap="viridis",
            origin="lower",
            vmin=std_vmin,
            vmax=std_vmax,
        )
        ax_std.set_title(f"{row_title} - Std")
        ax_std.set_xlabel("lon")
        ax_std.set_ylabel("lat")

        ax_mean = axes[row_idx, 1]
        im_mean = ax_mean.imshow(
            mean_maps[row_idx],
            cmap="coolwarm",
            origin="lower",
            vmin=mean_vmin,
            vmax=mean_vmax,
        )
        ax_mean.set_title(f"{row_title} - Mean")
        ax_mean.set_xlabel("lon")
        ax_mean.set_ylabel("lat")

        ax_anomaly = axes[row_idx, 2]
        im_anomaly = ax_anomaly.imshow(
            anomaly_maps[row_idx],
            cmap="RdBu_r",
            origin="lower",
            vmin=anomaly_vmin,
            vmax=anomaly_vmax,
        )
        ax_anomaly.set_title(f"{row_title} - Anomaly")
        ax_anomaly.set_xlabel("lon")
        ax_anomaly.set_ylabel("lat")

    if im_std is not None:
        cbar_std_ax = fig.add_axes([0.91, 0.53, 0.012, 0.38])
        fig.colorbar(im_std, cax=cbar_std_ax, label=f"{variable_name} std")

    if im_mean is not None:
        cbar_mean_ax = fig.add_axes([0.91, 0.09, 0.012, 0.38])
        fig.colorbar(im_mean, cax=cbar_mean_ax, label=f"{variable_name} mean")

    if im_anomaly is not None:
        cbar_anom_ax = fig.add_axes([0.935, 0.31, 0.012, 0.38])
        fig.colorbar(
            im_anomaly,
            cax=cbar_anom_ax,
            label=f"{variable_name} anomaly",
        )

    plt.tight_layout(rect=[0, 0, 0.9, 0.98])
    out_png = Path(out_dir) / "ml_eval_summary_maps.png"
    plt.savefig(str(out_png), bbox_inches="tight", dpi=150)
    out_paths.append(str(out_png))
    plt.close()

    return out_paths


def plot_training_curve(training_history: dict, out_path: Path):
    x_vals = training_history.get("x", [])
    train_vals = training_history.get("train", [])
    val_vals = training_history.get("val", None)
    metric = str(training_history.get("metric", "metric")).upper()
    model_type = str(training_history.get("model_type", "model"))

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


def save_merf_diagnostics_plot(
    diagnostics: dict,
    output_path: _t.Union[str, Path],
    decimals: int = 4,
    figsize: tuple = (14, 10),
):
    if diagnostics is None:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = f".{decimals}f"
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    table_data = []
    table_data.append(["MERF Diagnostics", ""])
    table_data.append(["", ""])

    table_data.append(["Group Effects Statistics", ""])
    n_groups = diagnostics["random_stats"]["n_groups"]
    table_data.append([f"  Number of Groups", f"{n_groups}"])
    table_data.append(
        [f"  Mean Effect", f"{diagnostics['random_stats']['mean']:{fmt}}"]
    )
    table_data.append([f"  Std Dev", f"{diagnostics['random_stats']['std']:{fmt}}"])
    table_data.append([f"  Min Effect", f"{diagnostics['random_stats']['min']:{fmt}}"])
    table_data.append([f"  Max Effect", f"{diagnostics['random_stats']['max']:{fmt}}"])
    table_data.append([f"  Range", f"{diagnostics['random_stats']['range']:{fmt}}"])
    table_data.append(
        [f"  % Non-zero Samples", f"{diagnostics['random_stats']['pct_nonzero']:.1f}%"]
    )
    table_data.append(["", ""])

    table_data.append(["Base (Fixed) Term Statistics", ""])
    table_data.append([f"  Mean", f"{diagnostics['base_stats']['mean']:{fmt}}"])
    table_data.append([f"  Std Dev", f"{diagnostics['base_stats']['std']:{fmt}}"])
    table_data.append([f"  Min", f"{diagnostics['base_stats']['min']:{fmt}}"])
    table_data.append([f"  Max", f"{diagnostics['base_stats']['max']:{fmt}}"])
    table_data.append([f"  Range", f"{diagnostics['base_stats']['range']:{fmt}}"])
    table_data.append(["", ""])

    table_data.append(["Contribution Analysis", ""])
    table_data.append(
        [
            f"  Avg Group Effect %",
            f"{diagnostics['contribution_ratio_groups']*100:.1f}%",
        ]
    )
    table_data.append(
        [f"  Max Group Effect %", f"{diagnostics['max_contribution_ratio']*100:.1f}%"]
    )
    table_data.append(["", ""])

    if "base_rmse" in diagnostics:
        table_data.append(["Error Metrics (Validation Set)", ""])
        table_data.append([f"  Base Model RMSE", f"{diagnostics['base_rmse']:{fmt}}"])
        table_data.append(
            [f"  MERF Model RMSE", f"{diagnostics['combined_rmse']:{fmt}}"]
        )
        table_data.append(
            [f"  RMSE Improvement", f"{diagnostics['rmse_improvement']:{fmt}}"]
        )
        if abs(diagnostics["rmse_improvement"]) > 1e-6:
            pct = (diagnostics["rmse_improvement"] / diagnostics["base_rmse"]) * 100
            table_data.append([f"  % Improvement", f"{pct:.1f}%"])
        table_data.append(["", ""])

    table = ax.table(
        cellText=table_data,
        cellLoc="left",
        loc="upper left",
        colWidths=[0.6, 0.4],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in [0, 2, 9, 12, 15]:
        if i < len(table_data):
            table[(i, 0)].set_facecolor("#4472C4")
            table[(i, 0)].set_text_props(weight="bold", color="white")
            table[(i, 1)].set_facecolor("#4472C4")
            table[(i, 1)].set_text_props(weight="bold", color="white")

    for i, row in enumerate(table_data):
        if i not in [0, 2, 9, 12, 15, 1] and row != ["", ""]:
            if i % 2 == 0:
                table[(i, 0)].set_facecolor("#E8F0F8")
                table[(i, 1)].set_facecolor("#E8F0F8")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    if "group_effects_df" in diagnostics:
        top_groups_path = output_path.parent / f"{output_path.stem}_top_groups.png"
        _save_group_effects_table(
            diagnostics["group_effects_df"], top_groups_path, decimals
        )


def _save_group_effects_table(
    group_effects_df: pd.DataFrame,
    output_path: _t.Union[str, Path],
    decimals: int = 4,
    n_top: int = 15,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_top = group_effects_df.head(n_top).copy()
    df_top.columns = ["Group", "Effect"]
    df_top["Effect"] = df_top["Effect"].apply(lambda x: f"{x:.{decimals}f}")

    fig, ax = plt.subplots(figsize=(10, max(6, n_top * 0.4)))
    ax.axis("off")

    table_data = [["Group ID", "Effect"]] + df_top.values.tolist()

    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.7, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    for i in range(2):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white")

    for i in range(1, len(table_data)):
        if i % 2 == 0:
            table[(i, 0)].set_facecolor("#E8F0F8")
            table[(i, 1)].set_facecolor("#E8F0F8")

    plt.title(f"Top {n_top} Group Effects", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
