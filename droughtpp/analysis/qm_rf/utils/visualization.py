import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_qm_distributions(
    distributions,
    var_name,
    out_dir="qm_plots",
    n_quantiles=200,
    figsize=(8, 6),
    bins=250,
):
    """
    distributions: dict produced by compute_qm_distributions()
    Creates one PNG per entry comparing histograms of original, qm and obs.
    Returns list of written file paths.
    """
    out_paths = []
    hind, qm = [], []
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for key, d in distributions.items():
        orig_vals = d["orig"]
        qm_vals = d["qm"]
        obs_vals = d["obs"]
        hind.append(orig_vals)
        qm.append(qm_vals)

    qm_mean = np.mean(qm, axis=0)
    orig_mean = np.mean(hind, axis=0)
    x_min = min(orig_mean.min(), qm_mean.min())
    x_max = max(orig_mean.max(), qm_mean.max())
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
    plt.hist(orig_mean, bins=bins, density=True, alpha=0.45, label="original")
    plt.hist(qm_mean, bins=bins, density=True, alpha=0.45, label="quantile-mapped")
    plt.hist(obs_vals, bins=bins, density=True, alpha=0.45, label="reference")
    plt.legend()
    plt.title(f"Ensemble Mean — {var_name} distribution (QM vs obs)")
    plt.xlabel(var_name)
    plt.xlim(x_min, x_max)
    plt.ylabel("density")

    out_png = Path(out_dir) / f"ensemble_mean_qm_distribution_n{n_quantiles}.png"
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

    Returns:
        list of written file paths
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_paths = []
    if file_prefix is None:
        file_prefix = str(metric_name).lower().replace(" ", "_")

    n_members = mae_orig.shape[0]
    members_to_plot = min(n_members_display, n_members)

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
    if finite_viridis.size > 0:
        vmin_viridis = np.nanmin(finite_viridis)
        vmax_viridis = np.nanmax(finite_viridis)
    else:
        vmin_viridis, vmax_viridis = None, None

    # Compute global vmin/vmax for differences (RdBu_r)
    diff_values = []
    for i in range(members_to_plot):
        mae_diff = mae_corrected[i] - mae_orig[i]
        diff_values.append(mae_diff.ravel())
    mae_diff_mean = mae_corrected_mean - mae_orig_mean
    diff_values.append(mae_diff_mean.ravel())

    diff_values = np.concatenate(diff_values)
    finite_diff = diff_values[np.isfinite(diff_values)]
    if finite_diff.size > 0:
        max_abs_diff = np.nanmax(np.abs(finite_diff))
        vmin_diff = -max_abs_diff
        vmax_diff = max_abs_diff
    else:
        max_abs_diff = 1e-12
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
        ax.set_title(f"{system_name} (<P{int(lower_percentile)})")

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
    land_mask = np.isfinite(reference_time_mean.values)

    baseline_ensemble = baseline_ensemble.where(land_mask)
    if qm_ensemble is not None:
        qm_ensemble = qm_ensemble.where(land_mask)
    if corrected_ensemble is not None:
        corrected_ensemble = corrected_ensemble.where(land_mask)
    reference = reference.where(land_mask)
    reference_time_mean = reference_time_mean.where(land_mask)

    system_columns = [("Original", baseline_ensemble)]
    if corrected_ensemble is not None:
        system_columns.append((corrected_label, corrected_ensemble))
    if qm_ensemble is not None:
        system_columns.append(("QM", qm_ensemble))

    n_cols = len(system_columns) + 1  # + reference column
    n_rows = (timesteps_to_plot * members_to_plot) + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.8 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    title_parts = ["Original"]
    if corrected_ensemble is not None:
        title_parts.append(corrected_label)
    if qm_ensemble is not None:
        title_parts.append("QM")
    title_parts.append("Reference")
    fig.suptitle(
        (
            f"Example maps ({timesteps_to_plot} timesteps/member) — {variable_name}: "
            + " / ".join(title_parts)
        ),
        fontsize=12,
        fontweight="bold",
    )

    # Compute global limits across all columns for consistent colorbar
    all_values = []

    for col_name, col_da in system_columns:
        col_values = []
        for member_idx in range(members_to_plot):
            for timestep_idx in timestep_indices:
                col_values.append(
                    col_da.isel(member=member_idx, time=timestep_idx).values
                )
        col_values.append(col_da.mean(dim="member").mean(dim="time").values)
        all_values.extend(np.asarray(col_values).ravel().tolist())

    ref_values = []
    for timestep_idx in timestep_indices:
        ref_values.append(reference.isel(time=timestep_idx).values)
    ref_values.append(reference_time_mean.values)
    all_values.extend(np.asarray(ref_values).ravel().tolist())

    # Compute global percentiles from all data
    all_values_arr = np.asarray(all_values)
    finite_values = all_values_arr[np.isfinite(all_values_arr)]
    if finite_values.size > 0:
        global_vmin, global_vmax = np.nanpercentile(finite_values, [2, 98])
    else:
        global_vmin, global_vmax = None, None

    row_idx = 0
    im_last = None
    for member_idx in range(members_to_plot):
        for timestep_idx in timestep_indices:
            for col_idx, (col_name, col_da) in enumerate(system_columns):
                ax = axes[row_idx, col_idx]
                im_last = ax.imshow(
                    col_da.isel(member=member_idx, time=timestep_idx),
                    cmap="coolwarm",
                    origin="lower",
                    vmin=global_vmin,
                    vmax=global_vmax,
                )
                ax.set_title(f"{col_name} m={member_idx} t={int(timestep_idx)}")
                ax.set_xlabel("lon")
                ax.set_ylabel("lat")

            ax = axes[row_idx, len(system_columns)]
            im_last = ax.imshow(
                reference.isel(time=timestep_idx),
                cmap="coolwarm",
                origin="lower",
                vmin=global_vmin,
                vmax=global_vmax,
            )
            ax.set_title(f"Reference t={int(timestep_idx)}")
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
            row_idx += 1

    mean_row = n_rows - 1
    for col_idx, (col_name, col_da) in enumerate(system_columns):
        ax = axes[mean_row, col_idx]
        im_last = ax.imshow(
            col_da.mean(dim="member").mean(dim="time"),
            cmap="coolwarm",
            origin="lower",
            vmin=global_vmin,
            vmax=global_vmax,
        )
        ax.set_title(f"{col_name} ensemble time mean")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

    ax = axes[mean_row, len(system_columns)]
    im_last = ax.imshow(
        reference_time_mean,
        cmap="coolwarm",
        origin="lower",
        vmin=global_vmin,
        vmax=global_vmax,
    )
    ax.set_title("Reference time mean")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")

    # Add single shared colorbar for all maps
    if im_last is not None:
        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        fig.colorbar(im_last, cax=cbar_ax, label=variable_name)

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
    ):
        """
        Plot timeseries with ensemble mean and spread (min/max as shaded region).

        Args:
            baseline_ensemble: xr.DataArray or np.ndarray, shape (time, member, lat, lon)
            corrected_ensemble: xr.DataArray or np.ndarray, shape (time, member, lat, lon)
            reference: xr.DataArray or np.ndarray, shape (time, lat, lon)
            out_dir: output directory for plots
            variable_name: variable label for y-axis
            qm_ensemble: xr.DataArray or np.ndarray, shape (time, member, lat, lon) (optional)

        Returns:
            list of written file paths
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_paths = []

        # Convert to numpy if xarray
        if hasattr(baseline_ensemble, "values"):
            baseline_np = baseline_ensemble.values
            time_coords = baseline_ensemble.time.values if hasattr(baseline_ensemble, "time") else np.arange(baseline_ensemble.shape[0])
        else:
            baseline_np = baseline_ensemble
            time_coords = np.arange(baseline_np.shape[0])

        if hasattr(corrected_ensemble, "values"):
            corrected_np = corrected_ensemble.values
        else:
            corrected_np = corrected_ensemble

        if hasattr(reference, "values"):
            reference_np = reference.values
        else:
            reference_np = reference

        if qm_ensemble is not None:
            if hasattr(qm_ensemble, "values"):
                qm_np = qm_ensemble.values
            else:
                qm_np = qm_ensemble
        else:
            qm_np = None

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

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot reference as black line
        ax.plot(time_coords, reference_spatial_mean, "k-", linewidth=2.5, label="Reference", zorder=10)

        # Plot Original with spread
        ax.fill_between(time_coords, baseline_min, baseline_max, alpha=0.2, color="C0", label="Original (min/max)")
        ax.plot(time_coords, baseline_mean, color="C0", linewidth=2, label="Original mean")

        # Plot RF-corrected with spread
        ax.fill_between(time_coords, corrected_min, corrected_max, alpha=0.2, color="C1", label="RF-corrected (min/max)")
        ax.plot(time_coords, corrected_mean, color="C1", linewidth=2, label="RF-corrected mean")

        # Plot QM with spread if available
        if qm_np is not None:
            ax.fill_between(time_coords, qm_min, qm_max, alpha=0.2, color="C2", label="QM (min/max)")
            ax.plot(time_coords, qm_mean, color="C2", linewidth=2, label="QM mean")

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
