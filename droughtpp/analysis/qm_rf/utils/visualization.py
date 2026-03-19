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

    for key, d in distributions.items():
        orig_vals = d["orig"]
        qm_vals = d["qm"]
        obs_vals = d["obs"]

        plt.figure(figsize=figsize)
        plt.hist(orig_vals, bins=bins, density=True, alpha=0.45, label="original")
        plt.hist(qm_vals, bins=bins, density=True, alpha=0.45, label="quantile-mapped")
        plt.hist(obs_vals, bins=bins, density=True, alpha=0.45, label="reference")
        plt.legend()
        plt.title(f"{key} — {var_name} distribution (QM vs obs)")
        plt.xlabel(var_name)
        plt.xlim(x_min, x_max)
        plt.ylabel("density")

        out_png = Path(out_dir) / f"{key}_qm_distribution_n{n_quantiles}.png"
        plt.savefig(str(out_png), bbox_inches="tight")
        out_paths.append(str(out_png))
        plt.close()

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

    # Plot first N members
    for i in range(members_to_plot):
        row_label = f"Member {i}"

        # Original
        ax = axes[i, 0]
        im = ax.imshow(mae_orig[i], cmap="viridis", origin="lower")
        ax.set_title(
            f"Original {row_label} (mean={np.nanmean(mae_orig[i]):.3f})",
            fontsize=9,
        )
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        plt.colorbar(im, ax=ax, label=str(metric_name))

        # Corrected
        ax = axes[i, 1]
        im = ax.imshow(mae_corrected[i], cmap="viridis", origin="lower")
        ax.set_title(
            f"RF-corrected {row_label} (mean={np.nanmean(mae_corrected[i]):.3f})",
            fontsize=9,
        )
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        plt.colorbar(im, ax=ax, label=str(metric_name))

        # QM (if available)
        if mae_qm is not None:
            ax = axes[i, 2]
            im = ax.imshow(mae_qm[i], cmap="viridis", origin="lower")
            ax.set_title(
                f"QM {row_label} (mean={np.nanmean(mae_qm[i]):.3f})",
                fontsize=9,
            )
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
            plt.colorbar(im, ax=ax, label=str(metric_name))

            # Difference RF vs orig
            ax = axes[i, 3]
            mae_diff = mae_corrected[i] - mae_orig[i]
            max_abs = np.nanmax(np.abs(mae_diff))
            if not np.isfinite(max_abs) or max_abs == 0:
                max_abs = 1e-12
            im = ax.imshow(
                mae_diff, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, origin="lower"
            )
            ax.set_title(
                f"Δ{metric_name} (RF-orig) {row_label} (mean={np.nanmean(mae_diff):.3f})",
                fontsize=9,
            )
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
            plt.colorbar(im, ax=ax, label=f"Δ{metric_name}")
        else:
            # Difference RF vs orig
            ax = axes[i, 2]
            mae_diff = mae_corrected[i] - mae_orig[i]
            max_abs = np.nanmax(np.abs(mae_diff))
            if not np.isfinite(max_abs) or max_abs == 0:
                max_abs = 1e-12
            im = ax.imshow(
                mae_diff, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, origin="lower"
            )
            ax.set_title(
                f"Δ{metric_name} (RF-orig) {row_label} (mean={np.nanmean(mae_diff):.3f})",
                fontsize=9,
            )
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
            plt.colorbar(im, ax=ax, label=f"Δ{metric_name}")

    # Plot ensemble mean row
    mean_row = members_to_plot
    ax = axes[mean_row, 0]
    im = ax.imshow(mae_orig_mean, cmap="viridis", origin="lower")
    ax.set_title(
        f"Original Ens. Mean (mean={np.nanmean(mae_orig_mean):.3f})",
        fontsize=9,
    )
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    plt.colorbar(im, ax=ax, label=str(metric_name))

    ax = axes[mean_row, 1]
    im = ax.imshow(mae_corrected_mean, cmap="viridis", origin="lower")
    ax.set_title(
        f"RF-corrected Ens. Mean (mean={np.nanmean(mae_corrected_mean):.3f})",
        fontsize=9,
    )
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    plt.colorbar(im, ax=ax, label=str(metric_name))

    if mae_qm is not None:
        ax = axes[mean_row, 2]
        im = ax.imshow(mae_qm_mean, cmap="viridis", origin="lower")
        ax.set_title(
            f"QM Ens. Mean (mean={np.nanmean(mae_qm_mean):.3f})",
            fontsize=9,
        )
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        plt.colorbar(im, ax=ax, label=str(metric_name))

        # Ensemble mean difference
        ax = axes[mean_row, 3]
        mae_diff_mean = mae_corrected_mean - mae_orig_mean
        max_abs_mean = np.nanmax(np.abs(mae_diff_mean))
        if not np.isfinite(max_abs_mean) or max_abs_mean == 0:
            max_abs_mean = 1e-12
        im = ax.imshow(
            mae_diff_mean,
            cmap="RdBu_r",
            vmin=-max_abs_mean,
            vmax=max_abs_mean,
            origin="lower",
        )
        ax.set_title(
            f"Δ{metric_name} (RF-orig) Ens. Mean (mean={np.nanmean(mae_diff_mean):.3f})",
            fontsize=9,
        )
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        plt.colorbar(im, ax=ax, label=f"Δ{metric_name}")
    else:
        # Ensemble mean difference
        ax = axes[mean_row, 2]
        mae_diff_mean = mae_corrected_mean - mae_orig_mean
        max_abs_mean = np.nanmax(np.abs(mae_diff_mean))
        if not np.isfinite(max_abs_mean) or max_abs_mean == 0:
            max_abs_mean = 1e-12
        im = ax.imshow(
            mae_diff_mean,
            cmap="RdBu_r",
            vmin=-max_abs_mean,
            vmax=max_abs_mean,
            origin="lower",
        )
        ax.set_title(
            f"Δ{metric_name} (RF-orig) Ens. Mean (mean={np.nanmean(mae_diff_mean):.3f})",
            fontsize=9,
        )
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        plt.colorbar(im, ax=ax, label=f"Δ{metric_name}")

    plt.tight_layout()
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


def plot_probability_skill_metrics(
    prob_corrected_upper,
    prob_baseline_upper,
    prob_corrected_lower,
    prob_baseline_lower,
    out_dir="qm_plots",
    figsize=(12, 8),
    upper_threshold_label="P90",
    lower_threshold_label="P10",
    file_prefix="probability",
    title_prefix="RF-corrected vs Original",
):
    """
    Plot mean forecast probabilities for upper/lower tail events.

    Args:
        prob_corrected_upper, prob_baseline_upper: np.ndarray (lat, lon)
            Mean event probability maps for upper-tail threshold.
        prob_corrected_lower, prob_baseline_lower: np.ndarray (lat, lon)
            Mean event probability maps for lower-tail threshold.
        out_dir: output directory for plots
        figsize: figure size
        upper_threshold_label: label for upper threshold (e.g. P90)
        lower_threshold_label: label for lower threshold (e.g. P10)
        file_prefix: prefix for output filename
        title_prefix: title prefix

    Returns:
        list of written file paths
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_paths = []

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        (
            f"Forecast Event Probability: {title_prefix} "
            f"(Upper: {upper_threshold_label}, Lower: {lower_threshold_label})"
        ),
        fontsize=12,
        fontweight="bold",
    )

    ax = axes[0, 0]
    im = ax.imshow(prob_corrected_upper, cmap="viridis", vmin=0, vmax=1, origin="lower")
    ax.set_title(
        f"RF-corrected P(event > {upper_threshold_label}) (mean={np.nanmean(prob_corrected_upper):.3f})",
        fontsize=9,
    )
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    plt.colorbar(im, ax=ax, label="Probability")

    ax = axes[0, 1]
    im = ax.imshow(prob_baseline_upper, cmap="viridis", vmin=0, vmax=1, origin="lower")
    ax.set_title(
        f"Original P(event > {upper_threshold_label}) (mean={np.nanmean(prob_baseline_upper):.3f})",
        fontsize=9,
    )
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    plt.colorbar(im, ax=ax, label="Probability")

    ax = axes[1, 0]
    im = ax.imshow(prob_corrected_lower, cmap="viridis", vmin=0, vmax=1, origin="lower")
    ax.set_title(
        f"RF-corrected P(event < {lower_threshold_label}) (mean={np.nanmean(prob_corrected_lower):.3f})",
        fontsize=9,
    )
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    plt.colorbar(im, ax=ax, label="Probability")

    ax = axes[1, 1]
    im = ax.imshow(prob_baseline_lower, cmap="viridis", vmin=0, vmax=1, origin="lower")
    ax.set_title(
        f"Original P(event < {lower_threshold_label}) (mean={np.nanmean(prob_baseline_lower):.3f})",
        fontsize=9,
    )
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    plt.colorbar(im, ax=ax, label="Probability")

    plt.tight_layout()
    out_png = Path(out_dir) / f"{file_prefix}_skill_metrics.png"
    plt.savefig(str(out_png), bbox_inches="tight", dpi=150)
    out_paths.append(str(out_png))
    plt.close()

    return out_paths


def plot_example_time_means(
    baseline_ensemble,
    qm_ensemble,
    corrected_ensemble,
    reference,
    out_dir="plots",
    n_members_display=3,
    n_lead_months=3,
    variable_name="CWB",
):
    """
    Plot example time means over the first N lead months for first N members + ensemble mean.
    Compares original hindcast, QM hindcast, and RF-corrected hindcast.

    Args:
        baseline_ensemble: xr.DataArray shape (time, member, lat, lon) - original hindcasts
        qm_ensemble: xr.DataArray shape (time, member, lat, lon) - QM hindcasts (can be None)
        corrected_ensemble: xr.DataArray shape (time, member, lat, lon) - RF-corrected hindcasts
        reference: xr.DataArray shape (time, lat, lon) - reference data
        out_dir: output directory for plots
        n_members_display: number of members to display
        n_lead_months: number of lead months to average over
        variable_name: name of variable for plot titles

    Returns:
        list of written file paths
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_paths = []

    n_members = baseline_ensemble.sizes["member"]
    members_to_plot = min(n_members_display, n_members)

    # Compute time means over first N lead months
    baseline_mean = baseline_ensemble.isel(time=slice(0, n_lead_months)).mean(
        dim="time"
    )
    corrected_mean = corrected_ensemble.isel(time=slice(0, n_lead_months)).mean(
        dim="time"
    )
    reference_mean = reference.isel(time=slice(0, n_lead_months)).mean(dim="time")
    qm_mean = (
        qm_ensemble.isel(time=slice(0, n_lead_months)).mean(dim="time")
        if qm_ensemble is not None
        else None
    )

    # Land-sea mask from reference NaNs: True over land (valid), False over sea (NaN)
    land_mask = np.isfinite(reference_mean.values)

    baseline_mean = baseline_mean.where(land_mask)
    corrected_mean = corrected_mean.where(land_mask)
    reference_mean = reference_mean.where(land_mask)
    if qm_mean is not None:
        qm_mean = qm_mean.where(land_mask)

    # Determine number of columns: 3 if no QM, 4 if QM available
    n_cols = 4 if qm_mean is not None else 3
    n_rows = members_to_plot + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    title_suffix = " / QM" if qm_mean is not None else ""
    fig.suptitle(
        f"Time means (first {n_lead_months} months) — {variable_name}: Original / RF-corrected{title_suffix} / Reference",
        fontsize=12,
        fontweight="bold",
    )

    # Get common vmin/vmax for each column
    baseline_all = np.concatenate(
        [baseline_mean.isel(member=i).values.ravel() for i in range(members_to_plot)]
        + [baseline_mean.mean(dim="member").values.ravel()]
    )
    corrected_all = np.concatenate(
        [corrected_mean.isel(member=i).values.ravel() for i in range(members_to_plot)]
        + [corrected_mean.mean(dim="member").values.ravel()]
    )
    reference_all = reference_mean.values.ravel()

    baseline_vmin, baseline_vmax = np.nanpercentile(baseline_all, [2, 98])
    corrected_vmin, corrected_vmax = np.nanpercentile(corrected_all, [2, 98])
    reference_vmin, reference_vmax = np.nanpercentile(reference_all, [2, 98])

    if qm_mean is not None:
        qm_all = np.concatenate(
            [qm_mean.isel(member=i).values.ravel() for i in range(members_to_plot)]
            + [qm_mean.mean(dim="member").values.ravel()]
        )
        qm_vmin, qm_vmax = np.nanpercentile(qm_all, [2, 98])

    # Plot first N members
    for i in range(members_to_plot):
        row_label = f"Member {i}"

        # Original
        ax = axes[i, 0]
        im = ax.imshow(
            baseline_mean.isel(member=i),
            cmap="coolwarm",
            origin="lower",
            vmin=baseline_vmin,
            vmax=baseline_vmax,
        )
        ax.set_title(f"Original {row_label}")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        plt.colorbar(im, ax=ax, label=variable_name)

        # Corrected (RF)
        col_idx = 1
        ax = axes[i, col_idx]
        im = ax.imshow(
            corrected_mean.isel(member=i),
            cmap="coolwarm",
            origin="lower",
            vmin=corrected_vmin,
            vmax=corrected_vmax,
        )
        ax.set_title(f"RF-corrected {row_label}")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        plt.colorbar(im, ax=ax, label=variable_name)

        # QM (if available)
        if qm_mean is not None:
            ax = axes[i, 2]
            im = ax.imshow(
                qm_mean.isel(member=i),
                cmap="coolwarm",
                origin="lower",
                vmin=qm_vmin,
                vmax=qm_vmax,
            )
            ax.set_title(f"QM {row_label}")
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
            plt.colorbar(im, ax=ax, label=variable_name)

            # Reference
            ax = axes[i, 3]
            im = ax.imshow(
                reference_mean,
                cmap="coolwarm",
                origin="lower",
                vmin=reference_vmin,
                vmax=reference_vmax,
            )
            ax.set_title(f"Reference {row_label}")
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
            plt.colorbar(im, ax=ax, label=variable_name)
        else:
            # Reference
            ax = axes[i, 2]
            im = ax.imshow(
                reference_mean,
                cmap="coolwarm",
                origin="lower",
                vmin=reference_vmin,
                vmax=reference_vmax,
            )
            ax.set_title(f"Reference {row_label}")
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
            plt.colorbar(im, ax=ax, label=variable_name)

    # Plot ensemble mean row
    mean_row = members_to_plot
    ax = axes[mean_row, 0]
    im = ax.imshow(
        baseline_mean.mean(dim="member"),
        cmap="coolwarm",
        origin="lower",
        vmin=baseline_vmin,
        vmax=baseline_vmax,
    )
    ax.set_title("Original Ens. Mean")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    plt.colorbar(im, ax=ax, label=variable_name)

    col_idx = 1
    ax = axes[mean_row, col_idx]
    im = ax.imshow(
        corrected_mean.mean(dim="member"),
        cmap="coolwarm",
        origin="lower",
        vmin=corrected_vmin,
        vmax=corrected_vmax,
    )
    ax.set_title("RF-corrected Ens. Mean")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    plt.colorbar(im, ax=ax, label=variable_name)

    if qm_mean is not None:
        ax = axes[mean_row, 2]
        im = ax.imshow(
            qm_mean.mean(dim="member"),
            cmap="coolwarm",
            origin="lower",
            vmin=qm_vmin,
            vmax=qm_vmax,
        )
        ax.set_title("QM Ens. Mean")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        plt.colorbar(im, ax=ax, label=variable_name)

        ax = axes[mean_row, 3]
        im = ax.imshow(
            reference_mean,
            cmap="coolwarm",
            origin="lower",
            vmin=reference_vmin,
            vmax=reference_vmax,
        )
        ax.set_title("Reference Ens. Mean")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        plt.colorbar(im, ax=ax, label=variable_name)
    else:
        ax = axes[mean_row, 2]
        im = ax.imshow(
            reference_mean,
            cmap="coolwarm",
            origin="lower",
            vmin=reference_vmin,
            vmax=reference_vmax,
        )
        ax.set_title("Reference Ens. Mean")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        plt.colorbar(im, ax=ax, label=variable_name)

    plt.tight_layout()
    out_png = Path(out_dir) / f"example_time_means_{n_lead_months}months.png"
    plt.savefig(str(out_png), bbox_inches="tight", dpi=150)
    out_paths.append(str(out_png))
    plt.close()

    return out_paths
