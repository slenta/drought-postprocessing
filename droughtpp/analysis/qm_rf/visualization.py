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
    mae_qm,
    out_dir="qm_plots",
    n_members_display=3,
    metric_name="MAE",
    file_prefix=None,
):
    """
    Plot skill metrics in a single figure with 3 columns:
    Original / QM / Difference, for first N members and ensemble mean.

    Args:
        mae_orig: np.ndarray shape (ensemble, lat, lon) - metric values for original hindcasts
        mae_qm: np.ndarray shape (ensemble, lat, lon) - metric values for QM hindcasts
        out_dir: output directory for plots
        n_members_display: number of ensemble members to display (default 3)
        figsize: figure size
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

    # Compute ensemble mean
    mae_orig_mean = np.nanmean(mae_orig, axis=0)
    mae_qm_mean = np.nanmean(mae_qm, axis=0)

    # Create one figure: rows = first N members + ensemble mean, cols = Original/QM/Difference
    n_rows = members_to_plot + 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(35, 25))
    if n_rows == 1:
        axes = np.array([axes])

    fig.suptitle(
        f"{metric_name}: Original / QM / Difference", fontsize=14, fontweight="bold"
    )

    # Plot first N members
    for i in range(members_to_plot):
        row_label = f"Member {i}"

        # Original
        ax = axes[i, 0]
        im = ax.imshow(mae_orig[i], cmap="viridis", origin="lower")
        ax.set_title(f"Original {row_label} (mean={np.nanmean(mae_orig[i]):.4f})")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        plt.colorbar(im, ax=ax, label=str(metric_name))

        # QM
        ax = axes[i, 1]
        im = ax.imshow(mae_qm[i], cmap="viridis", origin="lower")
        ax.set_title(f"QM {row_label} (mean={np.nanmean(mae_qm[i]):.4f})")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        plt.colorbar(im, ax=ax, label=str(metric_name))

        # Difference
        ax = axes[i, 2]
        mae_diff = mae_qm[i] - mae_orig[i]
        max_abs = np.nanmax(np.abs(mae_diff))
        if not np.isfinite(max_abs) or max_abs == 0:
            max_abs = 1e-12
        im = ax.imshow(
            mae_diff, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, origin="lower"
        )
        ax.set_title(f"Δ{metric_name} {row_label} (mean={np.nanmean(mae_diff):.4f})")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        plt.colorbar(im, ax=ax, label=f"Δ{metric_name}")

    # Plot ensemble mean row
    mean_row = members_to_plot
    ax = axes[mean_row, 0]
    im = ax.imshow(mae_orig_mean, cmap="viridis", origin="lower")
    ax.set_title(f"Original Ens. Mean (mean={np.nanmean(mae_orig_mean):.4f})")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    plt.colorbar(im, ax=ax, label=str(metric_name))

    ax = axes[mean_row, 1]
    im = ax.imshow(mae_qm_mean, cmap="viridis", origin="lower")
    ax.set_title(f"QM Ens. Mean (mean={np.nanmean(mae_qm_mean):.4f})")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    plt.colorbar(im, ax=ax, label=str(metric_name))

    # Ensemble mean difference
    ax = axes[mean_row, 2]
    mae_diff_mean = mae_qm_mean - mae_orig_mean
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
    ax.set_title(f"Δ{metric_name} Ens. Mean (mean={np.nanmean(mae_diff_mean):.4f})")
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
):
    """
    Plot BSS maps for upper and lower tail extremes.

    Args:
        bss_upper, bss_lower: np.ndarray shape (lat, lon) - BSS for upper/lower tails
        out_dir: output directory for plots
        figsize: figure size
        upper_threshold_label: label describing the upper-extreme threshold
        lower_threshold_label: label describing the lower-extreme threshold

    Returns:
        list of written file paths
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_paths = []

    # Create figure: Upper and lower tail BSS spatial maps
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(
        (
            "Brier Skill Score: QM vs. Original "
            f"(Upper threshold: {upper_threshold_label}, "
            f"Lower threshold: {lower_threshold_label})"
        ),
        fontsize=14,
        fontweight="bold",
    )

    # Upper tail BSS
    ax = axes[0]
    im = ax.imshow(bss_upper, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    ax.set_title(
        f"BSS Upper [{upper_threshold_label}] (mean={np.nanmean(bss_upper):.4f})"
    )
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    plt.colorbar(im, ax=ax, label="BSS")

    # Lower tail BSS
    ax = axes[1]
    im = ax.imshow(bss_lower, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    ax.set_title(
        f"BSS Lower [{lower_threshold_label}] (mean={np.nanmean(bss_lower):.4f})"
    )
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    plt.colorbar(im, ax=ax, label="BSS")

    plt.tight_layout()
    out_png = Path(out_dir) / "bss_skill_metrics.png"
    plt.savefig(str(out_png), bbox_inches="tight", dpi=150)
    out_paths.append(str(out_png))
    plt.close()

    return out_paths
