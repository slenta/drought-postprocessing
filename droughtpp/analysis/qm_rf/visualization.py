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
