import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboardX import SummaryWriter
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import properscoring as ps

from tqdm import tqdm

plt.rcParams.update({"font.size": 16})


def crps_ensemble_grid(ensemble, reference):
    """
    Compute CRPS at each grid point for an ensemble vs. a reference time-series.

    Assumes `ensemble` shape (TIME, ENSEMBLE, LAT, LON) and `reference` shape (TIME, LAT, LON).

    Args:
        ensemble: np.ndarray or torch.Tensor, shape (time, ensemble, lat, lon)
        reference: np.ndarray or torch.Tensor, shape (time, lat, lon)

    Returns:
        np.ndarray of CRPS values with shape (time, lat, lon) or (lat, lon) if collapse_time.
    """
    # convert torch tensors to numpy
    if hasattr(ensemble, "detach"):
        ensemble = ensemble.detach().cpu().numpy()
    if hasattr(reference, "detach"):
        reference = reference.detach().cpu().numpy()
    ensemble = np.asarray(ensemble)
    reference = np.asarray(reference)

    if ensemble.ndim != 4:
        raise ValueError("ensemble must have shape (time, ensemble, lat, lon)")
    if reference.ndim != 3:
        raise ValueError("reference must have shape (time, lat, lon)")

    # reorder to properscoring expected shape (ensemble, time, lat, lon)
    forecasts = np.transpose(ensemble, (1, 0, 2, 3))

    # check shapes match
    if reference.shape != forecasts.shape[1:]:
        raise ValueError(
            f"reference shape {reference.shape} does not match forecasts time/spatial shape {forecasts.shape[1:]}"
        )

    # compute CRPS (axis=0 is ensemble axis in forecasts)
    crps = ps.crps_ensemble(reference, forecasts, axis=0)  # returns (time, lat, lon)

    return crps


def mae_per_member_grid(ensemble, reference):
    """
    Compute MAE per ensemble member and grid cell.

    Expects:
      - ensemble: shape (time, ensemble, lat, lon)
      - reference: shape (time, lat, lon)

    Returns:
      - np.ndarray shape (ensemble, lat, lon) with mean absolute error over time.
    """
    # convert torch tensors to numpy
    if hasattr(ensemble, "detach"):
        ensemble = ensemble.detach().cpu().numpy()
    if hasattr(reference, "detach"):
        reference = reference.detach().cpu().numpy()
    ensemble = np.asarray(ensemble)
    reference = np.asarray(reference)

    if ensemble.ndim != 4:
        raise ValueError("ensemble must have shape (time, ensemble, lat, lon)")
    if reference.ndim != 3:
        raise ValueError("reference must have shape (time, lat, lon)")

    ntime, nmem, nlat, nlon = ensemble.shape
    if reference.shape != (ntime, nlat, nlon):
        raise ValueError(
            f"reference shape {reference.shape} does not match ensemble time/spatial shape {(ntime, nlat, nlon)}"
        )

    # Broadcast reference to (time, ensemble, lat, lon) and compute abs error
    abs_err = np.abs(ensemble - reference[:, None, :, :])  # shape (time, mem, lat, lon)

    # Mean over time, ignoring NaNs
    mae = np.nanmean(abs_err, axis=0)  # shape (mem, lat, lon)

    return mae


def rmse_per_member_grid(ensemble, reference):
    """
    Compute RMSE per ensemble member and grid cell.

    Expects:
      - ensemble: shape (time, ensemble, lat, lon)
      - reference: shape (time, lat, lon)

    Returns:
      - np.ndarray shape (ensemble, lat, lon) with root mean squared error over time.
    """
    # convert torch tensors to numpy
    if hasattr(ensemble, "detach"):
        ensemble = ensemble.detach().cpu().numpy()
    if hasattr(reference, "detach"):
        reference = reference.detach().cpu().numpy()
    ensemble = np.asarray(ensemble)
    reference = np.asarray(reference)

    ntime, nmem, nlat, nlon = ensemble.shape

    # Broadcast reference to (time, ensemble, lat, lon) and compute squared error
    sq_err = (ensemble - reference[:, None, :, :]) ** 2  # (time, mem, lat, lon)

    # RMSE over time, ignoring NaNs
    rmse = np.sqrt(np.nanmean(sq_err, axis=0))  # (mem, lat, lon)

    return rmse


def mean_bias_per_member_grid(ensemble, reference):
    """
    Compute mean bias per ensemble member and grid cell.

    Bias is defined as forecast minus reference. Positive values indicate overestimation.

    Expects:
      - ensemble: shape (time, ensemble, lat, lon)
      - reference: shape (time, lat, lon)

    Returns:
      - np.ndarray shape (ensemble, lat, lon) with mean signed bias over time.
    """
    # convert torch tensors to numpy
    if hasattr(ensemble, "detach"):
        ensemble = ensemble.detach().cpu().numpy()
    if hasattr(reference, "detach"):
        reference = reference.detach().cpu().numpy()
    ensemble = np.asarray(ensemble)
    reference = np.asarray(reference)

    ntime, nmem, nlat, nlon = ensemble.shape

    # Broadcast reference and compute signed error
    bias = ensemble - reference[:, None, :, :]  # (time, mem, lat, lon)

    # Mean bias over time, ignoring NaNs
    mean_bias = np.nanmean(bias, axis=0)  # (mem, lat, lon)

    return mean_bias


def brier_score_grid(ensemble, reference, std_multiplier=1.0, extreme_type="upper"):
    """
    Compute Brier Score per grid cell from an ensemble and reference.

    Args:
        ensemble: np.ndarray shape (time, ensemble, lat, lon)
        reference: np.ndarray shape (time, lat, lon)
        std_multiplier: threshold = mean + std_multiplier * std
        extreme_type: "upper" for values above threshold, "lower" for values below threshold

    Returns:
        np.ndarray shape (lat, lon) - Brier Score values
    """
    # compute a threshold per grid cell from reference climatology
    # here: mean + std_multiplier * std (e.g., std_multiplier=1 => > mean+1*std)
    ref_mean = np.nanmean(reference, axis=0)  # (lat, lon)
    ref_std = np.nanstd(reference, axis=0)
    threshold = ref_mean + std_multiplier * ref_std  # (lat, lon)

    # observed binary time series: 1 when reference exceeds (or falls below) threshold, else 0
    if extreme_type == "upper":
        obs = (reference > threshold[None, :, :]).astype(float)  # (time, lat, lon)
        p = np.mean(ensemble > threshold[None, None, :, :], axis=1)  # (time, lat, lon)
    elif extreme_type == "lower":
        obs = (reference < threshold[None, :, :]).astype(float)  # (time, lat, lon)
        p = np.mean(ensemble < threshold[None, None, :, :], axis=1)  # (time, lat, lon)

    # probabilistic forecast: fraction of ensemble members above/below threshold

    # Brier Score per grid cell: mean over time of (p - o)^2
    # Lower BS is better (0 = perfect).
    bs = np.nanmean((p - obs) ** 2, axis=0)  # (lat, lon)
    return bs


def brier_skill_score_between_ensembles(
    ens_a,
    ens_b,
    reference,
    std_multiplier=1.0,
    extreme_type="upper",
):
    """
    Compute Brier Skill Score (BSS) per grid cell comparing two ensembles.

    Assumes shapes:
      - ens_a, ens_b: (time, ensemble, lat, lon)
      - reference: (time, lat, lon)

    Args:
        extreme_type: "upper" for upper tail extremes, "lower" for lower tail extremes

    Returns:
      - bss: np.ndarray shape (lat, lon)
      - bs_a: np.ndarray shape (lat, lon)
      - bs_b: np.ndarray shape (lat, lon)
    """
    # convert torch tensors to numpy if necessary
    if hasattr(ens_a, "detach"):
        ens_a = ens_a.detach().cpu().numpy()
    if hasattr(ens_b, "detach"):
        ens_b = ens_b.detach().cpu().numpy()
    if hasattr(reference, "detach"):
        reference = reference.detach().cpu().numpy()

    ens_a = np.asarray(ens_a)
    ens_b = np.asarray(ens_b)
    reference = np.asarray(reference)

    # Compute Brier Scores using the shared function
    bs_a = brier_score_grid(
        ens_a, reference, std_multiplier=std_multiplier, extreme_type=extreme_type
    )
    bs_b = brier_score_grid(
        ens_b, reference, std_multiplier=std_multiplier, extreme_type=extreme_type
    )

    # Brier Skill Score (BSS) comparing A to B:
    # BSS = 1 - BS_A / BS_B
    # - BSS > 0: ensemble A better than B
    # - BSS = 0: equal performance
    # - BSS < 0: ensemble A worse than B
    with np.errstate(divide="ignore", invalid="ignore"):
        bss = 1.0 - (bs_a / bs_b)
        # if baseline BS_b is effectively zero, the ratio is undefined — mark as NaN
        bss[np.isclose(bs_b, 0.0)] = np.nan

    return bss, bs_a, bs_b


def standardize_longitude(data, lons, lats=None):
    """
    Standardize longitude to -180-180 range and latitude to -90-90 range, reordering data accordingly.

    Args:
        data: np.ndarray with longitude as last dimension and latitude as second-to-last
        lons: 1D array of longitude values
        lats: 1D array of latitude values (optional)

    Returns:
        data_reordered: data with longitudes reordered to -180-180 and latitudes to -90-90
        lons_reordered: longitude array in -180 - 180 range
        lats_reordered: latitude array in -90-90 range (if lats provided)
    """
    # Convert lons to -180-180 range
    lons_180 = np.where(lons > 180, lons - 360, lons)

    # Get sorting indices for longitude
    lon_sort_idx = np.argsort(lons_180)

    # Reorder lons
    lons_reordered = lons_180[lon_sort_idx]

    # Reorder data along longitude axis (last axis)
    data_reordered = np.take(data, lon_sort_idx, axis=-1)

    # Handle latitude reordering if provided
    if lats is not None:
        # Convert lats to -90 to 90 range
        lats_90 = np.where(lats > 90, lats - 180, lats)

        # Get sorting indices for latitude
        lat_sort_idx = np.argsort(lats_90)

        # Reorder lats
        lats_reordered = lats_90[lat_sort_idx]

        # Reorder data along latitude axis (second-to-last axis)
        data_reordered = np.take(data_reordered, lat_sort_idx, axis=-2)

        return data_reordered, lons_reordered, lats_reordered

    return data_reordered, lons_reordered


def plot_gridwise_correlation(
    tensor1,
    tensor2,
    save_path,
    mask=None,
    lat=None,
    lon=None,
    title="Gridwise Correlation",
    plot=True,
):
    """
    Calculate and plot gridwise correlation between two tensors of shape (time, lat, lon).

    Args:
        tensor1: np.ndarray or torch.Tensor, shape (time, lat, lon)
        tensor2: np.ndarray or torch.Tensor, shape (time, lat, lon)
        mask: np.ndarray or torch.Tensor, shape (lat, lon), binary mask (optional)
        lat: 1D array of latitude values (optional, for axis labeling)
        lon: 1D array of longitude values (optional, for axis labeling)
        title: Title for the plot
        save_path: If provided, saves the figure to this path
    """
    # Convert to numpy if torch tensor
    if hasattr(tensor1, "detach"):
        tensor1 = tensor1.detach().cpu().numpy()
    if hasattr(tensor2, "detach"):
        tensor2 = tensor2.detach().cpu().numpy()
    if mask is not None and hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()

    time, nlat, nlon = tensor1.shape

    # Reshape to (time, lat*lon)
    t1_flat = np.nan_to_num(tensor1.reshape(time, nlat * nlon), 0)
    t2_flat = np.nan_to_num(tensor2.reshape(time, nlat * nlon), 0)

    # Calculate correlation for each grid cell
    corr = np.array(
        [np.corrcoef(t1_flat[:, i], t2_flat[:, i])[0, 1] for i in range(nlat * nlon)]
    )
    corr_grid = corr.reshape(nlat, nlon)

    # Plot
    if plot == True:
        plt.figure(figsize=(8, 6))
        im = plt.imshow(corr_grid, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)

        # Add mask overlay with small dots
        if mask is not None:
            y_coords = np.where(mask == 1)[1]
            x_coords = np.where(mask == 1)[2]
            plt.scatter(
                x_coords,
                y_coords,
                s=0.003,
                c="black",
                alpha=0.8,
                marker="o",
                linewidths=0.1,
            )

        plt.colorbar(im, label="Correlation")
        plt.title(title)
        plt.xlabel("Longitude" if lon is not None else "Grid X")
        plt.ylabel("Latitude" if lat is not None else "Grid Y")
        if lat is not None and lon is not None:
            plt.xticks(np.arange(len(lon)), np.round(lon, 2), rotation=90)
            plt.yticks(np.arange(len(lat)), np.round(lat, 2))
        plt.tight_layout()
        plt.savefig(f"{save_path}/images/{title}.png", bbox_inches="tight")
        plt.close()

    return corr_grid


def plot_gridwise_rmse(
    tensor1,
    tensor2,
    save_path,
    mask=None,
    lat=None,
    lon=None,
    title="Gridwise RMSE",
    plot=True,
):
    """
    Calculate and plot gridwise RMSE between two tensors of shape (time, lat, lon).

    Args:
        tensor1: np.ndarray or torch.Tensor, shape (time, lat, lon)
        tensor2: np.ndarray or torch.Tensor, shape (time, lat, lon)
        mask: np.ndarray or torch.Tensor, shape (lat, lon), binary mask (optional)
        lat: 1D array of latitude values (optional, for axis labeling)
        lon: 1D array of longitude values (optional, for axis labeling)
        title: Title for the plot
        save_path: If provided, saves the figure to this path
        plot: If True, creates and saves the plot

    Returns:
        rmse_grid: np.ndarray, shape (lat, lon) - RMSE values for each grid cell
    """
    # Convert to numpy if torch tensor
    if hasattr(tensor1, "detach"):
        tensor1 = tensor1.detach().cpu().numpy()
    if hasattr(tensor2, "detach"):
        tensor2 = tensor2.detach().cpu().numpy()
    if mask is not None and hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()

    time, nlat, nlon = tensor1.shape

    # Calculate squared differences
    squared_diff = (tensor1 - tensor2) ** 2

    # Calculate RMSE for each grid cell (mean over time axis)
    rmse_grid = np.sqrt(np.nanmean(squared_diff, axis=0))  # shape: (nlat, nlon)

    # Plot
    if plot == True:
        plt.figure(figsize=(8, 6))
        im = plt.imshow(rmse_grid, origin="lower", cmap="YlOrRd")

        # Add mask overlay with small dots
        if mask is not None:
            y_coords, x_coords = np.where(mask == 1)
            plt.scatter(
                x_coords,
                y_coords,
                s=0.003,
                c="black",
                alpha=0.8,
                marker="o",
                linewidths=0.1,
            )

        plt.colorbar(im, label="RMSE")
        plt.title(title)
        plt.xlabel("Longitude" if lon is not None else "Grid X")
        plt.ylabel("Latitude" if lat is not None else "Grid Y")
        if lat is not None and lon is not None:
            plt.xticks(np.arange(len(lon)), np.round(lon, 2), rotation=90)
            plt.yticks(np.arange(len(lat)), np.round(lat, 2))
        plt.tight_layout()
        plt.savefig(f"{save_path}/images/{title}.png", bbox_inches="tight", dpi=300)
        plt.close()

    return rmse_grid


def plot_ensemble_correlation_maps(
    corr_array, lat=None, lon=None, save_path=None, title="Ensemble Correlation Maps"
):
    """
    Plots an array of correlation maps with shape (n_ens_members, 3 or 4, nlat, nlon).
    The 3 maps per member are: gt_correlation, output_correlation, differences.
    If 4 maps provided, the 4th is used as a binary mask for stippling.

    Args:
        corr_array: np.ndarray, shape (n_ens_members, 3 or 4, nlat, nlon)
        lat: 1D array of latitude values (optional)
        lon: 1D array of longitude values (optional)
        save_path: Directory to save the figure (optional)
        title: Title for the figure
    """
    n_ens, n_maps, nlat, nlon = corr_array.shape

    # Keep only first 3 members and last member
    indices = list(range(3)) + [-1]
    corr_array = corr_array[indices]
    n_ens = len(indices)

    # Extract mask if n_maps == 4
    mask = None
    if n_maps == 4:
        mask = corr_array[:, 3, :, :]  # shape: (n_ens, nlat, nlon)
        corr_array = corr_array[:, :3, :, :]  # Only plot first 3 maps
        n_maps = 3

    map_titles = ["GT Correlation", "Output Correlation", "Diff: Out - GT"]

    # Create figure
    fig = plt.figure(figsize=(6 * n_maps, 3 * n_ens))
    fig.suptitle(title, fontsize=18)

    for i in range(n_ens):
        for j in range(n_maps):
            # Create subplot with cartopy projection
            ax = fig.add_subplot(
                n_ens, n_maps, i * n_maps + j + 1, projection=ccrs.PlateCarree()
            )

            im = ax.imshow(
                corr_array[i, j],
                origin="lower",
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                extent=[min(lon), max(lon), min(lat), max(lat)],
                transform=ccrs.PlateCarree(),
            )

            # Add coastlines
            ax.coastlines(linewidth=0.5, color="black")
            ax.set_extent(
                [min(lon), max(lon), min(lat), max(lat)], crs=ccrs.PlateCarree()
            )

            # Add mask overlay with small dots (use ensemble-specific mask)
            if mask is not None:
                y_coords, x_coords = np.where(mask[i] == 1)
                ax.scatter(
                    x_coords * 360 / nlon + min(lon),
                    y_coords * 180 / nlat + min(lat),
                    s=0.1,
                    c="black",
                    alpha=0.5,
                    marker=".",
                    transform=ccrs.PlateCarree(),
                )

            # Set title - use "Ensemble Mean" for last member
            member_label = "Ensemble Mean" if i == n_ens - 1 else f"Member {i+1}"
            ax.set_title(f"{member_label}: {map_titles[j]}")

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(
        f"{save_path}/images/{title.replace(' ', '_')}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_ensemble_rmse_maps(
    rmse_array, lat=None, lon=None, save_path=None, title="Ensemble RMSE Maps"
):
    """
    Plots an array of RMSE maps with shape (n_ens_members, 3 or 4, nlat, nlon).
    The 3 maps per member are: gt_rmse, output_rmse, differences.
    If 4 maps provided, the 4th is used as a binary mask for stippling.

    Args:
        rmse_array: np.ndarray, shape (n_ens_members, 3 or 4, nlat, nlon)
        lat: 1D array of latitude values (optional)
        lon: 1D array of longitude values (optional)
        save_path: Directory to save the figure (optional)
        title: Title for the figure
    """
    n_ens, n_maps, nlat, nlon = rmse_array.shape

    # Keep only first 3 members and last member
    indices = list(range(3)) + [-1]
    rmse_array = rmse_array[indices]
    n_ens = len(indices)

    # Extract mask if n_maps == 4
    mask = None
    if n_maps == 4:
        mask = rmse_array[:, 3, :, :]  # shape: (n_ens, nlat, nlon)
        rmse_array = rmse_array[:, :3, :, :]  # Only plot first 3 maps
        n_maps = 3

    map_titles = ["GT RMSE", "Output RMSE", "RMSE Diff: Out - GT"]

    # Create figure
    fig = plt.figure(figsize=(6 * n_maps, 3 * n_ens))
    fig.suptitle(title, fontsize=18)

    # Compute global vmin/vmax for consistent color scale
    vmin = np.nanmin(rmse_array[-1, :2])
    vmax = np.nanmax(rmse_array[-1, :2])
    vmin_diff = np.nanmin(rmse_array[-1, 2])
    vmax_diff = np.nanmax(rmse_array[-1, 2])

    for i in range(n_ens):
        for j in range(n_maps):
            # Create subplot with cartopy projection
            ax = fig.add_subplot(
                n_ens, n_maps, i * n_maps + j + 1, projection=ccrs.PlateCarree()
            )

            if j != 2:
                im = ax.imshow(
                    rmse_array[i, j],
                    origin="lower",
                    cmap="YlOrRd",
                    vmin=vmin,
                    vmax=vmax,
                    extent=[min(lon), max(lon), min(lat), max(lat)],
                    transform=ccrs.PlateCarree(),
                )
            elif j == 2:
                im = ax.imshow(
                    rmse_array[i, j],
                    origin="lower",
                    cmap="coolwarm",
                    vmin=vmin_diff,
                    vmax=vmax_diff,
                    extent=[min(lon), max(lon), min(lat), max(lat)],
                    transform=ccrs.PlateCarree(),
                )

            # Add coastlines
            ax.coastlines(linewidth=0.5, color="black")
            ax.set_extent(
                [min(lon), max(lon), min(lat), max(lat)], crs=ccrs.PlateCarree()
            )

            # Add mask overlay with small dots (use ensemble-specific mask)
            if mask is not None:
                y_coords, x_coords = np.where(mask[i] == 1)
                ax.scatter(
                    x_coords * 360 / nlon + min(lon),
                    y_coords * 180 / nlat + min(lat),
                    s=0.1,
                    c="black",
                    alpha=0.5,
                    marker=".",
                    transform=ccrs.PlateCarree(),
                )

            # Set title - use "Ensemble Mean" for last member
            member_label = "Ensemble Mean" if i == n_ens - 1 else f"Member {i+1}"
            ax.set_title(f"{member_label}: {map_titles[j]}")

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="RMSE")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(
        f"{save_path}/images/{title.replace(' ', '_')}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)
