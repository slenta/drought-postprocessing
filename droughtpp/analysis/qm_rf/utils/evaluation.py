import json
from pathlib import Path
from functools import reduce

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from IPython import embed

from droughtpp.analysis.qm_rf.utils.visualization import (
    plot_bss_skill_metrics,
    plot_event_bss_comparison_maps,
    plot_mae_skill_metrics,
    plot_qm_distributions,
    plot_ensemble_timeseries,
    plot_ml_eval_summary_maps,
    plot_example_time_means,
)


def compute_qm_distributions(
    hind_subsets,
    qm_subsets,
    ref_subset,
):
    """
    For each hindcast/QM pair in the pre-aligned subsets, collect flattened numpy arrays:
      - original hindcast values ('orig')
      - quantile-mapped values ('qm')
      - reference values ('obs')

    Args:
        hind_subsets: list of pre-aligned, pre-subset xarray DataArrays (hindcasts)
        qm_subsets: list of pre-aligned, pre-subset xarray DataArrays (QM)
        ref_subset: pre-aligned, pre-subset xarray DataArray (reference)

    Returns a dict keyed by file-stem with keys: orig, qm, obs, file, qm_file.
    """

    dist_by_hind = {}
    for i, (hind_da, qm_da) in tqdm(
        enumerate(zip(hind_subsets, qm_subsets)),
        desc="Collecting distributions",
        total=len(hind_subsets),
    ):
        # Use index as stem since we don't have filenames
        stem = f"member_{i:02d}"

        ref_vals = ref_subset.values.ravel()
        hind_vals = hind_da.values.ravel()
        qm_vals = qm_da.values.ravel()

        dist_by_hind[stem] = {
            "orig": hind_vals,
            "qm": qm_vals,
            "obs": ref_vals,
            "file": f"hindcast_{i}",
            "qm_file": f"qm_{i}",
        }

    return dist_by_hind


def compute_pairwise_distributions(
    baseline_ensemble,
    primary_ensemble,
    reference,
    qm_ensemble=None,
):
    """
    Build flattened per-member distributions for histogram comparison plots.

    Args:
        baseline_ensemble: np.ndarray (time, member, lat, lon)
        primary_ensemble: np.ndarray (time, member, lat, lon)
        reference: np.ndarray (time, lat, lon)
        qm_ensemble: optional np.ndarray (time, member, lat, lon)

    Returns:
        dict keyed by member name with entries containing orig, corrected, obs,
        and optionally qm arrays.
    """
    baseline_np = np.asarray(baseline_ensemble)
    primary_np = np.asarray(primary_ensemble)
    reference_np = np.asarray(reference)
    qm_np = np.asarray(qm_ensemble) if qm_ensemble is not None else None

    n_members = baseline_np.shape[1]
    dist_by_member = {}
    for i in range(n_members):
        entry = {
            "orig": baseline_np[:, i, :, :].ravel(),
            "corrected": primary_np[:, i, :, :].ravel(),
            "obs": reference_np.ravel(),
        }
        if qm_np is not None and i < qm_np.shape[1]:
            entry["qm"] = qm_np[:, i, :, :].ravel()
        dist_by_member[f"member_{i:02d}"] = entry

    return dist_by_member


def mae_per_member_grid(ensemble, reference):
    """
    Compute MAE per ensemble member and grid cell.

    Expects:
      - ensemble: shape (time, ensemble, lat, lon)
      - reference: shape (time, lat, lon)

    Returns:
      - np.ndarray shape (ensemble, lat, lon) with mean absolute error over time.
    """
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

    abs_err = np.abs(ensemble - reference[:, None, :, :])
    return np.nanmean(abs_err, axis=0)


def rmse_per_member_grid(ensemble, reference):
    """
    Compute RMSE per ensemble member and grid cell.

    Expects:
      - ensemble: shape (time, ensemble, lat, lon)
      - reference: shape (time, lat, lon)

    Returns:
      - np.ndarray shape (ensemble, lat, lon) with root mean squared error over time.
    """
    if hasattr(ensemble, "detach"):
        ensemble = ensemble.detach().cpu().numpy()
    if hasattr(reference, "detach"):
        reference = reference.detach().cpu().numpy()
    ensemble = np.asarray(ensemble)
    reference = np.asarray(reference)

    ntime, nmem, nlat, nlon = ensemble.shape
    if reference.shape != (ntime, nlat, nlon):
        raise ValueError(
            f"reference shape {reference.shape} does not match ensemble time/spatial shape {(ntime, nlat, nlon)}"
        )

    sq_err = (ensemble - reference[:, None, :, :]) ** 2
    return np.sqrt(np.nanmean(sq_err, axis=0))


def mean_bias_per_member_grid(ensemble, reference):
    """
    Compute mean signed bias per ensemble member and grid cell.

    Bias is forecast minus reference.
    """
    if hasattr(ensemble, "detach"):
        ensemble = ensemble.detach().cpu().numpy()
    if hasattr(reference, "detach"):
        reference = reference.detach().cpu().numpy()
    ensemble = np.asarray(ensemble)
    reference = np.asarray(reference)

    ntime, nmem, nlat, nlon = ensemble.shape
    if reference.shape != (ntime, nlat, nlon):
        raise ValueError(
            f"reference shape {reference.shape} does not match ensemble time/spatial shape {(ntime, nlat, nlon)}"
        )

    bias = ensemble - reference[:, None, :, :]
    return np.nanmean(bias, axis=0)


def brier_score_grid(ensemble, reference, std_multiplier=1.0, extreme_type="upper"):
    """
    Compute Brier Score per grid cell from an ensemble and reference.
    """
    ref_mean = np.nanmean(reference, axis=0)
    ref_std = np.nanstd(reference, axis=0)
    threshold = ref_mean + std_multiplier * ref_std

    if extreme_type == "upper":
        obs = (reference > threshold[None, :, :]).astype(float)
        p = np.mean(ensemble > threshold[None, None, :, :], axis=1)
    elif extreme_type == "lower":
        obs = (reference < threshold[None, :, :]).astype(float)
        p = np.mean(ensemble < threshold[None, None, :, :], axis=1)
    else:
        raise ValueError("extreme_type must be either 'upper' or 'lower'")

    return np.nanmean((p - obs) ** 2, axis=0)


def brier_skill_score_between_ensembles(
    ens_a,
    ens_b,
    reference,
    std_multiplier=1.0,
    extreme_type="upper",
):
    """
    Compute Brier Skill Score (BSS) per grid cell comparing two ensembles.
    """
    if hasattr(ens_a, "detach"):
        ens_a = ens_a.detach().cpu().numpy()
    if hasattr(ens_b, "detach"):
        ens_b = ens_b.detach().cpu().numpy()
    if hasattr(reference, "detach"):
        reference = reference.detach().cpu().numpy()

    ens_a = np.asarray(ens_a)
    ens_b = np.asarray(ens_b)
    reference = np.asarray(reference)

    bs_a = brier_score_grid(
        ens_a,
        reference,
        std_multiplier=std_multiplier,
        extreme_type=extreme_type,
    )
    bs_b = brier_score_grid(
        ens_b,
        reference,
        std_multiplier=std_multiplier,
        extreme_type=extreme_type,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        bss = 1.0 - (bs_a / bs_b)
        bss[np.isclose(bs_b, 0.0)] = np.nan

    return bss, bs_a, bs_b


def load_paths_from_json(json_path: Path) -> list[Path]:
    with open(json_path, "r") as fh:
        return [Path(path) for path in json.load(fh)]


def ensure_spatial_dims(data_array: xr.DataArray) -> xr.DataArray:
    rename_map = {}
    if "lat" in data_array.dims:
        rename_map["lat"] = "latitude"
    if "lon" in data_array.dims:
        rename_map["lon"] = "longitude"
    if rename_map:
        data_array = data_array.rename(rename_map)
    return data_array


def select_eval_years(data_array: xr.DataArray, eval_years) -> xr.DataArray:

    years = pd.to_datetime(data_array["time"].values).year
    return data_array.isel(time=np.isin(years, eval_years))


def compute_threshold_grid_from_reference(
    reference_da: xr.DataArray,
    eval_years,
    percentile: float,
) -> np.ndarray:
    years = pd.to_datetime(reference_da["time"].values).year
    train_mask = ~np.isin(years, np.asarray(eval_years, dtype=int))

    if np.any(train_mask):
        reference_train = reference_da.isel(time=train_mask).values
    else:
        reference_train = reference_da.values

    return np.nanpercentile(reference_train, percentile, axis=0)


def build_mixed_group_labels(
    stacked_da: xr.DataArray,
    years: np.ndarray,
    group_by: str,
    X_full: pd.DataFrame | None = None,
    quantile_group_predictor: str | None = None,
    quantile_group_n: int | None = None,
) -> np.ndarray:
    mode = str(group_by).lower()

    if mode == "year":
        return np.asarray(years).astype(str)

    if mode == "calendar_month":
        months = pd.to_datetime(stacked_da.coords["time"].values).month
        return np.asarray(months).astype(str)

    if mode == "member":
        return np.asarray(stacked_da.coords["member"].values).astype(str)

    if mode == "predictor_quantile":
        predictor = str(quantile_group_predictor or "").strip()

        n_quantiles = int(quantile_group_n if quantile_group_n is not None else 10)
        n_quantiles = max(2, n_quantiles)

        values = pd.to_numeric(X_full[predictor], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(values)
        labels = np.full(values.shape[0], f"{predictor}_qnan", dtype=object)

        if not np.any(finite_mask):
            return labels.astype(str)

        unique_finite = np.unique(values[finite_mask])
        q_effective = min(n_quantiles, int(unique_finite.size))

        if q_effective < 2:
            labels[finite_mask] = f"{predictor}_q0"
            return labels.astype(str)

        try:
            bin_ids = pd.qcut(
                values[finite_mask],
                q=q_effective,
                labels=False,
                duplicates="drop",
            )
            labels[finite_mask] = np.char.add(
                f"{predictor}_q",
                np.asarray(bin_ids).astype(int).astype(str),
            )
        except ValueError:
            labels[finite_mask] = f"{predictor}_q0"

        return labels.astype(str)

    lat = np.asarray(stacked_da.coords["latitude"].values)
    lon = np.asarray(stacked_da.coords["longitude"].values)

    if mode == "grid_id":
        return np.char.add(
            np.round(lat, 6).astype(str), np.char.add("_", np.round(lon, 6).astype(str))
        )

    if mode == "grid_year":
        grid = np.char.add(
            np.round(lat, 6).astype(str), np.char.add("_", np.round(lon, 6).astype(str))
        )
        return np.char.add(grid, np.char.add("_", np.asarray(years).astype(str)))


def select_knn_feature_frame(
    X_full: pd.DataFrame,
    knn_feature_columns: list[str] | None,
) -> pd.DataFrame:
    if not knn_feature_columns:
        return X_full

    requested = [str(col) for col in knn_feature_columns]
    missing = [col for col in requested if col not in X_full.columns]
    if missing:
        available = ", ".join(X_full.columns.astype(str).tolist())
        raise ValueError(
            "ml_arguments.knn_feature_columns contains unknown feature(s): "
            f"{missing}. Available features: [{available}]"
        )

    return X_full.loc[:, requested]


def build_knn_neighbor_features(
    stacked_da: xr.DataArray,
    X_full: pd.DataFrame,
    group_by: str,
    knn_feature_columns: list[str] | None = None,
) -> np.ndarray | None:
    mode = str(group_by).lower()
    if mode == "knn_spatial":
        if "latitude" not in stacked_da.coords or "longitude" not in stacked_da.coords:
            raise ValueError("knn_spatial requires latitude/longitude coordinates")
        lat = np.asarray(stacked_da.coords["latitude"].values, dtype=float)
        lon = np.asarray(stacked_da.coords["longitude"].values, dtype=float)
        return np.column_stack([lat, lon])
    if mode in {"knn_feature", "knn_proximity"}:
        X_knn = select_knn_feature_frame(X_full, knn_feature_columns)
        return np.asarray(X_knn.values, dtype=float)
    return None


def load_eval_data(path: Path, var_name: str) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        data_array = ds[var_name].load()

    data_array = ensure_spatial_dims(data_array)

    return data_array.dropna("time", how="all")


def count_extreme_drought_events(
    ensemble,
    reference,
    threshold_lower,
    comparison: str = "below",
):
    """
    Count reference drought events, correctly hit droughts, and wrong drought predictions.

    Args:
        ensemble: np.ndarray (time, member, lat, lon)
        reference: np.ndarray (time, lat, lon)
        threshold_lower: np.ndarray (lat, lon)

    Returns:
        tuple[int, int, int]: (reference_count, hit_count, wrong_count)
    """
    ensemble = np.asarray(ensemble)
    reference = np.asarray(reference)
    threshold_lower = np.asarray(threshold_lower)

    if comparison == "below":
        reference_drought = reference < threshold_lower[None, :, :]
        predicted_drought = ensemble < threshold_lower[None, None, :, :]
    elif comparison == "above":
        reference_drought = reference > threshold_lower[None, :, :]
        predicted_drought = ensemble > threshold_lower[None, None, :, :]

    valid = (
        np.isfinite(ensemble)
        & np.isfinite(reference)[:, None, :, :]
        & np.isfinite(threshold_lower)[None, None, :, :]
    )
    reference_drought_expanded = reference_drought[:, None, :, :]

    reference_count = int(np.count_nonzero(reference_drought_expanded & valid))
    hit_count = int(
        np.count_nonzero(predicted_drought & reference_drought_expanded & valid)
    )
    wrong_count = int(
        np.count_nonzero(predicted_drought & (~reference_drought_expanded) & valid)
    )

    return reference_count, hit_count, wrong_count


def count_binary_event_outcomes(
    event_probability,
    reference_event_binary,
    probability_threshold: float = 0.5,
):
    """
    Count event outcomes from a probability field and a reference event mask.

    Args:
        event_probability: np.ndarray (time, lat, lon), event probabilities in [0, 1]
        reference_event_binary: np.ndarray (time, lat, lon), reference event mask
        probability_threshold: threshold to convert probability into binary event

    Returns:
        tuple: (
            event_binary, valid_mask,
            reference_count, hit_count, wrong_count,
        )
    """
    prob = np.asarray(event_probability)
    ref = np.asarray(reference_event_binary)
    ref_binary = ref >= 0.5

    event_binary = prob >= float(probability_threshold)
    valid_mask = np.isfinite(prob) & np.isfinite(ref)

    reference_count = int(np.count_nonzero(ref_binary & valid_mask))
    hit_count = int(np.count_nonzero(event_binary & ref_binary & valid_mask))
    wrong_count = int(np.count_nonzero(event_binary & (~ref_binary) & valid_mask))

    return event_binary, valid_mask, reference_count, hit_count, wrong_count


def compute_gridcell_drought_hit_rate_percent(
    ensemble,
    reference,
    threshold_lower,
    comparison: str = "below",
):
    """
    Compute per-gridcell percentage of correctly predicted drought events.

    Args:
        ensemble: np.ndarray (time, member, lat, lon)
        reference: np.ndarray (time, lat, lon)
        threshold_lower: np.ndarray (lat, lon)

    Returns:
        np.ndarray (lat, lon): hit-rate percentage in [0, 100], NaN where no
        reference drought events are available.
    """
    ensemble = np.asarray(ensemble)
    reference = np.asarray(reference)
    threshold_lower = np.asarray(threshold_lower)

    if comparison == "below":
        reference_drought = reference < threshold_lower[None, :, :]
        predicted_drought = ensemble < threshold_lower[None, None, :, :]
    elif comparison == "above":
        reference_drought = reference > threshold_lower[None, :, :]
        predicted_drought = ensemble > threshold_lower[None, None, :, :]

    valid = (
        np.isfinite(ensemble)
        & np.isfinite(reference)[:, None, :, :]
        & np.isfinite(threshold_lower)[None, None, :, :]
    )
    reference_drought_expanded = reference_drought[:, None, :, :]

    hit_count = np.sum(
        predicted_drought & reference_drought_expanded & valid, axis=(0, 1)
    )
    reference_count = np.sum(reference_drought_expanded & valid, axis=(0, 1))

    hit_rate = np.full(reference_count.shape, np.nan, dtype=float)
    has_events = reference_count > 0
    hit_rate[has_events] = (hit_count[has_events] / reference_count[has_events]) * 100.0

    return hit_rate


def intersect_time_coordinates(data_arrays) -> np.ndarray:
    common_time = reduce(
        np.intersect1d,
        [np.asarray(data_array["time"].values) for data_array in data_arrays],
    )
    if len(common_time) == 0:
        raise ValueError("No common time steps found across evaluation inputs.")
    return common_time


def brier_skill_score_between_ensembles_percentile(
    ens_a,
    ens_b,
    reference,
    upper_percentile: float = 90.0,
    lower_percentile: float = 10.0,
):
    ens_a = np.asarray(ens_a)
    ens_b = np.asarray(ens_b)
    reference = np.asarray(reference)

    threshold_upper = np.nanpercentile(reference, upper_percentile, axis=0)
    threshold_lower = np.nanpercentile(reference, lower_percentile, axis=0)

    obs_upper = (reference > threshold_upper[None, :, :]).astype(float)
    prob_a_upper = np.mean(ens_a > threshold_upper[None, None, :, :], axis=1)
    prob_b_upper = np.mean(ens_b > threshold_upper[None, None, :, :], axis=1)
    bs_a_upper = np.nanmean((prob_a_upper - obs_upper) ** 2, axis=0)
    bs_b_upper = np.nanmean((prob_b_upper - obs_upper) ** 2, axis=0)

    obs_lower = (reference < threshold_lower[None, :, :]).astype(float)
    prob_a_lower = np.mean(ens_a < threshold_lower[None, None, :, :], axis=1)
    prob_b_lower = np.mean(ens_b < threshold_lower[None, None, :, :], axis=1)
    bs_a_lower = np.nanmean((prob_a_lower - obs_lower) ** 2, axis=0)
    bs_b_lower = np.nanmean((prob_b_lower - obs_lower) ** 2, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        bss_upper = 1.0 - (bs_a_upper / bs_b_upper)
        bss_upper[np.isclose(bs_b_upper, 0.0)] = np.nan

        bss_lower = 1.0 - (bs_a_lower / bs_b_lower)
        bss_lower[np.isclose(bs_b_lower, 0.0)] = np.nan

    return (
        bss_upper,
        bss_lower,
        bs_a_upper,
        bs_b_upper,
        bs_a_lower,
        bs_b_lower,
    )


def brier_skill_score_between_ensembles_threshold(
    ens_a,
    ens_b,
    reference,
    threshold: float = 0.0,
    comparison: str = "above",
):
    ens_a = np.asarray(ens_a)
    ens_b = np.asarray(ens_b)
    reference = np.asarray(reference)

    if comparison == "above":
        obs = (reference > threshold).astype(float)
        prob_a = np.mean(ens_a > threshold, axis=1)
        prob_b = np.mean(ens_b > threshold, axis=1)
    elif comparison == "below":
        obs = (reference < threshold).astype(float)
        prob_a = np.mean(ens_a < threshold, axis=1)
        prob_b = np.mean(ens_b < threshold, axis=1)
    else:
        raise ValueError("comparison must be either 'below' or 'above'")

    bs_a = np.nanmean((prob_a - obs) ** 2, axis=0)
    bs_b = np.nanmean((prob_b - obs) ** 2, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        bss = 1.0 - (bs_a / bs_b)
        bss[np.isclose(bs_b, 0.0)] = np.nan

    return bss, bs_a, bs_b


def _infer_lat_lon_names(da):
    lat_name = "latitude" if "latitude" in da.dims else "lat"
    lon_name = "longitude" if "longitude" in da.dims else "lon"

    if lat_name not in da.dims or lon_name not in da.dims:
        raise ValueError(f"Could not infer latitude/longitude dims from {da.dims}")

    return lat_name, lon_name


def _monthly_anomalies(data_array: xr.DataArray) -> xr.DataArray:
    monthly_climatology = data_array.groupby("time.month").mean("time")
    return data_array.groupby("time.month") - monthly_climatology


def _gridwise_correlation(ensemble_np, reference_np):
    n_time, n_members, n_lat, n_lon = ensemble_np.shape
    correlation = np.full((n_members, n_lat * n_lon), np.nan, dtype=float)
    reference_flat = reference_np.reshape(n_time, -1)

    for member_idx in range(n_members):
        member_flat = ensemble_np[:, member_idx].reshape(n_time, -1)
        for cell_idx in range(n_lat * n_lon):
            member_values = member_flat[:, cell_idx]
            reference_values = reference_flat[:, cell_idx]
            valid = np.isfinite(member_values) & np.isfinite(reference_values)
            if np.count_nonzero(valid) > 1:
                member_valid = member_values[valid]
                reference_valid = reference_values[valid]
                if np.nanstd(member_valid) > 0 and np.nanstd(reference_valid) > 0:
                    correlation[member_idx, cell_idx] = np.corrcoef(
                        member_valid,
                        reference_valid,
                    )[0, 1]

    return correlation.reshape(n_members, n_lat, n_lon)


def _gridwise_correlation_with_pvalues(ensemble_np, reference_np):
    from scipy.stats import pearsonr

    n_time, n_members, n_lat, n_lon = ensemble_np.shape
    correlation = np.full((n_members, n_lat * n_lon), np.nan, dtype=float)
    pvalues = np.full((n_members, n_lat * n_lon), np.nan, dtype=float)
    reference_flat = reference_np.reshape(n_time, -1)

    for member_idx in range(n_members):
        member_flat = ensemble_np[:, member_idx].reshape(n_time, -1)
        for cell_idx in range(n_lat * n_lon):
            member_values = member_flat[:, cell_idx]
            reference_values = reference_flat[:, cell_idx]
            valid = np.isfinite(member_values) & np.isfinite(reference_values)
            if np.count_nonzero(valid) > 1:
                member_valid = member_values[valid]
                reference_valid = reference_values[valid]
                if np.nanstd(member_valid) > 0 and np.nanstd(reference_valid) > 0:
                    r_value, p_value = pearsonr(member_valid, reference_valid)
                    correlation[member_idx, cell_idx] = r_value
                    pvalues[member_idx, cell_idx] = p_value

    return correlation.reshape(n_members, n_lat, n_lon), pvalues.reshape(n_members, n_lat, n_lon)


def _std_over_rmse_maps(std_map: xr.DataArray, rmse_member: np.ndarray) -> np.ndarray:
    std_values = np.asarray(std_map.values, dtype=float)
    ratio_members = []
    for member_rmse in np.asarray(rmse_member, dtype=float):
        ratio = np.full_like(member_rmse, np.nan, dtype=float)
        valid = np.isfinite(member_rmse) & (member_rmse != 0)
        ratio[valid] = std_values[valid] / member_rmse[valid]
        ratio_members.append(ratio)
    return np.stack(ratio_members, axis=0)


def _load_var(path, var_name):
    with xr.open_dataset(path, decode_times=False) as ds:
        if var_name not in ds:
            raise KeyError(f"{var_name} not found in {path}")
        return ds[var_name].load()


def _to_time_member_lat_lon(da, member_dim):
    lat_name, lon_name = _infer_lat_lon_names(da)

    if "time" not in da.dims:
        raise ValueError("Expected a time dimension in data.")

    if member_dim not in da.dims:
        da = da.expand_dims({member_dim: [0]})

    return da.transpose("time", member_dim, lat_name, lon_name)


def _to_time_lat_lon(da):
    lat_name, lon_name = _infer_lat_lon_names(da)

    if "time" not in da.dims:
        raise ValueError("Expected a time dimension in data.")

    return da.transpose("time", lat_name, lon_name)


def prepare_pairwise_skill_evaluation(
    ml_members,
    baseline_members,
    reference_members,
    eval_years=None,
    qm_members=None,
    ml_label="ML-corrected",
    baseline_label="Original",
    std_multiplier: float = 1.0,
):
    if eval_years is None:
        eval_years = []

    common_time = intersect_time_coordinates(
        ml_members + baseline_members + reference_members + (qm_members if qm_members else [])
    )

    ml_members = [member.sel(time=common_time) for member in ml_members]
    baseline_members = [member.sel(time=common_time) for member in baseline_members]
    reference_members = [member.sel(time=common_time) for member in reference_members]
    if qm_members:
        qm_members = [member.sel(time=common_time) for member in qm_members]
    ml_ensemble = xr.concat(ml_members, dim="member").transpose(
        "time", "member", "latitude", "longitude"
    )
    baseline_ensemble = xr.concat(baseline_members, dim="member").transpose(
        "time", "member", "latitude", "longitude"
    )
    reference = reference_members[0].transpose("time", "latitude", "longitude")

    ml_np = ml_ensemble.values
    baseline_np = baseline_ensemble.values
    reference_np = reference.values

    result = {
        "common_time": common_time,
        "ml_ensemble": ml_ensemble,
        "baseline_ensemble": baseline_ensemble,
        "reference": reference,
        "ml_np": ml_np,
        "baseline_np": baseline_np,
        "reference_np": reference_np,
        "ml_label": ml_label,
        "baseline_label": baseline_label,
    }

    result["mae_ml_member"] = mae_per_member_grid(ml_np, reference_np)
    result["mae_baseline_member"] = mae_per_member_grid(baseline_np, reference_np)
    result["rmse_ml_member"] = rmse_per_member_grid(ml_np, reference_np)
    result["rmse_baseline_member"] = rmse_per_member_grid(baseline_np, reference_np)
    result["mean_bias_ml_member"] = np.nanmean(ml_np - reference_np[:, None, :, :], axis=0)
    result["mean_bias_baseline_member"] = np.nanmean(
        baseline_np - reference_np[:, None, :, :], axis=0
    )

    (
        bss_upper,
        bss_lower,
        bs_ml_upper,
        bs_baseline_upper,
        bs_ml_lower,
        bs_baseline_lower,
    ) = brier_skill_score_between_ensembles_percentile(
        ml_np,
        baseline_np,
        reference_np,
        upper_percentile=90.0,
        lower_percentile=10.0,
    )
    result["bss_upper"] = bss_upper
    result["bss_lower"] = bss_lower
    result["bss_p90_upper"], result["bss_p10_lower"] = (
        result["bss_upper"],
        result["bss_lower"],
    )
    result["bs_ml_upper"] = bs_ml_upper
    result["bs_baseline_upper"] = bs_baseline_upper
    result["bs_ml_lower"] = bs_ml_lower
    result["bs_baseline_lower"] = bs_baseline_lower

    if qm_members:
        qm_ensemble = xr.concat(qm_members, dim="member").transpose(
            "time", "member", "latitude", "longitude"
        )
        qm_np = qm_ensemble.values
        result["qm_ensemble"] = qm_ensemble
        result["qm_np"] = qm_np
        result["mae_qm_member"] = mae_per_member_grid(qm_np, reference_np)
        result["rmse_qm_member"] = rmse_per_member_grid(qm_np, reference_np)
        result["mean_bias_qm_member"] = np.nanmean(
            qm_np - reference_np[:, None, :, :], axis=0
        )
        # Use percentile-based extremes (P90/P10) for event BSS comparisons
        (
            result["bss_ml_vs_qm_upper"],
            result["bss_ml_vs_qm_lower"],
            _,
            _,
            _,
            _,
        ) = brier_skill_score_between_ensembles_percentile(
            ml_np,
            qm_np,
            reference_np,
            upper_percentile=90.0,
            lower_percentile=10.0,
        )

        (
            result["bss_qm_vs_baseline_upper"],
            result["bss_qm_vs_baseline_lower"],
            _,
            _,
            _,
            _,
        ) = brier_skill_score_between_ensembles_percentile(
            qm_np,
            baseline_np,
            reference_np,
            upper_percentile=90.0,
            lower_percentile=10.0,
        )

    return result


def plot_pairwise_skill_evaluation(
    plot_dir,
    skill,
    reference_all=None,
    variable_name=None,
    include_timeseries=False,
    include_example_time_means=False,
    include_difference_examples=False,
    land_mask_path=None,
    include_bss_maps=True,
    include_anomalies: bool = False,
    latitude=None,
    longitude=None,
):

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    mae_qm_member = skill.get("mae_qm_member")
    rmse_qm_member = skill.get("rmse_qm_member")
    mean_bias_qm_member = skill.get("mean_bias_qm_member")
    qm_ensemble = skill.get("qm_ensemble")
    qm_np = skill.get("qm_np")

    mae_reference_member = np.zeros_like(skill["mae_baseline_member"])
    rmse_reference_member = np.zeros_like(skill["rmse_baseline_member"])
    mean_bias_reference_member = np.zeros_like(skill["mean_bias_baseline_member"])

    distributions = compute_pairwise_distributions(
        baseline_ensemble=skill["baseline_np"],
        primary_ensemble=skill["ml_np"],
        reference=skill["reference_np"],
        qm_ensemble=qm_np,
    )
    if qm_np is not None:
        plot_qm_distributions(
            distributions,
            var_name=variable_name if variable_name is not None else "variable",
            out_dir=str(plot_dir),
            series_keys=("orig", "qm", "corrected", "obs"),
            series_labels=("Original", "QM", "RF-corrected", "Reference"),
            title_tag="Original vs QM vs RF-corrected vs reference",
            file_prefix="ensemble_mean_rf_distribution",
        )
    else:
        plot_qm_distributions(
            distributions,
            var_name=variable_name if variable_name is not None else "variable",
            out_dir=str(plot_dir),
            series_keys=("orig", "corrected", "obs"),
            series_labels=("Original", "RF-corrected", "Reference"),
            title_tag="Original vs RF-corrected vs reference",
            file_prefix="ensemble_mean_rf_distribution",
        )

    plot_mae_skill_metrics(
        skill["mae_baseline_member"],
        skill["mae_ml_member"],
        mae_qm_member,
        out_dir=str(plot_dir),
        n_members_display=3,
        metric_name="MAE",
        file_prefix="mae",
        land_mask_path=land_mask_path,
        latitude=latitude,
        longitude=longitude,
    )

    plot_mae_skill_metrics(
        skill["rmse_baseline_member"],
        skill["rmse_ml_member"],
        rmse_qm_member,
        out_dir=str(plot_dir),
        n_members_display=3,
        metric_name="RMSE",
        file_prefix="rmse",
        land_mask_path=land_mask_path,
        latitude=latitude,
        longitude=longitude,
    )

    plot_mae_skill_metrics(
        skill["mean_bias_baseline_member"],
        skill["mean_bias_ml_member"],
        mean_bias_qm_member,
        out_dir=str(plot_dir),
        n_members_display=3,
        metric_name="Mean Bias",
        file_prefix="mean_bias",
        land_mask_path=land_mask_path,
        latitude=latitude,
        longitude=longitude,
    )

    if include_bss_maps:
        plot_bss_skill_metrics(
            skill["bss_upper"],
            skill["bss_lower"],
            out_dir=str(plot_dir),
            upper_threshold_label="P90",
            lower_threshold_label="P10",
            land_mask_path=land_mask_path,
            latitude=latitude,
            longitude=longitude,
        )

        plot_bss_skill_metrics(
            skill["bss_p90_upper"],
            skill["bss_p10_lower"],
            out_dir=str(plot_dir),
            file_prefix="bss_p90_p10",
            upper_threshold_label="P90",
            lower_threshold_label="P10",
            title_prefix="ML vs Baseline",
            land_mask_path=land_mask_path,
            latitude=latitude,
            longitude=longitude,
        )

        if skill.get("bss_ml_vs_qm_upper") is not None:
            plot_bss_skill_metrics(
                skill["bss_ml_vs_qm_upper"],
                skill["bss_ml_vs_qm_lower"],
                out_dir=str(plot_dir),
                file_prefix="bss_ml_vs_qm",
                upper_threshold_label="mean + 1σ",
                lower_threshold_label="mean - 1σ",
                title_prefix="ML vs QM",
                land_mask_path=land_mask_path,
                latitude=latitude,
                longitude=longitude,
            )

        if skill.get("bss_qm_vs_baseline_upper") is not None:
            plot_event_bss_comparison_maps(
                bss_qm_vs_orig=skill["bss_qm_vs_baseline_upper"],
                bss_ml_vs_orig=skill["bss_upper"],
                bss_ml_vs_qm=skill["bss_ml_vs_qm_upper"],
                out_dir=str(plot_dir),
                threshold_label="90th percentile",
                file_prefix="bss_comparison_upper",
                land_mask_path=land_mask_path,
                latitude=latitude,
                longitude=longitude,
            )
            plot_event_bss_comparison_maps(
                bss_qm_vs_orig=skill["bss_qm_vs_baseline_lower"],
                bss_ml_vs_orig=skill["bss_lower"],
                bss_ml_vs_qm=skill["bss_ml_vs_qm_lower"],
                out_dir=str(plot_dir),
                threshold_label="10th percentile",
                file_prefix="bss_comparison_lower",
                land_mask_path=land_mask_path,
                latitude=latitude,
                longitude=longitude,
            )

    if include_timeseries:
        plot_ensemble_timeseries(
            skill["baseline_ensemble"].values,
            skill["ml_ensemble"].values,
            skill["reference"].values,
            out_dir=str(plot_dir),
            qm_ensemble=qm_ensemble.values if qm_ensemble is not None else None,
            land_mask_path=land_mask_path,
            time_coords=skill.get("common_time", None),
            compute_monthly_anomalies=include_anomalies,
        )

    if reference_all is not None and variable_name is not None:
        plot_ml_eval_summary_maps(
            reference_all=reference_all,
            reference_eval=skill["reference"].values,
            corrected_ensemble=skill["ml_ensemble"].values,
            baseline_ensemble=skill["baseline_ensemble"].values,
            qm_ensemble=qm_ensemble.values if qm_ensemble is not None else None,
            out_dir=str(plot_dir),
            variable_name=variable_name,
            land_mask_path=land_mask_path,
        )

    if include_example_time_means:
        plot_example_time_means(
            skill["baseline_ensemble"],
            qm_ensemble if qm_ensemble is not None else None,
            skill["ml_ensemble"],
            skill["reference"],
            out_dir=str(plot_dir),
            n_members_display=3,
            n_timesteps=3,
            land_mask_path=land_mask_path,
        )

    if include_difference_examples:
        plot_example_time_means(
            skill["baseline_ensemble"],
            qm_ensemble if qm_ensemble is not None else None,
            skill["ml_ensemble"],
            skill["reference"],
            out_dir=str(plot_dir),
            n_members_display=3,
            n_timesteps=3,
            difference_mode=True,
            land_mask_path=land_mask_path,
        )

    return plot_dir
