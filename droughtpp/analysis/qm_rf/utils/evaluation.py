import json
from pathlib import Path
from functools import reduce

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from droughtpp.evaluation.evaluation import (
    brier_skill_score_between_ensembles,
    mae_per_member_grid,
    rmse_per_member_grid,
)
from droughtpp.analysis.qm_rf.utils.visualization import (
    plot_bss_skill_metrics,
    plot_mae_skill_metrics,
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
    if not eval_years:
        return data_array

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


def load_eval_data(path: Path, var_name: str) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        data_array = ds[var_name].load()

    data_array = ensure_spatial_dims(data_array).squeeze()
    if "time" not in data_array.dims:
        raise ValueError(f"Expected time dimension in {path}")

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
    primary_members,
    secondary_members,
    reference_members,
    eval_years=None,
    qm_members=None,
    primary_label="Primary",
    secondary_label="Secondary",
    std_multiplier: float = 1.0,
):
    if eval_years is None:
        eval_years = []

    common_time = intersect_time_coordinates(
        primary_members
        + secondary_members
        + reference_members
        + (qm_members if qm_members else [])
    )

    primary_members = [member.sel(time=common_time) for member in primary_members]
    secondary_members = [member.sel(time=common_time) for member in secondary_members]
    reference_members = [member.sel(time=common_time) for member in reference_members]
    if qm_members:
        qm_members = [member.sel(time=common_time) for member in qm_members]

    primary_ensemble = xr.concat(primary_members, dim="member").transpose(
        "time", "member", "latitude", "longitude"
    )
    secondary_ensemble = xr.concat(secondary_members, dim="member").transpose(
        "time", "member", "latitude", "longitude"
    )
    reference = reference_members[0].transpose("time", "latitude", "longitude")

    primary_np = primary_ensemble.values
    secondary_np = secondary_ensemble.values
    reference_np = reference.values

    result = {
        "common_time": common_time,
        "primary_ensemble": primary_ensemble,
        "secondary_ensemble": secondary_ensemble,
        "reference": reference,
        "primary_np": primary_np,
        "secondary_np": secondary_np,
        "reference_np": reference_np,
        "primary_label": primary_label,
        "secondary_label": secondary_label,
    }

    result["mae_primary_member"] = mae_per_member_grid(primary_np, reference_np)
    result["mae_secondary_member"] = mae_per_member_grid(secondary_np, reference_np)
    result["rmse_primary_member"] = rmse_per_member_grid(primary_np, reference_np)
    result["rmse_secondary_member"] = rmse_per_member_grid(secondary_np, reference_np)
    result["mean_bias_primary_member"] = np.nanmean(
        primary_np - reference_np[:, None, :, :], axis=0
    )
    result["mean_bias_secondary_member"] = np.nanmean(
        secondary_np - reference_np[:, None, :, :], axis=0
    )

    result["bss_upper"], result["bss_lower"], *_ = brier_skill_score_between_ensembles(
        primary_np,
        secondary_np,
        reference_np,
        std_multiplier=std_multiplier,
        extreme_type="upper",
    )
    result["bss_p90_upper"], result["bss_p10_lower"], *_ = (
        brier_skill_score_between_ensembles_percentile(
            primary_np,
            secondary_np,
            reference_np,
            upper_percentile=90.0,
            lower_percentile=10.0,
        )
    )

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
        result["bss_primary_vs_qm_upper"], _, _ = brier_skill_score_between_ensembles(
            primary_np,
            qm_np,
            reference_np,
            std_multiplier=1.0,
            extreme_type="upper",
        )
        result["bss_primary_vs_qm_lower"], _, _ = brier_skill_score_between_ensembles(
            primary_np,
            qm_np,
            reference_np,
            std_multiplier=-1.0,
            extreme_type="lower",
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
):

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    mae_qm_member = skill.get("mae_qm_member")
    rmse_qm_member = skill.get("rmse_qm_member")
    mean_bias_qm_member = skill.get("mean_bias_qm_member")
    qm_ensemble = skill.get("qm_ensemble")
    qm_np = skill.get("qm_np")

    plot_mae_skill_metrics(
        skill["mae_secondary_member"],
        skill["mae_primary_member"],
        mae_qm_member,
        out_dir=str(plot_dir),
        n_members_display=3,
        metric_name="MAE",
        file_prefix="mae",
        land_mask_path=land_mask_path,
    )

    plot_mae_skill_metrics(
        skill["rmse_secondary_member"],
        skill["rmse_primary_member"],
        rmse_qm_member,
        out_dir=str(plot_dir),
        n_members_display=3,
        metric_name="RMSE",
        file_prefix="rmse",
        land_mask_path=land_mask_path,
    )

    plot_mae_skill_metrics(
        skill["mean_bias_secondary_member"],
        skill["mean_bias_primary_member"],
        mean_bias_qm_member,
        out_dir=str(plot_dir),
        n_members_display=3,
        metric_name="Mean Bias",
        file_prefix="mean_bias",
        land_mask_path=land_mask_path,
    )

    if include_bss_maps:
        plot_bss_skill_metrics(
            skill["bss_upper"],
            skill["bss_lower"],
            out_dir=str(plot_dir),
            upper_threshold_label="mean + 1σ",
            lower_threshold_label="mean - 1σ",
            land_mask_path=land_mask_path,
        )

        plot_bss_skill_metrics(
            skill["bss_p90_upper"],
            skill["bss_p10_lower"],
            out_dir=str(plot_dir),
            file_prefix="bss_p90_p10",
            upper_threshold_label="P90",
            lower_threshold_label="P10",
            title_prefix="RF-corrected vs Original",
            land_mask_path=land_mask_path,
        )

        if skill.get("bss_primary_vs_qm_upper") is not None:
            plot_bss_skill_metrics(
                skill["bss_primary_vs_qm_upper"],
                skill["bss_primary_vs_qm_lower"],
                out_dir=str(plot_dir),
                file_prefix="bss_rf_vs_qm",
                upper_threshold_label="mean + 1σ",
                lower_threshold_label="mean - 1σ",
                title_prefix="RF vs QM",
                land_mask_path=land_mask_path,
            )

    if include_timeseries:
        plot_ensemble_timeseries(
            skill["secondary_ensemble"].values,
            skill["primary_ensemble"].values,
            skill["reference"].values,
            out_dir=str(plot_dir),
            qm_ensemble=qm_ensemble.values if qm_ensemble is not None else None,
            land_mask_path=land_mask_path,
        )

    if reference_all is not None and variable_name is not None:
        plot_ml_eval_summary_maps(
            reference_all=reference_all,
            reference_eval=skill["reference"].values,
            corrected_ensemble=skill["primary_ensemble"].values,
            baseline_ensemble=skill["secondary_ensemble"].values,
            qm_ensemble=qm_ensemble.values if qm_ensemble is not None else None,
            out_dir=str(plot_dir),
            variable_name=variable_name,
            land_mask_path=land_mask_path,
        )

    if include_example_time_means:
        plot_example_time_means(
            skill["secondary_ensemble"],
            qm_ensemble if qm_ensemble is not None else None,
            skill["primary_ensemble"],
            skill["reference"],
            out_dir=str(plot_dir),
            n_members_display=3,
            n_timesteps=3,
            land_mask_path=land_mask_path,
        )

    if include_difference_examples:
        plot_example_time_means(
            skill["secondary_ensemble"],
            qm_ensemble if qm_ensemble is not None else None,
            skill["primary_ensemble"],
            skill["reference"],
            out_dir=str(plot_dir),
            n_members_display=3,
            n_timesteps=3,
            difference_mode=True,
            land_mask_path=land_mask_path,
        )

    return plot_dir
