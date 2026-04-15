import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt

from droughtpp.analysis.qm_rf.utils.visualization import (
    plot_drought_hit_rate_maps,
    plot_event_bss_comparison_maps,
    plot_example_time_means,
    plot_extreme_drought_hit_histogram,
    plot_residual_comparison_maps,
)
from droughtpp.analysis.qm_rf.utils.evaluation import (
    brier_skill_score_between_ensembles_threshold,
    count_binary_event_outcomes,
    compute_gridcell_drought_hit_rate_percent,
    count_extreme_drought_events,
    load_paths_from_json,
    load_eval_data,
    select_eval_years,
    prepare_pairwise_skill_evaluation,
    plot_pairwise_skill_evaluation,
)


def _plot_accumulated_event_timeseries(
    reference_counts_per_timestep,
    system_counts_per_timestep,
    out_dir: Path,
    threshold_label: str,
    file_prefix: str = "event_timeseries",
):
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_series = np.asarray(reference_counts_per_timestep, dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(ref_series, color="black", linewidth=2.0, label="Reference")

    for label, series in system_counts_per_timestep:
        values = np.asarray(series, dtype=float)
        ax.plot(values, linewidth=1.8, label=label)

    ax.set_xlabel("Evaluation timestep")
    ax.set_ylabel("Event count per timestep")
    ax.set_title(f"Event Counts ({threshold_label})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    out_png = out_dir / f"{file_prefix}.png"
    plt.savefig(str(out_png), bbox_inches="tight", dpi=150)
    plt.close()
    return str(out_png)


def evaluate_cwb(
    corrected_cwb_json: Path | None,
    hindcasts_json: Path,
    reference_data: Path,
    out_dir: Path,
    cwb_var: str = "CWB",
    eval_years=None,
    std_multiplier: float = 1.0,
    plot_dir: Path | str | None = None,
    qm_hindcasts_json: Path | None = None,
    qm_residuals_json: Path | None = None,
    ml_residuals_json: Path | None = None,
    land_mask_path: Path | None = None,
    ml_event_probability_json: Path | None = None,
    event_percentile: float = 90.0,
    probability_threshold: float = 0.5,
    ml_event_var: str = "event_probability",
):

    if ml_event_probability_json is not None:
        return evaluate_cwb_events(
            ml_event_probability_json=ml_event_probability_json,
            hindcasts_json=hindcasts_json,
            reference_data=reference_data,
            out_dir=out_dir,
            cwb_var=cwb_var,
            eval_years=eval_years,
            event_percentile=event_percentile,
            probability_threshold=probability_threshold,
            plot_dir=plot_dir,
            qm_hindcasts_json=qm_hindcasts_json,
            land_mask_path=land_mask_path,
            ml_event_var=ml_event_var,
        )

    if eval_years is None:
        eval_years = []

    if corrected_cwb_json is None:
        raise ValueError("corrected_cwb_json is required for residual CWB evaluation")

    corrected_cwb_paths = load_paths_from_json(corrected_cwb_json)
    hindcast_paths = load_paths_from_json(hindcasts_json)
    qm_hindcast_paths = (
        load_paths_from_json(qm_hindcasts_json) if qm_hindcasts_json else None
    )
    qm_residual_paths = (
        load_paths_from_json(qm_residuals_json) if qm_residuals_json else None
    )
    ml_residual_paths = (
        load_paths_from_json(ml_residuals_json) if ml_residuals_json else None
    )

    sample_corrected_cwb = load_eval_data(corrected_cwb_paths[0], cwb_var)
    eval_month = int(sample_corrected_cwb["time"].dt.month.values[0])

    reference_cwb_all = load_eval_data(Path(reference_data), cwb_var)
    reference_cwb_all = reference_cwb_all.isel(
        time=reference_cwb_all["time"].dt.month == eval_month
    )
    reference_cwb = select_eval_years(reference_cwb_all, eval_years)

    corrected_members = []
    baseline_members = []
    qm_members = []
    qm_residual_members = []
    ml_residual_members = []
    reference_members = []

    for i, (hindcast_path, corrected_cwb_path) in enumerate(
        zip(hindcast_paths, corrected_cwb_paths)
    ):
        baseline_cwb = load_eval_data(hindcast_path, cwb_var)
        corrected_cwb = load_eval_data(corrected_cwb_path, cwb_var)

        baseline_cwb = select_eval_years(baseline_cwb, eval_years)
        corrected_cwb = select_eval_years(corrected_cwb, eval_years)

        corrected_cwb, baseline_cwb, reference_member = xr.align(
            corrected_cwb,
            baseline_cwb,
            reference_cwb,
            join="inner",
        )

        corrected_members.append(corrected_cwb)
        baseline_members.append(baseline_cwb)
        reference_members.append(reference_member)

        # Load QM hindcasts if provided
        if qm_hindcast_paths:
            qm_cwb = load_eval_data(qm_hindcast_paths[i], cwb_var)
            qm_cwb = select_eval_years(qm_cwb, eval_years)
            qm_cwb, _, _ = xr.align(
                qm_cwb,
                baseline_cwb,
                reference_member,
                join="inner",
            )
            qm_members.append(qm_cwb)

        if qm_residual_paths is not None and ml_residual_paths is not None:
            qm_residual = load_eval_data(qm_residual_paths[i], cwb_var)
            ml_residual = load_eval_data(ml_residual_paths[i], cwb_var)
            qm_residual = select_eval_years(qm_residual, eval_years)
            ml_residual = select_eval_years(ml_residual, eval_years)
            qm_residual, ml_residual, _, _ = xr.align(
                qm_residual,
                ml_residual,
                baseline_cwb,
                reference_member,
                join="inner",
            )
            qm_residual_members.append(qm_residual)
            ml_residual_members.append(ml_residual)

    skill = prepare_pairwise_skill_evaluation(
        corrected_members,
        baseline_members,
        reference_members,
        eval_years=eval_years,
        qm_members=qm_members if qm_members else None,
        primary_label="RF-corrected",
        secondary_label="Original",
        std_multiplier=std_multiplier,
    )

    common_time = skill["common_time"]
    corrected_ensemble = skill["primary_ensemble"]
    baseline_ensemble = skill["secondary_ensemble"]
    reference = skill["reference"]

    corrected_np = skill["primary_np"]
    baseline_np = skill["secondary_np"]
    reference_np = skill["reference_np"]

    qm_ensemble = skill.get("qm_ensemble")
    qm_np = skill.get("qm_np")

    qm_residual_np = None
    ml_residual_np = None
    if qm_residual_members and ml_residual_members:
        qm_residual_ensemble = xr.concat(qm_residual_members, dim="member").transpose(
            "time", "member", "latitude", "longitude"
        )
        ml_residual_ensemble = xr.concat(ml_residual_members, dim="member").transpose(
            "time", "member", "latitude", "longitude"
        )
        qm_residual_np = qm_residual_ensemble.sel(time=common_time).values
        ml_residual_np = ml_residual_ensemble.sel(time=common_time).values

    if plot_dir is not None:
        plot_dir = Path(plot_dir)
        plot_pairwise_skill_evaluation(
            plot_dir=plot_dir,
            skill=skill,
            reference_all=reference_cwb_all.values,
            variable_name=cwb_var,
            include_timeseries=True,
            include_example_time_means=True,
            include_difference_examples=False,
            land_mask_path=land_mask_path,
        )

        threshold_lower_p10 = np.nanpercentile(reference_np, 10.0, axis=0)

        if qm_np is not None:
            baseline_hit_rate_p10 = compute_gridcell_drought_hit_rate_percent(
                baseline_np,
                reference_np,
                threshold_lower_p10,
            )
            qm_hit_rate_p10 = compute_gridcell_drought_hit_rate_percent(
                qm_np,
                reference_np,
                threshold_lower_p10,
            )
            corrected_hit_rate_p10 = compute_gridcell_drought_hit_rate_percent(
                corrected_np,
                reference_np,
                threshold_lower_p10,
            )
            plot_drought_hit_rate_maps(
                baseline_hit_rate_p10,
                qm_hit_rate_p10,
                corrected_hit_rate_p10,
                out_dir=str(plot_dir),
                lower_threshold_label="P10",
                file_prefix="drought_hit_rate_p10",
            )

        extreme_drought_counts = [
            (
                "Original",
                *count_extreme_drought_events(
                    baseline_np,
                    reference_np,
                    threshold_lower_p10,
                ),
            )
        ]
        if qm_np is not None:
            extreme_drought_counts.append(
                (
                    "QM",
                    *count_extreme_drought_events(
                        qm_np,
                        reference_np,
                        threshold_lower_p10,
                    ),
                )
            )
        extreme_drought_counts.append(
            (
                "ML-corrected",
                *count_extreme_drought_events(
                    corrected_np,
                    reference_np,
                    threshold_lower_p10,
                ),
            )
        )

        plot_extreme_drought_hit_histogram(
            extreme_drought_counts,
            out_dir=str(plot_dir),
            lower_percentile=10.0,
            file_prefix="extreme_drought_p10_histogram",
        )

        if qm_residual_np is not None and ml_residual_np is not None:
            plot_residual_comparison_maps(
                qm_residual_np,
                ml_residual_np,
                out_dir=str(plot_dir),
                variable_name=cwb_var,
                file_prefix="residual_qm_vs_ml",
                land_mask_path=land_mask_path,
            )


def evaluate_cwb_events(
    ml_event_probability_json: Path,
    hindcasts_json: Path,
    reference_data: Path,
    out_dir: Path,
    cwb_var: str = "CWB",
    eval_years=None,
    event_percentile: float = 90.0,
    probability_threshold: float = 0.5,
    plot_dir: Path | str | None = None,
    qm_hindcasts_json: Path | None = None,
    land_mask_path: Path | None = None,
    ml_event_var: str = "event_probability",
):

    if eval_years is None:
        eval_years = []

    ml_event_probability_paths = load_paths_from_json(ml_event_probability_json)
    hindcast_paths = load_paths_from_json(hindcasts_json)
    qm_hindcast_paths = (
        load_paths_from_json(qm_hindcasts_json) if qm_hindcasts_json else None
    )

    sample_ml_event = load_eval_data(ml_event_probability_paths[0], ml_event_var)
    eval_month = int(sample_ml_event["time"].dt.month.values[0])

    reference_cwb_all = load_eval_data(Path(reference_data), cwb_var)
    reference_cwb_all = reference_cwb_all.isel(
        time=reference_cwb_all["time"].dt.month == eval_month
    )
    reference_cwb = select_eval_years(reference_cwb_all, eval_years)

    baseline_members = []
    qm_members = []
    ml_event_members = []
    reference_members = []

    for i, (hindcast_path, ml_event_path) in enumerate(
        zip(hindcast_paths, ml_event_probability_paths)
    ):
        baseline_cwb = load_eval_data(hindcast_path, cwb_var)
        ml_event_probability = load_eval_data(ml_event_path, ml_event_var)

        baseline_cwb = select_eval_years(baseline_cwb, eval_years)
        ml_event_probability = select_eval_years(ml_event_probability, eval_years)

        ml_event_probability, baseline_cwb, reference_member = xr.align(
            ml_event_probability,
            baseline_cwb,
            reference_cwb,
            join="inner",
        )

        baseline_members.append(baseline_cwb)
        ml_event_members.append(ml_event_probability)
        reference_members.append(reference_member)

        if qm_hindcast_paths:
            qm_cwb = load_eval_data(qm_hindcast_paths[i], cwb_var)
            qm_cwb = select_eval_years(qm_cwb, eval_years)
            qm_cwb, _, _, _ = xr.align(
                qm_cwb,
                baseline_cwb,
                ml_event_probability,
                reference_member,
                join="inner",
            )
            qm_members.append(qm_cwb)

    baseline_ensemble = xr.concat(baseline_members, dim="member").transpose(
        "time", "member", "latitude", "longitude"
    )
    ml_event_probability_ensemble = xr.concat(ml_event_members, dim="member").transpose(
        "time", "member", "latitude", "longitude"
    )
    reference = reference_members[0].transpose("time", "latitude", "longitude")
    qm_ensemble = (
        xr.concat(qm_members, dim="member").transpose(
            "time", "member", "latitude", "longitude"
        )
        if qm_members
        else None
    )

    baseline_np = baseline_ensemble.values
    ml_event_probability_np = np.clip(ml_event_probability_ensemble.values, 0.0, 1.0)
    reference_np = reference.values
    qm_np = qm_ensemble.values if qm_ensemble is not None else None

    threshold_upper = np.nanpercentile(reference_np, event_percentile, axis=0)
    threshold_label = f">P{int(event_percentile)}"

    obs_event = (reference_np > threshold_upper[None, :, :]).astype(float)
    baseline_prob = np.mean(baseline_np > threshold_upper[None, None, :, :], axis=1)
    qm_prob = (
        np.mean(qm_np > threshold_upper[None, None, :, :], axis=1)
        if qm_np is not None
        else None
    )
    ml_prob = np.nanmean(ml_event_probability_np, axis=1)

    bss_ml_vs_orig, _, _ = brier_skill_score_between_ensembles_threshold(
        ml_prob[:, None, :, :],
        baseline_prob[:, None, :, :],
        obs_event,
        threshold=probability_threshold,
        comparison="above",
    )

    if qm_prob is not None:
        bss_qm_vs_orig, _, _ = brier_skill_score_between_ensembles_threshold(
            qm_prob[:, None, :, :],
            baseline_prob[:, None, :, :],
            obs_event,
            threshold=probability_threshold,
            comparison="above",
        )
        bss_ml_vs_qm, _, _ = brier_skill_score_between_ensembles_threshold(
            ml_prob[:, None, :, :],
            qm_prob[:, None, :, :],
            obs_event,
            threshold=probability_threshold,
            comparison="above",
        )
    else:
        bss_qm_vs_orig = np.full_like(bss_ml_vs_orig, np.nan, dtype=float)
        bss_ml_vs_qm = np.full_like(bss_ml_vs_orig, np.nan, dtype=float)

    reference_event_binary = obs_event >= 0.5
    (
        baseline_event_binary,
        valid_baseline,
        baseline_reference_count,
        baseline_hit_count,
        baseline_wrong_count,
    ) = count_binary_event_outcomes(
        baseline_prob,
        reference_event_binary,
        probability_threshold=probability_threshold,
    )
    if qm_prob is not None:
        (
            qm_event_binary,
            valid_qm,
            qm_reference_count,
            qm_hit_count,
            qm_wrong_count,
        ) = count_binary_event_outcomes(
            qm_prob,
            reference_event_binary,
            probability_threshold=probability_threshold,
        )
    else:
        qm_event_binary = None
        valid_qm = None
        qm_reference_count = 0
        qm_hit_count = 0
        qm_wrong_count = 0
    (
        ml_event_binary,
        valid_ml,
        ml_reference_count,
        ml_hit_count,
        ml_wrong_count,
    ) = count_binary_event_outcomes(
        ml_prob,
        reference_event_binary,
        probability_threshold=probability_threshold,
    )

    ml_hit_count_grid = np.sum(
        ml_event_binary & reference_event_binary & valid_ml, axis=0
    )
    ref_event_count_grid = np.sum(reference_event_binary & valid_ml, axis=0)
    ml_hit_rate_grid = np.full(ref_event_count_grid.shape, np.nan, dtype=float)
    has_ref_events = ref_event_count_grid > 0
    ml_hit_rate_grid[has_ref_events] = (
        ml_hit_count_grid[has_ref_events] / ref_event_count_grid[has_ref_events]
    ) * 100.0

    baseline_hit_rate = compute_gridcell_drought_hit_rate_percent(
        baseline_np,
        reference_np,
        threshold_upper,
        comparison="above",
    )
    qm_hit_rate = (
        compute_gridcell_drought_hit_rate_percent(
            qm_np,
            reference_np,
            threshold_upper,
            comparison="above",
        )
        if qm_np is not None
        else np.full_like(baseline_hit_rate, np.nan, dtype=float)
    )

    baseline_counts = (
        baseline_reference_count,
        baseline_hit_count,
        baseline_wrong_count,
    )
    qm_counts = (
        (
            qm_reference_count,
            qm_hit_count,
            qm_wrong_count,
        )
        if qm_event_binary is not None and valid_qm is not None
        else None
    )

    reference_counts_t = np.sum(
        reference_event_binary & np.isfinite(obs_event), axis=(1, 2)
    )
    baseline_counts_t = np.sum(
        baseline_event_binary & valid_baseline,
        axis=(1, 2),
    )
    ml_counts_t = np.sum(ml_event_binary & valid_ml, axis=(1, 2))
    system_series = [("Original", baseline_counts_t), ("ML", ml_counts_t)]
    if qm_event_binary is not None and valid_qm is not None:
        qm_counts_t = np.sum(qm_event_binary & valid_qm, axis=(1, 2))
        system_series.insert(1, ("QM", qm_counts_t))

    if plot_dir is not None:
        plot_dir = Path(plot_dir)

        plot_event_bss_comparison_maps(
            bss_qm_vs_orig,
            bss_ml_vs_orig,
            bss_ml_vs_qm,
            out_dir=str(plot_dir),
            threshold_label=threshold_label,
            file_prefix="event_bss_comparison",
            land_mask_path=land_mask_path,
        )

        plot_drought_hit_rate_maps(
            baseline_hit_rate,
            qm_hit_rate,
            ml_hit_rate_grid,
            out_dir=str(plot_dir),
            lower_threshold_label=threshold_label,
            file_prefix=f"event_hit_rate_p{int(event_percentile)}",
        )

        event_hist_counts = [("Original", *baseline_counts)]
        if qm_counts is not None:
            event_hist_counts.append(("QM", *qm_counts))
        event_hist_counts.append(
            ("ML-event", ml_reference_count, ml_hit_count, ml_wrong_count)
        )
        plot_extreme_drought_hit_histogram(
            event_hist_counts,
            out_dir=str(plot_dir),
            lower_percentile=event_percentile,
            threshold_label=threshold_label,
            file_prefix=f"event_histogram_p{int(event_percentile)}",
        )

        _plot_accumulated_event_timeseries(
            reference_counts_t,
            system_series,
            out_dir=plot_dir,
            threshold_label=threshold_label,
            file_prefix=f"event_counts_per_timestep_p{int(event_percentile)}",
        )

        # Example timestep maps for event probabilities (baseline/QM/ML) and
        # observed event mask, using a single pseudo-member for probability fields.
        baseline_prob_da = xr.DataArray(
            baseline_prob[:, None, :, :],
            dims=("time", "member", "latitude", "longitude"),
            coords={
                "time": reference["time"],
                "member": [0],
                "latitude": reference["latitude"],
                "longitude": reference["longitude"],
            },
        )
        qm_prob_da = (
            xr.DataArray(
                qm_prob[:, None, :, :],
                dims=("time", "member", "latitude", "longitude"),
                coords={
                    "time": reference["time"],
                    "member": [0],
                    "latitude": reference["latitude"],
                    "longitude": reference["longitude"],
                },
            )
            if qm_prob is not None
            else None
        )
        ml_prob_da = xr.DataArray(
            ml_prob[:, None, :, :],
            dims=("time", "member", "latitude", "longitude"),
            coords={
                "time": reference["time"],
                "member": [0],
                "latitude": reference["latitude"],
                "longitude": reference["longitude"],
            },
        )
        reference_event_da = xr.DataArray(
            obs_event,
            dims=("time", "latitude", "longitude"),
            coords={
                "time": reference["time"],
                "latitude": reference["latitude"],
                "longitude": reference["longitude"],
            },
        )

        plot_example_time_means(
            baseline_ensemble=baseline_prob_da,
            qm_ensemble=qm_prob_da,
            corrected_ensemble=ml_prob_da,
            reference=reference_event_da,
            out_dir=str(plot_dir),
            n_members_display=1,
            n_timesteps=3,
            variable_name=f"Event probability ({threshold_label})",
            corrected_label="ML-event",
            land_mask_path=land_mask_path,
        )
