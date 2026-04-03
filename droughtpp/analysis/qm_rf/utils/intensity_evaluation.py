import numpy as np
import xarray as xr
from pathlib import Path

from droughtpp.analysis.qm_rf.utils.visualization import (
    plot_drought_hit_rate_maps,
    plot_extreme_drought_hit_histogram,
)
from droughtpp.analysis.qm_rf.utils.evaluation import (
    compute_gridcell_drought_hit_rate_percent,
    count_extreme_drought_events,
    load_paths_from_json,
    load_eval_data,
    select_eval_years,
    prepare_pairwise_skill_evaluation,
    plot_pairwise_skill_evaluation,
)


def evaluate_intensity(
    corrected_intensity_json: Path,
    hindcasts_json: Path,
    reference_data: Path,
    out_dir: Path,
    intensity_var: str = "CWB_INTENSITY",
    eval_years=None,
    std_multiplier: float = 1.0,
    plot_dir: Path | str | None = None,
    qm_hindcasts_json: Path | None = None,
    land_mask_path: Path | None = None,
):

    if eval_years is None:
        eval_years = []

    corrected_intensity_paths = load_paths_from_json(corrected_intensity_json)
    hindcast_paths = load_paths_from_json(hindcasts_json)
    qm_hindcast_paths = (
        load_paths_from_json(qm_hindcasts_json) if qm_hindcasts_json else None
    )

    reference_intensity_all = load_eval_data(Path(reference_data), intensity_var)
    reference_intensity = select_eval_years(reference_intensity_all, eval_years)

    corrected_members = []
    baseline_members = []
    qm_members = []
    reference_members = []

    for i, (hindcast_path, corrected_intensity_path) in enumerate(
        zip(hindcast_paths, corrected_intensity_paths)
    ):
        baseline_intensity = load_eval_data(hindcast_path, intensity_var)
        corrected_intensity = load_eval_data(corrected_intensity_path, intensity_var)

        baseline_intensity = select_eval_years(baseline_intensity, eval_years)
        corrected_intensity = select_eval_years(corrected_intensity, eval_years)

        corrected_intensity, baseline_intensity, reference_member = xr.align(
            corrected_intensity,
            baseline_intensity,
            reference_intensity,
            join="inner",
        )

        corrected_members.append(corrected_intensity)
        baseline_members.append(baseline_intensity)
        reference_members.append(reference_member)

        if qm_hindcast_paths:
            qm_intensity = load_eval_data(qm_hindcast_paths[i], intensity_var)
            qm_intensity = select_eval_years(qm_intensity, eval_years)
            qm_intensity, _, _ = xr.align(
                qm_intensity,
                baseline_intensity,
                reference_member,
                join="inner",
            )
            qm_members.append(qm_intensity)

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

    if plot_dir is not None:
        plot_dir = Path(plot_dir)
        plot_pairwise_skill_evaluation(
            plot_dir=plot_dir,
            skill=skill,
            reference_all=reference_intensity_all.sel(time=common_time)
            .transpose("time", "latitude", "longitude")
            .values,
            variable_name=intensity_var,
            include_timeseries=True,
            include_example_time_means=True,
            include_difference_examples=True,
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
