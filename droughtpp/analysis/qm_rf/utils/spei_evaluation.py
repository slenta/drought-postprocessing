import json
from pathlib import Path

import numpy as np
import xarray as xr
from IPython import embed

from droughtpp.analysis.qm_rf.utils.evaluation import (
    brier_skill_score_between_ensembles,
    mae_per_member_grid,
    rmse_per_member_grid,
    _load_var,
    _to_time_lat_lon,
    _to_time_member_lat_lon,
    load_paths_from_json,
    load_eval_data,
    select_eval_years,
    prepare_pairwise_skill_evaluation,
    plot_pairwise_skill_evaluation,
    _monthly_anomalies,
    _gridwise_correlation,
    _gridwise_correlation_with_pvalues,
    _std_over_rmse_maps,
    compute_gridcell_drought_hit_rate_percent,
    count_extreme_drought_events,
)
from droughtpp.analysis.qm_rf.utils.visualization import (
    plot_mae_skill_metrics,
    plot_mean_ensemble_std_maps,
    plot_drought_hit_rate_maps,
    plot_extreme_drought_hit_histogram,
    plot_residual_comparison_maps,
    plot_event_bss_comparison_maps,
    plot_example_time_means,
    _mean_ensemble_std_map,
)


def evaluate_spei(
    corrected_spei_paths,
    hindcasts_spei_json: Path,
    reference_spei_json: Path,
    out_dir: Path,
    spei_var: str = "spei",
    eval_years=None,
    std_multiplier: float = 1.0,
    plot_dir: Path | None = None,
    land_mask_path: Path | None = None,
    qm_hindcasts_json: Path | None = None,
):
    if eval_years is None:
        eval_years = []

    corrected_spei_paths = [Path(path) for path in corrected_spei_paths]

    baseline_spei_paths = load_paths_from_json(hindcasts_spei_json)
    reference_spei_paths = load_paths_from_json(reference_spei_json)

    reference_spei = load_eval_data(reference_spei_paths[0], spei_var)
    reference_spei = select_eval_years(reference_spei, eval_years)

    corrected_members = []
    baseline_members = []
    reference_members = []
    qm_members = []

    qm_hindcast_paths = (
        load_paths_from_json(qm_hindcasts_json) if qm_hindcasts_json else None
    )

    for baseline_spei_path, corrected_spei_path in zip(
        baseline_spei_paths, corrected_spei_paths
    ):
        corrected_spei = load_eval_data(corrected_spei_path, spei_var)
        baseline_spei = load_eval_data(baseline_spei_path, spei_var)

        corrected_spei = select_eval_years(corrected_spei, eval_years)
        baseline_spei = select_eval_years(baseline_spei, eval_years)

        corrected_spei, baseline_spei, reference_member = xr.align(
            corrected_spei,
            baseline_spei,
            reference_spei,
            join="inner",
        )

        corrected_members.append(corrected_spei)
        baseline_members.append(baseline_spei)
        reference_members.append(reference_member)
        qm_spei = load_eval_data(qm_hindcast_paths[len(qm_members)], spei_var)
        qm_spei = select_eval_years(qm_spei, eval_years)
        qm_spei, _, _ = xr.align(qm_spei, baseline_spei, reference_member, join="inner")
        qm_members.append(qm_spei)
    

    skill = prepare_pairwise_skill_evaluation(
        corrected_members,
        baseline_members,
        reference_members,
        qm_members=qm_members if qm_members else None,
        eval_years=eval_years,
        ml_label="RF-corrected",
        baseline_label="Original",
        std_multiplier=std_multiplier
    )

    common_time = skill["common_time"]
    ml_ensemble = skill["ml_ensemble"]
    baseline_ensemble = skill["baseline_ensemble"]
    reference = skill["reference"]

    ml_np = skill["ml_np"]
    baseline_np = skill["baseline_np"]
    reference_np = skill["reference_np"]

    mae_ml_member = skill["mae_ml_member"]
    mae_baseline_member = skill["mae_baseline_member"]

    rmse_ml_member = skill["rmse_ml_member"]
    rmse_baseline_member = skill["rmse_baseline_member"]

    mae_ml = np.nanmean(mae_ml_member, axis=0)
    mae_baseline = np.nanmean(mae_baseline_member, axis=0)
    mae_diff = mae_ml - mae_baseline

    rmse_ml = np.nanmean(rmse_ml_member, axis=0)
    rmse_baseline = np.nanmean(rmse_baseline_member, axis=0)
    rmse_diff = rmse_ml - rmse_baseline

    bss_upper = skill["bss_upper"]
    bss_lower = skill["bss_lower"]
    bs_ml_upper = skill["bs_ml_upper"]
    bs_baseline_upper = skill["bs_baseline_upper"]
    bs_ml_lower = skill["bs_ml_lower"]
    bs_baseline_lower = skill["bs_baseline_lower"]
    latitude = reference_spei.latitude.values if "latitude" in reference_spei.coords else None
    longitude = reference_spei.longitude.values if "longitude" in reference_spei.coords else None

    if plot_dir is not None:
        plot_pairwise_skill_evaluation(
            plot_dir=Path(plot_dir),
            skill=skill,
            reference_all=reference.values,
            variable_name=spei_var,
            land_mask_path=land_mask_path,
            include_timeseries=True,
            include_example_time_means=True,
            include_difference_examples=False,
            include_anomalies=False,
            latitude=latitude,
            longitude=longitude,
        )


        # Additional CWB-like plots: correlations, std/RMSE, and event-based plots
        # extract ensembles and arrays
        ml_ensemble = skill["ml_ensemble"]
        baseline_ensemble = skill["baseline_ensemble"]
        reference_da = skill["reference"]
        ml_np = skill["ml_np"]
        baseline_np = skill["baseline_np"]
        reference_np = skill["reference_np"]


        qm_ensemble = skill.get("qm_ensemble")
        qm_np = skill.get("qm_np")

        # Correlation maps using monthly anomalies
        correlation_secondary_source = _monthly_anomalies(baseline_ensemble)
        correlation_primary_source = _monthly_anomalies(ml_ensemble)
        correlation_reference_source = _monthly_anomalies(reference_da)
        correlation_qm_source = _monthly_anomalies(qm_ensemble)

        correlation_secondary_np = correlation_secondary_source.values
        correlation_primary_np = correlation_primary_source.values
        correlation_reference_np = correlation_reference_source.values
        correlation_qm_np = correlation_qm_source.values if correlation_qm_source is not None else None

        correlation_secondary_member = _gridwise_correlation(
            correlation_secondary_np,
            correlation_reference_np,
        )
        correlation_primary_member = _gridwise_correlation(
            correlation_primary_np,
            correlation_reference_np,
        )
        correlation_qm_member = (
            _gridwise_correlation(correlation_qm_np, correlation_reference_np)
            if correlation_qm_np is not None
            else None
        )

        correlation_secondary_pvalues = None
        correlation_primary_pvalues = None
        correlation_qm_pvalues = None
        try:
            correlation_secondary_member, correlation_secondary_pvalues = _gridwise_correlation_with_pvalues(
                correlation_secondary_np,
                correlation_reference_np,
            )
            correlation_primary_member, correlation_primary_pvalues = _gridwise_correlation_with_pvalues(
                correlation_primary_np,
                correlation_reference_np,
            )
            if correlation_qm_np is not None:
                correlation_qm_member, correlation_qm_pvalues = _gridwise_correlation_with_pvalues(
                    correlation_qm_np,
                    correlation_reference_np,
                )
        except Exception:
            # p-values optional; ignore failures
            pass

        plot_mae_skill_metrics(
            correlation_secondary_member,
            correlation_primary_member,
            correlation_qm_member,
            out_dir=str(plot_dir),
            n_members_display=3,
            metric_name="Correlation",
            file_prefix="correlation",
            land_mask_path=land_mask_path,
            stipple_significant=False,
            latitude=latitude,
            longitude=longitude,
        )

        if qm_ensemble is not None:
            plot_mean_ensemble_std_maps(
                qm_dataarray=qm_ensemble,
                ml_dataarray=ml_ensemble,
                original_dataarray=baseline_ensemble,
                output_path=Path(plot_dir) / "mean_ensemble_std_maps.png",
                land_mask_path=land_mask_path,
                latitude=latitude,
                longitude=longitude,
            )

            qm_std_over_rmse = _std_over_rmse_maps(
                _mean_ensemble_std_map(qm_ensemble),
                skill["rmse_qm_member"],
            )
            ml_std_over_rmse = _std_over_rmse_maps(
                _mean_ensemble_std_map(ml_ensemble),
                skill["rmse_ml_member"],
            )
            original_std_over_rmse = _std_over_rmse_maps(
                _mean_ensemble_std_map(baseline_ensemble),
                skill["rmse_baseline_member"],
            )
            plot_mae_skill_metrics(
                original_std_over_rmse,
                ml_std_over_rmse,
                qm_std_over_rmse,
                out_dir=str(plot_dir),
                n_members_display=3,
                metric_name="Std/RMSE",
                file_prefix="std_over_rmse",
                land_mask_path=land_mask_path,
                latitude=latitude,
                longitude=longitude,
            )

        # Event-based diagnostics (SPEI droughts are 'below' extremes)
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
            ml_hit_rate_p10 = compute_gridcell_drought_hit_rate_percent(
                ml_np,
                reference_np,
                threshold_lower_p10,
            )
            plot_drought_hit_rate_maps(
                baseline_hit_rate_p10,
                qm_hit_rate_p10,
                ml_hit_rate_p10,
                out_dir=str(plot_dir),
                lower_threshold_label="P10",
                file_prefix="drought_hit_rate_p10",
                latitude=latitude,
                longitude=longitude,
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
                        ml_np,
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

    return {
        "spei_mae_corrected_mean": float(np.nanmean(mae_ml)),
        "spei_mae_baseline_mean": float(np.nanmean(mae_baseline)),
        "spei_mae_diff_mean": float(np.nanmean(mae_diff)),
        "spei_rmse_corrected_mean": float(np.nanmean(rmse_ml)),
        "spei_rmse_baseline_mean": float(np.nanmean(rmse_baseline)),
        "spei_rmse_diff_mean": float(np.nanmean(rmse_diff)),
        "spei_bs_corrected_upper_mean": float(np.nanmean(bs_ml_upper)),
        "spei_bs_baseline_upper_mean": float(np.nanmean(bs_baseline_upper)),
        "spei_bss_upper_mean": float(np.nanmean(bss_upper)),
        "spei_bs_corrected_lower_mean": float(np.nanmean(bs_ml_lower)),
        "spei_bs_baseline_lower_mean": float(np.nanmean(bs_baseline_lower)),
        "spei_bss_lower_mean": float(np.nanmean(bss_lower)),
    }


def evaluate_spei_pair(
    output_spei_path,
    gt_spei_path,
    reference_spei_path,
    output_var="spei",
    gt_var="spei",
    reference_var="spei",
    member_dim="member",
    std_multiplier=1.0,
    plot_dir=None,
    land_mask_path: Path | None = None,
):
    """
    Generic SPEI evaluation: compare two SPEI outputs against a reference.
    Returns xr.Dataset with detailed metrics.
    """
    output_da = _load_var(output_spei_path, output_var)
    gt_da = _load_var(gt_spei_path, gt_var)
    reference_da = _load_var(reference_spei_path, reference_var)

    output_da, gt_da, reference_da = xr.align(
        output_da, gt_da, reference_da, join="inner"
    )

    output_da = _to_time_member_lat_lon(output_da, member_dim)
    gt_da = _to_time_member_lat_lon(gt_da, member_dim)
    reference_da = _to_time_lat_lon(reference_da)

    output_np = output_da.values
    gt_np = gt_da.values
    reference_np = reference_da.values

    mae_output_member = mae_per_member_grid(output_np, reference_np)
    mae_gt_member = mae_per_member_grid(gt_np, reference_np)

    rmse_output_member = rmse_per_member_grid(output_np, reference_np)
    rmse_gt_member = rmse_per_member_grid(gt_np, reference_np)

    mae_output = np.nanmean(mae_output_member, axis=0)
    mae_gt = np.nanmean(mae_gt_member, axis=0)
    mae_diff = mae_output - mae_gt

    rmse_output = np.nanmean(rmse_output_member, axis=0)
    rmse_gt = np.nanmean(rmse_gt_member, axis=0)
    rmse_diff = rmse_output - rmse_gt

    bss_upper, bs_output_upper, bs_gt_upper = brier_skill_score_between_ensembles(
        output_np,
        gt_np,
        reference_np,
        std_multiplier=std_multiplier,
        extreme_type="upper",
    )
    bss_lower, bs_output_lower, bs_gt_lower = brier_skill_score_between_ensembles(
        output_np,
        gt_np,
        reference_np,
        std_multiplier=-1.0,
        extreme_type="lower",
    )

    if plot_dir is not None:
        helper_skill = {
            "ml_np": output_np,
            "baseline_np": gt_np,
            "reference_np": reference_np,
            "ml_ensemble": output_da,
            "baseline_ensemble": gt_da,
            "reference": reference_da,
            "mae_baseline_member": mae_gt_member,
            "mae_ml_member": mae_output_member,
            "rmse_baseline_member": rmse_gt_member,
            "rmse_ml_member": rmse_output_member,
            "mean_bias_baseline_member": np.nanmean(
                gt_np - reference_np[:, None, :, :], axis=0
            ),
            "mean_bias_ml_member": np.nanmean(
                output_np - reference_np[:, None, :, :], axis=0
            ),
            "bss_upper": bss_upper,
            "bss_lower": bss_lower,
            "bss_p90_upper": bss_upper,
            "bss_p10_lower": bss_lower,
        }
        plot_pairwise_skill_evaluation(
            plot_dir=Path(plot_dir),
            skill=helper_skill,
            land_mask_path=land_mask_path,
            variable_name=output_var,
        )

    return {
        "mae_output_mean": float(np.nanmean(mae_output)),
        "mae_gt_mean": float(np.nanmean(mae_gt)),
        "mae_diff_mean": float(np.nanmean(mae_diff)),
        "rmse_output_mean": float(np.nanmean(rmse_output)),
        "rmse_gt_mean": float(np.nanmean(rmse_gt)),
        "rmse_diff_mean": float(np.nanmean(rmse_diff)),
        "bss_upper_mean": float(np.nanmean(bss_upper)),
        "bss_lower_mean": float(np.nanmean(bss_lower)),
    }
