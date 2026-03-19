import numpy as np
import xarray as xr
from pathlib import Path

from droughtpp.evaluation.evaluation import (
    brier_skill_score_between_ensembles,
    mae_per_member_grid,
    rmse_per_member_grid,
)
from droughtpp.analysis.qm_rf.utils.visualization import (
    plot_mae_skill_metrics,
    plot_bss_skill_metrics,
    plot_probability_skill_metrics,
    plot_example_time_means,
)
from droughtpp.analysis.qm_rf.utils.evaluation import (
    load_paths_from_json,
    load_eval_data,
    select_eval_years,
    intersect_time_coordinates,
)


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
    prob_a_upper_mean = np.nanmean(prob_a_upper, axis=0)
    prob_b_upper_mean = np.nanmean(prob_b_upper, axis=0)

    obs_lower = (reference < threshold_lower[None, :, :]).astype(float)
    prob_a_lower = np.mean(ens_a < threshold_lower[None, None, :, :], axis=1)
    prob_b_lower = np.mean(ens_b < threshold_lower[None, None, :, :], axis=1)
    bs_a_lower = np.nanmean((prob_a_lower - obs_lower) ** 2, axis=0)
    bs_b_lower = np.nanmean((prob_b_lower - obs_lower) ** 2, axis=0)
    prob_a_lower_mean = np.nanmean(prob_a_lower, axis=0)
    prob_b_lower_mean = np.nanmean(prob_b_lower, axis=0)

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
        prob_a_upper_mean,
        prob_b_upper_mean,
        prob_a_lower_mean,
        prob_b_lower_mean,
    )


def evaluate_cwb(
    corrected_cwb_json: Path,
    hindcasts_json: Path,
    reference_data: Path,
    out_dir: Path,
    cwb_var: str = "CWB",
    eval_years=None,
    std_multiplier: float = 1.0,
    plot_dir: Path | None = None,
    qm_hindcasts_json: Path | None = None,
):
    if eval_years is None:
        eval_years = []

    corrected_cwb_paths = load_paths_from_json(corrected_cwb_json)
    hindcast_paths = load_paths_from_json(hindcasts_json)
    qm_hindcast_paths = (
        load_paths_from_json(qm_hindcasts_json) if qm_hindcasts_json else None
    )

    if len(hindcast_paths) != len(corrected_cwb_paths):
        raise ValueError(
            "Mismatch between number of hindcast files and corrected CWB files."
        )

    reference_cwb = load_eval_data(Path(reference_data), cwb_var)
    reference_cwb = select_eval_years(reference_cwb, eval_years)

    corrected_members = []
    baseline_members = []
    qm_members = []
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

    common_time = intersect_time_coordinates(
        corrected_members
        + baseline_members
        + reference_members
        + (qm_members if qm_members else [])
    )

    corrected_members = [member.sel(time=common_time) for member in corrected_members]
    baseline_members = [member.sel(time=common_time) for member in baseline_members]
    reference_members = [member.sel(time=common_time) for member in reference_members]
    if qm_members:
        qm_members = [member.sel(time=common_time) for member in qm_members]

    corrected_ensemble = xr.concat(corrected_members, dim="member").transpose(
        "time", "member", "latitude", "longitude"
    )
    baseline_ensemble = xr.concat(baseline_members, dim="member").transpose(
        "time", "member", "latitude", "longitude"
    )
    reference = reference_members[0].transpose("time", "latitude", "longitude")

    if qm_members:
        qm_ensemble = xr.concat(qm_members, dim="member").transpose(
            "time", "member", "latitude", "longitude"
        )

    corrected_np = corrected_ensemble.values
    baseline_np = baseline_ensemble.values
    reference_np = reference.values
    qm_np = qm_ensemble.values if qm_members else None

    mae_corrected_member = mae_per_member_grid(corrected_np, reference_np)
    mae_baseline_member = mae_per_member_grid(baseline_np, reference_np)
    mae_qm_member = (
        mae_per_member_grid(qm_np, reference_np) if qm_np is not None else None
    )

    rmse_corrected_member = rmse_per_member_grid(corrected_np, reference_np)
    rmse_baseline_member = rmse_per_member_grid(baseline_np, reference_np)
    rmse_qm_member = (
        rmse_per_member_grid(qm_np, reference_np) if qm_np is not None else None
    )

    mae_corrected = np.nanmean(mae_corrected_member, axis=0)
    mae_baseline = np.nanmean(mae_baseline_member, axis=0)
    mae_diff = mae_corrected - mae_baseline
    mae_qm = np.nanmean(mae_qm_member, axis=0) if mae_qm_member is not None else None

    rmse_corrected = np.nanmean(rmse_corrected_member, axis=0)
    rmse_baseline = np.nanmean(rmse_baseline_member, axis=0)
    rmse_diff = rmse_corrected - rmse_baseline
    rmse_qm = np.nanmean(rmse_qm_member, axis=0) if rmse_qm_member is not None else None

    bss_upper, bs_corrected_upper, bs_baseline_upper = (
        brier_skill_score_between_ensembles(
            corrected_np,
            baseline_np,
            reference_np,
            std_multiplier=std_multiplier,
            extreme_type="upper",
        )
    )
    bss_lower, bs_corrected_lower, bs_baseline_lower = (
        brier_skill_score_between_ensembles(
            corrected_np,
            baseline_np,
            reference_np,
            std_multiplier=-1.0,
            extreme_type="lower",
        )
    )

    (
        bss_upper_p90,
        bss_lower_p10,
        bs_corrected_upper_p90,
        bs_baseline_upper_p90,
        bs_corrected_lower_p10,
        bs_baseline_lower_p10,
        prob_corrected_upper_p90,
        prob_baseline_upper_p90,
        prob_corrected_lower_p10,
        prob_baseline_lower_p10,
    ) = brier_skill_score_between_ensembles_percentile(
        corrected_np,
        baseline_np,
        reference_np,
        upper_percentile=90.0,
        lower_percentile=10.0,
    )

    # BSS between RF-corrected and QM hindcasts
    bss_rf_vs_qm_upper = None
    bss_rf_vs_qm_lower = None
    if qm_np is not None:
        bss_rf_vs_qm_upper, _, _ = brier_skill_score_between_ensembles(
            corrected_np,
            qm_np,
            reference_np,
            std_multiplier=std_multiplier,
            extreme_type="upper",
        )
        bss_rf_vs_qm_lower, _, _ = brier_skill_score_between_ensembles(
            corrected_np,
            qm_np,
            reference_np,
            std_multiplier=-1.0,
            extreme_type="lower",
        )

    if plot_dir is not None:
        plot_dir = Path(plot_dir) / "rf_cwb_results"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Plot MAE with QM comparison if available
        if mae_qm_member is not None:
            plot_mae_skill_metrics(
                mae_baseline_member,
                mae_corrected_member,
                mae_qm_member,
                out_dir=str(plot_dir),
                n_members_display=3,
                metric_name="MAE",
                file_prefix="mae",
            )
        else:
            plot_mae_skill_metrics(
                mae_baseline_member,
                mae_corrected_member,
                out_dir=str(plot_dir),
                n_members_display=3,
                metric_name="MAE",
                file_prefix="mae",
            )

        # Plot RMSE with QM comparison if available
        if rmse_qm_member is not None:
            plot_mae_skill_metrics(
                rmse_baseline_member,
                rmse_corrected_member,
                rmse_qm_member,
                out_dir=str(plot_dir),
                n_members_display=3,
                metric_name="RMSE",
                file_prefix="rmse",
            )
        else:
            plot_mae_skill_metrics(
                rmse_baseline_member,
                rmse_corrected_member,
                out_dir=str(plot_dir),
                n_members_display=3,
                metric_name="RMSE",
                file_prefix="rmse",
            )

        plot_bss_skill_metrics(
            bss_upper,
            bss_lower,
            out_dir=str(plot_dir),
            upper_threshold_label="mean + 1σ",
            lower_threshold_label="mean - 1σ",
        )

        plot_bss_skill_metrics(
            bss_upper_p90,
            bss_lower_p10,
            out_dir=str(plot_dir),
            file_prefix="bss_p90_p10",
            upper_threshold_label="P90",
            lower_threshold_label="P10",
            title_prefix="RF-corrected vs Original",
        )

        plot_probability_skill_metrics(
            prob_corrected_upper_p90,
            prob_baseline_upper_p90,
            prob_corrected_lower_p10,
            prob_baseline_lower_p10,
            out_dir=str(plot_dir),
            file_prefix="probability_p90_p10",
            upper_threshold_label="P90",
            lower_threshold_label="P10",
            title_prefix="RF-corrected vs Original",
        )

        # Plot BSS between RF-corrected and QM if QM data available
        if bss_rf_vs_qm_upper is not None:
            plot_bss_skill_metrics(
                bss_rf_vs_qm_upper,
                bss_rf_vs_qm_lower,
                out_dir=str(plot_dir),
                file_prefix="bss_rf_vs_qm",
                upper_threshold_label="mean + 1σ",
                lower_threshold_label="mean - 1σ",
                title_prefix="RF vs QM",
            )

        # Plot example time means
        plot_example_time_means(
            baseline_ensemble,
            qm_ensemble if qm_members else None,
            corrected_ensemble,
            reference,
            out_dir=str(plot_dir),
            n_members_display=3,
            n_lead_months=3,
        )

    return {
        "cwb_mae_corrected_mean": float(np.nanmean(mae_corrected)),
        "cwb_mae_baseline_mean": float(np.nanmean(mae_baseline)),
        "cwb_mae_diff_mean": float(np.nanmean(mae_diff)),
        "cwb_mae_qm_mean": float(np.nanmean(mae_qm)) if mae_qm is not None else None,
        "cwb_rmse_corrected_mean": float(np.nanmean(rmse_corrected)),
        "cwb_rmse_baseline_mean": float(np.nanmean(rmse_baseline)),
        "cwb_rmse_diff_mean": float(np.nanmean(rmse_diff)),
        "cwb_rmse_qm_mean": float(np.nanmean(rmse_qm)) if rmse_qm is not None else None,
        "cwb_bs_corrected_upper_mean": float(np.nanmean(bs_corrected_upper)),
        "cwb_bs_baseline_upper_mean": float(np.nanmean(bs_baseline_upper)),
        "cwb_bss_upper_mean": float(np.nanmean(bss_upper)),
        "cwb_bs_corrected_lower_mean": float(np.nanmean(bs_corrected_lower)),
        "cwb_bs_baseline_lower_mean": float(np.nanmean(bs_baseline_lower)),
        "cwb_bss_lower_mean": float(np.nanmean(bss_lower)),
        "cwb_bs_corrected_upper_p90_mean": float(np.nanmean(bs_corrected_upper_p90)),
        "cwb_bs_baseline_upper_p90_mean": float(np.nanmean(bs_baseline_upper_p90)),
        "cwb_bss_upper_p90_mean": float(np.nanmean(bss_upper_p90)),
        "cwb_bs_corrected_lower_p10_mean": float(np.nanmean(bs_corrected_lower_p10)),
        "cwb_bs_baseline_lower_p10_mean": float(np.nanmean(bs_baseline_lower_p10)),
        "cwb_bss_lower_p10_mean": float(np.nanmean(bss_lower_p10)),
        "cwb_bss_rf_vs_qm_upper_mean": (
            float(np.nanmean(bss_rf_vs_qm_upper))
            if bss_rf_vs_qm_upper is not None
            else None
        ),
        "cwb_bss_rf_vs_qm_lower_mean": (
            float(np.nanmean(bss_rf_vs_qm_lower))
            if bss_rf_vs_qm_lower is not None
            else None
        ),
        "cwb_n_members": int(corrected_ensemble.sizes["member"]),
        "cwb_n_time": int(corrected_ensemble.sizes["time"]),
    }
