from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from droughtpp.preprocessing.config_loader import (
    add_preprocessing_config_arguments,
    load_preprocessing_config,
)
from droughtpp.preprocessing.utils.preprocessing import (
    combine_hindcast_month_window,
    compute_ensemble_monthly_thresholds,
    compute_threshold_exceedance_intensity,
    load_leadmonth_mapping_json,
    load_path_list_json,
    process_group,
    process_hindcast_leadmonth,
    resolve_path,
    write_paths_json,
)


class PreprocessingWorkflow:
    def __init__(self, config: Dict[str, Any], config_dir: Path):
        self.config = config
        self.config_dir = config_dir

    def run(self) -> Dict[str, Any]:
        output_data_dir = resolve_path(self.config["output_data_dir"], self.config_dir)
        output_paths_dir = resolve_path(
            self.config["output_paths_dir"], self.config_dir
        )
        output_data_dir.mkdir(parents=True, exist_ok=True)
        output_paths_dir.mkdir(parents=True, exist_ok=True)

        combine_cfg = self.config.get("combine_hindcast_month_window", {})
        start_month = int(combine_cfg.get("start_month", 1))
        end_month = int(combine_cfg.get("end_month", 12))

        hindcast_cwb_dir = output_data_dir / "hindcast_cwb"
        hindcast_combined_dir = (
            output_data_dir / f"hindcast_combined_{start_month:02d}-{end_month:02d}"
        )
        hindcast_intensity_dir = (
            output_data_dir / f"hindcast_intensity_{start_month:02d}-{end_month:02d}"
        )
        reference_cwb_dir = output_data_dir / "reference_cwb"
        reference_intensity_dir = output_data_dir / "reference_intensity"

        hindcast_cwb_dir.mkdir(parents=True, exist_ok=True)
        hindcast_combined_dir.mkdir(parents=True, exist_ok=True)
        hindcast_intensity_dir.mkdir(parents=True, exist_ok=True)
        reference_cwb_dir.mkdir(parents=True, exist_ok=True)
        reference_intensity_dir.mkdir(parents=True, exist_ok=True)

        hindcast_precip_map = load_leadmonth_mapping_json(
            resolve_path(self.config["hindcast_precip_json"], self.config_dir)
        )
        hindcast_temp_map = load_leadmonth_mapping_json(
            resolve_path(self.config["hindcast_temp_json"], self.config_dir)
        )

        reference_precip_paths = load_path_list_json(
            resolve_path(self.config["reference_precip_json"], self.config_dir)
        )
        reference_temp_paths = load_path_list_json(
            resolve_path(self.config["reference_temp_json"], self.config_dir)
        )

        results: Dict[str, Any] = {}

        workflow_cfg = self.config.get("workflow", {})
        intensity_cfg = self.config.get("intensity", {})
        lower_percentile = float(intensity_cfg.get("lower_threshold"))

        intensity_input_var = str(
            intensity_cfg.get(
                "input_var_name", self.config.get("output_var_name", "CWB")
            )
        )
        intensity_output_var = str(intensity_cfg.get("output_var_name"))

        if bool(workflow_cfg.get("process_hindcast", True)):
            leadmonths = [int(value) for value in self.config.get("leadmonths", [])]

            by_leadmonth_outputs: dict[int, list[str]] = {}

            for leadmonth in leadmonths:

                hindcast_outputs = process_hindcast_leadmonth(
                    precip_path=hindcast_precip_map[leadmonth],
                    temperature_path=hindcast_temp_map[leadmonth],
                    dataset_cfg=self.config["hindcast"],
                    output_dir=hindcast_cwb_dir,
                    output_var_name=self.config.get("output_var_name", "CWB"),
                    leadmonth=leadmonth,
                    config_dir=self.config_dir,
                )

                by_leadmonth_outputs[leadmonth] = hindcast_outputs

            hindcast_all_outputs: list[str] = []
            for leadmonth in leadmonths:
                hindcast_all_outputs.extend(by_leadmonth_outputs[leadmonth])

            hindcast_paths_json = output_paths_dir / "hindcast_cwb_paths_all_lm.json"
            write_paths_json(hindcast_all_outputs, hindcast_paths_json)

            results["hindcast"] = {
                "by_leadmonth_outputs": {
                    str(key): value for key, value in by_leadmonth_outputs.items()
                },
                "paths_json": str(hindcast_paths_json),
            }

            if bool(workflow_cfg.get("combine_hindcast_month_window", True)):
                merged_outputs, merged_paths_json = combine_hindcast_month_window(
                    by_leadmonth_outputs=by_leadmonth_outputs,
                    leadmonths=leadmonths,
                    start_month=start_month,
                    end_month=end_month,
                    output_dir=hindcast_combined_dir,
                    output_paths_dir=output_paths_dir,
                )
                results["hindcast"]["month_window_outputs"] = merged_outputs
                results["hindcast"]["month_window_paths_json"] = merged_paths_json

                if bool(workflow_cfg.get("intensity", True)):
                    monthly_lower_thresholds = compute_ensemble_monthly_thresholds(
                        cwb_paths=merged_outputs,
                        lower_threshold_percentile=lower_percentile,
                        input_var_name=intensity_input_var,
                    )
                    intensity_outputs: list[str] = []
                    for cwb_path in merged_outputs:
                        intensity_outputs.append(
                            compute_threshold_exceedance_intensity(
                                cwb_path=cwb_path,
                                monthly_lower_thresholds=monthly_lower_thresholds,
                                output_data_dir=hindcast_intensity_dir,
                                input_var_name=intensity_input_var,
                                output_var_name=intensity_output_var,
                            )
                        )
                    intensity_paths_json = output_paths_dir / (
                        f"hindcast_cwb_month_window_intensity_paths_{start_month:02d}-{end_month:02d}.json"
                    )
                    write_paths_json(intensity_outputs, intensity_paths_json)
                    results["hindcast"][
                        "month_window_intensity_outputs"
                    ] = intensity_outputs
                    results["hindcast"][
                        "month_window_intensity_paths_json"
                    ] = intensity_paths_json

        if bool(workflow_cfg.get("process_reference", True)):
            reference_outputs = process_group(
                precip_paths=reference_precip_paths,
                temperature_paths=reference_temp_paths,
                dataset_cfg=self.config["reference"],
                output_dir=reference_cwb_dir,
                output_var_name=self.config.get("output_var_name", "CWB"),
                group_prefix="reference",
                config_dir=self.config_dir,
            )
            reference_paths_json = output_paths_dir / "reference_cwb_paths.json"
            write_paths_json(reference_outputs, reference_paths_json)

            results["reference"] = {
                "outputs": reference_outputs,
                "paths_json": str(reference_paths_json),
            }

            if bool(workflow_cfg.get("intensity", True)) and reference_outputs:
                reference_monthly_lower_thresholds = (
                    compute_ensemble_monthly_thresholds(
                        cwb_paths=reference_outputs,
                        lower_threshold_percentile=lower_percentile,
                        input_var_name=intensity_input_var,
                    )
                )
                reference_intensity_outputs: list[str] = []
                for cwb_path in reference_outputs:
                    reference_intensity_outputs.append(
                        compute_threshold_exceedance_intensity(
                            cwb_path=cwb_path,
                            monthly_lower_thresholds=reference_monthly_lower_thresholds,
                            output_data_dir=reference_intensity_dir,
                            input_var_name=intensity_input_var,
                            output_var_name=intensity_output_var,
                        )
                    )

                reference_intensity_paths_json = (
                    output_paths_dir / "reference_cwb_intensity_paths.json"
                )
                write_paths_json(
                    reference_intensity_outputs,
                    reference_intensity_paths_json,
                )
                results["reference"]["intensity_outputs"] = reference_intensity_outputs
                results["reference"]["intensity_paths_json"] = str(
                    reference_intensity_paths_json
                )

        return results


def parse_args(argv=None) -> argparse.Namespace:
    default_config_path = Path(__file__).resolve().parent / "config.yaml"
    parser = argparse.ArgumentParser(description="Run preprocessing PET/CWB workflow")
    add_preprocessing_config_arguments(
        parser,
        default_config=str(default_config_path),
        config_required=False,
    )
    return parser.parse_args(argv)


def main(argv=None) -> Dict[str, Any]:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    config = load_preprocessing_config(
        config_path=config_path,
        overrides=args.config_overrides,
    )

    workflow = PreprocessingWorkflow(config=config, config_dir=config_path.parent)
    return workflow.run()


if __name__ == "__main__":
    main()
