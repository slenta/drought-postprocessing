from pathlib import Path
from typing import Optional

import yaml
import json
import tempfile

# relative imports within the package
from .quantile_mapping.quantile_mapping import quantile_map_json
from .random_forest.evaluate import evaluate as rf_evaluate
from .random_forest.train import train as rf_train
from .quantile_mapping.qm_evaluate import run_qm_evaluation


def run_qm_then_rf(
    config_path: str,
    train_rf: bool = False,
    qm_eval: bool = True,
):
    """
    Run additive quantile mapping for all hindcasts listed in hindcasts_json (or taken from the RF config),
    then run the RF training workflow using the provided RF YAML config.

    Parameters
    - config_path: path to RF training YAML config
    """
    # load RF/config YAML and allow it to supply QM inputs
    cfg = yaml.safe_load(Path(config_path).read_text())

    tmp_json_path = None

    # determine JSON with hindcasts (config may contain a path or an inline list)
    qm_json_path = Path(cfg["hindcasts_json"])
    ref_path = Path(cfg["reference_data"])

    # allow config to override var_name and output path for QM
    qm_var = cfg.get("var_name")
    qm_output = cfg.get("output_dir")
    plot_dir = cfg.get("plot_dir")
    n_quantiles = cfg.get("n_quantiles")

    # ensure output directory exists if provided
    if qm_output:
        Path(qm_output).mkdir(parents=True, exist_ok=True)

    # run quantile mapping (use or omit output_path depending on availability)
    # quantile_map_json(
    #     str(qm_json_path),
    #     ref_path=ref_path,
    #     var_name=qm_var,
    #     out_dir=str(qm_output),
    #     n_quantiles=n_quantiles,
    # )

    # if qm_eval:
    #     run_qm_evaluation(
    #         json_list_path=str(qm_json_path),
    #         ref_path=str(ref_path),
    #         var_name=qm_var,
    #         qm_out_dir=str(qm_output),
    #         plot_dir=str(plot_dir),
    #         n_quantiles=n_quantiles,
    #     )

    # 2) run RF evaluation and train if given
    if train_rf:
        rf_train(Path(config_path))

    rf_evaluate(Path(config_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run QM then RF workflow using a config YAML"
    )
    parser.add_argument("config", help="Path to RF/QM workflow YAML config")
    args = parser.parse_args()
    run_qm_then_rf(str(args.config), train_rf=False)
