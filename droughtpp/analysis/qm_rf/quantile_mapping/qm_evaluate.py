import argparse
from pathlib import Path
import yaml

from ..evaluation import compute_qm_distributions
from ..visualization import plot_qm_distributions


def run_qm_evaluation(
    json_list_path,
    ref_path,
    var_name="tas",
    qm_out_dir=None,
    plot_dir="qm_plots",
    n_quantiles=250,
):
    """
    High-level helper: compute distributions then plot.
    """
    dists = compute_qm_distributions(
        json_list_path, ref_path, var_name, qm_out_dir=qm_out_dir
    )
    plot_qm_distributions(dists, var_name, out_dir=plot_dir, n_quantiles=n_quantiles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate QM by plotting distributions"
    )
    parser.add_argument(
        "--config",
        "-c",
        help="Path to qm_rf/config.yaml (if omitted, uses config.yaml in qm_rf dir)",
        default="/work/bk1318/k202208/crai/hindcast-pp/Ml-Drought-Postprocessing/droughtpp/analysis/qm_rf/config.yaml",
    )
    args = parser.parse_args()

    # determine config path: CLI override or qm_rf/config.yaml next to this file's parent
    cfg_path = Path(args.config)

    cfg = yaml.safe_load(cfg_path.read_text())

    json_list = cfg["hindcasts_json"]
    reference = cfg["reference"]
    var = cfg.get("var_name", "tas")
    qm_out = cfg.get("qm_output_dir")
    plots_out = cfg.get("plot_dir")

    run_qm_evaluation(
        json_list,
        reference,
        var_name=var,
        qm_out_dir=qm_out,
        plot_dir=plots_out,
    )
