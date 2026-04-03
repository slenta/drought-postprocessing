from pathlib import Path

# relative imports within the package
from .config_loader import get_qm_rf_global_config
from .quantile_mapping.model_qm import quantile_map_json
from .random_forest.evaluate_rf import evaluate as rf_evaluate
from .random_forest.train_rf import train as rf_train
from .quantile_mapping.evaluate_qm import run_qm_evaluation


def run_qm_then_rf(
    config_path: str | None = None,
    train_rf: bool = False,
    qm_run: bool = False,
    qm_eval: bool = True,
    run_rf_evaluate: bool = True,
    config_overrides=None,
):
    """
    Run additive quantile mapping for all hindcasts listed in hindcasts_json (or taken from the RF config),
    then run the RF training workflow using the provided RF YAML config.

    Parameters
    - config_path: path to RF training YAML config
    """
    # load RF/config YAML and allow it to supply QM inputs
    default_cfg_path = Path(__file__).resolve().parent / "config.yaml"
    cfg = get_qm_rf_global_config(
        config_path=config_path,
        overrides=config_overrides,
        default_config=default_cfg_path,
    )

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    # run quantile mapping (use or omit output_path depending on availability)
    if qm_run:
        quantile_map_json()

    if qm_eval:
        run_qm_evaluation()

    # run RF training and evaluation if requested
    if train_rf:
        rf_train()

    if (
        run_rf_evaluate
        or cfg["workflow"].get("evaluate_cwb", False)
        or cfg["workflow"].get("evaluate_intensity", False)
        or cfg["workflow"].get("evaluate_spei", False)
    ):
        rf_evaluate()


if __name__ == "__main__":
    default_cfg_path = Path(__file__).resolve().parent / "config.yaml"
    cfg = get_qm_rf_global_config(default_config=default_cfg_path)

    run_qm_then_rf(
        None,
        train_rf=cfg["workflow"]["run_rf_train"],
        qm_run=cfg["workflow"]["run_qm"],
        qm_eval=cfg["workflow"]["run_qm_eval"],
        run_rf_evaluate=cfg["workflow"]["run_rf_evaluate"],
    )
