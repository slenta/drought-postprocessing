# QMRF — Quantile Mapping + Random Forest Postprocessing

This folder implements a two-stage post-processing pipeline that combines
additive quantile mapping (QM) with a random-forest-based residual correction
workflow (RF). It is designed to bias-correct and improve probabilistic
forecasts (hindcasts) of drought-relevant variables such as the Climate Water
Balance (CWB) used in the project.

This README documents the purpose, required inputs, configuration, execution
examples, and internals of the implementation found in this directory.

**Quick links**
- **Main module**: [droughtpp/analysis/qm_rf/main_qm_rf.py](droughtpp/analysis/qm_rf/main_qm_rf.py)
- **Primary config**: [droughtpp/analysis/qm_rf/config.yaml](droughtpp/analysis/qm_rf/config.yaml)
- **Config helper**: [droughtpp/analysis/qm_rf/config_loader.py](droughtpp/analysis/qm_rf/config_loader.py)
- **Quantile mapping code**: [droughtpp/analysis/qm_rf/quantile_mapping](droughtpp/analysis/qm_rf/quantile_mapping)
- **Random forest code**: [droughtpp/analysis/qm_rf/random_forest](droughtpp/analysis/qm_rf/random_forest)
- **Utilities & evaluation**: [droughtpp/analysis/qm_rf/utils](droughtpp/analysis/qm_rf/utils)

**Table of contents**
- Purpose and workflow
- Prerequisites
- Configuration (keys and patterns)
- Running the pipeline
- Running sub-steps and utilities
- Inputs and outputs
- Key modules explained
- Evaluation and plotting
- Tips, troubleshooting, and common overrides

**Purpose and workflow**

The pipeline follows two main stages:

1. Quantile mapping (QM): compute additive quantile corrections for hindcasts
   using a provided observational reference. The QM outputs residuals and
   corrected hindcasts.
2. Random-forest residual correction (RF): train and/or evaluate ML models on
   QM residuals to reconstruct corrected ensemble members or ensemble means,
   optionally including mixed-effect or KNN-based grouping strategies.

A small driver orchestrates these stages and evaluation steps: see
[droughtpp/analysis/qm_rf/main_qm_rf.py](droughtpp/analysis/qm_rf/main_qm_rf.py).

Prerequisites

- Python environment with the project requirements installed. See the project
  root `requirements.txt` or `pyproject.toml` for recommended packages.
- Access to the data layout expected by `config.yaml`: project root,
  external data roots, reference files, hindcast path jsons and output folders.

Configuration

The pipeline is driven by a YAML configuration file. The standard one is
[droughtpp/analysis/qm_rf/config.yaml](droughtpp/analysis/qm_rf/config.yaml). Key
features of the config system (see
[droughtpp/analysis/qm_rf/config_loader.py](droughtpp/analysis/qm_rf/config_loader.py)):

- Imports / composition: keys `import_config`, `imports`, or `extends` allow
  including and deep-merging external YAML fragments.
- Variable interpolation: values may contain `${var.name}` references which are
  resolved across the final merged config (supports dotted keys).
- `preprocessing_target` and `preprocessing_options` let you select one of
  multiple preprocessing presets (the loader will merge the selected preset
  into the top-level config).
- Runtime overrides: CLI-style overrides can be passed as `--set KEY=VALUE`
  (supporting dotted keys and YAML styles for values).

Important config keys (examples from the provided config):
- `project_root`, `external_data_root`: base paths used throughout the config.
- `experiment`, `leadtime_tag`: labels used for organizing outputs.
- `preprocessing_options`: candidate preprocessing presets (e.g. `cwb`,
  `cwb_intensity`). Use `preprocessing_target` to select one.
- `output_dir`, `model_dir`, `plot_dir`: output locations.
- `ml_arguments`, `qm_arguments`: model hyperparameters and QM settings.
- `workflow`: booleans controlling which steps run (run_qm, run_qm_eval,
  run_rf_train, run_rf_evaluate, etc.).

Running the pipeline

Run the high-level driver. This will load the YAML config, create the output
folder, and conditionally run QM, QM evaluation, RF training, and RF
evaluation according to the `workflow` section:

```bash
python -m droughtpp.analysis.qm_rf.main_qm_rf \
  --config droughtpp/analysis/qm_rf/config.yaml
```

Example: force running QM and RF train with overrides:

```bash
python -m droughtpp.analysis.qm_rf.main_qm_rf \
  --config droughtpp/analysis/qm_rf/config.yaml \
  --set workflow.run_qm=true --set workflow.run_rf_train=true
```

Notes on overrides: use `--set key=subvalue` and quoted lists for complex
values (see `config_loader.py` parsing). Example:

```bash
--set ml_arguments.n_estimators=100 --set output_dir=/tmp/qmrf_out
```

Running sub-steps and utilities

- Quantile mapping implementation: call the mapping routines or run the
  evaluation wrapper in `quantile_mapping/`. For automated workflows use the
  main driver.
- RF training & evaluation: see `random_forest/train_rf.py` and
  `random_forest/evaluate_rf.py` for entry points used by the driver.
- Timeline merge & evaluation helper: run
  `droughtpp.analysis.qm_rf.utils.merge_and_eval_full_timeline` to merge
  results and produce evaluation timelines:

```bash
python -m droughtpp.analysis.qm_rf.utils.merge_and_eval_full_timeline \
  --config droughtpp/analysis/qm_rf/config.yaml
```

Inputs and outputs

Inputs (configured in `config.yaml`):
- Hindcast path lists: JSON files listing input hindcast NetCDF paths.
- Observational/reference NetCDFs: used by QM (see
  `preprocessing_options.*.reference_data`).
- Additional reference features: can be attached via
  `additional_reference_features` (e.g. monthly variance, anomalies).

Outputs (written under `output_dir`):
- QM outputs: JSON path lists, corrected residuals, corrected hindcast
  NetCDFs and intermediate data.
- RF outputs: trained model files in `model_dir`, result NetCDFs and
  evaluation artifacts under `plot_dir` and `output_dir`.

File examples from the example config:
- Primary config: [droughtpp/analysis/qm_rf/config.yaml](droughtpp/analysis/qm_rf/config.yaml)
- Expected QM paths JSON: `${output_dir}/data/paths/qm_hindcast_paths.json`
- Example reference file: see `additional_reference_features` entries in
  the config for concrete reference NetCDF paths.

Key modules explained

- `main_qm_rf.py` — high-level orchestration for QM → RF workflows.
- `config_loader.py` — config merging, interpolation and CLI override handling.
- `quantile_mapping/` — QM model and evaluation logic. Produces corrected
  residuals and corrected hindcasts used as inputs to the RF stage.
- `random_forest/` — RF training, model wrappers, feature construction and
  alternate mixed-effect strategies (BLUP, KNN variants). The folder contains
  `train_rf.py`, `evaluate_rf.py`, `model_rf.py`, and helper code.
- `utils/` — evaluation and plotting helpers, preprocessing and small scripts
  used by orchestration. Notable scripts:
  - [droughtpp/analysis/qm_rf/utils/merge_and_eval_full_timeline.py](droughtpp/analysis/qm_rf/utils/merge_and_eval_full_timeline.py): merge results and compute time-series skill plots.
  - [droughtpp/analysis/qm_rf/utils/evaluation.py](droughtpp/analysis/qm_rf/utils/evaluation.py): evaluation helpers.
  - [droughtpp/analysis/qm_rf/utils/visualization.py](droughtpp/analysis/qm_rf/utils/visualization.py): plotting helpers used across runs.

Evaluation and plotting

The pipeline includes several evaluation routines: event probability evaluation,
CWB and SPEI evaluation, and plotting utilities for lead-month skill
timeseries. Enable the desired evaluation options in the `workflow` section of
the config. Output plots and summary tables are written to `plot_dir` and
`output_dir`.

Troubleshooting & tips

- Use `--set` overrides to experiment without editing the YAML file.
- If config interpolation fails, inspect `${...}` references for circular
  dependencies (config_loader will raise a helpful error).
- Large runs expect an established folder structure for `output_dir` and
  sufficient disk space for NetCDFs and model artefacts.
- Runtime warnings about empty slices often indicate missing or mis-shaped
  input arrays during evaluation — verify the JSON path lists and the input
  NetCDFs used by preprocessing.

Extending or adapting the pipeline

- Add preprocessing presets under `preprocessing_options` and select them via
  `preprocessing_target` for experiments.
- Experiment with model hyperparameters in `ml_arguments` or add new model
  types in `random_forest/model_rf.py`.
