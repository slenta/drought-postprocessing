# `droughtpp` (scope: `analysis/qm_rf`)

This README currently documents only the `droughtpp/analysis/qm_rf` workflow.

## What this folder does

`analysis/qm_rf` implements a postprocessing workflow for hindcasts with:

1. **Quantile Mapping (QM)** against a reference dataset
2. **Random Forest (RF)** residual modeling
3. Optional **SPEI generation and SPEI-based evaluation** for corrected outputs

The top-level orchestrator is `qm_rf_analysis.py`, which can run QM, QM evaluation, RF training, and RF evaluation in one sequence, controlled by config flags.

## Folder structure

```text
droughtpp/analysis/qm_rf/
├── config.yaml                      # Main runtime configuration
├── config_loader.py                 # YAML loading, imports/extends, ${var} interpolation, CLI --set overrides
├── qm_rf_analysis.py                # Main orchestrator for QM + RF workflow
├── evaluation.py                    # Distribution extraction helpers for QM outputs
├── visualization.py                 # Plotting utilities (histograms, MAE/RMSE/bias maps, BSS maps)
├── evaluate_spei.py                 # Generic SPEI-vs-reference evaluation helper
├── spei_from_rf.py                  # Build corrected CWB (QM + RF residual), compute SPEI
├── spei_evaluation.py               # SPEI evaluation for corrected ensemble vs baseline hindcasts
├── quantile_mapping/
│   ├── quantile_mapping.py          # Applies QM per file listed in JSON
│   └── qm_evaluate.py               # Runs QM distribution + skill metric evaluation
└── random_forest/
    ├── rf_net.py                    # RF bias-corrector helper class
    ├── train.py                     # RF training on residual targets
    └── evaluate.py                  # RF evaluation + residual export + optional SPEI workflow
```

## Configuration (`config.yaml`)

Key groups used by the pipeline:

- **Paths**
  - `project_root`, `external_data_root`
  - `hindcasts_json`, `residuals_json`, `qm_json_path`
  - `reference_data` / `reference`
  - `output_dir`, `model_path`, `corrected_spei_paths_json`
- **Variable and split setup**
  - `var_name`, `predictor_vars`, `leave_out_years`
- **Workflow toggles**
  - `workflow.run_qm`
  - `workflow.run_qm_eval`
  - `workflow.run_rf_train`
  - `workflow.run_rf_evaluate`
- **QM parameters**
  - `n_quantiles`, `plot_dir`
- **RF parameters**
  - `n_estimators`, `chunk_size`, `random_state`
- **SPEI options**
  - `evaluate_spei`, `generate_corrected_spei`
  - `spei_month_range`, `spei_std_multiplier`

`config_loader.py` supports:

- config inheritance via `import_config` / `imports` / `extends`
- variable interpolation like `${project_root}`
- runtime overrides via repeated `--set KEY=VALUE`

## Input expectations

- `hindcasts_json`: JSON list of hindcast NetCDF file paths
- `residuals_json`: JSON list of residual NetCDF file paths aligned with hindcasts
- `reference_data`: NetCDF reference dataset used by QM/evaluation
- For current default config, variable name is `CWB`

Most routines assume spatiotemporal grids with time and lat/lon dimensions.

## Outputs

Inside `output_dir` (for example `droughtpp/eval_results/<experiment>/qm_rf`), the workflow can create:

- QM files in `data/qm_hindcast/`: `<member>_qm.nc`
- QM file list in `data/qm_hindcast_paths.json`
- QM residuals: `<member>_qm_residual.nc`
- QM plots in `plot_dir`:
  - per-member and ensemble distribution plots
  - MAE / RMSE / Mean Bias comparison maps
  - BSS maps for upper/lower tails
- RF artifacts:
  - trained model (`rf_model.joblib`) when training is enabled
  - predicted residual files in `rf_residuals/`
  - corrected SPEI files in `data/spei_corrected/`
- Metrics:
  - `evaluation_metrics.yaml`
  - `spei_evaluation.nc` (when SPEI evaluation is enabled)

## How to run

Run from the project root (`Ml-Drought-Postprocessing`) so package imports resolve.

### 1) Full orchestrated run (recommended)

```bash
python -m droughtpp.analysis.qm_rf.qm_rf_analysis \
  --config droughtpp/analysis/qm_rf/config.yaml
```

### 2) Override config values at runtime

```bash
python -m droughtpp.analysis.qm_rf.qm_rf_analysis \
  --config droughtpp/analysis/qm_rf/config.yaml \
  --set workflow.run_qm=true \
  --set workflow.run_rf_train=false \
  --set n_quantiles=50
```

### 3) Run components directly

```bash
# RF training only
python -m droughtpp.analysis.qm_rf.random_forest.train \
  --config droughtpp/analysis/qm_rf/config.yaml

# RF evaluation (+ optional SPEI generation/evaluation based on config)
python -m droughtpp.analysis.qm_rf.random_forest.evaluate \
  --config droughtpp/analysis/qm_rf/config.yaml

# QM evaluation plots/metrics (expects QM outputs)
python -m droughtpp.analysis.qm_rf.quantile_mapping.qm_evaluate
```

## Notes and current limitations

- `hindcasts_json` and `residuals_json` are expected to be aligned (same ordering by member/file).
- Some scripts use `var_name="CWB"` assumptions in SPEI-related code paths.
- `quantile_mapping.py` relies on `cmethods.adjust` with method `quantile_mapping`.
- This README intentionally excludes other `droughtpp` modules for now.