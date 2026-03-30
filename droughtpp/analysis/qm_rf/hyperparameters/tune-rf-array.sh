#!/bin/bash

#SBATCH -J qm-rf_tune
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --account=bm1159
#SBATCH --time=10:00:00
#SBATCH --mem=480G
#SBATCH --constraint a100_80
#SBATCH --array=0-7
#SBATCH --output=outputs/tune-rf-%A_%a.out

set -euo pipefail

module load python3/2023.01-gcc-11.2.0
cd /work/bk1318/k202208/crai/hindcast-pp/Ml-Drought-Postprocessing/
source activate crai

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    MODEL_TYPE="rf"
    N_EST=300
    CHUNK=30
    RS=42
    VAL_FRAC=0.2
    ;;
  1)
    MODEL_TYPE="xgboost"
    N_EST=300
    CHUNK=30
    RS=42
    VAL_FRAC=0.2
    XGB_LR=0.10
    XGB_MD=6
    XGB_SS=0.80
    XGB_CS=0.80
    XGB_MCW=1
    XGB_METRIC="rmse"
    ;;
  2)
    MODEL_TYPE="xgboost"
    N_EST=300
    CHUNK=30
    RS=42
    VAL_FRAC=0.2
    XGB_LR=0.10
    XGB_MD=6
    XGB_SS=0.80
    XGB_CS=0.80
    XGB_MCW=1
    XGB_METRIC="rmse"
    ;;
  3)
    MODEL_TYPE="xgboost"
    N_EST=500
    CHUNK=30
    RS=42
    VAL_FRAC=0.2
    XGB_LR=0.05
    XGB_MD=6
    XGB_SS=0.80
    XGB_CS=0.80
    XGB_MCW=1
    XGB_METRIC="rmse"
    ;;
  4)
    MODEL_TYPE="xgboost"
    N_EST=500
    CHUNK=30
    RS=42
    VAL_FRAC=0.2
    XGB_LR=0.10
    XGB_MD=4
    XGB_SS=0.80
    XGB_CS=0.80
    XGB_MCW=1
    XGB_METRIC="rmse"
    ;;
  5)
    MODEL_TYPE="xgboost"
    N_EST=500
    CHUNK=30
    RS=42
    VAL_FRAC=0.2
    XGB_LR=0.10
    XGB_MD=8
    XGB_SS=0.80
    XGB_CS=0.80
    XGB_MCW=1
    XGB_METRIC="rmse"
    ;;
  6)
    MODEL_TYPE="xgboost"
    N_EST=500
    CHUNK=30
    RS=42
    VAL_FRAC=0.2
    XGB_LR=0.10
    XGB_MD=6
    XGB_SS=0.60
    XGB_CS=0.60
    XGB_MCW=1
    XGB_METRIC="rmse"
    ;;
  7)
    MODEL_TYPE="xgboost"
    N_EST=500
    CHUNK=30
    RS=42
    VAL_FRAC=0.2
    XGB_LR=0.10
    XGB_MD=6
    XGB_SS=0.90
    XGB_CS=0.90
    XGB_MCW=2
    XGB_METRIC="rmse"
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
    ;;
esac

CMD=(
  python -m droughtpp.analysis.qm_rf.main_qm_rf
  --config droughtpp/analysis/qm_rf/config.yaml
  --set workflow.run_qm=false
  --set workflow.run_qm_eval=false
  --set workflow.run_rf_train=true
  --set workflow.run_rf_evaluate=true
  --set workflow.evaluate_cwb=true
  --set workflow.evaluate_spei=false
  --set ml_arguments.model_type="${MODEL_TYPE}"
  --set ml_arguments.n_estimators="${N_EST}"
  --set ml_arguments.chunk_size="${CHUNK}"
  --set ml_arguments.random_state="${RS}"
  --set ml_arguments.val_fraction="${VAL_FRAC}"
)

if [[ "${MODEL_TYPE}" == "xgboost" ]]; then
  CMD+=(
    --set ml_arguments.xgb_learning_rate="${XGB_LR}"
    --set ml_arguments.xgb_max_depth="${XGB_MD}"
    --set ml_arguments.xgb_subsample="${XGB_SS}"
    --set ml_arguments.xgb_colsample_bytree="${XGB_CS}"
    --set ml_arguments.xgb_min_child_weight="${XGB_MCW}"
    --set ml_arguments.xgb_eval_metric="${XGB_METRIC}"
  )
fi

echo "Running sweep task ${SLURM_ARRAY_TASK_ID}: ${MODEL_TYPE}"
printf '%q ' "${CMD[@]}"; echo

"${CMD[@]}"
