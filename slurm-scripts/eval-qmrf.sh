#!/bin/bash

#SBATCH -J qm-rf_train
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --account=bm1159
#SBATCH --time=10:00:00
#SBATCH --mem=480G
#SBATCH --constraint a100_80
#SBATCH --output=outputs/eval-qmrf-job.o%j 

module load python3/2023.01-gcc-11.2.0
cd /work/bk1318/k202208/crai/hindcast-pp/Ml-Drought-Postprocessing/
source activate crai

python -m droughtpp.analysis.qm_rf.qm_rf_analysis droughtpp/analysis/qm_rf/config.yaml
