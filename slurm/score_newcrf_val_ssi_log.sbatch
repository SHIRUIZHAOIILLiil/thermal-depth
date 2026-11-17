#!/bin/bash
#SBATCH --job-name=crf_val_log
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# NeWCRF full official val under ssi and ssi_log (tools/score_newcrf_val_ssi_log.py). CPU only.
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
module load miniforge/24.7.1
conda activate iris
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
base="${IRIS_RUNS:-${SCRATCH:?}/runs}/outputs/lotus_line_v2"
python -u tools/score_newcrf_val_ssi_log.py --previous "$base/iris_pair_rmse_val_v1/job_9167854" \
    --data-root "${IRIS_MS2_ROOT:-$SCRATCH/from_sc23sz/data/ms2}" \
    --out "$base/newcrf_val_ssi_log_v1/job_${SLURM_JOB_ID:?}"
