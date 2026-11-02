#!/bin/bash
#SBATCH --job-name=iris_fs_recover
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
module load miniforge/24.7.1
conda activate iris
cd "${IRIS_REPO:-$HOME/Iris}"
python tools/recover_iris_far_stretch.py "${IRIS_RUNS:?}/outputs/lotus_line_v2/iris_far_stretch_l050_v1/job_${PARENT_ID:?}"
