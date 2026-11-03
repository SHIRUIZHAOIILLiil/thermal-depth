#!/bin/bash
#SBATCH --job-name=iris_cross_frame
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
parent_job=${1:?Pass the frame continuation job ID}
[[ "$parent_job" =~ ^[0-9]+$ ]]
F="${SCRATCH:?}/from_sc23sz"
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
base="$IRIS_RUNS/outputs/lotus_line_v2"
parent="$base/iris_far_pairs_l050_frame4k_v1/job_$parent_job"
out="$base/iris_cross_region_l050_frame_v1/job_${SLURM_JOB_ID:?}"
python tests/test_iris_cross_region_pairs.py
python -u tools/train_iris_cross_region_pairs.py --parent "$parent" --data-root "$IRIS_MS2_ROOT" --out "$out"
