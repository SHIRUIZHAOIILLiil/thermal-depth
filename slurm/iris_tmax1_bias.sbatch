#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
F="${SCRATCH:?}/from_sc23sz"
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
base="$IRIS_RUNS/outputs/lotus_line_v2"
phase=${1:?Missing phase}
run=${2:?Missing output directory}
source_weights=${3:?Missing tmax1 weights}
echo "Output: $run"
if [[ "$phase" == smoke ]]; then python tests/test_iris_tmax1_bias.py; fi
python -u tools/train_iris_tmax1_bias.py "$phase" --run "$run" --source "$source_weights" \
    --recipe-run "$base/iris_far_pairs_l050_frame4k_v1/job_9163317" --data-root "$IRIS_MS2_ROOT" \
    --archive "$base/iris_pair_rmse_val_v1/job_9167854"
