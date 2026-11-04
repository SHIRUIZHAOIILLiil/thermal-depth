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
parent="$base/iris_far_pairs_l050_frame4k_v1/job_9163317"
run=${2:?Missing output directory}
echo "Output: $run"
case "${1:?Missing phase}" in
    smoke|train)
        if [[ "$1" == smoke ]]; then python tests/test_iris_far_bias_square.py; fi
        python -u tools/train_iris_far_bias_square.py "$1" --run "$run" --parent "$parent" \
            --data-root "$IRIS_MS2_ROOT" --archive "$base/iris_pair_rmse_val_v1/job_9167854" ;;
    test)
        conditions=(day night rain)
        python -u tools/test_iris_cross_region_pairs.py --run "$run" --parent "$parent" \
            --condition "${conditions[$SLURM_ARRAY_TASK_ID]}" --data-root "$IRIS_MS2_ROOT" --manifests "$IRIS_MANIFEST_DIR" ;;
    *) exit 2 ;;
esac
