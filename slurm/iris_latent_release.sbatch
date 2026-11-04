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
phase=${1:?Missing phase}
run=${2:?Missing output directory}
arm=${3:-control}
echo "Output: $run"
case "$phase" in
    smoke)
        python tests/test_iris_latent_release.py
        python -u tools/train_iris_latent_release.py smoke --run "$run" --runs "$IRIS_RUNS" --data-root "$IRIS_MS2_ROOT" ;;
    train)
        python -u tools/train_iris_latent_release.py train --run "$run" --runs "$IRIS_RUNS" --data-root "$IRIS_MS2_ROOT" --arm "$arm" ;;
    test)
        conditions=(day night rain)
        python -u tools/test_iris_latent_release.py --run "$run" --runs "$IRIS_RUNS" --data-root "$IRIS_MS2_ROOT" \
            --manifests "$IRIS_MANIFEST_DIR" --arm "$arm" --condition "${conditions[$SLURM_ARRAY_TASK_ID]}" ;;
    *) exit 2 ;;
esac
