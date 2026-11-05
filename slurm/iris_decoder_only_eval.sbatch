#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
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
run=${2:?Missing decoder run directory}
common=(--runs "$IRIS_RUNS" --run "$run" --data-root "$IRIS_MS2_ROOT" --manifests "$IRIS_MANIFEST_DIR")
case "$phase" in
    reproduce)
        python tests/test_iris_decoder_only_eval.py
        python -u tools/eval_iris_decoder_only.py reproduce "${common[@]}" ;;
    test)
        conditions=(day night rain)
        python -u tools/eval_iris_decoder_only.py test "${common[@]}" \
            --condition "${conditions[${SLURM_ARRAY_TASK_ID:?}]}" ;;
    *) exit 2 ;;
esac
