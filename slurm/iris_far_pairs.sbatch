#!/bin/bash
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
echo "Output: ${PAIR_RUN:?}"
entry=(python -u tools/train_iris_far_pairs.py)
args=(--run "$PAIR_RUN" --manifests "$IRIS_MANIFEST_DIR" --data-root "$IRIS_MS2_ROOT"
      --pseudo-dir "${PSEUDO_DIR:-$F/pseudo_gt/official_train/calibrated_pseudo_depth}"
      --checkpoints "$IRIS_RUNS/iris_ms2" "$F/runs/iris_ms2" "$F/iris_ms2" /mnt/scratch/sc23sz/runs/iris_ms2)
case "${ACTION:?}" in
    smoke)
        python tests/test_iris_far_pairs.py
        for arm in frame empty; do "${entry[@]}" smoke --arm "$arm" "${args[@]}"; done
        "${entry[@]}" check --run "$PAIR_RUN" ;;
    train) "${entry[@]}" train --arm "${ARM:?}" "${args[@]}" ;;
    report) "${entry[@]}" report --run "$PAIR_RUN" ;;
    *) exit 2 ;;
esac
