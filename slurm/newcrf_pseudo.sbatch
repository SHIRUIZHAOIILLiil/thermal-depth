#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
F=${SCRATCH:?}/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
: "${RUN_DIR:?use submit_newcrf_pseudo.sh}" "${STAGE:?}"
args=(--out "$RUN_DIR" --data-root "$IRIS_MS2_ROOT" --manifest-dir "$IRIS_MANIFEST_DIR"
      --bmsd-root "${BMSD_ROOT:-$SCRATCH/baselines/SupDepth4Thermal}"
      --checkpoint "${NEWCRF_CKPT:-$SCRATCH/baselines/supdepth4thermal/MS2_MD_NeWCRF_THR_ckpt.ckpt}")
case "$STAGE" in
    smoke|train)
        args+=(--pseudo-dir "${PSEUDO_DIR:-$F/pseudo_gt/official_train/calibrated_pseudo_depth}")
        if [[ $STAGE == smoke ]]; then
            python tests/test_newcrf_pseudo.py
        fi ;;
    test) args+=(--condition "${CONDITION:?day/night/rain}") ;;
    *) exit 2 ;;
esac
python tools/newcrf_pseudo.py "$STAGE" "${args[@]}"
