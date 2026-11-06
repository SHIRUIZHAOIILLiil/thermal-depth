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
phase=${1:?mode}
run=${2:?run}
if [[ "$phase" == report ]]; then
    python -u tools/train_iris_decoder_joint.py report --run "$run"
    exit
fi
if [[ "$phase" == smoke ]]; then
    python tests/test_iris_decoder_joint.py
fi
args=(--run "$run" --source-run "${3:?source run}" --runs "$IRIS_RUNS" --data-root "$IRIS_MS2_ROOT")
if [[ "$phase" == train ]]; then args+=(--arm "${4:?arm}"); fi
python -u tools/train_iris_decoder_joint.py "$phase" "${args[@]}"
