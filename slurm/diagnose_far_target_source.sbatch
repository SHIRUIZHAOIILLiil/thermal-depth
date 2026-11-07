#!/bin/bash
#SBATCH --job-name=iris_far_source
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
F="${SCRATCH:?}/from_sc23sz"
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
for dir in "$SCRATCH/code/BridgeMultiSpectralDepth" "$F/code/BridgeMultiSpectralDepth" "$SCRATCH/baselines/SupDepth4Thermal"; do
    if [[ -d "$dir/models/network/midas" ]]; then export IRIS_BMSD_ROOT="${IRIS_BMSD_ROOT:-$dir}"; break; fi
done
for file in "$SCRATCH"/models/anythermal_midas/Midas_anythermal/*.ckpt "$F"/models/anythermal_midas/Midas_anythermal/*.ckpt; do
    if [[ -f "$file" ]]; then export IRIS_ANYTHERMAL_CKPT="${IRIS_ANYTHERMAL_CKPT:-$file}"; break; fi
done
base="$IRIS_RUNS/outputs/lotus_line_v2"
python tests/test_far_target_source.py
python -u tools/diagnose_far_target_source.py --runs "$IRIS_RUNS" --data-root "$IRIS_MS2_ROOT" \
    --bmsd-root "$SCRATCH/baselines/SupDepth4Thermal" --ckpt-dir "$SCRATCH/baselines/supdepth4thermal" \
    --out "$base/iris_far_target_source_fitonly_v2/job_${SLURM_JOB_ID:?}"
