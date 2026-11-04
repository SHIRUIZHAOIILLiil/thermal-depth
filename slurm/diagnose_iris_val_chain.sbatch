#!/bin/bash
#SBATCH --job-name=iris_val_chain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
F="${SCRATCH:?}/from_sc23sz"
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
# Only needed if the original AnyThermal val cache does not cover the frozen subset.
for dir in "$SCRATCH/code/BridgeMultiSpectralDepth" "$F/code/BridgeMultiSpectralDepth" "$SCRATCH/baselines/SupDepth4Thermal"; do
    if [[ -d "$dir/models/network/midas" ]]; then export IRIS_BMSD_ROOT="${IRIS_BMSD_ROOT:-$dir}"; break; fi
done
for file in "$SCRATCH"/models/anythermal_midas/Midas_anythermal/*.ckpt "$F"/models/anythermal_midas/Midas_anythermal/*.ckpt; do
    if [[ -f "$file" ]]; then export IRIS_ANYTHERMAL_CKPT="${IRIS_ANYTHERMAL_CKPT:-$file}"; break; fi
done
base="$IRIS_RUNS/outputs/lotus_line_v2"
out="$base/iris_val_chain_bias4k_v1/job_${SLURM_JOB_ID:?}"
python tests/test_iris_val_chain.py
python -u tools/diagnose_iris_val_chain.py --runs "$IRIS_RUNS" --data-root "$IRIS_MS2_ROOT" --out "$out" \
    --cache "${ANYTHERMAL_VAL_RUN:-$IRIS_RUNS/anythermal/Midas_anythermal}" \
    --cache "$F/runs/anythermal/Midas_anythermal" --cache "$base/anythermal_midas_val_full_rerun" \
    --cache "$base/anythermal_midas_val_full"
