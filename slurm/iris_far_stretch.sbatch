#!/bin/bash
#SBATCH --job-name=iris_far_stretch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
F="${SCRATCH:?}/from_sc23sz"
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
OUT="$IRIS_RUNS/outputs/lotus_line_v2/iris_far_stretch_l050_v1/job_${SLURM_JOB_ID:?}"
echo "Output: $OUT"
python tests/test_iris_far_stretch.py
python -u tools/iris_far_stretch.py --out "$OUT" --data-root "$IRIS_MS2_ROOT" \
    --manifests "$IRIS_MANIFEST_DIR" --checkpoints "$IRIS_RUNS/iris_ms2" \
    "$F/runs/iris_ms2" "$F/iris_ms2" /mnt/scratch/sc23sz/runs/iris_ms2
