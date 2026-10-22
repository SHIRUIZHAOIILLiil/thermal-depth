#!/bin/bash
#SBATCH --job-name=da2_cont_val
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
: "${ARM:?set ARM=no-caption or caption}"
[[ "$ARM" == no-caption || "$ARM" == caption ]] || exit 1
OUT="$IRIS_RUNS/da2_caption_v1/${ARM}_s43"
exec 9>"$OUT/.continuation.lock"
flock -n 9 || { echo 'Another continuation/validation job owns this arm'; exit 1; }
python tests/test_da2_continuation.py
python tools/eval_da2_continuation.py --run-dir "$OUT" \
  --assets "${DA2_CAPTION_ASSETS:-$IRIS_RUNS/da2_caption_assets/assets.json}" \
  --data-root "$IRIS_MS2_ROOT" \
  --val-manifest "$IRIS_MANIFEST_DIR/ms2_val_official3_thermalcap_20260821.jsonl"
