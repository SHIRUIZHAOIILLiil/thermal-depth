#!/bin/bash
#SBATCH --job-name=da2_if2_test
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
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
: "${ARM:?frame or empty}"
[[ "$ARM" == frame || "$ARM" == empty ]] || exit 1
BASE="$IRIS_RUNS/outputs/lotus_line_v2/da2_iris_fusion_matched_pair_v2"
exec 9>"$BASE/test_$ARM.lock"
flock -n 9 || { echo 'This arm already has a test writer'; exit 1; }
python tools/test_da2_iris_fusion_v2.py --arm "$ARM" --base "$BASE" \
  --assets "${DA2_CAPTION_ASSETS:-$IRIS_RUNS/da2_caption_assets/assets.json}" \
  --manifest-dir "$IRIS_MANIFEST_DIR" --data-root "$IRIS_MS2_ROOT"
