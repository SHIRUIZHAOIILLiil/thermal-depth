#!/bin/bash
#SBATCH --job-name=iris_text_corr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PARENT="$IRIS_RUNS/outputs/lotus_line_v2"
BASE="$PARENT/da2_iris_fusion_matched_pair_v2"
OUT="$PARENT/iris_text_correction_empty6k_ssi_v2"
test -r "$BASE/empty_s43/step006000.pt"
test -r "$BASE/empty_s43/summary.json"
mkdir -p "$OUT"
exec 9>"$OUT/writer.lock"
flock -n 9 || { echo 'Another correction job is running'; exit 1; }
python tests/test_iris_text_correction.py
python -u tools/iris_text_correction.py --base "$BASE" \
  --assets "${DA2_CAPTION_ASSETS:-$IRIS_RUNS/da2_caption_assets/assets.json}" \
  --data-root "$IRIS_MS2_ROOT" \
  --train "$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl" \
  --val "$IRIS_MANIFEST_DIR/ms2_val_official3_thermalcap_20260821.jsonl"
