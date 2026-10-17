#!/bin/bash
#SBATCH --job-name=depth_chain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# Diagnostic only: no training, optimizer, checkpoint save, or test-set inference.
# CHAIN_SMOKE=1: 8 train + 8 val frames; default: 256 train + 96 val.
set -euo pipefail
F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
export IRIS_THERMAL_STRETCH=percentile IRIS_TEXT_PADDING=legacy IRIS_DECODE_CLAMP=0
RUNS="$IRIS_RUNS/iris_ms2"
Q098="$RUNS/sd2_cap_pct_logtn_img_l0100_s43/converted/step16000_weights.pt"
Q0995="$RUNS/sd2_cap_pct_logtn_img_l0100_tmax0995_s43/converted/step16000_weights.pt"
Q100="$RUNS/sd2_cap_pct_logtn_img_l0100_tmax1_s43/converted/step16000_weights.pt"
for file in "$Q098" "$Q0995" "$Q100"; do
  [[ -s "$file" ]] || { echo "Missing checkpoint: $file; no training or conversion will be started"; exit 1; }
done
NTRAIN="${CHAIN_TRAIN_FRAMES:-256}"
NVAL="${CHAIN_VAL_FRAMES:-96}"
if [[ "${CHAIN_SMOKE:-0}" == 1 ]]; then NTRAIN=8; NVAL=8; fi
BASE="$IRIS_RUNS/analysis/depth_chain"
mkdir -p "$BASE"
OUT=$(mktemp -d "$BASE/attempt_${SLURM_JOB_ID:-local}_XXXXXX")
echo "DIAGNOSTIC_OUT=$OUT"
git log --oneline -1
python tools/diagnose_depth_chain.py --phase prepare --out "$OUT" \
  --data-root "$IRIS_MS2_ROOT" --pseudo-dir "${PSEUDO_DIR:-$F/pseudo_gt/official_train/calibrated_pseudo_depth}" \
  --train-manifest "$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl" \
  --val-manifest "$IRIS_MANIFEST_DIR/ms2_val_official3_thermalcap_20260821.jsonl" \
  --train-list "$IRIS_MS2_ROOT/train_list.txt" --val-list "$IRIS_MS2_ROOT/val_list.txt" \
  --train-frames "$NTRAIN" --val-frames "$NVAL" --seed 20261017 \
  --checkpoint "q098=$Q098" --checkpoint "q0995=$Q0995" --checkpoint "q100=$Q100"
for phase in targets models score; do
  python tools/diagnose_depth_chain.py --phase "$phase" --out "$OUT"
done
echo "COMPLETED: $OUT/analysis/summary.csv"
echo "No weights trained or modified; all predictions and frozen manifests retained."
