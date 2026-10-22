#!/bin/bash
#SBATCH --job-name=da2_continue
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
: "${ARM:?set ARM=no-caption or caption}"
: "${SMOKE_OUTPUT:?set original successful paired smoke directory}"
[[ "$ARM" == no-caption || "$ARM" == caption ]] || exit 1
OUT="$IRIS_RUNS/da2_caption_v1/${ARM}_s43"
# All new continuation/validation jobs cooperate on this arm-specific lock.
# The read-only 8000 ablation does not need the lock.
exec 9>"$OUT/.continuation.lock"
flock -n 9 || { echo 'Another continuation/validation job owns this arm'; exit 1; }
python tests/test_da2_continuation.py
python tools/da2_continuation.py preflight --run-dir "$OUT" --arm "$ARM" --smoke-output "$SMOKE_OUTPUT"
python tools/train_da2_caption.py --mode train --arm "$ARM" --resume --stop-step 20000 \
  --output "$OUT" --smoke-contract "$SMOKE_OUTPUT/no-caption/contract.json" \
  --assets "${DA2_CAPTION_ASSETS:-$IRIS_RUNS/da2_caption_assets/assets.json}" \
  --data-root "$IRIS_MS2_ROOT" \
  --train-manifest "$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl" \
  --val-manifest "$IRIS_MANIFEST_DIR/ms2_val_official3_thermalcap_20260821.jsonl" \
  --caption-source thermal --seed 43
