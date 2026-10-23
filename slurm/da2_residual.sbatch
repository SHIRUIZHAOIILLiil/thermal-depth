#!/bin/bash
#SBATCH --job-name=da2_residual
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
BASE="$IRIS_RUNS/da2_caption_residual_v1"
mkdir -p "$BASE"
ARGS=(--assets "${DA2_CAPTION_ASSETS:-$IRIS_RUNS/da2_caption_assets/assets.json}"
  --base-checkpoint "$IRIS_RUNS/da2_caption_v1/no-caption_s43/step018000.pt"
  --data-root "$IRIS_MS2_ROOT" --cache-dir "$BASE/tokens"
  --train-manifest "$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl"
  --val-manifest "$IRIS_MANIFEST_DIR/ms2_val_official3_thermalcap_20260821.jsonl")
python tests/test_da2_residual.py
if [[ "${ACTION:-smoke}" == smoke ]]; then
  OUT="$BASE/smoke_${SLURM_JOB_ID}"
  # Serial smoke prevents shared token-cache writers racing.
  exec 9>"$BASE/smoke.lock"
  flock -n 9 || { echo 'Another residual smoke owns the cache'; exit 1; }
  for arm in frame fixed; do
    python tools/train_da2_residual.py --mode smoke --arm "$arm" --output "$OUT/$arm" "${ARGS[@]}"
  done
  python tools/check_da2_residual.py --smoke --base "$OUT"
  echo "SMOKE_OUTPUT=$OUT"
elif [[ "$ACTION" == train ]]; then
  : "${ARM:?frame or fixed}"; : "${SMOKE_OUTPUT:?successful paired residual smoke directory}"
  [[ "$ARM" == frame || "$ARM" == fixed ]] || exit 1
  exec 9>"$BASE/$ARM.lock"
  flock -n 9 || { echo 'This residual arm is already running'; exit 1; }
  MORE=()
  [[ "${RESUME:-0}" == 1 ]] && MORE=(--resume)
  python tools/train_da2_residual.py --mode train --arm "$ARM" --output "$BASE/${ARM}_s43" \
    --smoke-output "$SMOKE_OUTPUT" "${ARGS[@]}" "${MORE[@]}"
else
  echo 'ACTION must be smoke or train'; exit 1
fi
