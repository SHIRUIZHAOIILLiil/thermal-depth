#!/bin/bash
#SBATCH --job-name=da2_iris_fusion_v2
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
BASE="$IRIS_RUNS/outputs/lotus_line_v2/da2_iris_fusion_matched_pair_v2"
mkdir -p "$BASE"
: "${IRIS_CAP_FEATURE_CKPT:?resolved by submit_da2_iris_fusion_v2.sh}"
: "${IRIS_EMPTY_FEATURE_CKPT:?resolved by submit_da2_iris_fusion_v2.sh}"
ARGS=(--assets "${DA2_CAPTION_ASSETS:-$IRIS_RUNS/da2_caption_assets/assets.json}"
  --iris-assets "$BASE/iris_assets.json"
  --base-checkpoint "$IRIS_RUNS/da2_caption_v1/no-caption_s43/step018000.pt"
  --iris-caption-checkpoint "$IRIS_CAP_FEATURE_CKPT"
  --iris-empty-checkpoint "$IRIS_EMPTY_FEATURE_CKPT"
  --data-root "$IRIS_MS2_ROOT" --cache-dir "$BASE/tokens"
  --train-manifest "$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl"
  --val-manifest "$IRIS_MANIFEST_DIR/ms2_val_official3_thermalcap_20260821.jsonl")
if [[ "${ACTION:-smoke}" == smoke ]]; then
  python tests/test_da2_iris_fusion.py
  python tests/test_da2_iris_fusion_v2.py
  OUT="$BASE/smoke_${SLURM_JOB_ID}"
  exec 9>"$BASE/smoke.lock"
  flock -n 9 || { echo 'Another smoke owns asset/token preparation'; exit 1; }
  for arm in frame empty; do
    python tools/train_da2_iris_fusion_v2.py --mode smoke --arm "$arm" --output "$OUT/$arm" "${ARGS[@]}"
  done
  python -c 'import sys; from pathlib import Path; from tools.train_da2_iris_fusion_v2 import pair_smoke; pair_smoke(Path(sys.argv[1]))' "$OUT"
  echo "SMOKE_OUTPUT=$OUT"
elif [[ "$ACTION" == train ]]; then
  : "${ARM:?frame or empty}"; : "${SMOKE_OUTPUT:?paired successful smoke}"
  [[ "$ARM" == frame || "$ARM" == empty ]] || exit 1
  exec 9>"$BASE/$ARM.lock"
  flock -n 9 || { echo 'This arm already has a writer'; exit 1; }
  MORE=()
  [[ "${RESUME:-0}" == 1 ]] && MORE=(--resume)
  python tools/train_da2_iris_fusion_v2.py --mode train --arm "$ARM" --output "$BASE/${ARM}_s43" \
    --smoke-output "$SMOKE_OUTPUT" "${ARGS[@]}" "${MORE[@]}"
else
  echo 'ACTION must be smoke or train'; exit 1
fi
