#!/bin/bash
#SBATCH --job-name=da2cap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
ACTION="${ACTION:-smoke}"
ASSETS="${DA2_CAPTION_ASSETS:-$IRIS_RUNS/da2_caption_assets/assets.json}"
BASE="$IRIS_RUNS/da2_caption_v1"
ARGS=(--assets "$ASSETS" --data-root "$IRIS_MS2_ROOT"
  --train-manifest "$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl"
  --val-manifest "$IRIS_MANIFEST_DIR/ms2_val_official3_thermalcap_20260821.jsonl"
  --caption-source thermal --seed 43)
for test in test_da2_caption; do
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python "tests/$test.py"
done
git log -1 --oneline
if [[ "$ACTION" == smoke ]]; then
  OUT="$BASE/smoke_${SLURM_JOB_ID}"
  for ARM in no-caption caption; do
    python tools/train_da2_caption.py --mode smoke --arm "$ARM" --output "$OUT/$ARM" "${ARGS[@]}"
  done
  # Marker is written only after both real GPU smoke runs and the pair audit succeed.
  python tools/check_da2_caption_pair.py --a "$OUT/no-caption" --b "$OUT/caption" --smoke
  echo "SMOKE_OUTPUT=$OUT"
elif [[ "$ACTION" == pilot ]]; then
  : "${ARM:?set ARM=no-caption or caption}"
  : "${SMOKE_OUTPUT:?set the successful smoke output path}"
  [[ "$ARM" == no-caption || "$ARM" == caption ]] || exit 1
  python tools/check_da2_caption_pair.py --a "$SMOKE_OUTPUT/no-caption" --b "$SMOKE_OUTPUT/caption" --smoke
  RESUME_ARGS=()
  [[ "${RESUME:-0}" == 1 ]] && RESUME_ARGS=(--resume)
  python tools/train_da2_caption.py --mode train --arm "$ARM" --stop-step 8000 \
    --output "$BASE/${ARM}_s43" --smoke-contract "$SMOKE_OUTPUT/no-caption/contract.json" \
    "${ARGS[@]}" "${RESUME_ARGS[@]}"
else
  echo "Unknown ACTION=$ACTION; only smoke/pilot are authorized here" >&2
  exit 1
fi
