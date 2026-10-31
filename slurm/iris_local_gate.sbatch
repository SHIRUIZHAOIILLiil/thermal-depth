#!/bin/bash
#SBATCH --job-name=iris_gate
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
ROOT="$IRIS_RUNS/outputs/lotus_line_v2"
OUT="$ROOT/iris_visual_gate_matched_v1"
ARGS=(--base "$ROOT/da2_iris_fusion_matched_pair_v2" --assets "${DA2_CAPTION_ASSETS:-$IRIS_RUNS/da2_caption_assets/assets.json}"
  --data-root "$IRIS_MS2_ROOT" --train "$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl"
  --val "$IRIS_MANIFEST_DIR/ms2_val_official3_thermalcap_20260821.jsonl")
mkdir -p "$OUT"
if [[ "${ACTION:?}" == smoke ]]; then
  python tests/test_iris_local_gate.py
  SMOKE="$OUT/smoke_$SLURM_JOB_ID"
  for arm in frame empty; do
    python -u tools/train_iris_local_gate.py smoke "${ARGS[@]}" --out "$SMOKE" --arm "$arm"
  done
  python tools/train_iris_local_gate.py pair-smoke --out "$SMOKE"
elif [[ "$ACTION" == train ]]; then
  [[ "${ARM:?}" == frame || "$ARM" == empty ]]
  exec 9>"$OUT/$ARM.lock"
  flock -n 9 || { echo "Another $ARM writer is running"; exit 1; }
  python -u tools/train_iris_local_gate.py train "${ARGS[@]}" --out "$OUT" --arm "$ARM" --smoke "${SMOKE:?}"
  if [[ -f "$OUT/empty/summary.json" && -f "$OUT/frame/summary.json" ]]; then
    python tools/train_iris_local_gate.py report --out "$OUT"
  fi
else
  exit 1
fi
