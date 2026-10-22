#!/bin/bash
#SBATCH --job-name=da2cap_ablate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python tests/test_da2_caption_ablation.py
python tools/eval_da2_caption_ablation.py \
  --run-dir "$IRIS_RUNS/da2_caption_v1/caption_s43" \
  --assets "${DA2_CAPTION_ASSETS:-$IRIS_RUNS/da2_caption_assets/assets.json}" \
  --data-root "$IRIS_MS2_ROOT" \
  --val-manifest "$IRIS_MANIFEST_DIR/ms2_val_official3_thermalcap_20260821.jsonl" \
  --shuffle-seed 43 \
  --output "$IRIS_RUNS/da2_caption_v1/ablation_s8000_${SLURM_JOB_ID}"
