#!/bin/bash
#SBATCH --job-name=iris_lg_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[[ "${ARM:?}" == frame || "$ARM" == empty ]]
parent="$IRIS_RUNS/outputs/lotus_line_v2"
run="$parent/iris_visual_gate_matched_v1"
mkdir -p "$run/test_selected_v1"
exec 9>"$run/test_selected_v1/$ARM.lock"
flock -n 9 || { echo "Another test is writing $ARM"; exit 1; }
python -u tools/test_iris_local_gate.py --arm "$ARM" --run "$run" \
  --base "$parent/da2_iris_fusion_matched_pair_v2" \
  --assets "${DA2_CAPTION_ASSETS:-$IRIS_RUNS/da2_caption_assets/assets.json}" \
  --manifest-dir "$IRIS_MANIFEST_DIR" --data-root "$IRIS_MS2_ROOT"
