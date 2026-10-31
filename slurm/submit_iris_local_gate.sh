#!/bin/bash
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
export IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
root="$IRIS_RUNS/outputs/lotus_line_v2"
base="$root/da2_iris_fusion_matched_pair_v2/empty_s43"
out="$root/iris_visual_gate_matched_v1"
test -r "$base/step006000.pt"
test -r "$base/summary.json"
test -r "$base/selected_full/val_step006000_frames.csv"
test -r "${DA2_CAPTION_ASSETS:-$IRIS_RUNS/da2_caption_assets/assets.json}"
for arm in frame empty; do
  [[ ! -e "$out/$arm" ]] || { echo "Existing $out/$arm; use explicit per-arm resubmission"; exit 1; }
done
smoke=$(sbatch --parsable -J iris_lg_smoke --time=02:00:00 --export=ALL,ACTION=smoke slurm/iris_local_gate.sbatch)
smoke=${smoke%%;*}
[[ "$smoke" =~ ^[0-9]+$ ]]
echo "paired smoke -> $smoke"
export SMOKE="$out/smoke_$smoke"
for arm in frame empty; do
  id=$(sbatch --parsable -J "iris_lg_$arm" --dependency="afterok:$smoke" --kill-on-invalid-dep=yes \
    --export="ALL,ACTION=train,ARM=$arm" slurm/iris_local_gate.sbatch)
  echo "$arm train+val -> ${id%%;*} (afterok:$smoke)"
done
echo "Outputs: $out"
