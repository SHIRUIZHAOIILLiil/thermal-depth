#!/bin/bash
# New experiment only: paired smoke -> two 20k train+validation jobs. No tests on test data.
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
export IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
BASE="$IRIS_RUNS/outputs/lotus_line_v2/da2_iris_fusion_pair_v1"
test -r "$IRIS_RUNS/da2_caption_v1/no-caption_s43/step018000.pt"
test -r "${DA2_CAPTION_ASSETS:-$IRIS_RUNS/da2_caption_assets/assets.json}"
# Exact same old run/step only; no fallback to l100, failed dual, or another seed.
if [[ -z "${IRIS_FEATURE_CKPT:-}" ]]; then
  for candidate in "$IRIS_RUNS/iris_ms2/sd2_cap_pct_logtn_s43/converted/step18000_weights.pt" \
    /mnt/scratch/sc23sz/runs/iris_ms2/sd2_cap_pct_logtn_s43/converted/step18000_weights.pt; do
    if [[ -r "$candidate" ]]; then export IRIS_FEATURE_CKPT="$candidate"; break; fi
  done
fi
: "${IRIS_FEATURE_CKPT:?Cannot find old caption Iris18k; set its exact converted checkpoint path}"
test -r "$IRIS_FEATURE_CKPT"
for arm in frame empty; do
  [[ ! -e "$BASE/${arm}_s43" ]] || { echo "$arm exists; explicit resume required"; exit 1; }
done
bash -n slurm/da2_iris_fusion.sbatch
echo "Iris feature source: $IRIS_FEATURE_CKPT"
smoke=$(sbatch --parsable -J da2_if_smoke --time=04:00:00 --export=ALL,ACTION=smoke slurm/da2_iris_fusion.sbatch)
smoke=${smoke%%;*}
[[ "$smoke" =~ ^[0-9]+$ ]] || { echo 'Invalid smoke job ID'; exit 1; }
echo "paired smoke -> $smoke"
export SMOKE_OUTPUT="$BASE/smoke_$smoke"
for arm in frame empty; do
  job=$(sbatch --parsable -J "da2_if_$arm" --dependency="afterok:$smoke" \
    --kill-on-invalid-dep=yes --export="ALL,ACTION=train,ARM=$arm,RESUME=0" slurm/da2_iris_fusion.sbatch)
  echo "$arm train+val -> ${job%%;*} (afterok:$smoke)"
done
echo "Outputs: $BASE"
