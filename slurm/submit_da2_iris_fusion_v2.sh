#!/bin/bash
# Matched weight/text arms; frame job also does selected-model permuted validation.
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
export IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
BASE="$IRIS_RUNS/outputs/lotus_line_v2/da2_iris_fusion_matched_pair_v2"
test -r "$IRIS_RUNS/da2_caption_v1/no-caption_s43/step018000.pt"
test -r "${DA2_CAPTION_ASSETS:-$IRIS_RUNS/da2_caption_assets/assets.json}"
# Resolve BOTH exact old runs before queuing. Never reuse v1's single weight override.
for arm in cap nocap; do
  if [[ "$arm" == cap ]]; then var=IRIS_CAP_FEATURE_CKPT; else var=IRIS_EMPTY_FEATURE_CKPT; fi
  run="sd2_${arm}_pct_logtn_s43"
  if [[ -z "${!var:-}" ]]; then
    for root in "$IRIS_RUNS" /mnt/scratch/sc23sz/runs; do
      candidate="$root/iris_ms2/$run/converted/step18000_weights.pt"
      if [[ -r "$candidate" ]]; then printf -v "$var" '%s' "$candidate"; export "$var"; break; fi
    done
  fi
  [[ -n "${!var:-}" && -r "${!var}" ]] || { echo "Missing $run 18k; set $var to its exact checkpoint"; exit 1; }
done
for arm in frame empty; do
  [[ ! -e "$BASE/${arm}_s43" ]] || { echo "$arm exists; explicit resume required"; exit 1; }
done
bash -n slurm/da2_iris_fusion_v2.sbatch
echo "Caption Iris: $IRIS_CAP_FEATURE_CKPT"
echo "No-caption Iris: $IRIS_EMPTY_FEATURE_CKPT"
smoke=$(sbatch --parsable -J da2_if2_smoke --time=04:00:00 --export=ALL,ACTION=smoke slurm/da2_iris_fusion_v2.sbatch)
smoke=${smoke%%;*}
[[ "$smoke" =~ ^[0-9]+$ ]] || { echo 'Invalid smoke job ID'; exit 1; }
echo "paired smoke -> $smoke"
export SMOKE_OUTPUT="$BASE/smoke_$smoke"
for arm in frame empty; do
  job=$(sbatch --parsable -J "da2_if2_$arm" --dependency="afterok:$smoke" \
    --kill-on-invalid-dep=yes --export="ALL,ACTION=train,ARM=$arm,RESUME=0" slurm/da2_iris_fusion_v2.sbatch)
  echo "$arm train+val -> ${job%%;*} (afterok:$smoke)"
done
echo "Outputs: $BASE"
