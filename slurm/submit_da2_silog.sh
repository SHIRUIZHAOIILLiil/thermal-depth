#!/bin/bash
# One paired smoke gates both 20k training+validation jobs; no test jobs.
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
export IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
BASE="$IRIS_RUNS/da2_caption_silog_v1"
test -f "${DA2_CAPTION_ASSETS:-$IRIS_RUNS/da2_caption_assets/assets.json}"
for arm in no-caption caption; do
  [[ ! -e "$BASE/${arm}_s43" ]] || { echo "$arm exists; use explicit resume, not a fresh chain"; exit 1; }
done
bash -n slurm/da2_silog.sbatch
smoke=$(sbatch --parsable -J da2_silog_smoke --time=02:00:00 --export=ALL,ACTION=smoke slurm/da2_silog.sbatch)
smoke=${smoke%%;*}
echo "paired smoke -> $smoke"
export SMOKE_OUTPUT="$BASE/smoke_$smoke"
for arm in no-caption caption; do
  job=$(sbatch --parsable -J "da2_silog_$arm" --dependency="afterok:$smoke" \
    --kill-on-invalid-dep=yes --export="ALL,ACTION=train,ARM=$arm,RESUME=0" slurm/da2_silog.sbatch)
  echo "$arm train+val -> ${job%%;*} (afterok:$smoke)"
done


