#!/bin/bash
# One paired smoke, then two independent formal jobs. Never submits test jobs.
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
BASE="$RUNS/da2_caption_residual_v1"
test -f "$RUNS/da2_caption_v1/no-caption_s43/step018000.pt"
for arm in frame fixed; do
  [[ ! -e "$BASE/${arm}_s43" ]] || { echo "$arm exists; use explicit resume, not a fresh chain"; exit 1; }
done
bash -n slurm/da2_residual.sbatch
smoke=$(sbatch --parsable -J da2_res_smoke --time=02:00:00 --export=ALL,ACTION=smoke slurm/da2_residual.sbatch)
smoke=${smoke%%;*}
echo "paired smoke -> $smoke"
export SMOKE_OUTPUT="$BASE/smoke_$smoke"
for arm in frame fixed; do
  job=$(sbatch --parsable -J "da2_res_$arm" --dependency="afterok:$smoke" \
    --export="ALL,ACTION=train,ARM=$arm,RESUME=0" slurm/da2_residual.sbatch)
  echo "$arm train+val -> ${job%%;*} (afterok:$smoke)"
done
