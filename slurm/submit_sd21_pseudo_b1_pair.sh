#!/bin/bash
# Only a paired smoke and two dependent training+validation jobs. No test jobs.
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
export IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
BASE="$IRIS_RUNS/outputs/lotus_line_v2/sd21_tadp_pseudo_b1_pair_v1"
for arm in frame empty; do
  [[ ! -e "$BASE/${arm}_s43" ]] || { echo "$arm exists; explicit resume required"; exit 1; }
done
bash -n slurm/sd21_pseudo_b1_pair.sbatch
smoke=$(sbatch --parsable -J sd21_pg1_smoke --time=04:00:00 --export=ALL,ACTION=smoke slurm/sd21_pseudo_b1_pair.sbatch)
smoke=${smoke%%;*}
[[ "$smoke" =~ ^[0-9]+$ ]] || { echo 'Invalid smoke job ID'; exit 1; }
echo "paired smoke -> $smoke"
export SMOKE_OUTPUT="$BASE/smoke_$smoke"
for arm in frame empty; do
  job=$(sbatch --parsable -J "sd21_pg1_$arm" --dependency="afterok:$smoke" \
    --kill-on-invalid-dep=yes --export="ALL,ACTION=train,ARM=$arm,RESUME=0" slurm/sd21_pseudo_b1_pair.sbatch)
  echo "$arm train+val -> ${job%%;*} (afterok:$smoke)"
done
echo "Outputs: $BASE"
