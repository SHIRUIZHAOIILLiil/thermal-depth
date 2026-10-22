#!/bin/bash
# Run on Aire login node; 2 resume jobs + 2 afterok validation jobs. No test jobs.
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
: "${SMOKE_OUTPUT:?export SMOKE_OUTPUT pointing to the successful original paired smoke}"
BASE="${IRIS_RUNS:-${SCRATCH:?}/runs}/da2_caption_v1"
for arm in no-caption caption; do
  test -f "$SMOKE_OUTPUT/$arm/completed.json"
  for step in 002000 004000 006000 008000; do
    test -f "$BASE/${arm}_s43/step${step}.pt"
  done
  test -f "$BASE/${arm}_s43/last.pt"
done
bash -n slurm/da2_continue.sbatch
bash -n slurm/da2_continue_val.sbatch
for arm in no-caption caption; do
  job=$(sbatch --parsable --job-name="da2_${arm}_20k" --export="ALL,ARM=$arm" slurm/da2_continue.sbatch)
  job=${job%%;*}
  echo "$arm training -> $job"
  val=$(sbatch --parsable --dependency="afterok:$job" --job-name="da2_${arm}_curve" --export="ALL,ARM=$arm" slurm/da2_continue_val.sbatch)
  echo "$arm validation -> ${val%%;*} (afterok:$job)"
done
