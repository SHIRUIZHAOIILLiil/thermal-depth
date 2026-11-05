#!/bin/bash
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
export IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
run=${1:?Usage: submit_iris_decoder_only_eval.sh <decoder run directory>}
for file in contract.json train/summary.json; do
    [[ -s "$run/$file" ]] || { echo "Missing $run/$file" >&2; exit 1; }
done
reproduce=$(sbatch --parsable --kill-on-invalid-dep=yes -J iris_vd_repro --time=03:00:00 \
    slurm/iris_decoder_only_eval.sbatch reproduce "$run")
reproduce=${reproduce%%;*}
[[ "$reproduce" =~ ^[0-9]+$ ]] || exit 1
echo "val reproduction -> $reproduce" | tee -a "$run/jobs.txt"
test_id=$(sbatch --parsable --kill-on-invalid-dep=yes -J iris_vd_test --array=0-2 --time=12:00:00 \
    --output=iris_vd_test-%A_%a.out --error=iris_vd_test-%A_%a.err \
    --dependency="afterok:$reproduce" slurm/iris_decoder_only_eval.sbatch test "$run")
test_id=${test_id%%;*}
[[ "$test_id" =~ ^[0-9]+$ ]] || exit 1
echo "day/night/rain -> $test_id (afterok:$reproduce)" | tee -a "$run/jobs.txt"
