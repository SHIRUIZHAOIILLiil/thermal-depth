#!/bin/bash
# Attach to a receipt from submit_iris_far_pairs.sh; never submits training.
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
export IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
base="$IRIS_RUNS/outputs/lotus_line_v2/iris_far_pairs_l050_v1"
[[ $# -le 1 ]] || { echo "Usage: bash $0 [run_directory]" >&2; exit 2; }
if [[ $# == 1 ]]; then
    PAIR_RUN=$(realpath -e -- "$1")
else
    shopt -s nullglob
    files=("$base"/run_*/jobs.txt)
    if (( ${#files[@]} != 1 )); then
        echo "Expected exactly one run; pass the Outputs: run directory explicitly." >&2
        printf '%s\n' "${files[@]}" >&2
        exit 2
    fi
    PAIR_RUN=$(realpath -e -- "$(dirname "${files[0]}")")
fi
[[ "$(dirname "$PAIR_RUN")" == "$(realpath -e -- "$base")" && -f "$PAIR_RUN/jobs.txt" ]] || { echo "Wrong experiment directory" >&2; exit 2; }
export PAIR_RUN
exec 9>"$PAIR_RUN/attach_test.lock"
flock -n 9 || { echo "Another attachment is in progress"; exit 1; }
receipt="$PAIR_RUN/test_array.id"
if [[ -s "$receipt" ]]; then echo "test already attached: $(cat "$receipt") ($PAIR_RUN)"; exit 0; fi
parent=$(awk '$1=="paired" && $2=="report" && $3=="->" {print $4}' "$PAIR_RUN/jobs.txt")
[[ "$parent" =~ ^[0-9]+$ ]] || { echo "Missing paired report job in $PAIR_RUN/jobs.txt" >&2; exit 1; }
deps=()
if [[ ! -s "$PAIR_RUN/paired.json" ]]; then
    deps=(--dependency="afterok:$parent")
fi
job=$(sbatch --parsable --kill-on-invalid-dep=yes "${deps[@]}" --export=ALL slurm/iris_far_pairs_test.sbatch)
job=${job%%;*}
[[ "$job" =~ ^[0-9]+$ ]]
printf '%s\n' "$job" > "$receipt.tmp"
mv "$receipt.tmp" "$receipt"
echo "day / night / rain selected tests -> array $job (paired report $parent, $PAIR_RUN)"
