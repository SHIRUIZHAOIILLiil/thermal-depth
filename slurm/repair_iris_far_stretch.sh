#!/bin/bash
# Original afterok dependencies cannot be released by reconstructing a file.
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD" IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}" PARENT_ID="${1:?validation job ID}"
[[ "$PARENT_ID" =~ ^[0-9]+$ ]]
parent="$IRIS_RUNS/outputs/lotus_line_v2/iris_far_stretch_l050_v1/job_$PARENT_ID"
for file in frozen_betas.json frame_val.csv empty_val.csv; do
    [[ -s "$parent/$file" ]] || { echo "Missing $parent/$file" >&2; exit 1; }
done
state=$(sacct -nXP -u "$(id -un)" -j "$PARENT_ID" -o JobIDRaw,JobName%48,State |
    awk -F'|' -v id="$PARENT_ID" '$1==id && $2=="iris_far_stretch" {print $3}')
[[ "$state" == FAILED ]] || { echo "Expected FAILED iris_far_stretch parent, got: $state" >&2; exit 1; }
exec 9>"$parent/repair.lock"
flock -n 9 || exit 1
receipt="$parent/recovery_test.id"
if [[ -s "$receipt" ]]; then echo "Already attached: $(cat "$receipt")"; exit 0; fi
cpu=$(sbatch --parsable --export=ALL slurm/recover_iris_far_stretch.sbatch)
cpu=${cpu%%;*}
[[ "$cpu" =~ ^[0-9]+$ ]]
echo "CPU recovery -> $cpu"
job=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$cpu" --export=ALL slurm/iris_far_stretch_test.sbatch)
job=${job%%;*}
[[ "$job" =~ ^[0-9]+$ ]]
printf '%s\n' "$job" > "$receipt"
echo "Replacement day/night/rain tests -> $job (afterok:$cpu)"
