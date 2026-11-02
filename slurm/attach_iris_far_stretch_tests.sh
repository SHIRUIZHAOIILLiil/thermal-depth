#!/bin/bash
# No argument attaches to the sole active validation; --new submits validation first.
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
export IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
base="$IRIS_RUNS/outputs/lotus_line_v2/iris_far_stretch_l050_v1"
mkdir -p "$base/submissions"
exec 9>"$base/submissions/attach.lock"
flock -n 9 || { echo "Another submission is in progress" >&2; exit 1; }
user=$(id -un)
queue=$(squeue -h -u "$user" -n iris_far_stretch -o '%i')
dependency=()
case "${1:-}" in
    --new)
        [[ -z "$queue" ]] || { echo "Validation already active: $queue. Attach without --new." >&2; exit 1; }
        id=$(sbatch --parsable slurm/iris_far_stretch.sbatch)
        id=${id%%;*}
        echo "validation -> $id"
        dependency=(--dependency="afterok:$id") ;;
    *)
        id=${1:-$queue}
        [[ "$id" =~ ^[0-9]+$ ]] || { echo "Specify one validation job ID, or --new for a new run." >&2; exit 2; }
        if grep -Fxq "$id" <<< "$queue"; then
            dependency=(--dependency="afterok:$id")
        else
            # Completed jobs can be too old for Slurm to accept a new afterok.
            [[ -s "$base/job_$id/summary.json" ]] || { echo "Missing completed validation $id" >&2; exit 1; }
            state=$(sacct -nXP -u "$user" -j "$id" -o JobIDRaw,JobName%48,State |
                    awk -F'|' -v id="$id" '$1==id && $2=="iris_far_stretch" {print $3}')
            [[ "$state" == COMPLETED ]] || { echo "Validation $id state: $state" >&2; exit 1; }
        fi ;;
esac
[[ "$id" =~ ^[0-9]+$ ]]
receipt="$base/submissions/test_after_$id.id"
if [[ -s "$receipt" ]]; then echo "Already attached to $id: $(cat "$receipt")"; exit 0; fi
export PARENT_ID="$id"
job=$(sbatch --parsable --kill-on-invalid-dep=yes "${dependency[@]}" --export=ALL slurm/iris_far_stretch_test.sbatch)
job=${job%%;*}
[[ "$job" =~ ^[0-9]+$ ]]
printf '%s\n' "$job" > "$receipt"
echo "test day/night/rain -> array $job (validation $id)"
