#!/bin/bash
# Attach one test array to the existing cross-region job. Never create its training directory.
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
cross=${1:?Pass the cross-region job ID}
parent=${2:?Pass the local-pair continuation job ID}
[[ "$cross" =~ ^[0-9]+$ && "$parent" =~ ^[0-9]+$ ]]
base="${IRIS_RUNS:-${SCRATCH:?}/runs}/outputs/lotus_line_v2"
cross_base="$base/iris_cross_region_l050_frame_v1"
parent_run="$base/iris_far_pairs_l050_frame4k_v1/job_$parent"
mkdir -p "$cross_base/test_receipts"
receipt="$cross_base/test_receipts/job_$cross"
exec 9>"$receipt.lock"
flock 9
if [[ -s "$receipt.id" ]]; then echo "Test array already submitted: $(cat "$receipt.id")"; exit 0; fi

deps=()
if [[ ! -s "$cross_base/job_$cross/summary.json" ]]; then deps+=("afterok:$cross"); fi
for condition in day night rain; do
    [[ -s "$parent_run/test_selected_v1/$condition/summary.json" ]] && continue
    array=${3:-}
    if [[ -z "$array" ]]; then
        # SubmitLine includes the parent argument even after Slurm clears a satisfied dependency.
        rows=$(sacct -X -nP -S now-7days --name=iris_fp4k_test -o 'JobID%64,SubmitLine%2048') || rows=''
        mapfile -t ids < <(awk -F'|' -v p="$parent" '
            $2 ~ ("iris_far_pairs_frame_4k_test[.]sbatch[[:space:]]+" p "([[:space:]]|$)") {
                split($1, id, "_"); if (id[1] ~ /^[0-9]+$/) print id[1]
            }' <<<"$rows" | sort -u)
        if (( ${#ids[@]} != 1 )); then
            # Queued arrays may not yet have a visible accounting row.
            rows=$(squeue -h -r -u "$USER" -n iris_fp4k_test -o '%F|%E')
            mapfile -t ids < <(awk -F'|' -v p="$parent" '$2 ~ ("afterok:" p "([^0-9]|$)") {print $1}' <<<"$rows" | sort -u)
        fi
        if (( ${#ids[@]} != 1 )); then
            echo "Pass the earlier iris_fp4k_test array ID as argument 3 (see squeue or sacct)." >&2
            exit 2
        fi
        array=${ids[0]}
    fi
    [[ "$array" =~ ^[0-9]+$ ]]
    deps+=("afterok:$array")
    break
done
options=()
if (( ${#deps[@]} )); then
    dependency=$(IFS=,; echo "${deps[*]}")
    options+=(--dependency="$dependency")
fi
id=$(sbatch --parsable --kill-on-invalid-dep=yes "${options[@]}" slurm/iris_cross_region_test.sbatch "$cross" "$parent")
id=${id%%;*}
[[ "$id" =~ ^[0-9]+$ ]]
printf '%s\n' "$id" > "$receipt.id.tmp"
mv "$receipt.id.tmp" "$receipt.id"
echo "day/night/rain test array -> $id (cross job $cross; dependencies: ${deps[*]:-already complete})"
