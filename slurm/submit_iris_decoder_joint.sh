#!/bin/bash
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
export IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
base="$IRIS_RUNS/outputs/lotus_line_v2"
matches=()
for file in "$base"/iris_vae_decoder_only_v1/run_*/jobs.txt; do
    if [[ -f "$file" ]] && grep -qx 'val reproduction -> 9213840' "$file"; then
        matches+=("${file%/jobs.txt}")
    fi
done
[[ ${#matches[@]} == 1 ]] || { echo 'Need exactly one q98 decoder run reproduced by 9213840' >&2; exit 1; }
source_run=${matches[0]}
for file in contract.json train/summary.json train/decoder_1000.pt train/test_selected_v1/reproduced.json; do
    [[ -s "$source_run/$file" ]] || { echo "Missing $source_run/$file" >&2; exit 1; }
done
mkdir -p "$base/iris_decoder_joint_q98_v1"
run=$(mktemp -d "$base/iris_decoder_joint_q98_v1/run_XXXXXXXX")
echo "Source: $source_run"
echo "Output: $run"
job() {
    local id
    id=$(sbatch --parsable --kill-on-invalid-dep=yes "$@") || return 1
    id=${id%%;*}
    [[ "$id" =~ ^[0-9]+$ ]] || return 1
    echo "$id"
}
gpu=(--partition=gpu --gres=gpu:1)
smoke=$(job "${gpu[@]}" -J iris_dj_smoke --time=03:00:00 \
    slurm/iris_decoder_joint.sbatch smoke "$run" "$source_run")
echo "paired smoke -> $smoke" | tee -a "$run/jobs.txt"
parents=()
for arm in decoder joint; do
    id=$(job "${gpu[@]}" -J "iris_dj_$arm" --time=12:00:00 --dependency="afterok:$smoke" \
        slurm/iris_decoder_joint.sbatch train "$run" "$source_run" "$arm")
    parents+=("$id")
    echo "$arm train+val -> $id (afterok:$smoke)" | tee -a "$run/jobs.txt"
done
id=$(job -J iris_dj_report --time=00:15:00 --mem=6G \
    --dependency="afterok:${parents[0]}:${parents[1]}" slurm/iris_decoder_joint.sbatch report "$run")
echo "paired CPU report -> $id (afterok:${parents[*]})" | tee -a "$run/jobs.txt"
echo 'No test jobs submitted.'
