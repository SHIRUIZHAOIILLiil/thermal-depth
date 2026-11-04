#!/bin/bash
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
export IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
base="$IRIS_RUNS/outputs/lotus_line_v2"
# Pin the completed 9171487 parent rather than choosing the newest directory.
matches=()
for file in "$base"/iris_far_bias_square_l050_frame_v1/run_*/jobs.txt; do
    if [[ -f "$file" ]] && grep -qx 'frame train+val -> 9171487 (afterok:9171486)' "$file"; then
        matches+=("${file%/jobs.txt}")
    fi
done
[[ ${#matches[@]} == 1 ]] || { echo "Expected exactly one parent job 9171487" >&2; exit 1; }
for file in summary.json contract.json conv_4000.pt; do
    [[ -s "${matches[0]}/$file" ]] || { echo "Missing parent $file" >&2; exit 1; }
done
mkdir -p "$base/iris_latent_release_pair_v1"
run=$(mktemp -d "$base/iris_latent_release_pair_v1/run_XXXXXXXX")
echo "Output: $run"
job() {
    local phase=$1 arm=$2 id
    shift 2
    id=$(sbatch --parsable --kill-on-invalid-dep=yes "$@" slurm/iris_latent_release.sbatch "$phase" "$run" "$arm")
    id=${id%%;*}
    [[ "$id" =~ ^[0-9]+$ ]] || return 1
    echo "$id"
}
smoke=$(job smoke control -J iris_lr_smoke --time=03:00:00)
echo "paired smoke -> $smoke" | tee -a "$run/jobs.txt"
control=$(job train control -J iris_lr_control --time=12:00:00 --dependency="afterok:$smoke")
echo "control train+val -> $control (afterok:$smoke)" | tee -a "$run/jobs.txt"
masked=$(job train masked -J iris_lr_masked --time=12:00:00 --dependency="afterok:$smoke")
echo "masked train+val -> $masked (afterok:$smoke)" | tee -a "$run/jobs.txt"
# Both completed arms are checked for identical random draws before either test.
for arm in control masked; do
    id=$(job test "$arm" -J "iris_lr_test_$arm" --time=12:00:00 --array=0-2 --dependency="afterok:$control:$masked")
    echo "$arm day/night/rain -> $id (afterok:$control:$masked)" | tee -a "$run/jobs.txt"
done
