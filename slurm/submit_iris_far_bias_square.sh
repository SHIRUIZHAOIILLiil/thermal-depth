#!/bin/bash
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
export IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
base="$IRIS_RUNS/outputs/lotus_line_v2"
parent="$base/iris_far_pairs_l050_frame4k_v1/job_9163317"
archive="$base/iris_pair_rmse_val_v1/job_9167854"
for file in "$parent/summary.json" "$archive/summary.json" "$archive/newcrf_moments.npz"; do
    [[ -r "$file" ]] || { echo "Missing $file" >&2; exit 1; }
done
dir="$base/iris_far_bias_square_l050_frame_v1"
mkdir -p "$dir"
run=$(mktemp -d "$dir/run_XXXXXXXX")
echo "Output: $run"
job() {
    local phase=$1 id
    shift
    id=$(sbatch --parsable --kill-on-invalid-dep=yes "$@" slurm/iris_far_bias_square.sbatch "$phase" "$run")
    id=${id%%;*}
    [[ "$id" =~ ^[0-9]+$ ]] || return 1
    echo "$id"
}
smoke=$(job smoke -J iris_fb_smoke --time=02:00:00)
echo "smoke -> $smoke" | tee -a "$run/jobs.txt"
train=$(job train -J iris_fb_frame --time=12:00:00 --dependency="afterok:$smoke")
echo "frame train+val -> $train (afterok:$smoke)" | tee -a "$run/jobs.txt"
test_id=$(job test -J iris_fb_test --array=0-2 --time=12:00:00 --dependency="afterok:$train")
echo "day/night/rain -> $test_id (afterok:$train)" | tee -a "$run/jobs.txt"
