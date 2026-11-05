#!/bin/bash
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
export IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
F="${SCRATCH:?}/from_sc23sz"
base="$IRIS_RUNS/outputs/lotus_line_v2"
weights=""
for root in "$IRIS_RUNS/iris_ms2" "$F/runs/iris_ms2" "$F/iris_ms2" /mnt/scratch/sc23sz/runs/iris_ms2; do
    candidate="$root/sd2_cap_pct_logtn_img_l0100_tmax1_s43/converted/step16000_weights.pt"
    # An ACL mask can hide read failures from -r, so read one real byte.
    if [[ -f "$candidate" ]] && [[ "$(head -c 1 "$candidate" 2>/dev/null | wc -c)" == 1 ]]; then
        weights=$candidate
        break
    fi
done
[[ -n "$weights" ]] || { echo "Missing readable tmax1 step 16000 weights" >&2; exit 1; }
for file in "$base/iris_far_pairs_l050_frame4k_v1/job_9163317/summary.json" \
            "$base/iris_pair_rmse_val_v1/job_9167854/newcrf_moments.npz"; do
    [[ -r "$file" ]] || { echo "Missing $file" >&2; exit 1; }
done
echo "Weights: $weights"
dir="$base/iris_tmax1_bias_newcrf_v2"
mkdir -p "$dir"
run=$(mktemp -d "$dir/run_XXXXXXXX")
echo "Output: $run"
job() {
    local phase=$1 id
    shift
    id=$(sbatch --parsable --kill-on-invalid-dep=yes "$@" slurm/iris_tmax1_bias.sbatch "$phase" "$run" "$weights")
    id=${id%%;*}
    [[ "$id" =~ ^[0-9]+$ ]] || return 1
    echo "$id"
}
smoke=$(job smoke -J iris_t1_smoke --time=02:00:00)
echo "smoke -> $smoke" | tee -a "$run/jobs.txt"
train=$(job train -J iris_t1_frame --time=12:00:00 --dependency="afterok:$smoke")
echo "frame train+val -> $train (afterok:$smoke)" | tee -a "$run/jobs.txt"
echo 'Validation only; no test jobs submitted. See [candidate-vs-newcrf] and [decision].'
