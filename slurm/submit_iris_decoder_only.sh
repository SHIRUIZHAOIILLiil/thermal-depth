#!/bin/bash
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
export IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
base="$IRIS_RUNS/outputs/lotus_line_v2"
matches=()
for file in "$base"/iris_far_bias_square_l050_frame_v1/run_*/jobs.txt; do
    if [[ -f "$file" ]] && grep -qx 'frame train+val -> 9171487 (afterok:9171486)' "$file"; then
        matches+=("${file%/jobs.txt}")
    fi
done
[[ ${#matches[@]} == 1 ]] || { echo 'Expected exactly one completed parent job 9171487' >&2; exit 1; }
for file in summary.json contract.json conv_4000.pt; do
    [[ -s "${matches[0]}/$file" ]] || { echo "Missing parent $file" >&2; exit 1; }
done
[[ -s "$base/iris_pair_rmse_val_v1/job_9167854/summary.json" ]] || exit 1
mkdir -p "$base/iris_vae_decoder_only_v1"
run=$(mktemp -d "$base/iris_vae_decoder_only_v1/run_XXXXXXXX")
echo "Output: $run"
smoke=$(sbatch --parsable --kill-on-invalid-dep=yes -J iris_vd_smoke --time=03:00:00 \
    slurm/iris_decoder_only.sbatch smoke "$run")
smoke=${smoke%%;*}
[[ "$smoke" =~ ^[0-9]+$ ]] || exit 1
echo "smoke -> $smoke" | tee -a "$run/jobs.txt"
train=$(sbatch --parsable --kill-on-invalid-dep=yes -J iris_vd_train --time=12:00:00 \
    --dependency="afterok:$smoke" slurm/iris_decoder_only.sbatch train "$run")
train=${train%%;*}
[[ "$train" =~ ^[0-9]+$ ]] || exit 1
echo "decoder train+val -> $train (afterok:$smoke)" | tee -a "$run/jobs.txt"
echo 'No test jobs submitted.'
