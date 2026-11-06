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
    if [[ -f "$candidate" ]] && [[ "$(head -c 1 "$candidate" 2>/dev/null | wc -c)" == 1 ]]; then
        weights=$candidate
        break
    fi
done
[[ -n "$weights" ]] || { echo 'Missing readable original tmax1 16k checkpoint' >&2; exit 1; }
matches=()
for file in "$base"/iris_vae_decoder_only_v1/run_*/jobs.txt; do
    if [[ -f "$file" ]] && grep -qx 'val reproduction -> 9213840' "$file"; then
        matches+=("${file%/jobs.txt}")
    fi
done
[[ ${#matches[@]} == 1 ]] || { echo 'Expected exactly one q98 decoder run reproduced by 9213840' >&2; exit 1; }
comparison=${matches[0]}
for file in "$comparison/contract.json" "$comparison/train/summary.json" \
            "$comparison/train/test_selected_v1/reproduced.json" \
            "$base/iris_far_pairs_l050_frame4k_v1/job_9163317/summary.json" \
            "$base/iris_pair_rmse_val_v1/job_9167854/newcrf_moments.npz"; do
    [[ -s "$file" ]] || { echo "Missing $file" >&2; exit 1; }
done
mkdir -p "$base/iris_tmax1_decoder_v1"
run=$(mktemp -d "$base/iris_tmax1_decoder_v1/run_XXXXXXXX")
echo "Source: $weights"
echo "Historical validation comparison: $comparison"
echo "Output: $run"
smoke=$(sbatch --parsable --kill-on-invalid-dep=yes -J iris_t1vd_sm --time=03:00:00 \
    slurm/iris_tmax1_decoder.sbatch smoke "$run" "$weights" "$comparison")
smoke=${smoke%%;*}
[[ "$smoke" =~ ^[0-9]+$ ]] || exit 1
echo "smoke -> $smoke" | tee -a "$run/jobs.txt"
train=$(sbatch --parsable --kill-on-invalid-dep=yes -J iris_t1vd_tr --time=12:00:00 \
    --dependency="afterok:$smoke" slurm/iris_tmax1_decoder.sbatch train "$run" "$weights" "$comparison")
train=${train%%;*}
[[ "$train" =~ ^[0-9]+$ ]] || exit 1
echo "decoder train+val -> $train (afterok:$smoke)" | tee -a "$run/jobs.txt"
echo 'Only decoder training; no U-Net training or test jobs submitted.'
