#!/bin/bash
# Independent of the output-only beta calibration.
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
export IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
F="$SCRATCH/from_sc23sz"
export PSEUDO_DIR="${PSEUDO_DIR:-$F/pseudo_gt/official_train/calibrated_pseudo_depth}"
[[ -d "$PSEUDO_DIR" && -r "$PSEUDO_DIR/../build_config.json" ]] || { echo "Missing calibrated labels: $PSEUDO_DIR" >&2; exit 1; }
for arm in cap nocap; do
    found=0
    for root in "$IRIS_RUNS" "$F/runs" "$F" /mnt/scratch/sc23sz/runs; do
        path="$root/iris_ms2/sd2_${arm}_pct_logtn_img_s43/converted/step16000_weights.pt"
        if [[ -r "$path" ]]; then echo "$arm parent: $path"; found=1; break; fi
    done
    (( found )) || { echo "Missing $arm lambda=.05 step16000" >&2; exit 1; }
done
base="$IRIS_RUNS/outputs/lotus_line_v2/iris_far_pairs_l050_v1"
mkdir -p "$base"
PAIR_RUN=$(mktemp -d "$base/run_XXXXXXXX")
export PAIR_RUN
echo "Outputs: $PAIR_RUN"
job() {
    local id
    id=$(sbatch --parsable --kill-on-invalid-dep=yes "$@" slurm/iris_far_pairs.sbatch) || return 1
    id=${id%%;*}
    [[ "$id" =~ ^[0-9]+$ ]] || return 1
    echo "$id"
}
gpu=(--partition=gpu --gres=gpu:1)
smoke=$(job "${gpu[@]}" -J iris_fp_smoke --time=02:00:00 --export=ALL,ACTION=smoke)
echo "paired smoke -> $smoke" | tee -a "$PAIR_RUN/jobs.txt"
parents=()
for arm in frame empty; do
    id=$(job "${gpu[@]}" -J "iris_fp_$arm" --time=24:00:00 --dependency="afterok:$smoke" --export="ALL,ACTION=train,ARM=$arm")
    parents+=("$id")
    echo "$arm train+val -> $id (afterok:$smoke)" | tee -a "$PAIR_RUN/jobs.txt"
done
id=$(job -J iris_fp_report --mem=6G --time=00:15:00 --dependency="afterok:${parents[0]}:${parents[1]}" --export=ALL,ACTION=report)
echo "paired report -> $id (afterok:${parents[*]})" | tee -a "$PAIR_RUN/jobs.txt"
