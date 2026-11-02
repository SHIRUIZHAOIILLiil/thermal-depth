#!/bin/bash
# One independent run. Reserve a fresh directory even for simultaneous submissions.
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
F=${SCRATCH:?}/from_sc23sz
base="${IRIS_RUNS:-$SCRATCH/runs}/outputs/lotus_line_v2/newcrf_pseudo_v1"
for path in "${NEWCRF_CKPT:-$SCRATCH/baselines/supdepth4thermal/MS2_MD_NeWCRF_THR_ckpt.ckpt}" \
            "${BMSD_ROOT:-$SCRATCH/baselines/SupDepth4Thermal}/models/network/newcrf/NewCRFDepth.py" \
            "${PSEUDO_DIR:-$F/pseudo_gt/official_train/calibrated_pseudo_depth}/../build_config.json"; do
    [[ -f "$path" ]] || { echo "Missing $path" >&2; exit 1; }
done
mkdir -p "$base"
RUN_DIR=$(mktemp -d "$base/run_XXXXXXXX")
export RUN_DIR
echo "Outputs: $RUN_DIR"
job() {
    local id
    id=$(sbatch --parsable --kill-on-invalid-dep=yes "$@" slurm/newcrf_pseudo.sbatch)
    id=${id%%;*}
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "Invalid Slurm ID: $id" >&2; return 1; }
    printf '%s\n' "$id"
}
export STAGE=smoke
smoke=$(job -J crf_pg_smoke --time=06:00:00 --export=ALL)
echo "smoke + pseudo inventory -> $smoke"
export STAGE=train
train=$(job -J crf_pg_train --dependency="afterok:$smoke" --export=ALL)
echo "fine-tune + validation -> $train (afterok:$smoke)"
export STAGE=test
for CONDITION in day night rain; do
    export CONDITION
    id=$(job -J "crf_pg_$CONDITION" --time=12:00:00 --dependency="afterok:$train" --export=ALL)
    echo "$CONDITION original + best fine-tuned test -> $id (afterok:$train)"
done
