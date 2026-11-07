#!/bin/bash
#SBATCH --job-name=ftr_build
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
run=${1:?Usage: far_target_build.sbatch <pipeline run directory>}
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
raw="$run/raw_predictions"
target="$run/target"
# Interrupted frames are skipped on resubmission; calibration always rebuilds.
python -u tools/far_target_pipeline.py infer --runs "$IRIS_RUNS" --choice "$run/choice.json" \
    --manifest "${TRAIN_MANIFEST:?}" --data-root "$IRIS_MS2_ROOT" \
    --bmsd-root "$SCRATCH/baselines/SupDepth4Thermal" --ckpt-dir "$SCRATCH/baselines/supdepth4thermal" --out "$raw"
python -u tools/build_anythermal_pseudo_gt.py --manifest "$TRAIN_MANIFEST" --ms2-root "$IRIS_MS2_ROOT" \
    --raw-pred-dir "$raw" --output-dir "$target" --dtype float32
built=$(find "$target/calibrated_pseudo_depth" -name '*.npy' | wc -l)
rows=$(grep -c . "$TRAIN_MANIFEST")
[[ "$built" == "$rows" ]] || { echo "Built $built of $rows targets" >&2; exit 1; }
du -sh "$raw" "$target"
