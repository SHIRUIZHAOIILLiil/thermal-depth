#!/bin/bash
#SBATCH --job-name=iris_fp_frame4k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
parent_job=${1:?Pass the completed frame training job ID}
[[ "$parent_job" =~ ^[0-9]+$ ]]
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-${SCRATCH:?}/from_sc23sz/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$SCRATCH/from_sc23sz/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
base="$IRIS_RUNS/outputs/lotus_line_v2"
matches=()
shopt -s nullglob
for receipt in "$base"/iris_far_pairs_l050_v1/run_*/jobs.txt; do
    if grep -qE "^frame train\+val -> ${parent_job} " "$receipt"; then
        matches+=("${receipt%/jobs.txt}")
    fi
done
if [[ ${#matches[@]} != 1 ]]; then
    echo "Expected one completed far-pair run for frame job $parent_job; found ${#matches[@]}" >&2
    exit 1
fi
out="$base/iris_far_pairs_l050_frame4k_v1/job_${SLURM_JOB_ID:?}"
echo "Parent: ${matches[0]}"
echo "Output: $out"
python tests/test_iris_far_pairs_frame_4k.py
python -u tools/extend_iris_far_pairs.py --parent "${matches[0]}" --data-root "$IRIS_MS2_ROOT" --out "$out"
