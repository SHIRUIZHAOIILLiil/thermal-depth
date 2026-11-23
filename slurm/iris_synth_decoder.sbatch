#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# Decoder stage on a caption arm's frozen U-Net (docs/SYNTH_POSTHOC_V1.md, docs/SD2_POSTHOC_V1.md).
# SOURCE_RUN: the synth arm (default) or sd2_cap_pct_logtn_img_l0100_tmax1_s43.
# Args: smoke|train <run dir> <averaging job dir> <q98 decoder comparison run>
set -euo pipefail
F="${SCRATCH:?}/from_sc23sz"
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
phase=${1:?mode}
if [[ "$phase" == smoke ]]; then
    python tests/test_iris_decoder_only.py
    python tests/test_iris_tmax1_decoder.py
    python tests/test_synth_posthoc.py
fi
python -u tools/train_iris_synth_decoder.py "$phase" --run "${2:?run}" \
    --average-summary "${3:?averaging job dir}/summary.json" --comparison-run "${4:?comparison run}" \
    --runs "$IRIS_RUNS" --data-root "$IRIS_MS2_ROOT" \
    --source-run "${SOURCE_RUN:-sd2_cap_pct_logtn_img_l0100_tmax1_synth_s43}"
