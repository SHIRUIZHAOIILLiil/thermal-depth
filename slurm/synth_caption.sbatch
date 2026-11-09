#!/bin/bash
#SBATCH --job-name=iris_synth_cap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
# Iris's captions for the synthetic sets (tools/caption_synth.py).
#   array task, COUNT=<total shards>; resumable, resubmit the same command.
#   Merging is STEP=caption-finalize in synth_data.sbatch (no GPU needed).
# L40S captions MS2 at 2.0-2.36 s/frame (caption_ms2.sbatch); ~93k images / 16 shards
# -> about 3.9 h per shard at 2.4 s, hence 6 h.
set -euo pipefail
F="${SCRATCH:?}/from_sc23sz"
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
conda activate caption
cd "$IRIS_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SYN="${SYNTH_ROOT:-$SCRATCH/data/synth}"
[[ -d "$HF_HOME/hub/models--OpenGVLab--InternVL3-8B" ]] || { echo "!! InternVL3-8B not in $HF_HOME"; exit 1; }
# CUDA frees memory after Slurm hands the card to the next array task (caption_ms2.sbatch).
for attempt in $(seq 1 20); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
  (( free >= 20000 )) && break
  echo "[wait] GPU free ${free} MiB"; sleep 30
done
(( free >= 20000 )) || { echo "!! GPU free ${free} MiB after 10 min"; exit 1; }
python -u tools/caption_synth.py shard --hypersim "$SYN/hypersim" --vkitti "$SYN/vkitti" --out "$SYN/captions" \
  --count "${COUNT:?}" --index "${SLURM_ARRAY_TASK_ID:-0}"
