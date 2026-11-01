#!/bin/bash
#SBATCH --job-name=cap_review80
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
conda activate caption
cd "$IRIS_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

repo="${CAPTION_REPO:-}"
if [[ -z "$repo" ]]; then
  for dir in "$SCRATCH/captioning" "$F/captioning" /mnt/scratch/sc23sz/captioning "$HOME/captioning"; do
    if [[ -r "$dir/scripts/generate_captions.py" ]]; then repo="$dir"; break; fi
  done
fi
[[ -r "$repo/scripts/generate_captions.py" ]] || { echo 'Set CAPTION_REPO to the existing captioning checkout with scripts/generate_captions.py' >&2; exit 1; }

snapshot="${INTERNVL_SNAPSHOT:-}"
if [[ -z "$snapshot" ]]; then
  for hub in "${HF_HUB_CACHE:-$HF_HOME/hub}" "$SCRATCH/hf_cache/hub" "$F/hf_cache/hub" /mnt/scratch/sc23sz/hf_cache/hub; do
    model="$hub/models--OpenGVLab--InternVL3-8B"
    if [[ -r "$model/refs/main" ]]; then
      candidate="$model/snapshots/$(cat "$model/refs/main")"
      if [[ -r "$candidate/config.json" ]]; then snapshot="$candidate"; break; fi
    fi
  done
fi
[[ -n "$snapshot" && -r "$snapshot/config.json" ]] || { echo 'Set INTERNVL_SNAPSHOT to the cached OpenGVLab/InternVL3-8B snapshot' >&2; exit 1; }
out="$IRIS_RUNS/outputs/lotus_line_v2/caption_grounded_train80_v1"
mkdir -p "$out"
exec 9>"$out/job.lock"
flock -n 9 || { echo 'Another review job is active'; exit 1; }
args=(--manifest "$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl"
      --train-contract "$IRIS_RUNS/outputs/lotus_line_v2/iris_visual_gate_matched_v1/frame/contract.json"
      --data-root "$IRIS_MS2_ROOT" --caption-repo "$repo" --snapshot "$snapshot" --out "$out")
srun python tools/review_train80_captions.py "${args[@]}"
