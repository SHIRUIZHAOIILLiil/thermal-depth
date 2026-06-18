#!/usr/bin/env bash
set -euo pipefail

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-wsl-pytorch}"

cd "$(dirname "${BASH_SOURCE[0]}")"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/e/dataset/Iris/runs/lotus_pipeline_demo}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/e/dataset/Iris/runs/lotus_pipeline_demo_eval}"
EVAL_ROOT="${EVAL_ROOT:-/mnt/e/dataset/Iris/eval}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. This demo requires an NVIDIA CUDA GPU.")

name = torch.cuda.get_device_name(0)
print(f"Using CUDA device 0: {name}")
if "nvidia" not in name.lower() and "geforce" not in name.lower() and "rtx" not in name.lower():
    raise SystemExit(f"Unexpected CUDA device: {name}")
PY

python eval_rng.py \
  --pretrained_model_name_or_path "$CHECKPOINT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --base_test_data_dir "$EVAL_ROOT" \
  --eval_config_dir /mnt/e/project/Iris/lotus/datasets/eval/depth \
  --half_precision \
  --task_name depth \
  --mode regression \
  --disparity \
  --eval_datasets scannet \
  --max_eval_samples 1 \
  --scannet_text_descriptions_path "$EVAL_ROOT/descriptions/scannet_depth_descriptions_smoke.json" \
  --strict_text_mode

echo "Saved demo eval outputs to: $OUTPUT_DIR"
