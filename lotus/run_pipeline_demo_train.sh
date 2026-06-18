#!/usr/bin/env bash
set -euo pipefail

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-wsl-pytorch}"

cd "$(dirname "${BASH_SOURCE[0]}")"

SMOKE_ROOT="${SMOKE_ROOT:-/mnt/e/dataset/Iris/smoke_train}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/e/dataset/Iris/models/lotus-depth-d-v2-0-disparity}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/e/dataset/Iris/runs/lotus_pipeline_demo}"
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

python ../tools/validate_lotus_data.py \
  --lotus-root . \
  --smoke-root "$SMOKE_ROOT" \
  --eval-depth-root "${EVAL_DEPTH_ROOT:-/mnt/e/dataset/Iris/eval/depth}"

accelerate launch \
  --config_file=accelerate_configs/0.yaml \
  --mixed_precision=fp16 \
  train_iris_d.py \
  --pretrained_model_name_or_path "$MODEL_ROOT" \
  --train_data_dir_hypersim "$SMOKE_ROOT/hypersim" \
  --train_data_dir_vkitti "$SMOKE_ROOT/vkitti" \
  --resolution_hypersim 256 \
  --resolution_vkitti 352 \
  --norm_type trunc_disparity \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_train_samples 1 \
  --max_train_steps 1 \
  --learning_rate 3e-5 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --task_name depth \
  --timestep 999 \
  --dataloader_num_workers 0 \
  --checkpointing_steps 9999 \
  --validation_steps 9999 \
  --output_dir "$OUTPUT_DIR" \
  --use_8bit_adam \
  --hypersim_text_descriptions_path "$SMOKE_ROOT/descriptions/hypersim_depth_descriptions.json" \
  --vkitti_text_descriptions_path "$SMOKE_ROOT/descriptions/vkitti_depth_descriptions.json"

echo "Saved demo checkpoint to: $OUTPUT_DIR"
