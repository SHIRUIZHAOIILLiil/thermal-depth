#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/project/Iris
CONDA=/home/dawn/miniconda3/bin/conda
TRAIN_MANIFEST=/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/ms2_fixed_sequence_internvl3_8b_filtered_caption_train.jsonl
VAL_MANIFEST=/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/ms2_fixed_sequence_internvl3_8b_filtered_caption_val.jsonl

echo "TRAIN_MANIFEST_SHA256"
sha256sum "$TRAIN_MANIFEST"
echo "VAL_MANIFEST_SHA256"
sha256sum "$VAL_MANIFEST"

"$CONDA" run --no-capture-output -n wsl-pytorch \
  python tools/train_ms2_adapter_v0.py \
  --train-manifest "$TRAIN_MANIFEST" \
  --val-manifest "$VAL_MANIFEST" \
  --ms2-root /mnt/e/dataset/ms2 \
  --output-dir outputs/lotus_line_v1/adapter_only_no_caption_v1 \
  --train-mode adapter_only \
  --caption-mode disabled \
  --batch-size 4 \
  --num-epochs 1 \
  --max-steps 1000 \
  --learning-rate 1e-4 \
  --adapter-lr 1e-4 \
  --weight-decay 1e-4 \
  --lr-scheduler constant \
  --mixed-precision no \
  --gradient-accumulation-steps 1 \
  --max-grad-norm 1.0 \
  --seed 42 \
  --timestep 999 \
  --validation-interval 100 \
  --val-batch-limit 100 \
  --checkpoint-interval 100 \
  --visualization-interval 0 \
  --num-fixed-val-samples 8 \
  --num-workers 0 \
  --local-files-only

echo "RUN_COMPLETE adapter_only_no_caption_v1"
