#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/project/Iris
CONDA=/home/dawn/miniconda3/bin/conda

"$CONDA" run --no-capture-output -n wsl-pytorch \
  python tools/run_ms2_lotus_direct_official.py \
  --manifest /mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/ms2_fixed_sequence_internvl3_8b_filtered_caption_val.jsonl \
  --ms2-root /mnt/e/dataset/ms2 \
  --output-dir outputs/lotus_line_v1/direct_baseline_official_val_full \
  --max-samples 0 \
  --local-files-only

echo "RUN_COMPLETE direct_baseline_official_val_full"
