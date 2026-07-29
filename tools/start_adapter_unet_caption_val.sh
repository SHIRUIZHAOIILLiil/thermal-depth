#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/project/Iris

output_dir="outputs/lotus_line_v1/adapter_unet_correct_caption_official_val_full"
log_file="logs/adapter_unet_correct_caption_official_val_full.log"
pid_file="logs/adapter_unet_correct_caption_official_val_full.pid"

if [[ -d "$output_dir" ]] && [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing non-empty output directory: $output_dir"
  exit 1
fi

PYTHONUNBUFFERED=1 nohup python tools/run_ms2_lotus_trained_official.py \
  --manifest /mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/ms2_fixed_sequence_internvl3_8b_filtered_caption_val.jsonl \
  --ms2-root /mnt/e/dataset/ms2 \
  --checkpoint outputs/lotus_line_v1/adapter_unet_correct_caption_v1/checkpoint_final.pt \
  --output-dir "$output_dir" \
  --max-samples 0 \
  --caption-mode correct \
  --local-files-only \
  > "$log_file" 2>&1 &

pid=$!
printf '%s\n' "$pid" > "$pid_file"
echo "Started PID $pid"
echo "Log: $log_file"
