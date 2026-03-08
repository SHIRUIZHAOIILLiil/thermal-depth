#!/usr/bin/env bash
set -e
set -x

# Use specified checkpoint path, otherwise, default value
ckpt=${1:-"checkpoint/marigold-v1-0"}
subfolder=${2:-"eval_baseline_retrain"}

CUDA_VISIBLE_DEVICES=3 python infer.py  \
    --checkpoint $ckpt \
    --seed 1234 \
    --base_data_dir data \
    --denoise_steps 50 \
    --ensemble_size 1 \
    --dataset_config config/dataset/data_diode_all.yaml \
    --output_dir output/${subfolder}/diode/prediction \
    --processing_res 640 \
    --resample_method bilinear \
    --dataset diode
