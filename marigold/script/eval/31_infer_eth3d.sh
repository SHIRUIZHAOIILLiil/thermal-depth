#!/usr/bin/env bash
set -e
set -x

# Use specified checkpoint path, otherwise, default value
ckpt=${1:-"checkpoint/marigold-v1-0"}
subfolder=${2:-"eval_PD_iter_20000"}

CUDA_VISIBLE_DEVICES=2 python infer.py  \
    --checkpoint $ckpt \
    --seed 1234 \
    --base_data_dir data \
    --denoise_steps 50 \
    --ensemble_size 1 \
    --dataset_config config/dataset/data_eth3d.yaml \
    --output_dir output/${subfolder}/eth3d/prediction \
    --processing_res 756 \
    --resample_method bilinear \
    --dataset eth3d