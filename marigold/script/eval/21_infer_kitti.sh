#!/usr/bin/env bash
set -e
set -x

# Use specified checkpoint path, otherwise, default value
ckpt=${1:-"checkpoint/marigold-v1-0"}
subfolder=${2:-"eval_PD_iter_20000"}

CUDA_VISIBLE_DEVICES=1 python infer.py  \
    --checkpoint $ckpt \
    --seed 1234 \
    --base_data_dir data \
    --denoise_steps 50 \
    --ensemble_size 1 \
    --processing_res 0 \
    --dataset_config config/dataset/data_kitti_eigen_test.yaml \
    --output_dir output/${subfolder}/kitti_eigen_test/prediction \
    --dataset kitti
