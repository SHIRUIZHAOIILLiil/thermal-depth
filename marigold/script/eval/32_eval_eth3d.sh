#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval_PD_iter_20000"}

CUDA_VISIBLE_DEVICES=2 python eval.py \
    --base_data_dir data \
    --dataset_config config/dataset/data_eth3d.yaml \
    --alignment least_square \
    --prediction_dir  output/${subfolder}/eth3d/prediction \
    --output_dir output/${subfolder}/eth3d/eval_metric \
    --alignment_max_res 1024 \