#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval_PD_iter_20000"}

CUDA_VISIBLE_DEVICES=1 python eval.py \
    --base_data_dir data \
    --dataset_config config/dataset/data_kitti_eigen_test.yaml \
    --alignment least_square \
    --prediction_dir output/${subfolder}/kitti_eigen_test/prediction \
    --output_dir output/${subfolder}/kitti_eigen_test/eval_metric \
