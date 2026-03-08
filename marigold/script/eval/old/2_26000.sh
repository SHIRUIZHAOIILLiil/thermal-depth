#!/usr/bin/env bash
set -e
set -x

ckpt=${1:-"checkpoint/marigold_26000"}
subfolder=${2:-"eval_PD_iter_26000"}

CUDA_VISIBLE_DEVICES=2 python infer.py  \
    --checkpoint $ckpt \
    --seed 1234 \
    --base_data_dir data \
    --denoise_steps 50 \
    --ensemble_size 1 \
    --processing_res 0 \
    --dataset_config config/dataset/data_nyu_test.yaml \
    --output_dir output/${subfolder}/nyu_test/prediction \
    --dataset nyuv2

CUDA_VISIBLE_DEVICES=2 python eval.py \
    --base_data_dir data \
    --dataset_config config/dataset/data_nyu_test.yaml \
    --alignment least_square \
    --prediction_dir output/${subfolder}/nyu_test/prediction \
    --output_dir output/${subfolder}/nyu_test/eval_metric \

CUDA_VISIBLE_DEVICES=2 python infer.py  \
    --checkpoint $ckpt \
    --seed 1234 \
    --base_data_dir data \
    --denoise_steps 50 \
    --ensemble_size 1 \
    --processing_res 0 \
    --dataset_config config/dataset/data_kitti_eigen_test.yaml \
    --output_dir output/${subfolder}/kitti_eigen_test/prediction \
    --dataset kitti

CUDA_VISIBLE_DEVICES=2 python eval.py \
    --base_data_dir data \
    --dataset_config config/dataset/data_kitti_eigen_test.yaml \
    --alignment least_square \
    --prediction_dir output/${subfolder}/kitti_eigen_test/prediction \
    --output_dir output/${subfolder}/kitti_eigen_test/eval_metric \

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

CUDA_VISIBLE_DEVICES=2 python eval.py \
    --base_data_dir data \
    --dataset_config config/dataset/data_eth3d.yaml \
    --alignment least_square \
    --prediction_dir  output/${subfolder}/eth3d/prediction \
    --output_dir output/${subfolder}/eth3d/eval_metric \
    --alignment_max_res 1024 \

CUDA_VISIBLE_DEVICES=2 python infer.py  \
    --checkpoint $ckpt \
    --seed 1234 \
    --base_data_dir data \
    --denoise_steps 50 \
    --ensemble_size 1 \
    --processing_res 0 \
    --dataset_config config/dataset/data_scannet_val.yaml \
    --output_dir output/${subfolder}/scannet/prediction \
    --dataset scannet

CUDA_VISIBLE_DEVICES=2 python eval.py \
    --base_data_dir data \
    --dataset_config config/dataset/data_scannet_val.yaml \
    --alignment least_square \
    --prediction_dir output/${subfolder}/scannet/prediction \
    --output_dir output/${subfolder}/scannet/eval_metric \
