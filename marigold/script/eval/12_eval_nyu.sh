#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval_PriorDiffusion"}

CUDA_VISIBLE_DEVICES=0 python eval.py \
    --base_data_dir data \
    --dataset_config config/dataset/data_nyu_test.yaml \
    --alignment least_square \
    --prediction_dir output/${subfolder}/nyu_test/prediction \
    --output_dir output/${subfolder}/nyu_test/eval_metric \
