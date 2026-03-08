#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval_baseline_retrain"}

CUDA_VISIBLE_DEVICES=3 python eval.py \
    --base_data_dir data \
    --dataset_config config/dataset/data_diode_all.yaml \
    --alignment least_square \
    --prediction_dir output/${subfolder}/diode/prediction \
    --output_dir output/${subfolder}/diode/eval_metric \
