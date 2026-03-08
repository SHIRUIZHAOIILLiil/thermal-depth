#!/usr/bin/env bash
set -e
set -x

# add
checkpoint="stable-diffusion-e2e-ft-depth_infer_only"

# add
python Marigold/eval.py \
    --base_data_dir="path/to/datasets/eval/depth/" \
    --dataset_config Marigold/config/dataset/data_eth3d.yaml \
    --alignment least_square \
    --prediction_dir="experiments/depth/eval/e2e_ft/$checkpoint/eth3d/prediction" \
    --output_dir="experiments/depth/eval/e2e_ft/$checkpoint/eth3d/eval_metric" \
    --alignment_max_res 1024