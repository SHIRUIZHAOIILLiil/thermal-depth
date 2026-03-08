#!/usr/bin/env bash
set -e
set -x

# add
checkpoint="stable-diffusion-e2e-ft-depth_infer_only"

checkpoint_path="model-finetuned/stable_diffusion_e2e_ft_depth"
# add
python Marigold/infer.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path" \
    --base_data_dir="path/to/datasets/eval/depth/" \
    --processing_res 756 \
    --dataset_config Marigold/config/dataset/data_eth3d.yaml \
    --output_dir="experiments/depth/eval/e2e_ft/$checkpoint/eth3d/prediction" \
    --resample_method bilinear \
    --model_type "marigold"