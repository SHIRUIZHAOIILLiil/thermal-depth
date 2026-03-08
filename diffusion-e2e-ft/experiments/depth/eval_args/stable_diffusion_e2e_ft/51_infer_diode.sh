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
    --processing_res 640 \
    --dataset_config Marigold/config/dataset/data_diode_all.yaml \
    --output_dir="experiments/depth/eval/e2e_ft/$checkpoint/diode/prediction" \
    --resample_method bilinear \
    --model_type "marigold"