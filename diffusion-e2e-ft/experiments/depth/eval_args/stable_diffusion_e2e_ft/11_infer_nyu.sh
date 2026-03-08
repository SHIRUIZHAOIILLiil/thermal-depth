#!/usr/bin/env bash

set -e
set -x

checkpoint="stable-diffusion-e2e-ft-depth_infer_only"

checkpoint_path="yourstable_diffusion_e2e_ft_depth"

# add
python Marigold/infer.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path" \
    --base_data_dir="path/to/datasets/eval/depth/" \
    --processing_res 0 \
    --dataset_config Marigold/config/dataset/data_nyu_test.yaml \
    --output_dir="experiments/depth/eval/e2e_ft/$checkpoint/nyu_test/prediction" \
    --model_type "marigold"