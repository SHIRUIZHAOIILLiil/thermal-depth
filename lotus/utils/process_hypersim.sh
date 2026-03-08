#!/bin/bash
export PATH_TO_HYPERSIM_SPLIT_CSV=lotus/scripts/downloads/metadata_images_split_scene_v1.csv
export PATH_TO_RAW_HYPERSIM_DATA=lotus/scripts/downloads
export PATH_TO_HYPERSIM_DATA=lotus/data/hypersim

python utils/process_hypersim.py \
    --csv_path=$PATH_TO_HYPERSIM_SPLIT_CSV \
    --src_path=$PATH_TO_RAW_HYPERSIM_DATA \
    --trg_path=$PATH_TO_HYPERSIM_DATA \
    --split='train' \
    --filter_nan
