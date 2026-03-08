export BASE_TEST_DATA_DIR="datasets/eval/"

export CHECKPOINT_DIR="output/iris_g/train-lotus-g-depth"
export OUTPUT_DIR="output/Depth_G_Eval_Iris"
export TASK_NAME="depth"

export MODE="generation"

CUDA_VISIBLE_DEVICES=$CUDA python eval_rng.py \
        --pretrained_model_name_or_path=$CHECKPOINT_DIR \
        --output_dir=$OUTPUT_DIR \
        --seed=42 \
        --mode=$MODE \
        --task_name=$TASK_NAME \
        --nyuv2_text_descriptions_path="internVL/outputs/eval/nyu_depth_descriptions.json" \
        --kitti_text_descriptions_path="internVL/outputs/eval/kitti_depth_descriptions.json" \
        --scannet_text_descriptions_path="internVL/outputs/eval/scannet_depth_descriptions.json" \
        --eth3d_text_descriptions_path="internVL/outputs/eval/eth3d_depth_descriptions.json" \
        --diode_text_descriptions_path="internVL/outputs/eval/diode_depth_descriptions.json" \
        --base_test_data_dir=$BASE_TEST_DATA_DIR \
        --half_precision \
        --disparity \