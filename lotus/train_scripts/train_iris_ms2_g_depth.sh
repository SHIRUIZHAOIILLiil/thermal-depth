# Iris's Lotus-G recipe on MS2 thermal, against the completed pseudo GT.
#
# A copy of train_iris_g_depth.sh. Every training hyper-parameter below is the
# value Iris ships -- batch, accumulation, timestep, normalisation, learning
# rate, schedule, optimiser, precision, gradient clipping, step budget, seed.
# Only the dataset flags differ, plus MODEL_NAME, which selects the one thing
# the two runs of this line are meant to differ by:
#
#   MODEL_NAME=stabilityai/stable-diffusion-2-base   -> Iris's own starting point,
#                                                       conv_in expanded 4->8
#   MODEL_NAME=jingheya/lotus-depth-g-v2-1-disparity -> the trained Lotus-G
#                                                       checkpoint, expansion skipped
#
# NO_CAPTIONS=1 runs the no-text ablation of the same recipe.

export MODEL_NAME="${MODEL_NAME:-stabilityai/stable-diffusion-2-base}"

# training dataset -- MS2 thermal + calibrated pseudo depth
export MS2_MANIFEST="${MS2_MANIFEST:?set MS2_MANIFEST}"
export MS2_ROOT="${MS2_ROOT:?set MS2_ROOT}"
export PSEUDO_GT_DIR="${PSEUDO_GT_DIR:?set PSEUDO_GT_DIR}"
export NORMTYPE="trunc_disparity"

# training configs
export BATCH_SIZE=4
export CUDA="${CUDA:-0}"
export GAS=3
export TOTAL_BSZ=$(($BATCH_SIZE * ${#CUDA} * $GAS))

# model configs
export TIMESTEP=999
export TASK_NAME="depth"

# output dir
export OUTPUT_DIR="${OUTPUT_DIR:-output/iris_ms2_g/train-lotus-g-${TASK_NAME}-bsz${TOTAL_BSZ}/}"

# Iris's own validation renders and scores its eval datasets, which are not on
# disk here; MS2 checkpoints are scored out of band under the official BridgeMSD
# protocol. Leaving VALIDATION_IMAGES unset skips it. Checkpoints still land
# every CKPT_STEP steps, and those are what gets evaluated.
export CKPT_STEP="${CKPT_STEP:-1000}"

CAPTION_FLAG=""
[[ "${NO_CAPTIONS:-0}" == "1" ]] && CAPTION_FLAG="--no_captions"

accelerate launch --config_file=accelerate_configs/$CUDA.yaml --mixed_precision="fp16" \
  --main_process_port="${MAIN_PORT:-13224}" \
  train_iris_ms2_g.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --ms2_manifest=$MS2_MANIFEST \
  --ms2_root=$MS2_ROOT \
  --pseudo_gt_dir=$PSEUDO_GT_DIR \
  $CAPTION_FLAG \
  --random_flip \
  --norm_type=$NORMTYPE \
  --dataloader_num_workers=0 \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=$GAS \
  --gradient_checkpointing \
  --max_grad_norm=1 \
  --seed="${SEED:-42}" \
  --max_train_steps="${MAX_TRAIN_STEPS:-20000}" \
  --learning_rate=3e-05 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --task_name=$TASK_NAME \
  --timestep=$TIMESTEP \
  --validation_steps=$CKPT_STEP \
  --checkpointing_steps=$CKPT_STEP \
  --output_dir=$OUTPUT_DIR \
  --resume_from_checkpoint="latest" \
  --use_8bit_adam
