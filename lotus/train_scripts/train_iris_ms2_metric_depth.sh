# Metric GT adaptation, stage 2 of the MS2 thermal line.
#
# A sibling of train_iris_ms2_g_depth.sh rather than a mode inside it. That file
# is the recipe every published checkpoint was trained with, hyper-parameter for
# hyper-parameter, and a run of it must keep meaning exactly that. This one
# starts from the checkpoint that file produced and changes four things:
#
#   1. --init_unet_from      continue from step 20000 instead of the base weights
#   2. --metric_adaptation   the target is normalised by two frozen TRAIN constants
#   3. --lambda_metric       a masked image-space L1 against real lidar is added
#   4. LR                    3e-5 -> 1e-6 by default: this is adaptation, not training
#
# Everything else -- batch, accumulation, timestep, optimiser, precision,
# clipping, seed -- is inherited unchanged, so the two stages remain comparable.
#
# Required:
#   MODEL_NAME          pipeline the VAE/CLIP/scheduler come from
#   MS2_MANIFEST        TRAIN manifest
#   MS2_ROOT            dataset root
#   PSEUDO_GT_DIR       calibrated_pseudo_depth
#   METRIC_NORM_JSON    tools/fit_metric_norm.py artifact, source_split=train
#   INIT_UNET_FROM      checkpoint to adapt (step20000_weights.pt)
#
# Optional:
#   LAMBDA_METRIC / LAMBDA_DENSE / LAMBDA_RECON   default 1.0 / 1.0 / 1.0
#   LEARNING_RATE       default 1e-6
#   MAX_TRAIN_STEPS     default 4000
#   NO_CAPTIONS=1       no-text ablation. NOT for the main run: the checkpoint
#                       being adapted was trained with captions, and the task is
#                       to isolate geometry, not to change the text condition.

export MODEL_NAME="${MODEL_NAME:?set MODEL_NAME}"
export MS2_MANIFEST="${MS2_MANIFEST:?set MS2_MANIFEST}"
export MS2_ROOT="${MS2_ROOT:?set MS2_ROOT}"
export PSEUDO_GT_DIR="${PSEUDO_GT_DIR:?set PSEUDO_GT_DIR}"
export METRIC_NORM_JSON="${METRIC_NORM_JSON:?set METRIC_NORM_JSON}"
export INIT_UNET_FROM="${INIT_UNET_FROM:?set INIT_UNET_FROM}"

# Inherited from the base recipe, unchanged.
export BATCH_SIZE=4
export CUDA="${CUDA:-0}"
export GAS=3
export TIMESTEP=999
export TASK_NAME="depth"

# Changed, deliberately.
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-4000}"
export LAMBDA_METRIC="${LAMBDA_METRIC:-1.0}"
export LAMBDA_DENSE="${LAMBDA_DENSE:-1.0}"
export LAMBDA_RECON="${LAMBDA_RECON:-1.0}"

export OUTPUT_DIR="${OUTPUT_DIR:-output/iris_ms2_metric/}"
export CKPT_STEP="${CKPT_STEP:-500}"

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
  --metric_adaptation \
  --metric_norm_json=$METRIC_NORM_JSON \
  --init_unet_from=$INIT_UNET_FROM \
  --lambda_metric=$LAMBDA_METRIC \
  --lambda_dense=$LAMBDA_DENSE \
  --lambda_recon=$LAMBDA_RECON \
  --metric_log_every="${METRIC_LOG_EVERY:-200}" \
  --random_flip \
  --dataloader_num_workers="${DATALOADER_WORKERS:-0}" \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=$GAS \
  --gradient_checkpointing \
  --max_grad_norm=1 \
  --seed="${SEED:-42}" \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --learning_rate=$LEARNING_RATE \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --task_name=$TASK_NAME \
  --timestep=$TIMESTEP \
  --validation_steps=$CKPT_STEP \
  --checkpointing_steps=$CKPT_STEP \
  --output_dir=$OUTPUT_DIR \
  --resume_from_checkpoint="latest" \
  --use_8bit_adam
