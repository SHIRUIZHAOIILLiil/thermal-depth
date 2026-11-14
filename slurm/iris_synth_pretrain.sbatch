#!/bin/bash
#SBATCH --job-name=iris_synth_pt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# Iris's synthetic pretraining (docs/IRIS_SYNTH_PRETRAIN_V1.md, stage 2), from SD2.
#
# The argument list is train_scripts/train_iris_g_depth.sh line for line, except:
#   - validation is off: Iris scores NYU/KITTI/... from BASE_TEST_DATA_DIR, which we
#     do not have. --validation_images unset skips the step-0 pass; the periodic one
#     at validation_steps is not guarded by it, so validation_steps is pushed past the
#     budget. Checkpoints still land every 1000 steps.
#   - --checkpoints_total_limit=2 (disk; the step budget is unchanged);
#   - --dataloader_num_workers=6 instead of 0 (loading only);
#   - SD2 is the 2.1-base mirror every sd2 arm uses (iris_ms2.sbatch, START=sd2).
#
#   SMOKE=1   20 steps into <out>_smoke, never resumes
#   default   20000 steps into $IRIS_RUNS/synth_pretrain/$RUN_TAG; resubmitting the same
#             command resumes from the latest checkpoint, and a finished run (unet/
#             saved by the trainer's final save_pretrained) exits at once, so a second
#             copy can be chained with afterany to cover a wall-clock timeout.
set -euo pipefail
# common.sh refuses to start without the MS2 tree (2026-11-10: job 9399130 died in 1 s
# on exactly that, the submit block unsets these). Same prelude as synth_data.sbatch.
F="${SCRATCH:?}/from_sc23sz"
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

SYN="${SYNTH_ROOT:-$SCRATCH/data/synth}"
MODEL_NAME="${SD2_REPO:-Manojb/stable-diffusion-2-1-base}"
TAG="${RUN_TAG:-iris_g_synth_sd2_s42}"
# GAS: gradient accumulation. 3 is train_iris_g_depth.sh's single-GPU value (effective
# batch 12). Iris's supplementary (Lotus paragraph) trained 4 per step x 3 accumulation x
# 3 RTX 3090s = 36; on one GPU that is GAS=9 (same 20000 optimizer steps, 3x the samples).
GAS="${GAS:-3}"
OUT="$IRIS_RUNS/synth_pretrain/$TAG"
HYP_TEXT="$SYN/captions/hypersim_depth_descriptions.json"
VK_TEXT="$SYN/captions/vkitti_depth_descriptions.json"

for f in "$SYN/hypersim/train/metadata.jsonl" "$HYP_TEXT" "$VK_TEXT"; do
  [[ -s "$f" ]] || { echo "!! missing $f (synth data / caption chain not finished)"; exit 1; }
done
[[ -d "$HF_HOME/hub/models--${MODEL_NAME//\//--}" ]] || { echo "!! $MODEL_NAME not in $HF_HOME"; exit 1; }

STEPS=20000 CKPT=1000 RESUME=(--resume_from_checkpoint=latest)
if [[ "${SMOKE:-0}" == "1" ]]; then
  OUT="${OUT}_smoke" STEPS=20 CKPT=20 RESUME=()
  rm -rf "$OUT"
elif [[ -s "$OUT/unet/diffusion_pytorch_model.safetensors" ]]; then
  echo "[done] $OUT/unet already saved; nothing to do"; exit 0
fi
mkdir -p "$OUT"
echo "model=$MODEL_NAME  out=$OUT  steps=$STEPS  effective batch=$((4 * GAS))"
echo "captions: hypersim $(python -c "import json;print(len(json.load(open('$HYP_TEXT'))))")  vkitti $(python -c "import json;print(len(json.load(open('$VK_TEXT'))))")"

cd "$IRIS_REPO/lotus"
git -C "$IRIS_REPO" log --oneline -1
accelerate launch --config_file=accelerate_configs/0.yaml --mixed_precision="fp16" \
  --main_process_port="${MAIN_PROCESS_PORT:-13224}" \
  train_iris_g.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir_hypersim="$SYN/hypersim" \
  --resolution_hypersim=576 \
  --train_data_dir_vkitti="$SYN/vkitti" \
  --resolution_vkitti=375 \
  --prob_hypersim=0.9 \
  --mix_dataset \
  --random_flip \
  --norm_type=trunc_disparity \
  --dataloader_num_workers=6 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=$GAS \
  --gradient_checkpointing \
  --max_grad_norm=1 \
  --seed=42 \
  --max_train_steps=$STEPS \
  --learning_rate=3e-05 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --task_name=depth \
  --timestep=999 \
  --validation_steps=1000000000 \
  --checkpointing_steps=$CKPT \
  --checkpoints_total_limit=2 \
  --output_dir="$OUT" \
  "${RESUME[@]}" \
  --use_8bit_adam \
  --hypersim_text_descriptions_path="$HYP_TEXT" \
  --vkitti_text_descriptions_path="$VK_TEXT"
[[ -s "$OUT/unet/diffusion_pytorch_model.safetensors" ]] || { echo "!! trainer exited without $OUT/unet"; exit 1; }
echo "[done] $OUT"
