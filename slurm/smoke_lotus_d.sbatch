#!/bin/bash
#SBATCH --job-name=iris_smoke_d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 冒烟测试：20 步就退出。验证环境能激活、数据路径对、显存够、checkpoint 落到 scratch。
# 用法（在 $SCRATCH/logs 下提交，日志才不会写进 home）：
#   cd $SCRATCH/logs && sbatch $HOME/Iris/slurm/smoke_lotus_d.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

cd "$IRIS_REPO/lotus"

OUT="$IRIS_RUNS/smoke_d_$SLURM_JOB_ID"

accelerate launch --config_file=accelerate_configs/0.yaml --mixed_precision="fp16" \
  train_iris_d.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-base" \
  --train_data_dir_hypersim="$IRIS_DATA/hypersim" \
  --train_data_dir_vkitti="$IRIS_DATA/vkitti" \
  --resolution_hypersim=576 \
  --resolution_vkitti=375 \
  --prob_hypersim=0.9 \
  --mix_dataset \
  --random_flip \
  --norm_type="trunc_disparity" \
  --dataloader_num_workers="$DATALOADER_WORKERS" \
  --train_batch_size=4 \
  --gradient_accumulation_steps=3 \
  --gradient_checkpointing \
  --max_grad_norm=1 \
  --seed=42 \
  --max_train_steps=20 \
  --learning_rate=3e-05 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --task_name="depth" \
  --timestep=999 \
  --validation_images="$IRIS_REPO/datasets/quick_validation/" \
  --validation_steps=10 \
  --checkpointing_steps=10 \
  --base_test_data_dir="$IRIS_REPO/datasets/eval/" \
  --output_dir="$OUT" \
  --use_8bit_adam

echo "=== smoke done, 检查输出 ==="
ls -la "$OUT"
du -sh "$OUT"
