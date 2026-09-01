#!/bin/bash
#SBATCH --job-name=irisms2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 「完整伪 GT + Iris 训练方式」线。跑的是 lotus/train_iris_ms2_g.py ——
# Iris 自己那份 trainer 的复制件，只换数据集，超参一个没动。
# 与 route_suite.sbatch 不是一条流水线，别混用。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   START            起点权重，二选一（这条线两个臂唯一的差别）：
#                      ckpt = jingheya/lotus-depth-g-v2-1-disparity（已训好的
#                             Lotus-G，跳过 conv_in 扩展）—— 默认，也是目前唯一能跑的
#                      sd2  = stabilityai/stable-diffusion-2-base（Iris 自己的起点，
#                             conv_in 做 4→8 扩展）
#                    ⚠️ 2026-08-11：**sd2 那一臂暂时跑不了。** Stability AI 已把
#                    SD 2.0/2.1 全系从 HuggingFace 下架（合规原因），匿名与带 token
#                    请求一律 401、网页 404。要恢复这一臂，需要先找到社区镜像并验真：
#                    Lotus 的 vae/ 与 text_encoder/ 是**未经修改的 SD2-base 组件**
#                    （Lotus 只训 U-Net），拿镜像的这两份与缓存里的 lotus-depth
#                    逐张量比对即可。比张量不要比文件哈希 —— 重新序列化会让字节不等
#                    而权重相等。换更新的基座不是替代方案：SD3/3.5 是 MMDiT + 16 通道
#                    latent（`UNet2DConditionModel` 根本加载不了），SDXL 自己占用了
#                    `class_embed_type`（Lotus 正靠它塞任务嵌入），且两者都会换掉 VAE，
#                    而两臂共用同一个 VAE 正是这个对照成立的前提。
#   NO_CAPTIONS=1    无文本消融臂（每帧喂空串）。不设＝按 Iris 用真 caption 训练。
#   SKY_MASKS        天空掩码目录（tools/build_sky_masks.py 的 masks/）。不设＝关闭。
#                    设了就启用优先级规则：激光 > 天空掩码 > 伪深度，天空像素的目标
#                    被改写成 d_max=80m。这不是加损失项，是修目标 —— 实测稠密补全图
#                    在天空里中位数只有 26.8m，而模型忠实地学成了 24.5m。
#                    ⚠️ 掩码必须覆盖 TRAIN_MANIFEST 的每一帧，缺一帧就报错退出。
#   PSEUDO_DIR       calibrated_pseudo_depth 目录，默认 official_train
#                    ⚠️ 2026-08-26：默认值原本是 train_full，而盘上那个目录并不存在
#                    —— 官方 8 序列那三条臂用的一直是 official_train。一个指向不存在
#                    路径的默认值，唯一的作用是让每次忘记传 PSEUDO_DIR 的提交排完队
#                    再失败。改成实际在用的那个。
#   TRAIN_MANIFEST   默认 clip75_20260728（19,949 帧，≤75 token）
#   MAX_TRAIN_STEPS  默认 20000（Iris 的步数）。SMOKE 时会被覆盖
#   SMOKE=1          20 步就退，只为验开关是否生效
#   SEED             训练随机种子，默认 42。换种子必须同时换 RUN_TAG，否则
#                    两个种子写进同一个输出目录、并被 resume 逻辑接上。
#   RUN_TAG          覆盖输出目录名
#
# ⚠️ lr 3e-05 是 Iris 从 SD2-base 起训的量级。START=ckpt 那一臂沿用它是
#    刻意的（两臂只差起点权重才构成受控对照），但它有把 checkpoint 冲掉的
#    风险 —— 先看 SMOKE 的 SL_A 曲线，再决定要不要投全量。
#
# ── 用法 ────────────────────────────────────────────────────────────
#   冒烟（两臂各一次，各约 10 分钟）：
#     START=sd2  SMOKE=1 sbatch -J sm_sd2  --time=00:40:00 --export=ALL,START,SMOKE ~/Iris/slurm/iris_ms2.sbatch
#     START=ckpt SMOKE=1 sbatch -J sm_ckpt --time=00:40:00 --export=ALL,START,SMOKE ~/Iris/slurm/iris_ms2.sbatch
#
#   全量（两臂并行）：
#     START=sd2  sbatch -J iris_sd2  --export=ALL,START ~/Iris/slurm/iris_ms2.sbatch
#     START=ckpt sbatch -J iris_ckpt --export=ALL,START ~/Iris/slurm/iris_ms2.sbatch
#
# 续跑：同一 RUN_TAG 重投即自动 resume（trainer 的 --resume_from_checkpoint=latest）。

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

START="${START:-ckpt}"
case "$START" in
  sd2)  MODEL_NAME="stabilityai/stable-diffusion-2-base" ;;
  ckpt) MODEL_NAME="jingheya/lotus-depth-g-v2-1-disparity" ;;
  # Lotus 的 direct 变体。conv_in 是 4 通道而不是 8 —— 这是两个底座唯一的架构差别，
  # 任务开关、双分支 loss、caption、伪 GT、评估协议全部共用。BACKBONE 跟着 START 走，
  # 不单独暴露，免得两者对不上。
  lotus_d) MODEL_NAME="jingheya/lotus-depth-d-v2-0-disparity"; BACKBONE=d ;;
  # Marigold：多步生成式，v 目标，单分支无任务开关。归一化和预测类型跟着一起换，
  # 三者是一套，分开设很容易只改一半 —— 依据是 Iris 补充材料 B.2 和他们自己的
  # marigold_trainer.py.PD_text_prior。
  marigold) MODEL_NAME="prs-eth/marigold-v1-0"; BACKBONE=marigold
            NORMTYPE="${NORMTYPE:-truncnorm}"
            PREDICTION_TYPE="${PREDICTION_TYPE:-v_prediction}"
            # config/train_marigold.yaml: IterExponential, total_iter 25000,
            # final_ratio 0.01, warmup 100. Their total is 25000 against a
            # 30000-step budget, so a 20000-step run stops with the rate still
            # at 0.025 -- that is their curve, cut short by our budget. Passing
            # 20000 here would be a different schedule, not a shorter one.
            LR_SCHEDULER="${LR_SCHEDULER:-iter_exponential}"
            LR_WARMUP="${LR_WARMUP:-100}"
            LR_TOTAL_ITER="${LR_TOTAL_ITER:-25000}" ;;
  # E2E-FT：单步确定性，v 目标，损失在**解码后的图像**上做尺度平移不变 L1。
  # 归一化同 Marigold（0.02/0.98 分位），推理 1 步，所以评估和 Lotus 一样便宜。
  e2eft) MODEL_NAME="GonzaloMG/stable-diffusion-e2e-ft-depth"; BACKBONE=e2eft
         NORMTYPE="${NORMTYPE:-truncnorm}"
         PREDICTION_TYPE="${PREDICTION_TYPE:-v_prediction}"
         # train_marigold_e2e_ft_depth.sh: --lr_total_iter_length 20000
         # --lr_exp_warmup_steps 100. Their budget is 20000 steps, the same as
         # ours, so this one lands on their endpoint exactly.
         LR_SCHEDULER="${LR_SCHEDULER:-iter_exponential}"
         LR_WARMUP="${LR_WARMUP:-100}"
         LR_TOTAL_ITER="${LR_TOTAL_ITER:-20000}" ;;
  *)    echo "!! START 只能是 sd2 / ckpt / lotus_d / marigold / e2eft，收到 '$START'"; exit 1 ;;
esac
BACKBONE="${BACKBONE:-g}"

# 权重不在缓存里就立刻停，别等 from_pretrained 去下载 —— SD2-base 已被 Stability
# 下架（见上），那条路只会走到一个看不懂的 401 traceback，而这时排队已经花掉几小时。
HF_REPO_DIR="$HF_HOME/hub/models--${MODEL_NAME//\//--}"
if [[ ! -d "$HF_REPO_DIR" ]]; then
  echo "!! $MODEL_NAME 不在 HF 缓存里（找不到 $HF_REPO_DIR）"
  if [[ "$START" == "sd2" ]]; then
    echo "!! sd2 臂当前无官方来源：Stability 已下架 SD 2.x，见本脚本开头的说明。"
  fi
  echo "!! 先用 CPU 分区预取，别让 GPU 作业去下载。"
  exit 1
fi

TAG="${RUN_TAG:-iris_ms2_$START}"
[[ "${NO_CAPTIONS:-0}" == "1" && -z "${RUN_TAG:-}" ]] && TAG="${TAG}_nocap"
if [[ "${SMOKE:-0}" == "1" ]]; then
  TAG="smoke_$TAG"
  # ⚠️ 无条件覆盖，不是 ${MAX_TRAIN_STEPS:-20}。登录 shell 里 export 过一次，
  # --export=ALL 就会把它带进来，于是「20 步冒烟」跑成全量、墙钟到点被砍。
  # 2026-08-27 在 metric_adapt.sbatch 上正是这么栽的。别的长度用 SMOKE_STEPS。
  MAX_TRAIN_STEPS="${SMOKE_STEPS:-20}"
  CKPT_STEP=20
  # And never resume. The smoke tag is smoke_iris_ms2_<backbone> unless
  # RUN_TAG says otherwise, so the second smoke of a backbone lands on the
  # first one's directory, finds its checkpoint-20, and exits having trained
  # nothing -- reporting COMPLETED, with the previous run's learning rate
  # restored into the progress bar. 2026-08-31 lost a round to exactly that
  # while checking whether a new scheduler was wired up.
  RESUME_FROM=""
  echo "[smoke] 强制 MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS（忽略环境里可能残留的值）"
fi

export MODEL_NAME
export BACKBONE
export NORMTYPE="${NORMTYPE:-trunc_disparity}"
export PREDICTION_TYPE="${PREDICTION_TYPE:-sample}"
# Lotus ships a constant rate; the other two decay it. Defaulted here so the
# Lotus lines are untouched by this.
# Both default to the original objective: no flag, and a zero weight.
export PURE_PSEUDO="${PURE_PSEUDO:-0}"
export LAMBDA_IMAGE="${LAMBDA_IMAGE:-0}"
export LR_SCHEDULER="${LR_SCHEDULER:-constant}"
export LR_WARMUP="${LR_WARMUP:-0}"
export LR_TOTAL_ITER="${LR_TOTAL_ITER:-0}"
# Plain - not :- , so the smoke block's empty value survives.
export RESUME_FROM="${RESUME_FROM-latest}"
export OUTPUT_DIR="$IRIS_RUNS/iris_ms2/$TAG"
export MS2_MANIFEST="${TRAIN_MANIFEST:-$IRIS_MANIFEST_DIR/ms2_train_day2seq_clip75_20260728.jsonl}"
export MS2_ROOT="$IRIS_MS2_ROOT"
export PSEUDO_GT_DIR="${PSEUDO_DIR:-$IRIS_RUNS/pseudo_gt/official_train/calibrated_pseudo_depth}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20000}"
export CKPT_STEP="${CKPT_STEP:-1000}"
export MAIN_PORT="$MAIN_PROCESS_PORT"
export NO_CAPTIONS="${NO_CAPTIONS:-0}"
export SKY_MASK_DIR="${SKY_MASKS:-}"

[[ -f "$MS2_MANIFEST" ]]     || { echo "!! 找不到 manifest: $MS2_MANIFEST"; exit 1; }
if [[ -n "$SKY_MASK_DIR" ]]; then
  [[ -d "$SKY_MASK_DIR" ]] || { echo "!! 找不到天空掩码目录: $SKY_MASK_DIR"; exit 1; }
  n_masks=$(ls "$SKY_MASK_DIR" | wc -l)
  n_frames=$(wc -l < "$MS2_MANIFEST")
  echo "sky masks = $SKY_MASK_DIR  ($n_masks 张 / manifest $n_frames 帧)"
  (( n_masks >= n_frames )) || {
    echo "!! 掩码数少于 manifest 帧数 —— 训练会在第一帧缺掩码时报错退出。"
    echo "!! 用 MAX_SAMPLES=0 SAVE_MASKS=1 对同一份 manifest 重出掩码。"
    exit 1
  }
fi

[[ -d "$PSEUDO_GT_DIR" ]]    || {
  echo "!! 找不到伪 GT 目录: $PSEUDO_GT_DIR"
  echo "!! $IRIS_RUNS/pseudo_gt/ 下现有的是："
  ls -1 "$IRIS_RUNS/pseudo_gt/" 2>/dev/null | sed 's/^/!!   /'
  echo "!! 用 PSEUDO_DIR=<其中之一>/calibrated_pseudo_depth 显式指定。"
  exit 1
}
mkdir -p "$OUTPUT_DIR"

echo "--- 本次配置 ---"
echo "start     = $START ($MODEL_NAME)"
echo "captions  = $([[ "$NO_CAPTIONS" == "1" ]] && echo off || echo on)"
echo "sky rule  = $([[ -n "$SKY_MASK_DIR" ]] && echo "on ($SKY_MASK_DIR)" || echo off)"
echo "manifest  = $MS2_MANIFEST"
echo "pseudo gt = $PSEUDO_GT_DIR"
echo "steps     = $MAX_TRAIN_STEPS (ckpt every $CKPT_STEP)"
echo "target    = $([[ "$PURE_PSEUDO" == "1" ]] && echo pure-pseudo || echo completed-with-lidar)   lambda_image=$LAMBDA_IMAGE"
echo "resume    = ${RESUME_FROM:-<off>}"
echo "lr        = 3e-05 / $LR_SCHEDULER (warmup $LR_WARMUP, total_iter $LR_TOTAL_ITER)"
echo "output    = $OUTPUT_DIR"
echo "----------------"
echo "门禁：日志里必须同时出现 SL_A（深度分支）与 SL_R（热像重建分支），"
echo "      以及一行 'target covers ...% of pixels' 且该数应为 100.0%。"

cd "$IRIS_REPO/lotus"
git -C "$IRIS_REPO" log --oneline -1
bash train_scripts/train_iris_ms2_g_depth.sh
