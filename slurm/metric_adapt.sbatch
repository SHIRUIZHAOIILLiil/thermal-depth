#!/bin/bash
#SBATCH --job-name=metric_adapt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 度量适配训练。从 iris_ms2.sbatch 产出的 checkpoint 继续，只改几何/尺度，
# 不动 caption 条件。跑的是 lotus/train_scripts/train_iris_ms2_metric_depth.sh。
#
# 与 iris_ms2.sbatch 的关系：那条线训出来的权重是这条线的**起点**，不是替代品。
# 两个作业写不同的输出目录，不要互相 resume。
#
# ── 前置 ────────────────────────────────────────────────────────────
#   先跑 metric_prep.sbatch，拿到 metric_norm_train.json。没有它这个作业起不来。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   METRIC_NORM     必填。metric_prep 产出的 metric_norm_train.json
#   INIT_CKPT       必填。起点权重（step20000_weights.pt）
#   TRAIN_MANIFEST  必填。官方 8 序列 train 清单，**必须与起点那条臂训练时同一份**
#   PSEUDO_DIR      默认 official_train 的 calibrated_pseudo_depth
#                   ⚠️ 原默认值写的是 train_full，那个目录在集群上不存在 —— 官方 8
#                   序列那三条臂用的一直是 official_train。指向不存在路径的默认值，
#                   唯一作用是让忘记传 PSEUDO_DIR 的提交排完队再失败。
#   LR              默认 1e-6
#   STEPS           默认 4000
#   LAMBDA_METRIC / LAMBDA_DENSE / LAMBDA_RECON   默认 1.0 / 1.0 / 1.0
#   CKPT_STEP       默认 500（比基线的 1000 密，因为这一阶段短得多）
#   RUN_TAG         输出目录名，默认 metric_adapt_lm<LAMBDA_METRIC>_lr<LR>
#   SMOKE=1         20 步就退，只验开关
#
# ⚠️ 门禁：日志里必须同时出现
#      SL_A（稠密 latent 分支）、SL_R（热像重建分支）、SL_M（度量分支）
#    以及 '[metric step N]' 三行 —— y / q_hat / D_hat 的量程、clip 规则、目标 clip 比例。
#    SL_M 恒为 0，或 D_hat 贴着某个边界不动，都说明这一阶段没在做它该做的事。
#
# ── 用法 ────────────────────────────────────────────────────────────
#   冒烟：
#     SMOKE=1 METRIC_NORM=$IRIS_RUNS/metric_adapt/metric_norm_train.json \
#     INIT_CKPT=$IRIS_RUNS/iris_ms2/iris_ms2_full8_thermalcap/converted/step20000_weights.pt \
#     TRAIN_MANIFEST=$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl \
#       sbatch -J ma_smoke --time=00:40:00 \
#       --export=ALL,SMOKE,METRIC_NORM,INIT_CKPT,TRAIN_MANIFEST ~/Iris/slurm/metric_adapt.sbatch
#
#   全量：去掉 SMOKE，-J ma_full，用默认墙钟。

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

: "${METRIC_NORM:?必须指定 METRIC_NORM（metric_prep 产出的 JSON）}"
: "${INIT_CKPT:?必须指定 INIT_CKPT（起点权重）}"
: "${TRAIN_MANIFEST:?必须指定 TRAIN_MANIFEST}"

# BACKBONE 驱动仓库路径，不给两个可能对不上的旋钮。不设＝g，与改动前一致。
BACKBONE="${BACKBONE:-g}"
case "$BACKBONE" in
  g) MODEL_NAME="jingheya/lotus-depth-g-v2-1-disparity" ;;
  d) MODEL_NAME="jingheya/lotus-depth-d-v2-0-disparity" ;;
  *) echo "!! BACKBONE 只能是 g 或 d，收到 '$BACKBONE'"; exit 1 ;;
esac
HF_REPO_DIR="$HF_HOME/hub/models--${MODEL_NAME//\//--}"
[[ -d "$HF_REPO_DIR" ]] || { echo "!! $MODEL_NAME 不在 HF 缓存里；先用 CPU 分区预取"; exit 1; }

[[ -f "$METRIC_NORM" ]]    || { echo "!! 找不到 $METRIC_NORM —— 先跑 metric_prep.sbatch"; exit 1; }
[[ -f "$INIT_CKPT" ]]      || { echo "!! 找不到 $INIT_CKPT"; exit 1; }
[[ -f "$TRAIN_MANIFEST" ]] || { echo "!! 找不到 $TRAIN_MANIFEST"; exit 1; }

# 常数只能是 train 上拟合的。产物自己带 source_split，这里再挡一次 ——
# 拟合脚本和训练脚本各自独立地拒绝非 train，比只在一处检查便宜得多。
python - "$METRIC_NORM" <<'PY' || exit 1
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
if d.get("source_split") != "train":
    sys.exit(f"!! {sys.argv[1]} 的 source_split 是 {d.get('source_split')!r}，只接受 'train'")
print(f"[check] 常数来自 train：q_lo={d['q_lo']:.6f} q_hi={d['q_hi']:.6f} 1/m "
      f"(深度 {1/d['q_hi']:.2f}-{1/d['q_lo']:.2f} m)，"
      f"{d['valid_pixels']:,} 个像素 / {d['frames']:,} 帧")
PY

LAMBDA_METRIC="${LAMBDA_METRIC:-1.0}"
LR="${LR:-1e-6}"
TAG="${RUN_TAG:-metric_adapt_lm${LAMBDA_METRIC}_lr${LR}}"
if [[ "${SMOKE:-0}" == "1" ]]; then
  TAG="smoke_$TAG"
  # ⚠️ 无条件覆盖，不是 ${STEPS:-20}。SMOKE 的含义就是"20 步"，而 STEPS 这个名字在
  # 本仓库的三个作业脚本里含义不同（这里是训练步数，iris_ms2_pipeline 里是要评的
  # checkpoint 步数，eval 里是去噪步数），登录 shell 里 export 过一次就会被
  # --export=ALL 带进来。2026-08-27 就是这么让一次"20 步冒烟"跑成了 4000 步的全量，
  # 40 分钟墙钟到点被砍。想要别的冒烟长度用 SMOKE_STEPS。
  STEPS="${SMOKE_STEPS:-20}"
  CKPT_STEP=20
  echo "[smoke] 强制 STEPS=$STEPS（忽略环境里可能残留的 STEPS）"
fi

export MODEL_NAME
export BACKBONE
export OUTPUT_DIR="$IRIS_RUNS/iris_ms2/$TAG"
export MS2_MANIFEST="$TRAIN_MANIFEST"
export MS2_ROOT="$IRIS_MS2_ROOT"
export PSEUDO_GT_DIR="${PSEUDO_DIR:-$IRIS_RUNS/pseudo_gt/official_train/calibrated_pseudo_depth}"
export METRIC_NORM_JSON="$METRIC_NORM"
export INIT_UNET_FROM="$INIT_CKPT"
export LAMBDA_METRIC
export LAMBDA_DENSE="${LAMBDA_DENSE:-1.0}"
export LAMBDA_RECON="${LAMBDA_RECON:-1.0}"
export LEARNING_RATE="$LR"
export MAX_TRAIN_STEPS="${STEPS:-4000}"
export CKPT_STEP="${CKPT_STEP:-500}"
export MAIN_PORT="$MAIN_PROCESS_PORT"
export METRIC_LOG_EVERY="${METRIC_LOG_EVERY:-200}"
export DATALOADER_WORKERS

[[ -d "$PSEUDO_GT_DIR" ]] || {
  echo "!! 找不到伪 GT 目录: $PSEUDO_GT_DIR"
  echo "!! $IRIS_RUNS/pseudo_gt/ 下现有的是："
  ls -1 "$IRIS_RUNS/pseudo_gt/" 2>/dev/null | sed 's/^/!!   /'
  exit 1
}
mkdir -p "$OUTPUT_DIR"

echo "--- 本次配置 ---"
echo "起点权重  = $INIT_CKPT"
echo "常数      = $METRIC_NORM"
echo "manifest  = $MS2_MANIFEST"
echo "pseudo gt = $PSEUDO_GT_DIR"
echo "loss      = ${LAMBDA_METRIC} * L_metric + ${LAMBDA_DENSE} * L_dense + ${LAMBDA_RECON} * L_I"
echo "lr        = $LEARNING_RATE   steps = $MAX_TRAIN_STEPS (ckpt every $CKPT_STEP)"
echo "output    = $OUTPUT_DIR"
echo "----------------"

cd "$IRIS_REPO/lotus"
git -C "$IRIS_REPO" log --oneline -1
bash train_scripts/train_iris_ms2_metric_depth.sh
