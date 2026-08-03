#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00  # 全量 5810 帧实测 8.3 分钟
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 用 --eval-checkpoint 对某个 checkpoint 跑全量评估（--val-stride 1）。
# 全量评估比训练更吃卡（5,810 帧 × 多个 checkpoint），所以单独成作业、按需投。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   ROUTE         必填，必须与该 checkpoint 训练时的 route 一致
#   CKPT          必填，checkpoint 路径
#   TAG           必填，输出目录名（自己带上区分度，如 b_e05_on_val）
#   VAL_MANIFEST  评测集；不设则用脚本自带默认（11-23-45 val）
#   CAPTION_MODE  empty | correct | shuffled，默认 empty
#                 —— 必须与该 checkpoint 训练时的输入模式一致，否则是在它没训过的
#                    模式下打分
#   VAL_STRIDE    默认 1（全量）
#   SHUFFLE_COND  证伪对照：all | anythermal。不设＝关闭
#
# ── 用法 ────────────────────────────────────────────────────────────
#   跨机器标定（§12.6，投六线之前必做）：
#     ROUTE=b_thermal_unet TAG=calib_b_e05 \
#     CKPT=$SCRATCH/checkpoints/calibration/epoch05_weights.pt \
#       sbatch -J calib_b --time=01:00:00 \
#       --export=ALL,ROUTE,TAG,CKPT ~/Iris/slurm/eval.sbatch
#     # 与本地全量 0.0775 比：差 <0.0005 可并表，否则集群结果自成一组
#
#   独立 test 集（任务 2）：
#     ROUTE=b_thermal_unet TAG=b_e03cap_on_test CAPTION_MODE=correct \
#     CKPT=$SCRATCH/runs/route_suite/b_thermal_unet_caption/epoch03_weights.pt \
#     VAL_MANIFEST=$SCRATCH/manifests/sequence_level_internvl3_8b/ms2_test_16-08-46_rgb_depth_v1_clip75_20260728.jsonl \
#       sbatch -J eval_b_test --export=ALL,ROUTE,TAG,CKPT,CAPTION_MODE,VAL_MANIFEST \
#       ~/Iris/slurm/eval.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

: "${ROUTE:?必须指定 ROUTE}"
: "${CKPT:?必须指定 CKPT}"
: "${TAG:?必须指定 TAG（输出目录名）}"

[[ -f "$CKPT" ]] || { echo "!! 找不到 checkpoint: $CKPT"; exit 1; }

OUT="$IRIS_RUNS/eval/$TAG"
mkdir -p "$OUT"

cd "$IRIS_REPO"

ARGS=(
  --route "$ROUTE"
  --output-dir "$OUT"
  --ms2-root "$IRIS_MS2_ROOT"
  # 评估不读训练集，但脚本仍会哈希校验它并写进 config 记录；
  # 与训练作业保持同一份，免得记录里留下旧 manifest
  --train-manifest "${TRAIN_MANIFEST:-$IRIS_MANIFEST_DIR/ms2_train_day2seq_clip75_20260728.jsonl}"
  --eval-checkpoint "$CKPT"
  --val-stride "${VAL_STRIDE:-1}"
  --val-caption-mode "${CAPTION_MODE:-empty}"
)
[[ -n "${VAL_MANIFEST:-}" ]] && ARGS+=(--val-manifest "$VAL_MANIFEST")
[[ "${LOCAL_FILES_ONLY:-1}" == "1" ]] && ARGS+=(--local-files-only)
# 任务 3 的可视化和九分层都吃 raw_predictions/<id>.npy。
# 每帧约 0.5 MB + 1 个 inode，所以默认关，要用时显式 SAVE_RAW=1。
[[ "${SAVE_RAW:-0}" == "1" ]] && ARGS+=(--save-raw-pred)

# 证伪对照：把 condition 换成随机 donor 帧的。指标几乎不动 = 该分支没在起作用。
# all = 换整个 condition；anythermal = 只换 AnyThermal 特征、保留正确的热像张量。
[[ -n "${SHUFFLE_COND:-}" ]] && ARGS+=(--shuffle-condition "$SHUFFLE_COND")

echo "=== 评估 $CKPT ==="
echo "=== python tools/train_route_suite.py ${ARGS[*]} ==="
python tools/train_route_suite.py "${ARGS[@]}"
rc=$?

echo "=== 退出码 $rc ==="
ls -la "$OUT"
exit $rc
