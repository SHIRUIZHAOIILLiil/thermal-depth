#!/bin/bash
#SBATCH --job-name=route
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 六线训练作业。所有开关走环境变量，配 --export=ALL,<VAR>,... 传入。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   ROUTE            必填，六选一
#   CAPTION=1        caption 臂：同时给 --caption-mode / --val-caption-mode correct
#                    （只给前者会用空 prompt 给 caption 模型选 best，等于用它没训过的
#                     输入模式挑冠军）
#   SMOKE=1          5 个 update 即退出
#   OVERFIT=1        300 步 / 32 样本自检
#   EPOCHS           默认 20
#   SNAPSHOT_EPOCHS  默认 1,2,3,5,10,15（含 e3 —— 本地 caption 臂最好点落在 e3）
#   TRAIN_MANIFEST   默认 clip75_20260728（≤75 token，逐帧同序同路径）
#   VAL_MANIFEST     不设则用脚本自带默认
#   INIT_FROM        跨 route 搬权重（任务 1b 阶段二用 c1 的 best）
#   FREEZE_ADAPTER=1 冻结 adapter 只训 U-Net，必须配 INIT_FROM
#   LOCAL_FILES_ONLY 默认 1 —— 权重缺失立即报错，而不是并发去写同一个 HF 缓存目录
#   RUN_TAG          覆盖输出目录名
#
# ── 用法 ────────────────────────────────────────────────────────────
#   冒烟：
#     ROUTE=b_thermal_unet SMOKE=1 sbatch -J smoke_b --time=00:30:00 \
#       --export=ALL,ROUTE,SMOKE ~/Iris/slurm/route_suite.sbatch
#
#   全量（--time 按线报，见 docs/AIRE_HPC_MIGRATION_STATUS_20260729.md §12.5，
#   贵的线直接报 48h，别统一 24h）：
#     ROUTE=c1_vae_adapter sbatch -J route_c1 --time=1-00:00:00 \
#       --export=ALL,ROUTE ~/Iris/slurm/route_suite.sbatch
#     ROUTE=a_rgb_unet     sbatch -J route_a  --time=2-00:00:00 \
#       --export=ALL,ROUTE ~/Iris/slurm/route_suite.sbatch
#
#   caption 臂：
#     ROUTE=b_thermal_unet CAPTION=1 sbatch -J route_b_cap --time=2-00:00:00 \
#       --export=ALL,ROUTE,CAPTION ~/Iris/slurm/route_suite.sbatch
#
#   任务 1b 阶段二（冻结 c1 的 adapter，只训 U-Net）：
#     ROUTE=c2_vae_adapter_unet FREEZE_ADAPTER=1 \
#     INIT_FROM=$SCRATCH/checkpoints/c1_stage1/best_weights.pt \
#       sbatch -J route_c2_stage2 --time=2-00:00:00 \
#       --export=ALL,ROUTE,FREEZE_ADAPTER,INIT_FROM ~/Iris/slurm/route_suite.sbatch
#
# 续跑：同一 ROUTE/TAG 重投即自动 --resume latest.pt。
# latest.pt 只在每个 epoch 末尾写，被墙钟砍最多丢一个 epoch。

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

: "${ROUTE:?必须指定 ROUTE，例如 ROUTE=b_thermal_unet}"
EPOCHS="${EPOCHS:-20}"

# caption 臂与 empty 臂必须写到不同目录，否则互相覆盖
TAG="${RUN_TAG:-$ROUTE}"
[[ "${CAPTION:-0}" == "1" && -z "${RUN_TAG:-}" ]] && TAG="${ROUTE}_caption"

# validate_args() 要求：--smoke-updates 的输出目录名必须含 "smoke"，
# --overfit-steps 的必须含 "overfit"。目录名要跟着模式走。
if [[ "${SMOKE:-0}" == "1" ]]; then
  OUT="$IRIS_RUNS/route_suite/smoke_$TAG"
elif [[ "${OVERFIT:-0}" == "1" ]]; then
  OUT="$IRIS_RUNS/route_suite/overfit_$TAG"
else
  OUT="$IRIS_RUNS/route_suite/$TAG"
fi
mkdir -p "$OUT"

cd "$IRIS_REPO"

ARGS=(
  --route "$ROUTE"
  --output-dir "$OUT"
  --ms2-root "$IRIS_MS2_ROOT"
  --train-manifest "${TRAIN_MANIFEST:-$IRIS_MANIFEST_DIR/ms2_train_day2seq_clip75_20260728.jsonl}"
)
[[ -n "${VAL_MANIFEST:-}" ]] && ARGS+=(--val-manifest "$VAL_MANIFEST")

# 预取完成后权重都在 $HF_HOME；缺失就立即失败，而不是六个作业并发重下 4 GB
[[ "${LOCAL_FILES_ONLY:-1}" == "1" ]] && ARGS+=(--local-files-only)

if [[ "${CAPTION:-0}" == "1" ]]; then
  echo "=== caption 臂：train 与 val 都用 correct ==="
  ARGS+=(--caption-mode correct --val-caption-mode correct)
fi

# --init-from 起新跑、--resume 接旧跑，脚本层面互斥。已有 latest.pt 时优先续跑并
# 忽略 INIT_FROM —— 否则一个被墙钟砍掉的 48h 作业，按原命令重投会从头再来。
RESUMING=0
if [[ "${SMOKE:-0}" != "1" && "${OVERFIT:-0}" != "1" && -f "$OUT/latest.pt" ]]; then
  RESUMING=1
fi

if [[ -n "${INIT_FROM:-}" && "$RESUMING" == "0" ]]; then
  [[ -f "$INIT_FROM" ]] || { echo "!! 找不到 INIT_FROM=$INIT_FROM"; exit 1; }
  ARGS+=(--init-from "$INIT_FROM")
elif [[ -n "${INIT_FROM:-}" && "$RESUMING" == "1" ]]; then
  echo "=== 已有 latest.pt，忽略 INIT_FROM 转为续跑 ==="
fi
[[ "${FREEZE_ADAPTER:-0}" == "1" ]] && ARGS+=(--freeze-adapter)

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "=== 冒烟模式：${SMOKE_UPDATES:-5} 个 update，跳过验证 ==="
  ARGS+=(--smoke-updates "${SMOKE_UPDATES:-5}" --skip-val)
elif [[ "${OVERFIT:-0}" == "1" ]]; then
  echo "=== 过拟合自检：${OVERFIT_STEPS:-300} 步 / ${OVERFIT_SAMPLES:-32} 样本 ==="
  ARGS+=(--overfit-steps "${OVERFIT_STEPS:-300}" --overfit-samples "${OVERFIT_SAMPLES:-32}" --skip-val)
else
  ARGS+=(--epochs "$EPOCHS" --snapshot-epochs "${SNAPSHOT_EPOCHS:-1,2,3,5,10,15}")
  if [[ "$RESUMING" == "1" ]]; then
    echo "=== 发现 $OUT/latest.pt，续跑 ==="
    ARGS+=(--resume "$OUT/latest.pt")
  fi
fi

echo "=== python tools/train_route_suite.py ${ARGS[*]} ==="
python tools/train_route_suite.py "${ARGS[@]}"
rc=$?

echo "=== 退出码 $rc ==="

# ── 跑完即清 latest.pt ──────────────────────────────────────────────
# latest.pt 含优化器动量，约 10.4 GB，唯一用途是断点续训。12 条线全投时它占
# 约 125 GB / 1 TB，盘满会让后续作业静默失败（本地踩过：41 个训练攒到 1165 GB）。
# 三道闸（沿用本地清盘纪律）：① 退出码 0 ② 有 summary.json ③ end_weights.pt 幸存。
# 被墙钟砍掉的作业 rc≠0，latest.pt 会保住，续投不受影响。KEEP_LATEST=1 可关闭。
if [[ "$rc" == "0" && "${SMOKE:-0}" != "1" && "${OVERFIT:-0}" != "1" && "${KEEP_LATEST:-0}" != "1" ]]; then
  if [[ -f "$OUT/summary.json" && -f "$OUT/end_weights.pt" && -f "$OUT/latest.pt" ]]; then
    sz=$(du -h "$OUT/latest.pt" | cut -f1)
    rm -f "$OUT/latest.pt"
    echo "=== 清理：已删 latest.pt（$sz）；summary.json 与 end_weights.pt 均在 ==="
  else
    echo "=== 清理：跳过（缺 summary.json / end_weights.pt，或已无 latest.pt）==="
  fi
fi

du -sh "$OUT" 2>/dev/null
ls -la "$OUT"
lfs quota -h -u "$USER" /scratch
exit $rc
