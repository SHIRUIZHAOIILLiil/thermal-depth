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
# 六线对比的训练作业。route 通过环境变量传入。
#
# 冒烟（5 个 update 就退出，验证路径/显存/数据加载）：
#   cd $SCRATCH/logs
#   ROUTE=b_thermal_unet SMOKE=1 sbatch -J smoke_b --time=00:30:00 \
#     --export=ALL,ROUTE,SMOKE $HOME/Iris/slurm/route_suite.sbatch
#
# 全量 20 epoch：
#   ROUTE=b_thermal_unet sbatch -J route_b --export=ALL,ROUTE \
#     $HOME/Iris/slurm/route_suite.sbatch
#
# 六线一次全投（QoS 允许 15 卡并发、作业数不限，六个能同时排队）：
#   for r in a_rgb_unet b_thermal_unet c1_vae_adapter c2_vae_adapter_unet \
#            d1_anythermal_adapter d2_anythermal_adapter_unet; do
#     ROUTE=$r sbatch -J "route_${r%%_*}" --export=ALL,ROUTE \
#       $HOME/Iris/slurm/route_suite.sbatch
#   done
#
# 续跑：同一个 ROUTE 重投即可，脚本会自动接上 latest.pt。
# gpu 分区上限 48h，这里报 24h —— backfill 更容易插队，靠 resume 分段跑完。

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

: "${ROUTE:?必须指定 ROUTE，例如 ROUTE=b_thermal_unet}"
EPOCHS="${EPOCHS:-20}"
OUT="$IRIS_RUNS/route_suite/${RUN_TAG:-$ROUTE}"
mkdir -p "$OUT"

cd "$IRIS_REPO"

ARGS=(
  --route "$ROUTE"
  --output-dir "$OUT"
  --ms2-root "$IRIS_MS2_ROOT"
)

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "=== 冒烟模式：5 个 update，跳过验证 ==="
  ARGS+=(--smoke-updates 5 --skip-val)
else
  ARGS+=(--epochs "$EPOCHS" --snapshot-epochs "${SNAPSHOT_EPOCHS:-1,2,5,10,15}")
  # 断点续跑：48h 墙钟不够跑完 20 epoch 时，重投同一个 ROUTE 自动接上
  if [[ -f "$OUT/latest.pt" ]]; then
    echo "=== 发现 $OUT/latest.pt，续跑 ==="
    ARGS+=(--resume "$OUT/latest.pt")
  fi
fi

echo "=== python tools/train_route_suite.py ${ARGS[*]} ==="
python tools/train_route_suite.py "${ARGS[@]}"
rc=$?

echo "=== 退出码 $rc ==="
ls -la "$OUT"
exit $rc
