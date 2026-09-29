#!/bin/bash
#SBATCH --job-name=da2train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 把 Depth Anything V2 适配到 MS2 热像，**用它自己的训练配方**。
#
# 数据是共享的（官方 8 序列划分、补全伪深度 + 激光覆写、同一条选点规则），
# 目标函数和训练过程是他们的。他们只开源了米制微调代码，所以相对深度的配方
# 按他们发表的东西拼起来，每一项的出处写在 tools/finetune_depth_anything_thermal.py
# 的文件头里：
#
#   L_ssi（中位数/MAD 归一化）+ 2 × L_gm      DA2 论文 + DAv1 论文
#   伪标签样本忽略损失最大的 10%              DA2 论文
#   短边 518 + 随机裁 518×518 + 随机翻转      他们的 metric_depth 代码
#   AdamW wd 0.01，编码器 5e-6 / 解码器 ×10    他们的 metric_depth 代码
#   poly 衰减 (1-it/total)^0.9，不裁梯度       他们的 metric_depth 代码
#
# ⛔ 2026-09-29 之前那一版（v1）训的是**我方**的损失：最小二乘 SSI-L1、无 GM、
#    整帧、不翻转、常数学习率、梯度裁到 1.0。那不是 DA2 的配方，报它等于和一个
#    被换了目标函数的对手比。LEGACY=1 逐位复现它，留作消融。
#
# ⚠️ 没有 resume：每 2000 步 save_pretrained，被时限杀掉接不上，只能重来。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   RUN_TAG   输出目录名，默认 v2_recipe（LEGACY=1 时默认 v1_repro）
#   LEGACY    1 = 复现 v1（我方损失）；默认 0 = DA2 自己的配方
#   STEPS     默认 20000，与 v1 一致
#   SEED      默认 42，与 v1 一致
#   SMOKE     1 = 只跑 40 步，最后报稳态 s/step 和 STEPS 步的预计小时数。
#             ⚠️ 正式跑之前先冒烟定墙钟：v2 第一次投的 10 小时是拍的，而实际
#             大约 5 s/step，两万步要一天多 —— 就算没崩也会超时。
#
#   sbatch --export=ALL ~/Iris/slurm/da2_train.sbatch
#   LEGACY=1 sbatch --export=ALL,LEGACY ~/Iris/slurm/da2_train.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

LEGACY="${LEGACY:-0}"
if [[ "$LEGACY" == "1" ]]; then
  RUN_TAG="${RUN_TAG:-v1_repro}"
  RECIPE=(--ssi-norm lstsq --gm-weight 0 --trim-top 0 --crop 0 --no-flip
          --lr-schedule constant --grad-clip 1.0)
  RECIPE_NAME="v1（我方损失，复现用）"
else
  RUN_TAG="${RUN_TAG:-v2_recipe}"
  RECIPE=()      # 脚本默认值就是 DA2 的配方
  RECIPE_NAME="DA2 自己的配方"
fi
STEPS="${STEPS:-20000}"
SEED="${SEED:-42}"
SMOKE="${SMOKE:-0}"
[[ "$SMOKE" == "1" ]] && RUN_TAG="${RUN_TAG}_smoke"
OUT="$IRIS_RUNS/da2_thermal/$RUN_TAG"

# 与 v1、与我方 log 线**同一份**训练数据和目标。换了这里，重跑就白跑。
MANIFEST="${MANIFEST:-$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl}"
PSEUDO="${PSEUDO:-$IRIS_RUNS/pseudo_gt/official_train/calibrated_pseudo_depth}"

[[ -f "$MANIFEST" ]] || { echo "!! 找不到 manifest $MANIFEST"; exit 1; }
[[ -d "$PSEUDO" ]]   || { echo "!! 找不到伪深度目录 $PSEUDO"; exit 1; }
if [[ -d "$OUT" ]] && compgen -G "$OUT/step*" >/dev/null; then
  echo "!! $OUT 下已有 checkpoint。没有 resume 逻辑，接着写会把两次训练混在一起。"
  echo "!! 换 RUN_TAG，或先把旧的移走。"
  exit 1
fi

export IRIS_THERMAL_STRETCH="${IRIS_THERMAL_STRETCH:-percentile}"

mkdir -p "$OUT"
cd "$IRIS_REPO"
git log --oneline -1
echo "=== 配置 ==="
echo "配方      = $RECIPE_NAME"
echo "输出      = $OUT"
echo "步数      = $STEPS   batch 4   seed $SEED"
echo "manifest  = $MANIFEST"
echo "伪深度    = $PSEUDO"
echo "thermal   = $IRIS_THERMAL_STRETCH"
echo "============"

# 第 0 步的 AbsRel 必须落在零样本那一档（实测 0.153 / 0.165 / 0.172）。
# 这道门禁挡下过两个 bug：没走 DPTImageProcessor，以及诊断参照拿了伪深度而非激光。
# ⚠️ 裁剪模式下第 0 步量的是一个 518×518 的随机块，不是整帧，会偏离 0.153 一些；
#    门禁带宽 [0.08, 0.30] 足够容下这个偏离，落在带外仍然说明管线有问题。
python tools/finetune_depth_anything_thermal.py \
  --manifest "$MANIFEST" \
  --ms2-root "$IRIS_MS2_ROOT" \
  --pseudo-dir "$PSEUDO" \
  --out-dir "$OUT" \
  --steps "$STEPS" \
  --seed "$SEED" \
  $([[ "$SMOKE" == "1" ]] && echo --smoke) \
  "${RECIPE[@]}"

echo
echo "[完成] $OUT"
echo "下一步：官方 val 扫 10 个点选点（select_checkpoint.sbatch），规则不变 ——"
echo "        val AbsRel 最低、并列取更早。推理仍走整帧 + DPTImageProcessor，"
echo "        那是他们自己的推理约定（训练裁块、推理整帧）。"
