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
# 把 Depth Anything V2 适配到 MS2 热像，**用它自己论文里的目标**。
#
# 2026-09-29：这条臂重跑，因为第一版只训了 `L_ssi`，而 DA2 的相对深度模型训的是
# `L_ssi + L_gm`，论文原话「The weight ratio of L_ssi and L_gm is set as 1:2」——
# 梯度匹配项的权重是数据项的两倍。只用 SSI 等于让它在自己目标三分之一的权重下
# 迎战，输了不算数。⚠️ 旧那一版是 GM_WEIGHT=0，逐位可复现，留作消融。
#
# 他们仓库只开源了**米制**变体的微调代码（SiLog），相对深度的训练代码没发布，
# 所以配方写在我们这边；模型和预处理都是 HuggingFace 上的原版，没有改动。
#
# ⚠️ 没有 resume：每 2000 步 save_pretrained，但被时限杀掉接不上，只能重来。
#    墙钟给够（第一版 20000 步在 L40S 上跑得完）。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   RUN_TAG    输出目录名，默认 v2_gm
#   GM_WEIGHT  梯度匹配项的权重，默认 2.0（＝他们的 1:2）。0 ＝ 退回第一版
#   STEPS      默认 20000，与第一版一致
#   SEED       默认 42，与第一版一致
#
#   sbatch --export=ALL ~/Iris/slurm/da2_train.sbatch
#   GM_WEIGHT=0 RUN_TAG=v1_repro sbatch --export=ALL,GM_WEIGHT,RUN_TAG ~/Iris/slurm/da2_train.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

RUN_TAG="${RUN_TAG:-v2_gm}"
GM_WEIGHT="${GM_WEIGHT:-2.0}"
STEPS="${STEPS:-20000}"
SEED="${SEED:-42}"
OUT="$IRIS_RUNS/da2_thermal/$RUN_TAG"

# 与第一版、与我方 log 线**同一份**训练数据和目标。换了这里，重跑就白跑。
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
echo "输出      = $OUT"
echo "目标      = L_ssi + ${GM_WEIGHT} x L_gm   (DA2 论文的比例是 1:2)"
echo "步数      = $STEPS   batch 4   seed $SEED"
echo "学习率    = 编码器 5e-6 / 解码器 5e-5   (照他们米制配方的 1:10)"
echo "manifest  = $MANIFEST"
echo "伪深度    = $PSEUDO"
echo "thermal   = $IRIS_THERMAL_STRETCH"
echo "============"

# 第 0 步的 AbsRel 必须落在零样本那一档（实测 0.153 / 0.165 / 0.172）。
# 这道门禁挡下过两个 bug：没走 DPTImageProcessor，以及诊断参照拿了伪深度而非激光。
python tools/finetune_depth_anything_thermal.py \
  --manifest "$MANIFEST" \
  --ms2-root "$IRIS_MS2_ROOT" \
  --pseudo-dir "$PSEUDO" \
  --out-dir "$OUT" \
  --gm-weight "$GM_WEIGHT" \
  --steps "$STEPS" \
  --seed "$SEED"

echo
echo "[完成] $OUT"
echo "下一步：官方 val 扫 10 个点选点（select_checkpoint.sbatch，MODEL=depth_anything_v2，"
echo "        ALIGN=ssi_disparity），规则不变 —— val AbsRel 最低、并列取更早。"
