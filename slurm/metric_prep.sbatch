#!/bin/bash
#SBATCH --job-name=metric_prep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 度量适配的两个前置产物，一个作业跑完。两个都只读 TRAIN，写成 JSON 存档。
#
#   1. tools/fit_metric_norm.py     -> q_lo / q_hi（全局逆深度常数）
#   2. --fit-global-affine          -> (a, b)（原始 checkpoint 的训练集标定基线）
#
# ⛔ 两个产物都**只能**在官方 train 的 8 条序列上拟合。脚本对每一行做序列白名单
#    检查（--split-list / --fit-affine-split-list），清单里出现 val 或 test 的序列
#    就直接退出。这不是形式检查：常数一旦带上 val/test 的信息，就被烤进模型的单位
#    里，后面任何一步都看不出来。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   TRAIN_MANIFEST  必填。官方 8 序列 train 清单
#                   （热像 caption 臂 = ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl，
#                     ⚠️ 名字里的 untrimmed 是错的，认 75,688 这个行数）
#   CKPT            必填。要标定的 checkpoint（.pt，route_suite 格式）
#   OUT_DIR         默认 $IRIS_RUNS/metric_adapt
#   NORM_STRIDE     常数拟合的抽帧步长，默认 1（全部）。改了会写进产物
#   AFFINE_STRIDE   仿射拟合的抽帧步长，默认 20
#   CAPTION_MODE    仿射拟合时喂的 prompt，默认 correct
#                   ⚠️ 必须与这条臂**被评估时**用的 prompt 一致 —— (a,b) 是在某个
#                   文本条件下的预测上拟合的，换条件用就是拿另一套预测的标定
#
# ── 用法 ────────────────────────────────────────────────────────────
#   TRAIN_MANIFEST=$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl \
#   CKPT=$IRIS_RUNS/iris_ms2/iris_ms2_full8_thermalcap/converted/step20000_weights.pt \
#     sbatch --export=ALL,TRAIN_MANIFEST,CKPT ~/Iris/slurm/metric_prep.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

: "${TRAIN_MANIFEST:?必须指定 TRAIN_MANIFEST（官方 8 序列 train 清单）}"
: "${CKPT:?必须指定 CKPT（要标定的 checkpoint）}"
OUT_DIR="${OUT_DIR:-$IRIS_RUNS/metric_adapt}"
NORM_STRIDE="${NORM_STRIDE:-1}"
AFFINE_STRIDE="${AFFINE_STRIDE:-20}"
CAPTION_MODE="${CAPTION_MODE:-correct}"
SPLIT_LIST="${SPLIT_LIST:-$IRIS_MS2_ROOT/train_list.txt}"

[[ -f "$TRAIN_MANIFEST" ]] || { echo "!! 找不到 $TRAIN_MANIFEST"; exit 1; }
[[ -f "$CKPT" ]]           || { echo "!! 找不到 $CKPT"; exit 1; }
[[ -f "$SPLIT_LIST" ]]     || { echo "!! 找不到 $SPLIT_LIST（官方 train 序列表）"; exit 1; }

mkdir -p "$OUT_DIR"
cd "$IRIS_REPO"
git log --oneline -1

NORM_JSON="$OUT_DIR/metric_norm_train.json"
AFFINE_JSON="$OUT_DIR/global_affine_train.json"

echo "=== 配置 ==="
echo "train manifest = $TRAIN_MANIFEST  ($(wc -l < "$TRAIN_MANIFEST") 帧)"
echo "checkpoint     = $CKPT"
echo "序列白名单     = $SPLIT_LIST"
echo "caption mode   = $CAPTION_MODE  (须与该臂被评估时一致)"
echo "输出           = $NORM_JSON"
echo "                 $AFFINE_JSON"
echo "============"

# --- 1. 全局逆深度常数 -----------------------------------------------------
if [[ -f "$NORM_JSON" ]]; then
  echo "[skip] $NORM_JSON 已存在"
else
  python tools/fit_metric_norm.py \
    --manifest "$TRAIN_MANIFEST" \
    --ms2-root "$IRIS_MS2_ROOT" \
    --split-list "$SPLIT_LIST" \
    --output "$NORM_JSON" \
    --frame-stride "$NORM_STRIDE" \
    --notes "official 8 training sequences, thermal-view lidar, official validity range"
fi

# --- 2. 训练集全局仿射基线 -------------------------------------------------
if [[ -f "$AFFINE_JSON" ]]; then
  echo "[skip] $AFFINE_JSON 已存在"
else
  python tools/train_route_suite.py \
    --route b_thermal_unet \
    --output-dir "$OUT_DIR/affine_fit" \
    --ms2-root "$IRIS_MS2_ROOT" \
    --train-manifest "$TRAIN_MANIFEST" \
    --val-manifest "$TRAIN_MANIFEST" \
    --val-stride 1 \
    --eval-checkpoint "$CKPT" \
    --fit-global-affine "$AFFINE_JSON" \
    --fit-affine-split-list "$SPLIT_LIST" \
    --fit-affine-stride "$AFFINE_STRIDE" \
    --val-caption-mode "$CAPTION_MODE" \
    --num-inference-steps 1 \
    --condition-latent mode \
    --local-files-only
fi

echo "=== 产物 ==="
python - "$NORM_JSON" "$AFFINE_JSON" <<'PY'
import json, sys
for path in sys.argv[1:]:
    d = json.load(open(path, encoding="utf-8"))
    print(f"--- {path}")
    for k in ("q_lo", "q_hi", "a", "b", "source_split", "valid_pixels", "frames",
              "frames_used", "sequences", "quantile_lo", "quantile_hi", "caption_mode"):
        if k in d:
            print(f"    {k} = {d[k]}")
PY
echo "下一步：把 metric_norm_train.json 传给 metric_adapt.sbatch 的 METRIC_NORM_JSON"
