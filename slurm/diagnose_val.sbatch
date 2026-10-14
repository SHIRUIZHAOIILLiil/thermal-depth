#!/bin/bash
#SBATCH --job-name=diag_val
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# ⛔ 诊断，只在 val 上跑，不进汇报和论文。回答「我们比 NeWCRF 输在哪」：
# 远距离 / 深度边界 / 少数坏帧（tools/diagnose_error_sources.py）。
# 放在 val 而不是 test：这些答案会决定方法怎么改，在 test 上查就等于拿 test 调方法。
#
#   1. 我们的模型在 val 全帧推理并存逐像素预测（种子、prompt、拉伸与选点评估相同）；
#   2. NeWCRF 公开权重在 val 序列上推理并存逐像素预测；
#   3. 同一批像素、各自官方对齐后，做三项拆解。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   CKPT   我们模型的 converted 权重（必填）
#   TAG    输出子目录名（默认取 CKPT 的臂名 + 步数）
#   KEEP_RAW=1  保留逐像素预测（默认分析完删掉）
#
#   CKPT=.../converted/step16000_weights.pt sbatch --export=ALL,CKPT ~/Iris/slurm/diagnose_val.sbatch

F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
set +e

: "${CKPT:?必须指定 CKPT（converted 权重）}"
[[ -f "$CKPT" ]] || { echo "!! 找不到 $CKPT"; exit 1; }
run=$(basename "$(dirname "$(dirname "$CKPT")")"); step=$(basename "$CKPT" _weights.pt)
TAG="${TAG:-${run}_${step}}"
BASE="${BASE:-/mnt/scratch/dwdh0363/baselines}"
OUT="$IRIS_RUNS/analysis/diag_val/$TAG"
VAL="$IRIS_MANIFEST_DIR/ms2_val_official3_thermalcap_20260821.jsonl"
TRAIN="$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl"
mkdir -p "$OUT"
cd "$IRIS_REPO"
git log --oneline -1
[[ -f "$IRIS_MS2_ROOT/val_list.txt" ]] || { echo "!! $IRIS_MS2_ROOT/val_list.txt 不存在，NeWCRF 无法按序列取 val 帧"; exit 1; }
echo "ckpt = $CKPT"; echo "out  = $OUT"

echo "=== 1. 我们的模型：val 全帧，存逐像素预测 ==="
if [[ "$(ls "$OUT/ours/raw_predictions" 2>/dev/null | wc -l)" -eq 0 ]]; then
  IRIS_THERMAL_STRETCH=percentile IRIS_TEXT_PADDING=legacy python tools/train_route_suite.py \
    --route b_thermal_unet --output-dir "$OUT/ours" --ms2-root "$IRIS_MS2_ROOT" \
    --train-manifest "$TRAIN" --val-manifest "$VAL" --eval-checkpoint "$CKPT" \
    --val-caption-mode correct --align-mode ssi_log --num-inference-steps 1 \
    --condition-latent mode --val-stride 1 --save-raw-pred --local-files-only \
    || { echo "!! 我们的模型推理失败"; exit 1; }
fi
python -c "import json,glob;d=json.load(open(glob.glob('$OUT/ours/eval_*log_space.json')[0]));print('ours val (自检，应与同步数的 stride1 val 评估一致):',d['abs_rel'],d['rmse'],d['a1'])"

echo "=== 2. NeWCRF：val 序列全帧，存逐像素预测 ==="
if [[ "$(ls "$OUT/newcrf/raw_predictions/newcrf" 2>/dev/null | wc -l)" -eq 0 ]]; then
  python tools/run_ms2_supdepth_baselines.py --models newcrf --bmsd-root "$BASE/SupDepth4Thermal" \
    --ckpt-dir "$BASE/supdepth4thermal" --ms2-root "$IRIS_MS2_ROOT" --test-env val \
    --sample-step 1 --save-all-preds --output-dir "$OUT/newcrf" 2>&1 | tail -n 12 \
    || { echo "!! NeWCRF 推理失败"; exit 1; }
fi

echo "=== 3. 拆解：远距离 / 边界 / 坏帧 ==="
python tools/diagnose_error_sources.py --manifest "$VAL" --data-root "$IRIS_MS2_ROOT" \
  --model "ours=$OUT/ours/raw_predictions:ssi_log" \
  --model "newcrf=$OUT/newcrf/raw_predictions/newcrf:ssi" \
  --out "$OUT/analysis" | tee "$OUT/analysis_stdout.txt"
status=${PIPESTATUS[0]}

if [[ "$status" -eq 0 && "${KEEP_RAW:-0}" != "1" ]]; then
  rm -rf "$OUT/ours/raw_predictions" "$OUT/newcrf/raw_predictions"
  echo "[清理] 逐像素预测已删（KEEP_RAW=1 可保留）"
fi
echo "[完成] $OUT  (exit $status)"
exit "$status"
