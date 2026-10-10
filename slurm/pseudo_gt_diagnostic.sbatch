#!/bin/bash
#SBATCH --job-name=pgt_diag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# ⛔ 诊断，不是基准结果，不进论文表格。
#
# 在官方测试子集上，用和训练目标同一种构造的稠密参照（激光处用激光，其余用按本帧
# 激光校准的 AnyThermal）给四个模型打分：我们加米制 loss 的主模型、不加的那条、
# NeWCRF、AnyThermal。分三块：lidar（只激光像素，必须复现已有表格 = 自检）、
# filled（只补出来的像素，那里的参照就是 AnyThermal）、dense（两者合起来）。
# 参照是用测试激光校准出来的；filled 区上 AnyThermal 是在和自己比。读结果时带着这两点。
#
# 前置（登录节点，一次）：
#   SupDepth4Thermal 代码与 MS2_MD_NeWCRF_THR_ckpt.ckpt 在 $BASE（见下），
#   且 run_ms2_supdepth_baselines.py --build-only --models newcrf 已通过。
#
#   sbatch ~/Iris/slurm/pseudo_gt_diagnostic.sbatch

F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
set +e

S=/mnt/scratch/sc23sz/runs
BASE="${BASE:-/mnt/scratch/dwdh0363/baselines}"
OUT="${OUT:-$IRIS_RUNS/analysis/pseudo_gt_diagnostic}"
MAN="$OUT/manifests"
mkdir -p "$OUT"
cd "$IRIS_REPO"
git log --oneline -1

echo "=== 1. 官方子集清单 ==="
python tools/make_official_subset_manifest.py --ms2-root "$IRIS_MS2_ROOT" \
  --manifest-dir "$IRIS_MANIFEST_DIR" --out "$MAN" || { echo "!! 清单失败，停"; exit 1; }

echo; echo "=== 2. 我们的两个模型（补空行清单，保住噪声种子）==="
for spec in "img|$S/iris_ms2/sd2_cap_pct_logtn_img_s43/converted/step16000_weights.pt" \
            "cap|$S/iris_ms2/sd2_cap_pct_logtn_s43/converted/step18000_weights.pt"; do
  k=${spec%%|*}; ckpt=${spec#*|}
  for m in day3_common night3 rainy3; do
    dest="$OUT/ours_${k}_$m"
    if [[ "$(ls "$dest/raw_predictions" 2>/dev/null | wc -l)" -gt 0 ]]; then echo "[skip] $dest"; continue; fi
    man="$MAN/ms2_test_${m}_thermalcap_20260821_official_padded.jsonl"
    IRIS_THERMAL_STRETCH=percentile python tools/train_route_suite.py \
      --route b_thermal_unet --output-dir "$dest" --ms2-root "$IRIS_MS2_ROOT" \
      --train-manifest "$man" --val-manifest "$man" --eval-checkpoint "$ckpt" \
      --val-caption-mode correct --align-mode ssi_log --num-inference-steps 1 \
      --condition-latent mode --val-stride 1 --save-raw-pred --local-files-only \
      || echo "!! ours $k $m 失败"
  done
done

echo; echo "=== 3. NeWCRF（重新下载的公开权重）==="
for pair in "day3_common test_day" "night3 test_night" "rainy3 test_rain"; do
  set -- $pair
  dest="$OUT/newcrf_$2"
  if [[ "$(ls "$dest/raw_predictions/newcrf" 2>/dev/null | wc -l)" -gt 0 ]]; then echo "[skip] $dest"; continue; fi
  ids=$(python -c "import json,sys;print(' '.join(json.loads(l)['id'] for l in open(sys.argv[1],encoding='utf-8') if l.strip()))" \
        "$MAN/ms2_test_$1_thermalcap_20260821_official.jsonl")
  python tools/run_ms2_supdepth_baselines.py --models newcrf --bmsd-root "$BASE/SupDepth4Thermal" \
    --ckpt-dir "$BASE/supdepth4thermal" --ms2-root "$IRIS_MS2_ROOT" --test-env "$2" \
    --output-dir "$dest" --save-pred-ids $ids 2>&1 | grep -v "saved prediction" | tail -n 15 \
    || echo "!! newcrf $2 失败"
done

echo; echo "=== 4. 打分（诊断）==="
python tools/score_on_pseudo_gt.py --ms2-root "$IRIS_MS2_ROOT" --manifest-dir "$MAN" \
  --anythermal-root "$S/anythermal" \
  --model "ours_img=$OUT/ours_img_{m}/raw_predictions:ssi_log" \
  --model "ours_cap=$OUT/ours_cap_{m}/raw_predictions:ssi_log" \
  --model "newcrf=$OUT/newcrf_{env}/raw_predictions/newcrf:ssi" \
  --out "$OUT/score" | tee "$OUT/score_summary.txt"

echo; echo "[完成] $OUT/score_summary.txt"
echo "自检：lidar 那几行应复现主表 / 补充表的官方子集数字（ours_img 6.88/2.733/95.17 day 等）"
