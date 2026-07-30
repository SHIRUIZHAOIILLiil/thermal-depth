#!/usr/bin/env bash
# Does the caption effect survive when the TRAINING supervision is thinned?
#
# The dense-GT 2x2 on RGBDT500 gave a caption effect of +0.0073* (win 70.4%),
# the same magnitude Iris reports (Lotus-G + Text, five test sets, mean 0.0072).
# On MS2 the same recipe gives ~0. The one variable never aligned between the two
# is TRAINING supervision density: Iris always learns its text pathway on dense
# synthetic GT and only *reports* on sparse test sets (KITTI included), while we
# always train on sparse LiDAR.
#
# This queue evaluates the arms trained with --gt-sparsify. Evaluation is left
# DENSE and identical to the existing 2x2, so the only changed variable is what
# the model was taught on -- not what it is measured against.
#
#   ms2_lidar arm : real LiDAR blindness (sky and far field go dark)
#   random arm    : same pixel COUNT per frame, uniformly scattered
#                   -> separates "fewer pixels" from "blind where text helps"
#
# Usage: nohup bash tools/run_rgbdt500_sparsity_eval_queue.sh > outputs/lotus_line_v2/rgbdt500_sparsity_eval.log 2>&1 &

set -u
cd "$(dirname "$0")/.."

M=/mnt/e/dataset/RGBDT500/clean_test/rgbdt500_test_manifest_iris_prose.jsonl
ROOT=/mnt/e/dataset/RGBDT500/clean_test
OUT=outputs/lotus_line_v2
THR=tools/run_ms2_lotus_thermal_vae_official.py
SCALE="--depth-scale 1000 --max-depth 20"
# RGBDT500 needs --min-depth 0.1: the default 1e-3 treats millimetre sensor
# noise as GT and inflated all four cells to 0.82 (correct value 0.36).
MIND="--min-depth 0.1"

if [ ! -f "$M" ]; then echo "FATAL: test manifest missing: $M"; exit 1; fi

declare -a NAMES DIRS CKPTS MODES
add() { NAMES+=("$1"); DIRS+=("$2"); CKPTS+=("$3"); MODES+=("$4"); }

# structured arm: full 2x2 (cell 3 also reports the dependency cost)
S=$OUT/rgbdt500_sparse_ms2_empty/arm6_end.pt
SC=$OUT/rgbdt500_sparse_ms2_caption/arm6_end.pt
add "ms2_1_emptytrain_x_empty"   "rgbdt500_sparse_ms2_emptytrain_empty"   "$S"  "empty"
add "ms2_2_emptytrain_x_correct" "rgbdt500_sparse_ms2_emptytrain_correct" "$S"  "correct"
add "ms2_3_capttrain_x_empty"    "rgbdt500_sparse_ms2_capttrain_empty"    "$SC" "empty"
add "ms2_4_capttrain_x_correct"  "rgbdt500_sparse_ms2_capttrain_correct"  "$SC" "correct"

# random control: only the two cells the total effect needs
R=$OUT/rgbdt500_sparse_rand_empty/arm6_end.pt
RC=$OUT/rgbdt500_sparse_rand_caption/arm6_end.pt
add "rand_1_emptytrain_x_empty"  "rgbdt500_sparse_rand_emptytrain_empty"  "$R"  "empty"
add "rand_4_capttrain_x_correct" "rgbdt500_sparse_rand_capttrain_correct" "$RC" "correct"

declare -a RESULTS
for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"; dir="$OUT/${DIRS[$i]}"; ckpt="${CKPTS[$i]}"; mode="${MODES[$i]}"
    if [ ! -f "$ckpt" ]; then echo "=== $name: checkpoint missing ($ckpt)"; RESULTS+=("$name NO_CKPT"); continue; fi

    if [ ! -f "$dir/official_run_metadata.json" ]; then
        echo "=== [$((i+1))/${#NAMES[@]}] $name: inference $(date +%H:%M:%S)"
        if ! python $THR --manifest "$M" --ms2-root $ROOT --output-dir "$dir" --max-samples 0 \
             --condition-posterior mode --unet-checkpoint "$ckpt" --caption-mode "$mode" \
             $SCALE --save-raw-pred > "$dir.log" 2>&1; then
            echo "=== $name FAILED"; tail -5 "$dir.log"; RESULTS+=("$name FAILED"); continue
        fi
    fi

    official="outputs/ms2_official/${DIRS[$i]}_ssi_disparity"
    if [ ! -f "$official/metrics/summary.json" ]; then
        python tools/run_official_ms2_evaluation.py --manifest "$dir/selected_manifest.jsonl" \
            --data-root $ROOT --prediction-dir "$dir/raw_predictions" --route thermal-unet \
            --output-dir "$official" $SCALE $MIND > "$official.log" 2>&1
    fi
    line=$(python - "$official/metrics/summary.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
s = d["image_wise"]["statistics"]
print(f"absrel={s['abs_rel']['mean']:.4f} d1={s['a1']['mean']:.4f} n={d['sample_count']}")
EOF
)
    echo "=== $name: $line"
    RESULTS+=("$name $line")
done

echo
echo "================ RGBDT500 training-supervision sparsity ================"
for r in "${RESULTS[@]}"; do echo "  $r"; done
cat <<'EOF'

对照锚点（稠密训练，已有）: cell1 0.3616 | cell4 0.3543 | 总效果 +0.0073* 胜率 70.4%
判决规则（事先立好）:
  结构化臂总效果 |x| <= 0.002 且 CI 含零 -> 训练监督稠密性就是那个开关
  结构化臂总效果 >= 0.005 且显著        -> 稠密性不是开关，嫌疑转向合成数据的精确性
  中间区间                              -> 只报衰减比例，不加戏
  随机臂仍 ~0.007 而结构化臂塌          -> 是盲区位置（天空/远端），不是像素数量
逐帧配对: python tools/compare_route_evals.py <cell1>/metrics/per_image.csv <cell4>/metrics/per_image.csv
EOF
