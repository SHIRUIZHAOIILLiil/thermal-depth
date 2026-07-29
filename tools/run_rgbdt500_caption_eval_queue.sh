#!/usr/bin/env bash
# RGBDT500 caption ablation: the 2x2 (train arm x inference mode) on the
# held-out test split, scored with the SAME official protocol as MS2 so the
# caption EFFECT SIZES are comparable across the two datasets.
#
#                     inference: empty        inference: correct
#   train empty       cell 1 (baseline)       cell 2 (injection value)
#   train caption     cell 3 (dependency)     cell 4 (train+use total)
#
# Cell 3 is the decisive one: on MS2 the caption-trained f-line got WORSE when
# the caption was withheld (-0.0041), i.e. caption training bought dependency,
# not value. This queue reproduces that test on dense GT.
#
# Absolute numbers from this dataset are NOT a performance claim (thermal<->depth
# registration is only approximate); only the paired caption deltas are.
#
# Usage: nohup bash tools/run_rgbdt500_caption_eval_queue.sh > outputs/lotus_line_v2/rgbdt500_eval_queue.log 2>&1 &

set -u
cd "$(dirname "$0")/.."

M=/mnt/e/dataset/RGBDT500/clean_test/rgbdt500_test_manifest_iris_prose.jsonl
ROOT=/mnt/e/dataset/RGBDT500/clean_test
OUT=outputs/lotus_line_v2
THR=tools/run_ms2_lotus_thermal_vae_official.py
SCALE="--depth-scale 1000 --max-depth 20"
EMPTY_CKPT=$OUT/rgbdt500_thermal_unet_gt/arm6_end.pt
CAPT_CKPT=$OUT/rgbdt500_thermal_unet_gt_caption/arm6_end.pt

if [ ! -f "$M" ]; then
    echo "FATAL: test manifest with captions not found: $M"
    echo "Run the captioner on the cleaned test manifest first."
    exit 1
fi

declare -a NAMES DIRS CKPTS MODES
add() { NAMES+=("$1"); DIRS+=("$2"); CKPTS+=("$3"); MODES+=("$4"); }
add "1_emptytrain_x_empty"   "rgbdt500_eval_emptytrain_empty"   "$EMPTY_CKPT" "empty"
add "2_emptytrain_x_correct" "rgbdt500_eval_emptytrain_correct" "$EMPTY_CKPT" "correct"
add "3_capttrain_x_empty"    "rgbdt500_eval_capttrain_empty"    "$CAPT_CKPT"  "empty"
add "4_capttrain_x_correct"  "rgbdt500_eval_capttrain_correct"  "$CAPT_CKPT"  "correct"

declare -a RESULTS
for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"; dir="$OUT/${DIRS[$i]}"; ckpt="${CKPTS[$i]}"; mode="${MODES[$i]}"
    if [ ! -f "$ckpt" ]; then echo "=== $name: checkpoint missing ($ckpt)"; RESULTS+=("$name NO_CKPT"); continue; fi

    if [ ! -f "$dir/official_run_metadata.json" ]; then
        echo "=== [$((i+1))/4] $name: inference $(date +%H:%M:%S)"
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
            --align ssi_disparity $SCALE --min-gt-valid-fraction 0.10 \
            --output-dir "$official" > "$official.log" 2>&1 \
            || { echo "$name rescore FAILED"; RESULTS+=("$name RESCORE_FAILED"); continue; }
    fi

    read -r absrel d1 n <<< "$(python3 -c "
import json
s = json.load(open('$official/metrics/summary.json'))
st = s['image_wise']['statistics']
print(round(st['abs_rel']['mean'], 4), round(st['a1']['mean'], 4), s['sample_count'])")"
    neg="$(python3 -c "
import numpy as np, glob
files = sorted(glob.glob('$dir/raw_predictions/*.npy'))[::40]
print(f'{np.mean([float((np.load(f) <= 0).mean()) for f in files])*100:.1f}%')")"
    RESULTS+=("$name absrel=$absrel d1=$d1 negpx=$neg n=$n")
    echo "=== [$((i+1))/4] $name: absrel=$absrel d1=$d1 negpx=$neg"
done

echo
echo "================ RGBDT500 caption 2x2 (official protocol, dense GT) ================"
for line in "${RESULTS[@]}"; do echo "  $line"; done
echo
echo "读法: cell2-cell1 = 推理注入价值 | cell3-cell1 = caption 训练的依赖代价 | cell4-cell1 = 训练+使用总效果"
echo "MS2 对照: 注入 -0.004(RGB线) / 0(thermal线) / +0.0013(f线) | f线依赖代价 -0.0041"
echo "下一步: python tools/analyze_rgbdt500_caption.py  (逐样本配对 + bootstrap CI)"
