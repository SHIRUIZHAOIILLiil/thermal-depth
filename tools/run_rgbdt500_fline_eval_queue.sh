#!/usr/bin/env bash
# RGBDT500 f-line caption 2x2 (AnyThermal features -> trained Adapter -> frozen U-Net).
#
# Purpose beyond the caption question itself: the f-line reaches a DIFFERENT
# model strength than the d-line already measured on this dataset (d-line
# abs_rel 0.3616). The GT-density explanation for the RGBDT500 caption effect is
# already refuted three ways, so model strength is now the leading suspect --
# comparing the caption effect at two strengths tests it directly.
#
# Usage: nohup bash tools/run_rgbdt500_fline_eval_queue.sh > outputs/lotus_line_v2/rgbdt500_fline_eval.log 2>&1 &

set -u
cd "$(dirname "$0")/.."

M=/mnt/e/dataset/RGBDT500/clean_test/rgbdt500_test_manifest_iris_prose.jsonl
ROOT=/mnt/e/dataset/RGBDT500/clean_test
OUT=outputs/lotus_line_v2
RUN=tools/run_ms2_lotus_trained_official.py
# min-depth 0.1: the MS2 default of 1e-3 admits this sensor's millimetre noise
SCALE="--depth-scale 1000 --min-depth 0.1 --max-depth 20"
EMPTY_CKPT=$OUT/rgbdt500_f_line/adapter_end.pt
CAPT_CKPT=$OUT/rgbdt500_f_line_caption/adapter_end.pt

declare -a NAMES DIRS CKPTS MODES
add() { NAMES+=("$1"); DIRS+=("$2"); CKPTS+=("$3"); MODES+=("$4"); }
add "f1_emptytrain_x_empty"   "rgbdt500_fline_emptytrain_empty"   "$EMPTY_CKPT" "empty"
add "f2_emptytrain_x_correct" "rgbdt500_fline_emptytrain_correct" "$EMPTY_CKPT" "correct"
add "f3_capttrain_x_empty"    "rgbdt500_fline_capttrain_empty"    "$CAPT_CKPT"  "empty"
add "f4_capttrain_x_correct"  "rgbdt500_fline_capttrain_correct"  "$CAPT_CKPT"  "correct"

declare -a RESULTS
for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"; dir="$OUT/${DIRS[$i]}"; ckpt="${CKPTS[$i]}"; mode="${MODES[$i]}"
    if [ ! -f "$ckpt" ]; then echo "=== $name: checkpoint missing ($ckpt)"; RESULTS+=("$name NO_CKPT"); continue; fi

    if [ ! -f "$dir/official_run_metadata.json" ]; then
        echo "=== [$((i+1))/4] $name: inference $(date +%H:%M:%S)"
        if ! python $RUN --manifest "$M" --ms2-root $ROOT --checkpoint "$ckpt" \
             --output-dir "$dir" --max-samples 0 --caption-mode "$mode" \
             $SCALE --save-raw-pred > "$dir.log" 2>&1; then
            echo "=== $name FAILED"; tail -5 "$dir.log"; RESULTS+=("$name FAILED"); continue
        fi
    fi

    official="outputs/ms2_official/${DIRS[$i]}_mind01"
    if [ ! -f "$official/metrics/summary.json" ]; then
        python tools/run_official_ms2_evaluation.py --manifest "$dir/selected_manifest.jsonl" \
            --data-root $ROOT --prediction-dir "$dir/raw_predictions" --route adapter-only \
            --align ssi_disparity $SCALE --min-gt-valid-fraction 0.10 \
            --output-dir "$official" > "$official.log" 2>&1 \
            || { echo "$name rescore FAILED"; RESULTS+=("$name RESCORE_FAILED"); continue; }
    fi

    read -r absrel d1 <<< "$(python3 -c "
import json
st = json.load(open('$official/metrics/summary.json'))['image_wise']['statistics']
print(round(st['abs_rel']['mean'], 4), round(st['a1']['mean'], 4))")"
    RESULTS+=("$name absrel=$absrel d1=$d1")
    echo "=== [$((i+1))/4] $name: absrel=$absrel d1=$d1"
done

echo
echo "================ RGBDT500 f 线 caption 2x2 ================"
for line in "${RESULTS[@]}"; do echo "  $line"; done
echo
echo "d 线对照 (同数据集/同协议): 0.3616 / 0.3597 / 0.3601 / 0.3543  总效果 +0.00729*"
echo "MS2 f 线对照: 0.1347 (empty) / 0.1334 (correct)  注入 +0.0013*  caption训练净收益 -0.0001 n.s."
echo "下一步: python tools/analyze_rgbdt500_caption.py --line f"
