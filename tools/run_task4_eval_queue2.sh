#!/usr/bin/env bash
# Task-4 evaluation queue #2: the five newly trained arms.
#   C'  (ssi teacher, no anchor)          -> empty inference
#   C'' (l1 teacher, no anchor)           -> empty inference
#   B_capt   (pure GT, caption-trained)   -> empty + correct inference
#   C'_capt  (ssi teacher, caption-train) -> empty + correct
#   C''_capt (l1 teacher, caption-train)  -> empty + correct
# Each cell: inference (+raw export) -> official BMSD re-score -> summary row.
#
# Usage: nohup bash tools/run_task4_eval_queue2.sh > outputs/lotus_line_v2/task4_eval_queue2.log 2>&1 &

set -u
cd "$(dirname "$0")/.."

M=/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/ms2_fixed_sequence_internvl3_8b_filtered_caption_val_rgb_depth_v1_clip75_rerun_20260714.jsonl
MS2=/mnt/e/dataset/ms2
OUT=outputs/lotus_line_v2
THR=tools/run_ms2_lotus_thermal_vae_official.py

declare -a NAMES DIRS CKPTS MODES
add() { NAMES+=("$1"); DIRS+=("$2"); CKPTS+=("$3"); MODES+=("$4"); }
add "cprime_x_empty"          "route_cprime_val_full"                 "full_train_epoch1_thermal_unet_cprime/arm6_end.pt"            "empty"
add "cprime_l1_x_empty"       "route_cprime_l1_val_full"              "full_train_epoch1_thermal_unet_cprime_l1/arm6_end.pt"         "empty"
add "B_capt_x_empty"          "route_gt_only_capttrain_val_full_empty"    "full_train_epoch1_thermal_unet_gt_only_caption/arm6_end.pt"   "empty"
add "B_capt_x_correct"        "route_gt_only_capttrain_val_full_correct"  "full_train_epoch1_thermal_unet_gt_only_caption/arm6_end.pt"   "correct"
add "cprime_capt_x_empty"     "route_cprime_capttrain_val_full_empty"     "full_train_epoch1_thermal_unet_cprime_caption/arm6_end.pt"    "empty"
add "cprime_capt_x_correct"   "route_cprime_capttrain_val_full_correct"   "full_train_epoch1_thermal_unet_cprime_caption/arm6_end.pt"    "correct"
add "cprime_l1_capt_x_empty"  "route_cprime_l1_capttrain_val_full_empty"  "full_train_epoch1_thermal_unet_cprime_l1_caption/arm6_end.pt" "empty"
add "cprime_l1_capt_x_correct" "route_cprime_l1_capttrain_val_full_correct" "full_train_epoch1_thermal_unet_cprime_l1_caption/arm6_end.pt" "correct"

TOTAL=${#NAMES[@]}
declare -a RESULTS
for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"; dir="$OUT/${DIRS[$i]}"; ckpt="$OUT/${CKPTS[$i]}"; mode="${MODES[$i]}"
    if [ ! -f "$ckpt" ]; then echo "=== [$((i+1))/$TOTAL] $name: checkpoint missing ($ckpt)"; RESULTS+=("$name NO_CKPT"); continue; fi
    if [ ! -f "$dir/official_run_metadata.json" ]; then
        echo "=== [$((i+1))/$TOTAL] $name: inference $(date +%H:%M:%S)"
        if ! python $THR --manifest $M --ms2-root $MS2 --output-dir "$dir" --max-samples 0 --condition-posterior mode --unet-checkpoint "$ckpt" --caption-mode "$mode" --save-raw-pred > "$dir.log" 2>&1; then
            echo "=== [$((i+1))/$TOTAL] $name: FAILED"; tail -5 "$dir.log"; RESULTS+=("$name FAILED"); continue
        fi
    fi
    official="outputs/ms2_official/${DIRS[$i]}_ssi_disparity"
    if [ ! -f "$official/metrics/summary.json" ]; then
        python tools/run_official_ms2_evaluation.py --manifest "$dir/selected_manifest.jsonl" --data-root $MS2 --prediction-dir "$dir/raw_predictions" --route thermal-unet --align ssi_disparity --output-dir "$official" > "$official.log" 2>&1 || { echo "$name rescore FAILED"; RESULTS+=("$name RESCORE_FAILED"); continue; }
    fi
    read -r absrel d1 <<< "$(python3 -c "import json;s=json.load(open('$official/metrics/summary.json'))['image_wise']['statistics'];print(round(s['abs_rel']['mean'],4), round(s['a1']['mean'],4))")"
    neg="$(python3 -c "
import numpy as np, glob
files = sorted(glob.glob('$dir/raw_predictions/*.npy'))[::500]
vals = [float((np.load(f) <= 0).mean()) for f in files]
print(f'{np.mean(vals)*100:.1f}%')
")"
    RESULTS+=("$name OK absrel=$absrel d1=$d1 negpx=$neg")
    echo "=== [$((i+1))/$TOTAL] $name: absrel=$absrel d1=$d1 negpx=$neg"
done

echo
echo "================ task-4 eval queue #2 summary ================"
echo "  ref: A(anchor+GT)=0.1275 | B(pure GT)=0.1184 negpx~21% | champion=0.1172"
for line in "${RESULTS[@]}"; do echo "  $line"; done
