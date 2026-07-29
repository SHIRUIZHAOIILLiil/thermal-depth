#!/usr/bin/env bash
# Task-4 closing queue: evaluate arms B (GT-only) and C (+AnyThermal dense
# teacher) on the full val split, then re-score both under the official BMSD
# protocol, then print the A/B/C summary line-up.
#
# Usage:  nohup bash tools/run_task4_eval_queue.sh > outputs/lotus_line_v2/task4_eval_queue.log 2>&1 &
# Idempotent: finished stages are skipped on rerun.

set -u
cd "$(dirname "$0")/.."

M=/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/ms2_fixed_sequence_internvl3_8b_filtered_caption_val_rgb_depth_v1_clip75_rerun_20260714.jsonl
MS2=/mnt/e/dataset/ms2
OUT=outputs/lotus_line_v2
THR=tools/run_ms2_lotus_thermal_vae_official.py

declare -a NAMES DIRS CKPTS
NAMES+=("armB_gt_only");   DIRS+=("route_gt_only_val_full");   CKPTS+=("full_train_epoch1_thermal_unet_gt_only/arm6_end.pt")
NAMES+=("armC_amteacher"); DIRS+=("route_amteacher_val_full"); CKPTS+=("full_train_epoch1_thermal_unet_amteacher/arm6_end.pt")

declare -a RESULTS
for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"; dir="$OUT/${DIRS[$i]}"; ckpt="$OUT/${CKPTS[$i]}"
    if [ -f "$dir/official_run_metadata.json" ]; then
        echo "=== $name inference: already complete, skipping"
    else
        echo "=== $name inference: starting $(date +%H:%M:%S)"
        if ! python $THR --manifest $M --ms2-root $MS2 --output-dir "$dir" --max-samples 0 --condition-posterior mode --unet-checkpoint "$ckpt" --caption-mode empty --save-raw-pred > "$dir.log" 2>&1; then
            echo "=== $name inference FAILED (see $dir.log)"; tail -5 "$dir.log"; RESULTS+=("$name FAILED"); continue
        fi
    fi
    official="outputs/ms2_official/${DIRS[$i]}_ssi_disparity"
    if [ ! -f "$official/metrics/summary.json" ]; then
        echo "=== $name official re-score"
        python tools/run_official_ms2_evaluation.py --manifest "$dir/selected_manifest.jsonl" --data-root $MS2 --prediction-dir "$dir/raw_predictions" --route thermal-unet --align ssi_disparity --output-dir "$official" > "$official.log" 2>&1 || { echo "re-score FAILED"; RESULTS+=("$name RESCORE_FAILED"); continue; }
    fi
    absrel=$(python3 -c "import json;s=json.load(open('$official/metrics/summary.json'));print(round(s['image_wise']['statistics']['abs_rel']['mean'],4))")
    d1=$(python3 -c "import json;s=json.load(open('$official/metrics/summary.json'));print(round(s['image_wise']['statistics']['a1']['mean'],4))")
    RESULTS+=("$name OK absrel=$absrel d1=$d1")
    echo "=== $name done: absrel=$absrel d1=$d1"
done

echo
echo "================ task-4 A/B/C line-up (official protocol) ================"
echo "  armA_anchor_gt   absrel=0.1275 d1=0.8185  (route_d_fp32dec_val_full_empty)"
for line in "${RESULTS[@]}"; do echo "  $line"; done
echo "  reference: champion 0.1172 / AnyThermal-MiDaS 0.0929"
