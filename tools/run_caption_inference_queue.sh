#!/usr/bin/env bash
# Sequential queue for every pending caption-ablation inference cell (task 2).
# clip75 manifest throughout. Idempotent: cells with official_run_metadata.json
# are skipped, so rerunning after a crash continues where it left off.
#
# Usage:  nohup bash tools/run_caption_inference_queue.sh > outputs/lotus_line_v2/caption_queue.log 2>&1 &
# Stop:   pkill -9 -f run_caption_inference_queue; then pkill -9 -f 'run_ms2_lotus'

set -u
cd "$(dirname "$0")/.."

M=/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/ms2_fixed_sequence_internvl3_8b_filtered_caption_val_rgb_depth_v1_clip75_rerun_20260714.jsonl
MS2=/mnt/e/dataset/ms2
OUT=outputs/lotus_line_v2
RGB=tools/run_ms2_lotus_rgb_official.py
THR=tools/run_ms2_lotus_thermal_vae_official.py
TRN=tools/run_ms2_lotus_trained_official.py

declare -a NAMES CMDS
# --- line b (2x2 remaining cells) ---
NAMES+=("b2_emptytrain_x_correct"); CMDS+=("python $RGB --manifest $M --ms2-root $MS2 --output-dir $OUT/route_b_rgb_unet_val_full_caption --max-samples 0 --unet-checkpoint $OUT/full_train_epoch1_rgb_unet_gt/rgb_unet_end.pt --caption-mode correct --save-raw-pred")
NAMES+=("b3_capttrain_x_correct");  CMDS+=("python $RGB --manifest $M --ms2-root $MS2 --output-dir $OUT/route_b_capttrain_val_full_correct --max-samples 0 --unet-checkpoint $OUT/full_train_epoch1_rgb_unet_gt_caption/rgb_unet_end.pt --caption-mode correct --save-raw-pred")
NAMES+=("b4_capttrain_x_empty");    CMDS+=("python $RGB --manifest $M --ms2-root $MS2 --output-dir $OUT/route_b_capttrain_val_full_empty --max-samples 0 --unet-checkpoint $OUT/full_train_epoch1_rgb_unet_gt_caption/rgb_unet_end.pt --caption-mode empty --save-raw-pred")
# --- line c (frozen; single caption cell) ---
NAMES+=("c_frozen_x_correct");      CMDS+=("python $THR --manifest $M --ms2-root $MS2 --output-dir $OUT/route_c_thermal_frozen_val_full_caption --max-samples 0 --condition-posterior mode --caption-mode correct --save-raw-pred")
# --- line d (fp32dec 2x2) ---
NAMES+=("d1_fp32dec_x_empty");      CMDS+=("python $THR --manifest $M --ms2-root $MS2 --output-dir $OUT/route_d_fp32dec_val_full_empty --max-samples 0 --condition-posterior mode --unet-checkpoint $OUT/full_train_epoch1_thermal_unet_fp32dec/arm6_end.pt --caption-mode empty --save-raw-pred")
NAMES+=("d2_fp32dec_x_correct");    CMDS+=("python $THR --manifest $M --ms2-root $MS2 --output-dir $OUT/route_d_fp32dec_val_full_caption --max-samples 0 --condition-posterior mode --unet-checkpoint $OUT/full_train_epoch1_thermal_unet_fp32dec/arm6_end.pt --caption-mode correct --save-raw-pred")
NAMES+=("d3_capttrain_x_correct");  CMDS+=("python $THR --manifest $M --ms2-root $MS2 --output-dir $OUT/route_d_capttrain_val_full_correct --max-samples 0 --condition-posterior mode --unet-checkpoint $OUT/full_train_epoch1_thermal_unet_fp32dec_caption/arm6_end.pt --caption-mode correct --save-raw-pred")
NAMES+=("d4_capttrain_x_empty");    CMDS+=("python $THR --manifest $M --ms2-root $MS2 --output-dir $OUT/route_d_capttrain_val_full_empty --max-samples 0 --condition-posterior mode --unet-checkpoint $OUT/full_train_epoch1_thermal_unet_fp32dec_caption/arm6_end.pt --caption-mode empty --save-raw-pred")
# --- line e (frozen U-Net + trained latent adapter; single caption cell) ---
NAMES+=("e_vae_adapter_x_correct"); CMDS+=("python $THR --manifest $M --ms2-root $MS2 --output-dir $OUT/route_e_vae_adapter_val_full_caption --max-samples 0 --condition-posterior mode --latent-adapter-checkpoint $OUT/full_train_epoch1_vae_latent_adapter_gt/vae_adapter_end.pt --caption-mode correct --save-raw-pred")
# --- line f (adapter-only; single caption cell) ---
NAMES+=("f_adapter_only_x_correct"); CMDS+=("python $TRN --manifest $M --ms2-root $MS2 --checkpoint $OUT/full_train_epoch1_v2_4_response/adapter_end.pt --output-dir $OUT/route_f_adapter_only_val_full_caption --caption-mode correct --save-raw-pred")

TOTAL=${#NAMES[@]}
declare -a RESULTS
for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    cmd="${CMDS[$i]}"
    outdir="$(echo "$cmd" | grep -o -- '--output-dir [^ ]*' | cut -d' ' -f2)"
    if [ -f "$outdir/official_run_metadata.json" ]; then
        echo "=== [$((i+1))/$TOTAL] $name: already complete, skipping"
        RESULTS+=("$name SKIPPED(done)")
        continue
    fi
    log="$outdir.log"
    mkdir -p "$(dirname "$log")"
    echo "=== [$((i+1))/$TOTAL] $name: starting $(date +%H:%M:%S)  (log: $log)"
    if $cmd > "$log" 2>&1; then
        absrel=$(python3 -c "import json;print(round(json.load(open('$outdir/official_run_metadata.json'))['metrics']['abs_relative_difference'],4))" 2>/dev/null || echo "?")
        echo "=== [$((i+1))/$TOTAL] $name: OK  abs_rel=$absrel"
        RESULTS+=("$name OK abs_rel=$absrel")
    else
        echo "=== [$((i+1))/$TOTAL] $name: FAILED (last lines below)"
        tail -5 "$log"
        RESULTS+=("$name FAILED")
    fi
done

echo
echo "================ caption inference queue summary ================"
for line in "${RESULTS[@]}"; do echo "  $line"; done
