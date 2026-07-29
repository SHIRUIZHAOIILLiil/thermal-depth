#!/usr/bin/env bash
# Six-route caption ablation (task 2): rerun every route's val-full inference
# with --caption-mode correct. Checkpoints untouched; only inference-time text
# changes, seeds are per-manifest-row so caption/empty runs stay pairable.
#
# Usage:            bash tools/run_six_route_caption_ablation.sh
# Smoke (8 imgs):   SMOKE=1 bash tools/run_six_route_caption_ablation.sh
#
# Idempotent: a route whose output dir already contains official_run_metadata.json
# is skipped, so a crashed batch can simply be rerun.

set -u
cd "$(dirname "$0")/.."

# clip75 = Iris-format captions clipped to <=75 CLIP tokens (verified: max 75,
# zero truncation; id/path rows identical to spatial_v2 so seeds & pairing with
# existing empty-caption runs stay exact). This is the canonical caption set.
M=/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/ms2_fixed_sequence_internvl3_8b_filtered_caption_val_rgb_depth_v1_clip75_rerun_20260714.jsonl
MS2=/mnt/e/dataset/ms2
OUT=outputs/lotus_line_v2

if [ "${SMOKE:-0}" = "1" ]; then
    SAMPLES=8; SUFFIX="_caption_smoke8"
else
    SAMPLES=0; SUFFIX="_caption"
fi

declare -a NAMES CMDS
NAMES+=("route_a"); CMDS+=("python tools/run_ms2_lotus_rgb_official.py --manifest $M --ms2-root $MS2 --output-dir $OUT/route_a_rgb_frozen_val_full$SUFFIX --max-samples $SAMPLES --caption-mode correct --save-raw-pred")
NAMES+=("route_b"); CMDS+=("python tools/run_ms2_lotus_rgb_official.py --manifest $M --ms2-root $MS2 --output-dir $OUT/route_b_rgb_unet_val_full$SUFFIX --max-samples $SAMPLES --unet-checkpoint $OUT/full_train_epoch1_rgb_unet_gt/rgb_unet_end.pt --caption-mode correct --save-raw-pred")
NAMES+=("route_c"); CMDS+=("python tools/run_ms2_lotus_thermal_vae_official.py --manifest $M --ms2-root $MS2 --output-dir $OUT/route_c_thermal_frozen_val_full$SUFFIX --max-samples $SAMPLES --condition-posterior mode --caption-mode correct --save-raw-pred")
NAMES+=("route_d"); CMDS+=("python tools/run_ms2_lotus_thermal_vae_official.py --manifest $M --ms2-root $MS2 --output-dir $OUT/route_d_thermal_unet_val_full$SUFFIX --max-samples $SAMPLES --condition-posterior mode --unet-checkpoint $OUT/full_train_epoch1_thermal_vae_unet_gt/arm6_end.pt --caption-mode correct --save-raw-pred")
NAMES+=("route_e"); CMDS+=("python tools/run_ms2_lotus_thermal_vae_official.py --manifest $M --ms2-root $MS2 --output-dir $OUT/route_e_vae_adapter_val_full$SUFFIX --max-samples $SAMPLES --condition-posterior mode --latent-adapter-checkpoint $OUT/full_train_epoch1_vae_latent_adapter_gt/vae_adapter_end.pt --caption-mode correct --save-raw-pred")
NAMES+=("route_f"); CMDS+=("python tools/run_ms2_lotus_trained_official.py --manifest $M --ms2-root $MS2 --checkpoint $OUT/full_train_epoch1_v2_4_response/adapter_end.pt --output-dir $OUT/route_f_adapter_only_val_full$SUFFIX --max-samples $SAMPLES --caption-mode correct --save-raw-pred")

declare -a RESULTS
for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    cmd="${CMDS[$i]}"
    outdir="$(echo "$cmd" | grep -o -- '--output-dir [^ ]*' | cut -d' ' -f2)"
    if [ -f "$outdir/official_run_metadata.json" ]; then
        echo "=== [$((i+1))/6] $name: already complete, skipping ($outdir)"
        RESULTS+=("$name SKIPPED(done)")
        continue
    fi
    log="$outdir.log"
    mkdir -p "$(dirname "$log")"
    echo "=== [$((i+1))/6] $name: starting $(date +%H:%M:%S)  (log: $log)"
    if $cmd > "$log" 2>&1; then
        absrel=$(python3 -c "import json;print(round(json.load(open('$outdir/official_run_metadata.json'))['metrics']['abs_relative_difference'],4))" 2>/dev/null || echo "?")
        echo "=== [$((i+1))/6] $name: OK  abs_rel=$absrel"
        RESULTS+=("$name OK abs_rel=$absrel")
    else
        echo "=== [$((i+1))/6] $name: FAILED (see $log, last lines below)"
        tail -5 "$log"
        RESULTS+=("$name FAILED")
    fi
done

echo
echo "================ caption ablation batch summary ================"
for line in "${RESULTS[@]}"; do echo "  $line"; done
