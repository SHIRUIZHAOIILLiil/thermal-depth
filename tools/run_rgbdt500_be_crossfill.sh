#!/usr/bin/env bash
# Fill the 4 missing off-diagonal cells for RGBDT500 lines b and e:
#   b  train-empty  + infer-caption   (训✗ 推✓)
#   b  train-caption + infer-empty     (训✓ 推✗)
#   e  train-empty  + infer-caption
#   e  train-caption + infer-empty
#
# NO RETRAINING -- reuses the already-trained checkpoints and only re-runs
# inference (with the crossed caption mode) + official BMSD rescore. Commands
# are copied verbatim from run_rgbdt500_six_route_completion.sh so the numbers
# land in the same protocol (ssi_disparity / 0.1-20 / min-gt-valid 0.10).
#
# e is expected to reproduce line c (adapter degenerated to identity) -- a null
# there is the expected result, not a bug.
#
# Usage: nohup bash tools/run_rgbdt500_be_crossfill.sh > outputs/lotus_line_v2/rgbdt500_be_crossfill.log 2>&1 &
# Idempotent: finished stages are skipped.

set -u
cd "$(dirname "$0")/.."

ROOT_TEST=/mnt/e/dataset/RGBDT500/clean_test
MTEST=$ROOT_TEST/rgbdt500_test_manifest_iris_prose.jsonl
MTEST_PLAIN=$ROOT_TEST/rgbdt500_test_manifest.jsonl
OUT=outputs/lotus_line_v2
EVAL_SCALE="--depth-scale 1000 --min-depth 0.1 --max-depth 20"
RESCORE="--data-root $ROOT_TEST --align ssi_disparity --min-depth 0.1 --max-depth 20 --depth-scale 1000 --min-gt-valid-fraction 0.10"

for p in "$MTEST" "$MTEST_PLAIN"; do
    [ -f "$p" ] || { echo "FATAL: missing $p"; exit 1; }
done

declare -a RESULTS
say() { echo; echo "=== $* ($(date +%H:%M:%S))"; }

# ---- b line (RGB + trained U-Net) : $1 infer-caption-mode  $2 checkpoint-dir  $3 tag ----
cross_b() {
    local ckpt="$OUT/$2/rgb_unet_end.pt"
    [ -f "$ckpt" ] || { echo "MISSING CKPT: $ckpt"; RESULTS+=("b_$3 NO_CKPT"); return 1; }
    local ev="$OUT/rgbdt500_eval_bline_$3"
    if [ ! -f "$ev/official_run_metadata.json" ]; then
        say "b/$3 inference (infer=$1)"
        python tools/run_ms2_lotus_rgb_official.py --dataset rgbdt500 --manifest "$MTEST" \
            --ms2-root $ROOT_TEST --output-dir "$ev" --max-samples 0 --caption-mode "$1" \
            --unet-checkpoint "$ckpt" $EVAL_SCALE --save-raw-pred > "$ev.log" 2>&1 \
            || { echo "INFER FAILED:"; tail -8 "$ev.log"; RESULTS+=("b_$3 INFER_FAILED"); return 1; }
    fi
    local off="outputs/ms2_official/rgbdt500_eval_bline_${3}_mind01"
    if [ ! -f "$off/metrics/summary.json" ]; then
        say "b/$3 rescore"
        python tools/run_official_ms2_evaluation.py --manifest "$MTEST_PLAIN" \
            --prediction-dir "$ev/raw_predictions" --route rgb-unet --gt-view thermal \
            $RESCORE --output-dir "$off" > "$off.log" 2>&1 \
            || { echo "RESCORE FAILED:"; tail -6 "$off.log"; RESULTS+=("b_$3 RESCORE_FAILED"); return 1; }
    fi
    RESULTS+=("b_$3 $(python3 -c "
import json; s=json.load(open('$off/metrics/summary.json'))['image_wise']['statistics']
print(f\"abs_rel={s['abs_rel']['mean']:.4f}\")")")
}

# ---- e line (VAE latent adapter) : $1 infer-caption-mode  $2 checkpoint-dir  $3 tag ----
cross_e() {
    local ckpt="$OUT/$2/vae_adapter_end.pt"
    [ -f "$ckpt" ] || { echo "MISSING CKPT: $ckpt"; RESULTS+=("e_$3 NO_CKPT"); return 1; }
    local ev="$OUT/rgbdt500_eval_eline_$3"
    if [ ! -f "$ev/official_run_metadata.json" ]; then
        say "e/$3 inference (infer=$1)"
        python tools/run_ms2_lotus_thermal_vae_official.py --manifest "$MTEST" --ms2-root $ROOT_TEST \
            --output-dir "$ev" --max-samples 0 --condition-posterior mode --caption-mode "$1" \
            --latent-adapter-checkpoint "$ckpt" $EVAL_SCALE --save-raw-pred > "$ev.log" 2>&1 \
            || { echo "INFER FAILED:"; tail -8 "$ev.log"; RESULTS+=("e_$3 INFER_FAILED"); return 1; }
    fi
    local off="outputs/ms2_official/rgbdt500_eval_eline_${3}_mind01"
    if [ ! -f "$off/metrics/summary.json" ]; then
        say "e/$3 rescore"
        python tools/run_official_ms2_evaluation.py --manifest "$ev/selected_manifest.jsonl" \
            --prediction-dir "$ev/raw_predictions" --route vae-adapter \
            $RESCORE --output-dir "$off" > "$off.log" 2>&1 \
            || { echo "RESCORE FAILED:"; tail -6 "$off.log"; RESULTS+=("e_$3 RESCORE_FAILED"); return 1; }
    fi
    RESULTS+=("e_$3 $(python3 -c "
import json; s=json.load(open('$off/metrics/summary.json'))['image_wise']['statistics']
print(f\"abs_rel={s['abs_rel']['mean']:.4f}\")")")
}

# 训✗ 推✓ : train-empty checkpoint, infer with caption
cross_b correct full_rgbdt500_bline_empty   emptytrain_infcap
cross_e correct full_rgbdt500_eline_empty   emptytrain_infcap
# 训✓ 推✗ : train-caption checkpoint, infer empty
cross_b empty   full_rgbdt500_bline_caption capttrain_infempty
cross_e empty   full_rgbdt500_eline_caption capttrain_infempty

echo
echo "===================== CROSSFILL RESULTS ====================="
printf '%s\n' "${RESULTS[@]}"
echo "b: emptytrain_infcap = 训✗推✓ | capttrain_infempty = 训✓推✗ (e same)"
