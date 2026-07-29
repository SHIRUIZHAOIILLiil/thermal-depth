#!/usr/bin/env bash
# The zero-training caption baseline on RGBDT500 -- the control that was missing.
#
# Every RGBDT500 caption cell measured so far (d-line, f-line, RGB-teacher line,
# 12 evals) sits on a TRAINED model, so each carries its own training-induced
# interaction with the text pathway. None of them can answer the prior question:
#
#     does the caption carry usable depth information on this data AT ALL,
#     before any training of ours touches the model?
#
# A frozen pretrained Lotus with vs without the caption answers exactly that,
# with zero training and therefore zero training confound. It is also the
# cheapest control available: pure inference.
#
# Cells (all frozen pretrained Lotus-G, no checkpoint loaded):
#   1. RGB input     x empty     -- ALREADY DONE (rgbdt500_rgb_direct, 0.2481)
#   2. RGB input     x caption   -- this queue
#   3. thermal input x empty     -- this queue (never measured on RGBDT500 at all)
#   4. thermal input x caption   -- this queue
#
# Cell 3 is a bonus: it is the untrained starting point of every thermal model
# we have trained on this set, so it also tells us how much our training bought.
#
# Pre-registered reading (set before running):
#   caption null or harmful on BOTH modalities -> the text pathway carries no
#     usable depth signal here; every effect measured on trained models is a
#     property of our training, not of the caption.
#   caption clearly helpful when frozen -> the signal is real and our trained
#     models are failing to exploit it; that reframes the whole caption line.
#
# Usage: nohup bash tools/run_rgbdt500_frozen_caption_baseline.sh > outputs/lotus_line_v2/rgbdt500_frozen_caption_baseline.log 2>&1 &
# Idempotent: stages whose output already exists are skipped.

set -u
cd "$(dirname "$0")/.."

ROOT=/mnt/e/dataset/RGBDT500/clean_test
M=$ROOT/rgbdt500_test_manifest_iris_prose.jsonl
OUT=outputs/lotus_line_v2
SCALE="--depth-scale 1000 --min-depth 0.1 --max-depth 20"
RESCORE="--data-root $ROOT --align ssi_disparity --min-depth 0.1 --max-depth 20 --depth-scale 1000 --min-gt-valid-fraction 0.10"

[ -f "$M" ] || { echo "FATAL: missing manifest $M"; exit 1; }

declare -a RESULTS
say() { echo; echo "=== $* ($(date +%H:%M:%S))"; }

rescore() {  # $1 = run dir name, $2 = official route
    local official="outputs/ms2_official/${1}_mind01"
    if [ ! -f "$official/metrics/summary.json" ]; then
        say "$1: official rescore"
        if ! python tools/run_official_ms2_evaluation.py --manifest "$OUT/$1/selected_manifest.jsonl" \
             --prediction-dir "$OUT/$1/raw_predictions" --route "$2" $RESCORE \
             --output-dir "$official" > "$official.log" 2>&1; then
            echo "RESCORE FAILED:"; tail -5 "$official.log"; RESULTS+=("$1 RESCORE_FAILED"); return 1
        fi
    fi
    RESULTS+=("$1 $(python3 -c "
import json
s = json.load(open('$official/metrics/summary.json'))['image_wise']['statistics']
print(f\"abs_rel={s['abs_rel']['mean']:.4f} d1={s['a1']['mean']:.4f}\")")")
}

# --- cell 2: frozen Lotus, RGB input, caption injected ---
if [ ! -f "$OUT/rgbdt500_rgb_direct_capt/official_run_metadata.json" ]; then
    say "1/3  frozen RGB x caption"
    python tools/run_ms2_lotus_rgb_official.py --dataset rgbdt500 --manifest "$M" --ms2-root $ROOT \
        --output-dir "$OUT/rgbdt500_rgb_direct_capt" --max-samples 0 --caption-mode correct \
        $SCALE --save-raw-pred > "$OUT/rgbdt500_rgb_direct_capt.log" 2>&1 \
        || { echo "FAILED:"; tail -8 "$OUT/rgbdt500_rgb_direct_capt.log"; RESULTS+=("rgb_capt INFER_FAILED"); }
fi
rescore "rgbdt500_rgb_direct_capt" "rgb-frozen"

# --- cell 3: frozen Lotus, thermal input, empty prompt (no --unet-checkpoint = frozen) ---
if [ ! -f "$OUT/rgbdt500_thermal_frozen_empty/official_run_metadata.json" ]; then
    say "2/3  frozen thermal x empty  (never measured on RGBDT500 before)"
    python tools/run_ms2_lotus_thermal_vae_official.py --manifest "$M" --ms2-root $ROOT \
        --output-dir "$OUT/rgbdt500_thermal_frozen_empty" --max-samples 0 \
        --condition-posterior mode --caption-mode empty $SCALE --save-raw-pred \
        > "$OUT/rgbdt500_thermal_frozen_empty.log" 2>&1 \
        || { echo "FAILED:"; tail -8 "$OUT/rgbdt500_thermal_frozen_empty.log"; RESULTS+=("thermal_frozen_empty INFER_FAILED"); }
fi
rescore "rgbdt500_thermal_frozen_empty" "thermal-frozen"

# --- cell 4: frozen Lotus, thermal input, caption injected ---
if [ ! -f "$OUT/rgbdt500_thermal_frozen_capt/official_run_metadata.json" ]; then
    say "3/3  frozen thermal x caption"
    python tools/run_ms2_lotus_thermal_vae_official.py --manifest "$M" --ms2-root $ROOT \
        --output-dir "$OUT/rgbdt500_thermal_frozen_capt" --max-samples 0 \
        --condition-posterior mode --caption-mode correct $SCALE --save-raw-pred \
        > "$OUT/rgbdt500_thermal_frozen_capt.log" 2>&1 \
        || { echo "FAILED:"; tail -8 "$OUT/rgbdt500_thermal_frozen_capt.log"; RESULTS+=("thermal_frozen_capt INFER_FAILED"); }
fi
rescore "rgbdt500_thermal_frozen_capt" "thermal-frozen"

say "summary"
echo "frozen RGB x empty = abs_rel=0.2481 d1=0.6948   [already measured, rgbdt500_rgb_direct]"
printf '%s\n' "${RESULTS[@]}"
echo
echo "Paired CIs are NOT computed here. Hand the per_image.csv files to the paired"
echo "analysis before concluding -- mean differences on this set have been"
echo "overturned by paired tests before, and every caption effect measured so far"
echo "has a per-image win rate in the 38-57% band (i.e. near coin-flip)."
