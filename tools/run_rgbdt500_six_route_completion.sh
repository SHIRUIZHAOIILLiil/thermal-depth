#!/usr/bin/env bash
# Fill the two missing six-route lines on RGBDT500 so the 12-cell table closes.
#
# Already on disk (nothing to redo):
#   a  RGB frozen direct         0.2481 / caption 0.2486
#   c  thermal frozen direct     0.3825 / caption 0.3770
#   d  thermal + trained U-Net   0.3616 / full 2x2
#   f  AnyThermal + adapter      0.3985 / full 2x2
#
# This queue adds, for the 6x2 diagonal the report needs:
#   b  RGB + trained U-Net       empty arm + caption arm
#   e  VAE latent + adapter      empty arm + caption arm
#
# EXPECTATION ON e (frozen MS2 result, recorded so a null is not read as a bug):
# on MS2 the latent adapter converged back to the identity twice -- step-1 grad
# 11 decaying to 0.16, delta peak 0.027 ending at 1e-4, val bit-identical to
# line c, and the caption arm's train effect was +-0.00000. If e lands on top of
# c here too, that is the expected reproduction, not a failure.
#
# Each arm runs smoke(5) -> full -> eval -> official rescore, and deletes its own
# intermediate checkpoints once summary.json and the end checkpoint both exist
# (line b writes ~10GB per save; without the cleanup the two b arms alone cost
# ~80GB).
#
# Usage: nohup bash tools/run_rgbdt500_six_route_completion.sh > outputs/lotus_line_v2/rgbdt500_six_route_completion.log 2>&1 &
# Idempotent: finished stages are skipped.

set -u
cd "$(dirname "$0")/.."

ROOT_TRAIN=/mnt/e/dataset/RGBDT500/clean_train
ROOT_TEST=/mnt/e/dataset/RGBDT500/clean_test
MTRAIN=$ROOT_TRAIN/rgbdt500_train_manifest_iris_prose.jsonl
MTEST=$ROOT_TEST/rgbdt500_test_manifest_iris_prose.jsonl
MTEST_PLAIN=$ROOT_TEST/rgbdt500_test_manifest.jsonl   # line b rescore needs thermal_path
OUT=outputs/lotus_line_v2
SCALE="--depth-scale 1000 --gt-max-depth 20 --min-gt-valid-fraction 0.10"
EVAL_SCALE="--depth-scale 1000 --min-depth 0.1 --max-depth 20"
RESCORE="--data-root $ROOT_TEST --align ssi_disparity --min-depth 0.1 --max-depth 20 --depth-scale 1000 --min-gt-valid-fraction 0.10"

for p in "$MTRAIN" "$MTEST" "$MTEST_PLAIN"; do
    [ -f "$p" ] || { echo "FATAL: missing $p"; exit 1; }
done

declare -a RESULTS
say() { echo; echo "=== $* ($(date +%H:%M:%S))"; }

prune() {  # $1 = train dir, $2 = end checkpoint filename
    if [ -f "$1/summary.json" ] && [ -f "$1/$2" ]; then
        find "$1" -maxdepth 1 -name '*.pt' ! -name "$2" -delete 2>/dev/null
        echo "  pruned intermediate checkpoints in $1"
    fi
}

# ---- line b -----------------------------------------------------------------
run_b() {  # $1 = caption mode (empty|correct), $2 = tag
    local dir="$OUT/full_rgbdt500_bline_$2" ckpt
    ckpt="$dir/rgb_unet_end.pt"
    if [ ! -f "$ckpt" ]; then
        if [ ! -f "$OUT/smoke_rgbdt500_bline_$2/smoke_summary.json" ]; then
            say "b/$2 smoke"
            python tools/train_ms2_rgb_unet_gt.py --gt-decode-fp32 --train-manifest "$MTRAIN" \
                --ms2-root $ROOT_TRAIN $SCALE --caption-mode "$1" \
                --output-dir "$OUT/smoke_rgbdt500_bline_$2" --smoke-updates 5 \
                > "$OUT/smoke_rgbdt500_bline_$2.log" 2>&1 \
                || { echo "SMOKE FAILED:"; tail -12 "$OUT/smoke_rgbdt500_bline_$2.log"; RESULTS+=("b_$2 SMOKE_FAILED"); return 1; }
        fi
        say "b/$2 full training"
        python tools/train_ms2_rgb_unet_gt.py --gt-decode-fp32 --train-manifest "$MTRAIN" \
            --ms2-root $ROOT_TRAIN $SCALE --caption-mode "$1" --output-dir "$dir" \
            > "$dir.log" 2>&1 \
            || { echo "TRAIN FAILED:"; tail -12 "$dir.log"; RESULTS+=("b_$2 TRAIN_FAILED"); return 1; }
        prune "$dir" "rgb_unet_end.pt"
    fi
    local ev="$OUT/rgbdt500_eval_bline_$2"
    if [ ! -f "$ev/official_run_metadata.json" ]; then
        say "b/$2 inference"
        python tools/run_ms2_lotus_rgb_official.py --dataset rgbdt500 --manifest "$MTEST" \
            --ms2-root $ROOT_TEST --output-dir "$ev" --max-samples 0 --caption-mode "$1" \
            --unet-checkpoint "$ckpt" $EVAL_SCALE --save-raw-pred > "$ev.log" 2>&1 \
            || { echo "INFER FAILED:"; tail -8 "$ev.log"; RESULTS+=("b_$2 INFER_FAILED"); return 1; }
    fi
    local off="outputs/ms2_official/rgbdt500_eval_bline_${2}_mind01"
    if [ ! -f "$off/metrics/summary.json" ]; then
        say "b/$2 rescore"
        # NOTE: the RGB runner's selected_manifest.jsonl carries no thermal_path,
        # which ms2_eval.io refuses -- use the plain test manifest (same id order).
        python tools/run_official_ms2_evaluation.py --manifest "$MTEST_PLAIN" \
            --prediction-dir "$ev/raw_predictions" --route rgb-unet --gt-view thermal \
            $RESCORE --output-dir "$off" > "$off.log" 2>&1 \
            || { echo "RESCORE FAILED:"; tail -6 "$off.log"; RESULTS+=("b_$2 RESCORE_FAILED"); return 1; }
    fi
    RESULTS+=("b_$2 $(python3 -c "
import json; s=json.load(open('$off/metrics/summary.json'))['image_wise']['statistics']
print(f\"abs_rel={s['abs_rel']['mean']:.4f} d1={s['a1']['mean']:.4f}\")")")
}

# ---- line e -----------------------------------------------------------------
run_e() {  # $1 = caption mode, $2 = tag
    local dir="$OUT/full_rgbdt500_eline_$2" ckpt
    ckpt="$dir/vae_adapter_end.pt"
    if [ ! -f "$ckpt" ]; then
        if [ ! -f "$OUT/smoke_rgbdt500_eline_$2/smoke_summary.json" ]; then
            say "e/$2 smoke"
            python tools/train_ms2_vae_latent_adapter_gt.py --gt-decode-fp32 --train-manifest "$MTRAIN" \
                --ms2-root $ROOT_TRAIN $SCALE --caption-mode "$1" \
                --output-dir "$OUT/smoke_rgbdt500_eline_$2" --smoke-updates 5 \
                > "$OUT/smoke_rgbdt500_eline_$2.log" 2>&1 \
                || { echo "SMOKE FAILED:"; tail -12 "$OUT/smoke_rgbdt500_eline_$2.log"; RESULTS+=("e_$2 SMOKE_FAILED"); return 1; }
        fi
        say "e/$2 full training"
        python tools/train_ms2_vae_latent_adapter_gt.py --gt-decode-fp32 --train-manifest "$MTRAIN" \
            --ms2-root $ROOT_TRAIN $SCALE --caption-mode "$1" --output-dir "$dir" \
            > "$dir.log" 2>&1 \
            || { echo "TRAIN FAILED:"; tail -12 "$dir.log"; RESULTS+=("e_$2 TRAIN_FAILED"); return 1; }
        prune "$dir" "vae_adapter_end.pt"
    fi
    local ev="$OUT/rgbdt500_eval_eline_$2"
    if [ ! -f "$ev/official_run_metadata.json" ]; then
        say "e/$2 inference"
        python tools/run_ms2_lotus_thermal_vae_official.py --manifest "$MTEST" --ms2-root $ROOT_TEST \
            --output-dir "$ev" --max-samples 0 --condition-posterior mode --caption-mode "$1" \
            --latent-adapter-checkpoint "$ckpt" $EVAL_SCALE --save-raw-pred > "$ev.log" 2>&1 \
            || { echo "INFER FAILED:"; tail -8 "$ev.log"; RESULTS+=("e_$2 INFER_FAILED"); return 1; }
    fi
    local off="outputs/ms2_official/rgbdt500_eval_eline_${2}_mind01"
    if [ ! -f "$off/metrics/summary.json" ]; then
        say "e/$2 rescore"
        python tools/run_official_ms2_evaluation.py --manifest "$ev/selected_manifest.jsonl" \
            --prediction-dir "$ev/raw_predictions" --route vae-adapter \
            $RESCORE --output-dir "$off" > "$off.log" 2>&1 \
            || { echo "RESCORE FAILED:"; tail -6 "$off.log"; RESULTS+=("e_$2 RESCORE_FAILED"); return 1; }
    fi
    RESULTS+=("e_$2 $(python3 -c "
import json; s=json.load(open('$off/metrics/summary.json'))['image_wise']['statistics']
print(f\"abs_rel={s['abs_rel']['mean']:.4f} d1={s['a1']['mean']:.4f}\")")")
}

say "1/4  line b, empty prompt";   run_b empty   empty
say "2/4  line b, caption";        run_b correct caption
say "3/4  line e, empty prompt";   run_e empty   empty
say "4/4  line e, caption";        run_e correct caption

say "summary"
printf '%s\n' "${RESULTS[@]}"
echo
echo "Already measured (no rerun): a 0.2481/0.2486  c 0.3825/0.3770  d 0.3616  f 0.3985"
echo "Hand the results back for the 12-cell table and the paired tests."
