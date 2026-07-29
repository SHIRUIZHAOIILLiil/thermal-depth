#!/usr/bin/env bash
# Caption 2x2 on the RGB-teacher model (experiment-5 recipe, RGBDT500).
#
#                     inference: empty        inference: correct
#   train empty       cell 1 (DONE, 0.2595)   cell 2 (injection value)
#   train caption     cell 3 (dependency)     cell 4 (train+use total)
#
# Cell 1 already exists (outputs/lotus_line_v2/rgbdt500_eval_rgbteacher_l1),
# so this queue fills the other three. Cell 2 needs no training and runs first,
# so a signal arrives early even if a later stage fails.
#
# Reference point: on the WEAK model (0.3616) the same cells gave injection
# +0.0019*, caption-training net +0.0054*, total +0.0073*. The open question is
# whether those shrink now that the model is much stronger.
#
# CAVEAT (do not lose this when reporting): this model's dense teacher is
# RGB-derived and the captions are also RGB-derived, so a SHRUNK effect here is
# ambiguous -- it could be model strength OR the caption being redundant with
# information the teacher already supplied. A POSITIVE effect is unambiguous.
# Disentangling a null needs a same-recipe/thermal-teacher control.
#
# Usage: nohup bash tools/run_rgbdt500_rgbteacher_caption_queue.sh > outputs/lotus_line_v2/rgbdt500_rgbteacher_caption_queue.log 2>&1 &
# Idempotent: stages whose output already exists are skipped.

set -u
cd "$(dirname "$0")/.."

ROOT_TRAIN=/mnt/e/dataset/RGBDT500/clean_train
ROOT_TEST=/mnt/e/dataset/RGBDT500/clean_test
MTRAIN=$ROOT_TRAIN/rgbdt500_train_manifest_iris_prose.jsonl
MTEST=$ROOT_TEST/rgbdt500_test_manifest_iris_prose.jsonl
OUT=outputs/lotus_line_v2
TEACHER=$OUT/rgbdt500_rgb_teacher_train/teacher_disparity
CKPT_EMPTY=$OUT/full_rgbdt500_rgbteacher_l1/arm6_end.pt
CKPT_CAPT=$OUT/full_rgbdt500_rgbteacher_l1_caption/arm6_end.pt

TRAIN_COMMON="--gt-decode-fp32 --train-manifest $MTRAIN --ms2-root $ROOT_TRAIN --depth-scale 1000 --gt-max-depth 20 --min-gt-valid-fraction 0.10 --response-weight 0 --dense-teacher-dir $TEACHER --dense-teacher-weight 1.0 --dense-teacher-align l1"
EVAL_COMMON="--manifest $MTEST --ms2-root $ROOT_TEST --max-samples 0 --condition-posterior mode --depth-scale 1000 --min-depth 0.1 --max-depth 20 --save-raw-pred"
RESCORE_COMMON="--data-root $ROOT_TEST --route thermal-unet --align ssi_disparity --min-depth 0.1 --max-depth 20 --depth-scale 1000 --min-gt-valid-fraction 0.10"

for path in "$MTRAIN" "$MTEST" "$CKPT_EMPTY" "$TEACHER"; do
    if [ ! -e "$path" ]; then echo "FATAL: missing prerequisite $path"; exit 1; fi
done

declare -a RESULTS
say() { echo; echo "=== $* ($(date +%H:%M:%S))"; }

run_eval() {   # $1 = output name, $2 = checkpoint, $3 = caption mode
    local dir="$OUT/$1"
    if [ ! -f "$dir/official_run_metadata.json" ]; then
        say "$1: inference"
        if ! python tools/run_ms2_lotus_thermal_vae_official.py $EVAL_COMMON \
             --unet-checkpoint "$2" --caption-mode "$3" --output-dir "$dir" > "$dir.log" 2>&1; then
            echo "FAILED (last lines):"; tail -5 "$dir.log"; RESULTS+=("$1 INFER_FAILED"); return 1
        fi
    else
        say "$1: inference already done, skipping"
    fi
    local official="outputs/ms2_official/${1}_mind01"
    if [ ! -f "$official/metrics/summary.json" ]; then
        say "$1: official rescore"
        if ! python tools/run_official_ms2_evaluation.py --manifest "$dir/selected_manifest.jsonl" \
             --prediction-dir "$dir/raw_predictions" $RESCORE_COMMON \
             --output-dir "$official" > "$official.log" 2>&1; then
            echo "RESCORE FAILED:"; tail -5 "$official.log"; RESULTS+=("$1 RESCORE_FAILED"); return 1
        fi
    fi
    RESULTS+=("$1 $(python3 -c "
import json
s = json.load(open('$official/metrics/summary.json'))['image_wise']['statistics']
print(f\"abs_rel={s['abs_rel']['mean']:.4f} d1={s['a1']['mean']:.4f}\")")")
}

# --- stage 0: smoke the caption training path (free, no checkpoints written) ---
if [ ! -f "$CKPT_CAPT" ] && [ ! -f "$OUT/smoke_rgbdt500_rgbteacher_capt/smoke_summary.json" ]; then
    say "stage 0/5  smoke caption training (5 updates)"
    if ! python tools/train_ms2_thermal_vae_unet_gt.py $TRAIN_COMMON --caption-mode correct \
         --output-dir "$OUT/smoke_rgbdt500_rgbteacher_capt" --smoke-updates 5 \
         > "$OUT/smoke_rgbdt500_rgbteacher_capt.log" 2>&1; then
        echo "SMOKE FAILED -- stopping before anything expensive:"
        tail -15 "$OUT/smoke_rgbdt500_rgbteacher_capt.log"
        exit 1
    fi
    echo "smoke OK"
fi

# --- stage 1: cell 2, needs no training, gives the earliest signal ---
say "stage 1/5  cell 2: empty-trained x caption-injected"
run_eval "rgbdt500_eval_rgbteacher_l1_capt" "$CKPT_EMPTY" "correct"

# --- stage 2: the caption-trained twin ---
if [ ! -f "$CKPT_CAPT" ]; then
    say "stage 2/5  caption-trained twin (~30 min)"
    if ! python tools/train_ms2_thermal_vae_unet_gt.py $TRAIN_COMMON --caption-mode correct \
         --output-dir "$OUT/full_rgbdt500_rgbteacher_l1_caption" \
         > "$OUT/full_rgbdt500_rgbteacher_l1_caption.log" 2>&1; then
        echo "TRAINING FAILED (last lines):"; tail -10 "$OUT/full_rgbdt500_rgbteacher_l1_caption.log"
        RESULTS+=("caption_training FAILED")
        printf '%s\n' "${RESULTS[@]}"; exit 1
    fi
else
    say "stage 2/5  caption twin already trained, skipping"
fi

# --- stages 3-4: the two caption-trained cells ---
say "stage 3/5  cell 3: caption-trained x empty inference (the dependency test)"
run_eval "rgbdt500_eval_rgbteacher_l1_capttrain_empty" "$CKPT_CAPT" "empty"

say "stage 4/5  cell 4: caption-trained x caption-injected"
run_eval "rgbdt500_eval_rgbteacher_l1_capttrain_correct" "$CKPT_CAPT" "correct"

# --- stage 5: report ---
say "stage 5/5  summary"
echo "cell 1 (empty x empty) = abs_rel=0.2595 d1=0.6644   [previously measured]"
printf '%s\n' "${RESULTS[@]}"
echo
echo "Paired significance tests are NOT done here -- mean differences on this"
echo "dataset have been overturned by paired CIs before. Hand the per_image.csv"
echo "files under outputs/ms2_official/rgbdt500_eval_rgbteacher_l1*_mind01/ to"
echo "the paired analysis before drawing any conclusion."
echo
echo "Disk: after this completes, the intermediate arm6_step_*.pt under"
echo "$OUT/full_rgbdt500_rgbteacher_l1_caption/ can be deleted (keep arm6_end.pt)."
