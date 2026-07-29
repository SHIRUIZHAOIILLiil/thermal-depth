#!/usr/bin/env bash
# Task-4 training queue: C-prime arms (dense AnyThermal teacher, no response
# anchor) plus the caption-trained twins of every unanchored arm.
#
#   1. smoke  C'      (ssi teacher, resp=0)          -- 5 updates
#   2. smoke  C''     (l1 teacher, resp=0)           -- 5 updates, new l1 path
#   3. full   C'      ssi teacher + GT               -- ~1h
#   4. full   C''     l1  teacher + GT               -- ~1h
#   5. full   B_capt  pure GT      + caption train   -- ~1h
#   6. full   C'_capt ssi teacher  + caption train   -- ~1h
#   7. full   C''_capt l1 teacher  + caption train   -- ~1h
#
# Usage: nohup bash tools/run_task4_training_queue.sh > outputs/lotus_line_v2/task4_train_queue.log 2>&1 &
# Idempotent: stages whose summary file exists are skipped.

set -u
cd "$(dirname "$0")/.."

MTRAIN=/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/ms2_fixed_sequence_internvl3_8b_filtered_caption_train_rgb_depth_v1_clip75_rerun_20260714.jsonl
OUT=outputs/lotus_line_v2
TEACH=$OUT/anythermal_teacher_train_full/teacher_disparity
TRAIN="python tools/train_ms2_thermal_vae_unet_gt.py --gt-decode-fp32 --train-manifest $MTRAIN"

declare -a NAMES DIRS ARGS DONEFILES
NAMES+=("smoke_cprime");     DIRS+=("smoke_thermal_unet_cprime");            ARGS+=("--response-weight 0 --dense-teacher-dir $TEACH --dense-teacher-weight 1.0 --smoke-updates 5");                                DONEFILES+=("smoke_summary.json")
NAMES+=("smoke_cprime_l1");  DIRS+=("smoke_thermal_unet_cprime_l1");         ARGS+=("--response-weight 0 --dense-teacher-dir $TEACH --dense-teacher-weight 1.0 --dense-teacher-align l1 --smoke-updates 5");        DONEFILES+=("smoke_summary.json")
NAMES+=("full_cprime");      DIRS+=("full_train_epoch1_thermal_unet_cprime");    ARGS+=("--response-weight 0 --dense-teacher-dir $TEACH --dense-teacher-weight 1.0");                                              DONEFILES+=("summary.json")
NAMES+=("full_cprime_l1");   DIRS+=("full_train_epoch1_thermal_unet_cprime_l1"); ARGS+=("--response-weight 0 --dense-teacher-dir $TEACH --dense-teacher-weight 1.0 --dense-teacher-align l1");                    DONEFILES+=("summary.json")
NAMES+=("full_B_caption");   DIRS+=("full_train_epoch1_thermal_unet_gt_only_caption");   ARGS+=("--response-weight 0 --caption-mode correct");                                                                     DONEFILES+=("summary.json")
NAMES+=("full_cprime_caption");    DIRS+=("full_train_epoch1_thermal_unet_cprime_caption");    ARGS+=("--response-weight 0 --dense-teacher-dir $TEACH --dense-teacher-weight 1.0 --caption-mode correct");         DONEFILES+=("summary.json")
NAMES+=("full_cprime_l1_caption"); DIRS+=("full_train_epoch1_thermal_unet_cprime_l1_caption"); ARGS+=("--response-weight 0 --dense-teacher-dir $TEACH --dense-teacher-weight 1.0 --dense-teacher-align l1 --caption-mode correct"); DONEFILES+=("summary.json")

declare -a RESULTS
for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"; dir="$OUT/${DIRS[$i]}"
    if [ -f "$dir/${DONEFILES[$i]}" ]; then
        echo "=== [$((i+1))/7] $name: already complete, skipping"
        RESULTS+=("$name SKIPPED(done)")
        continue
    fi
    echo "=== [$((i+1))/7] $name: starting $(date +%H:%M:%S)"
    if $TRAIN ${ARGS[$i]} --output-dir "$dir" > "$dir.log" 2>&1; then
        echo "=== [$((i+1))/7] $name: OK"
        RESULTS+=("$name OK")
    else
        echo "=== [$((i+1))/7] $name: FAILED (last lines below)"
        tail -5 "$dir.log"
        RESULTS+=("$name FAILED")
    fi
done

echo
echo "================ task-4 training queue summary ================"
for line in "${RESULTS[@]}"; do echo "  $line"; done
echo "Next: run the evaluation queue (Claude will extend run_task4_eval_queue.sh)."
