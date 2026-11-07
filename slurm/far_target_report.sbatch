#!/bin/bash
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# Runs after the chain ends or stops; it must mail even when nothing else finished.
tag=$(basename "${REPORT_RUN:?}")
{
    F="${SCRATCH:?}/from_sc23sz"
    export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
    export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
    source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
    cd "$IRIS_REPO"
    cat "$REPORT_RUN/jobs.txt"
    python tools/far_target_pipeline.py report --choice "$REPORT_RUN/choice.json" --summary "$REPORT_SUMMARY" \
        --reference "$REPORT_REFERENCE" --caption "$REPORT_CAPTION" --empty "$REPORT_EMPTY"
    python tools/full_metric_table.py --ms2-root "$IRIS_MS2_ROOT" \
        --model "caption=$REPORT_CAPTION_TEST" --model "empty=$REPORT_EMPTY_TEST"
    sacct -j "$REPORT_JOBS" -X --format=JobID,JobName%16,State,Elapsed,ExitCode
    grep -h truncnorm_max ftr_sm_caption-*.err ftr_sm_empty-*.err 2>/dev/null | head -n 4
    du -sh "$REPORT_RUN/raw_predictions" "$REPORT_RUN/target" 2>/dev/null
} 2>&1 | mail -s "far target retrain $tag" anmolidawn@gmail.com
