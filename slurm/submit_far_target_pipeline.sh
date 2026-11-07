#!/bin/bash
# Diagnostic -> choose -> full target -> smoke -> 20k caption/empty -> full val at 16k
# -> frozen gate -> three-weather test of both arms -> mail.
# Every step is afterok, so a stop cancels what follows; the report still mails the reason.
# Usage: bash slurm/submit_far_target_pipeline.sh <diagnostic job id>
set -euo pipefail
diagnostic=${1:?Usage: submit_far_target_pipeline.sh <diagnostic job id>}
[[ "$diagnostic" =~ ^[0-9]+$ ]] || { echo 'Diagnostic job id must be numeric' >&2; exit 1; }
cd "${IRIS_REPO:-$HOME/Iris}"
F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_REPO="$PWD" IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
export IRIS_MS2_ROOT="$F/data/ms2" IRIS_MANIFEST_DIR="$F/manifests/sequence_level_internvl3_8b"
base="$IRIS_RUNS/outputs/lotus_line_v2"
summary="$base/iris_far_target_source_fitonly_v2/job_$diagnostic/summary.json"

# The tmax1 caption arm recipe from submit_arm.sh; only the pseudo target and NO_CAPTIONS differ.
export START=sd2 SEED=43 RANDOM_FLIP=1 PURE_PSEUDO=0
export NORMTYPE=log_truncnorm PREDICTION_TYPE=sample
export IRIS_THERMAL_STRETCH=percentile IRIS_TEXT_PADDING=legacy
export LAMBDA_IMAGE=0.1 LAMBDA_TAIL=0 TAIL_LOSS=none TAIL_ALIGN=none TAIL_HUBER_DELTA_M=5
export TRAIN_MANIFEST="$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl"
export VAL_MANIFEST="$IRIS_MANIFEST_DIR/ms2_val_official3_thermalcap_20260821.jsonl"
export EXTRA_TRAIN_ARGS="--truncnorm_max 1.0"
# Same 20000 updates as submit_arm.sh. Checkpoint frequency does not touch training;
# every 4000 steps still saves 16000 and 20000 and keeps two arms inside the scratch quota.
export MAX_TRAIN_STEPS=20000 CKPT_STEP=4000

for file in "$TRAIN_MANIFEST" "$VAL_MANIFEST" "$SCRATCH/baselines/supdepth4thermal/MS2_MD_BTS_THR_ckpt.ckpt" \
    "$SCRATCH/baselines/supdepth4thermal/MS2_MD_AdaBins_THR_ckpt.ckpt"; do
    [[ -s "$file" ]] || { echo "Missing $file" >&2; exit 1; }
done
fp=$(sha256sum "$VAL_MANIFEST" | cut -c1-8)
suffix=eval_eval_affine_invariant_log_space.json
reference="$IRIS_RUNS/eval/sd2_cap_pct_logtn_img_l0100_tmax1_s43_val_${fp}_stride1_correct_s16000/$suffix"
[[ -s "$reference" ]] || { echo "Missing $reference" >&2; exit 1; }
test_manifests=()
for condition in day3_common night3 rainy3; do
    file="$IRIS_MANIFEST_DIR/ms2_test_${condition}_thermalcap_20260821.jsonl"
    [[ -s "$file" ]] || { echo "Missing $file" >&2; exit 1; }
    test_manifests+=("$file")
done

after=()
state=$(squeue -h -j "$diagnostic" -o %T 2>/dev/null || true)
if [[ -n "$state" ]]; then
    after=(--dependency="afterok:$diagnostic")
elif [[ ! -s "$summary" ]]; then
    echo "Job $diagnostic is not queued and $summary is missing" >&2
    exit 1
fi

mkdir -p "$base/far_target_retrain_v1"
run=$(mktemp -d "$base/far_target_retrain_v1/run_XXXXXXXX")
tag=$(basename "$run")
tag=${tag#run_}
export PSEUDO_DIR="$run/target/calibrated_pseudo_depth"
echo "Output: $run"
job() {
    local id
    id=$(sbatch --parsable --kill-on-invalid-dep=yes "$@") || return 1
    id=${id%%;*}
    [[ "$id" =~ ^[0-9]+$ ]] || return 1
    echo "$id"
}
choose=$(job -J ftr_choose --time=00:10:00 --mem=1G -c 1 -o "$run/choose-%j.out" "${after[@]}" \
    --wrap="python3 tools/far_target_pipeline.py choose --summary '$summary' --out '$run/choice.json'")
echo "choose -> $choose (${after[*]:-diagnostic finished})" | tee -a "$run/jobs.txt"
build=$(job --dependency="afterok:$choose" --export=ALL slurm/far_target_build.sbatch "$run")
echo "target -> $build (afterok:$choose)" | tee -a "$run/jobs.txt"

ids=("$choose" "$build")
vals=()
results=()
runs=()
for arm in caption empty; do
    if [[ "$arm" == caption ]]; then
        export NO_CAPTIONS=0 SEL_PROMPT=correct
    else
        export NO_CAPTIONS=1 SEL_PROMPT=empty
    fi
    RUN="ftr_${tag}_${arm}_s43"
    export RUN_TAG="${RUN}_smoke" SMOKE=1
    smoke=$(job -J "ftr_sm_$arm" --time=00:45:00 --dependency="afterok:$build" --export=ALL slurm/iris_ms2.sbatch)
    export RUN_TAG="$RUN" SMOKE=0
    train=$(job -J "ftr_tr_$arm" --dependency="afterok:$smoke" --export=ALL slurm/iris_ms2.sbatch)
    export RUN STEPS=16000 FIXED_STEP=16000 VAL_ONLY=1 VAL_STRIDE=1
    export TEST_PROMPTS="$SEL_PROMPT" ALIGN_MODE=ssi_log ROUTE=b_thermal_unet
    val=$(job -J "ftr_val_$arm" --time=05:00:00 --dependency="afterok:$train" --export=ALL slurm/iris_ms2_pipeline.sbatch)
    echo "$arm smoke -> $smoke, train -> $train, val -> $val" | tee -a "$run/jobs.txt"
    ids+=("$smoke" "$train" "$val")
    vals+=("$val")
    results+=("$IRIS_RUNS/eval/${RUN}_val_${fp}_stride1_${SEL_PROMPT}_s16000/$suffix")
    runs+=("$RUN")
done
gate=$(job -J ftr_gate --time=00:10:00 --mem=1G -c 1 -o "$run/gate-%j.out" \
    --dependency="afterok:${vals[0]}:${vals[1]}" \
    --wrap="python3 tools/far_target_pipeline.py gate --reference '$reference' --caption '${results[0]}'")
echo "gate -> $gate (afterok:${vals[*]})" | tee -a "$run/jobs.txt"
ids+=("$gate")

tests=()
export VAL_ONLY=0
for index in 0 1; do
    export RUN="${runs[$index]}"
    if (( index == 0 )); then export SEL_PROMPT=correct; else export SEL_PROMPT=empty; fi
    export TEST_PROMPTS="$SEL_PROMPT"
    for manifest in "${test_manifests[@]}"; do
        export TEST_MANIFEST="$manifest"
        id=$(job -J "ftr_test_$SEL_PROMPT" --time=08:00:00 --dependency="afterok:$gate" \
            --export=ALL slurm/iris_ms2_pipeline.sbatch)
        tests+=("$id")
    done
done
echo "test (day, night, rain x caption, empty) -> ${tests[*]} (afterok:$gate)" | tee -a "$run/jobs.txt"
ids+=("${tests[@]}")
export REPORT_RUN="$run" REPORT_SUMMARY="$summary" REPORT_REFERENCE="$reference"
export REPORT_CAPTION="${results[0]}" REPORT_EMPTY="${results[1]}" REPORT_JOBS=$(IFS=,; echo "${ids[*]}")
export REPORT_CAPTION_TEST="$IRIS_RUNS/eval/${runs[0]}_test_{fp}_s16000_correct/${suffix%.json}_per_sample.csv"
export REPORT_EMPTY_TEST="$IRIS_RUNS/eval/${runs[1]}_test_{fp}_s16000_empty/${suffix%.json}_per_sample.csv"
# Plain sbatch on purpose: afterany must also fire when the chain above was cancelled.
waiting=$(IFS=:; echo "${tests[*]}")
report=$(sbatch --parsable -J ftr_report --time=00:30:00 --mem=4G -c 1 \
    --dependency="afterany:$waiting" --export=ALL slurm/far_target_report.sbatch)
report=${report%%;*}
echo "mail -> $report (afterany:$waiting)" | tee -a "$run/jobs.txt"
