#!/bin/bash
# The empty-caption twin of the SD2-start final model (docs/SD2_POSTHOC_V1.md):
#   tmax1 recipe with NO_CAPTIONS=1 (smoke -> train -> full val, submit_arm.sh)
#   -> checkpoint average (not_worse) -> decoder smoke -> decoder train + full val
#   -> reproduce val for both rows -> test day/night/rain for both. Empty prompt everywhere.
# Resume: AVG_JOB=<finished averaging job id> skips the arm and the averaging and starts at the
# decoder smoke (first used after 9622938 failed on a hash-pinned file, 2026-11-25).
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
export IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
MAIL_TO="${MAIL_TO:?set MAIL_TO}"
AVG_JOB="${AVG_JOB:-}"
RUN=sd2_nocap_pct_logtn_img_l0100_tmax1_s43
base="$IRIS_RUNS/outputs/lotus_line_v2"

for f in "$base/newcrf_val_ssi_log_v1/job_9488948/summary.json" \
         "$base/iris_pair_rmse_val_v1/job_9167854/newcrf_moments.npz" \
         "$base/iris_far_pairs_l050_frame4k_v1/job_9163317/summary.json"; do
    [[ -e "$f" ]] || { echo "!! missing $f" >&2; exit 1; }
done
matches=()
for file in "$base"/iris_vae_decoder_only_v1/run_*/jobs.txt; do
    if [[ -f "$file" ]] && grep -qx 'val reproduction -> 9213840' "$file"; then
        matches+=("${file%/jobs.txt}")
    fi
done
[[ ${#matches[@]} == 1 ]] || { echo '!! expected exactly one q98 decoder run reproduced by 9213840' >&2; exit 1; }
comparison=${matches[0]}
# The decoder stage re-checks the comparator's pinned code hashes; check them before queueing anything.
python3 tools/check_contract_code.py "$comparison/contract.json"

mailafter() {
    sbatch --dependency=afterany:$1 -J mail_$1 -t 00:10:00 --mem=1G -c 1 -o /dev/null \
        --wrap="bash $IRIS_REPO/slurm/mail_summary.sh $MAIL_TO 'iris job $1' $1" >/dev/null
}

if [[ -z "$AVG_JOB" ]]; then
    [[ ! -e "$IRIS_RUNS/iris_ms2/$RUN" ]] || { echo "!! $IRIS_RUNS/iris_ms2/$RUN already exists" >&2; exit 1; }
    # 1. The arm itself: same recipe as sd2_cap_pct_logtn_img_l0100_tmax1_s43, captions off.
    out=$(NO_CAPTIONS=1 bash slurm/submit_arm.sh "$RUN" "--truncnorm_max 1.0")
    echo "$out"
    SM=$(echo "$out" | sed -n 's/^smoke -> \([0-9]*\).*/\1/p')
    TR=$(echo "$out" | sed -n 's/^train -> \([0-9]*\).*/\1/p')
    VA=$(echo "$out" | sed -n 's/^val   -> \([0-9]*\).*/\1/p')
    [[ "$SM" =~ ^[0-9]+$ && "$TR" =~ ^[0-9]+$ && "$VA" =~ ^[0-9]+$ ]] || { echo '!! could not read the arm job ids' >&2; exit 1; }
    # 2. Post-hoc chain, after the arm's val job has converted step 16000.
    avg=$(sbatch --parsable --kill-on-invalid-dep=yes -J iris_ckpt_avg --time=03:00:00 --dependency=afterok:$VA \
        --export=ALL,SOURCE_RUN=$RUN,SAVE_RULE=not_worse slurm/diagnose_checkpoint_average.sbatch)
    after_avg="--dependency=afterok:$avg"
    head_jobs="$VA:$avg"
else
    [[ "$AVG_JOB" =~ ^[0-9]+$ ]] || { echo "!! AVG_JOB must be a job id" >&2; exit 1; }
    avg=$AVG_JOB
    summary="$base/iris_checkpoint_average_tmax1_nocap_v1/job_$avg/summary.json"
    python3 -c 'import json, sys; s = json.load(open(sys.argv[1])); sys.exit(not (s["source_run"] == sys.argv[2] and s["control"]["passed"]))' \
        "$summary" "$RUN" || { echo "!! $summary is not a passed $RUN averaging job" >&2; exit 1; }
    after_avg=""
    head_jobs=""
fi
avg_dir="$base/iris_checkpoint_average_tmax1_nocap_v1/job_$avg"
mkdir -p "$base/iris_tmax1_nocap_avg_decoder_v1"
run=$(mktemp -d "$base/iris_tmax1_nocap_avg_decoder_v1/run_XXXXXXXX")
D="--kill-on-invalid-dep=yes --export=ALL,SOURCE_RUN=$RUN"
smoke=$(sbatch --parsable $D -J iris_n1ad_sm --time=03:00:00 $after_avg \
    slurm/iris_synth_decoder.sbatch smoke "$run" "$avg_dir" "$comparison")
train=$(sbatch --parsable $D -J iris_n1ad_tr --time=12:00:00 --dependency=afterok:$smoke \
    slurm/iris_synth_decoder.sbatch train "$run" "$avg_dir" "$comparison")
E="--kill-on-invalid-dep=yes --dependency=afterok:$train"
rs=$(sbatch --parsable $E -J iris_n1ad_rs --time=02:00:00 slurm/eval_stage_decoder.sbatch reproduce "$run" start)
rd=$(sbatch --parsable $E -J iris_n1ad_rd --time=02:00:00 slurm/eval_stage_decoder.sbatch reproduce "$run" selected)
ts=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$rs -J iris_n1ad_ts --time=03:00:00 --array=0-2 \
    -o "%x-%A_%a.out" -e "%x-%A_%a.err" slurm/eval_stage_decoder.sbatch test "$run" start)
td=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$rd -J iris_n1ad_td --time=03:00:00 --array=0-2 \
    -o "%x-%A_%a.out" -e "%x-%A_%a.err" slurm/eval_stage_decoder.sbatch test "$run" selected)
{ [[ -z "$AVG_JOB" ]] && echo "arm smoke -> $SM | train -> $TR | val -> $VA"; echo "average -> $avg"; echo "decoder smoke -> $smoke";
  echo "decoder train+val -> $train"; echo "reproduce start -> $rs | selected -> $rd"; echo "test start -> $ts | selected -> $td"; } | tee "$run/jobs.txt"
# The arm's smoke and long training mail on their own (submit_arm.sh jobs stall silently on a failed
# dependency); everything else goes into one summary. Every post-hoc job uses --kill-on-invalid-dep,
# so a failure anywhere still lets the summary fire, with the failing job's traceback in it.
if [[ -z "$AVG_JOB" ]]; then mailafter $SM; mailafter $TR; fi
rest="${head_jobs:+$head_jobs:}$smoke:$train:$rs:$rd:$ts:$td"
sbatch --dependency=afterany:$rest -J mail_nocap_summary -t 00:10:00 --mem=1G -c 1 -o /dev/null \
    --wrap="bash $IRIS_REPO/slurm/mail_summary.sh $MAIL_TO 'iris nocap twin summary' ${rest//:/ }" >/dev/null
echo "run dir: $run | averaging dir: $avg_dir | summary mail after: ${rest//:/ }"
