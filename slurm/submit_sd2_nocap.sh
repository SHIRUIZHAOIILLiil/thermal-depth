#!/bin/bash
# The empty-caption twin of the SD2-start final model (docs/SD2_POSTHOC_V1.md):
#   tmax1 recipe with NO_CAPTIONS=1 (smoke -> train -> full val, submit_arm.sh)
#   -> checkpoint average (not_worse) -> decoder smoke -> decoder train + full val
#   -> reproduce val for both rows -> test day/night/rain for both. Empty prompt everywhere.
# Every job gets an afterany mail to $MAIL_TO.
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
export IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
MAIL_TO="${MAIL_TO:?set MAIL_TO}"
RUN=sd2_nocap_pct_logtn_img_l0100_tmax1_s43
base="$IRIS_RUNS/outputs/lotus_line_v2"

for f in "$base/newcrf_val_ssi_log_v1/job_9488948/summary.json" \
         "$base/iris_pair_rmse_val_v1/job_9167854/newcrf_moments.npz" \
         "$base/iris_far_pairs_l050_frame4k_v1/job_9163317/summary.json"; do
    [[ -e "$f" ]] || { echo "!! missing $f" >&2; exit 1; }
done
[[ ! -e "$IRIS_RUNS/iris_ms2/$RUN" ]] || { echo "!! $IRIS_RUNS/iris_ms2/$RUN already exists" >&2; exit 1; }
matches=()
for file in "$base"/iris_vae_decoder_only_v1/run_*/jobs.txt; do
    if [[ -f "$file" ]] && grep -qx 'val reproduction -> 9213840' "$file"; then
        matches+=("${file%/jobs.txt}")
    fi
done
[[ ${#matches[@]} == 1 ]] || { echo '!! expected exactly one q98 decoder run reproduced by 9213840' >&2; exit 1; }
comparison=${matches[0]}

mailafter() {
    sbatch --dependency=afterany:$1 -J mail_$1 -t 00:10:00 --mem=1G -c 1 -o /dev/null --wrap="{ sacct -X -j $1 -o JobID,JobName%30,State,Elapsed,ExitCode; for f in $IRIS_REPO/*-$1.out $IRIS_REPO/*-$1.err $IRIS_REPO/*-$1_*.out $IRIS_REPO/*-$1_*.err; do [[ -f \"\$f\" ]] || continue; echo \"--- \$f\"; grep -h -E '^\[(control|row|gate|use_average|reproduced|selected|verdict|test-official|test-full)\]|captions  =|affine_invariant_log_space: abs_rel' \"\$f\" || true; tail -c 3000 \"\$f\" | tr '\r' '\n' | tail -n 15; done; } 2>&1 | mail -s \"iris job $1\" $MAIL_TO" >/dev/null
}

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
avg_dir="$base/iris_checkpoint_average_tmax1_nocap_v1/job_$avg"
mkdir -p "$base/iris_tmax1_nocap_avg_decoder_v1"
run=$(mktemp -d "$base/iris_tmax1_nocap_avg_decoder_v1/run_XXXXXXXX")
D="--kill-on-invalid-dep=yes --export=ALL,SOURCE_RUN=$RUN"
smoke=$(sbatch --parsable $D -J iris_n1ad_sm --time=03:00:00 --dependency=afterok:$avg \
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
{ echo "arm smoke -> $SM | train -> $TR | val -> $VA"; echo "average -> $avg"; echo "decoder smoke -> $smoke";
  echo "decoder train+val -> $train"; echo "reproduce start -> $rs | selected -> $rd"; echo "test start -> $ts | selected -> $td"; } | tee "$run/jobs.txt"
# Three mails: the arm's smoke and its long training (submit_arm.sh jobs stall silently on a failed
# dependency, so these two must mail on their own), then one summary after everything else ends.
# Every post-hoc job uses --kill-on-invalid-dep, so a failure anywhere still lets the summary fire.
mailafter $SM
mailafter $TR
rest="$VA:$avg:$smoke:$train:$rs:$rd:$ts:$td"
sbatch --dependency=afterany:$rest -J mail_nocap_summary -t 00:10:00 --mem=1G -c 1 -o /dev/null --wrap="{ sacct -X -j ${rest//:/,} -o JobID,JobName%30,State,Elapsed,ExitCode; for j in ${rest//:/ }; do for f in $IRIS_REPO/*-\$j.out $IRIS_REPO/*-\${j}_*.out; do [[ -f \"\$f\" ]] || continue; echo \"--- \$f\"; grep -h -E '^\[(control|row|use_average|reproduced|selected|verdict|test-official)\]|affine_invariant_log_space: abs_rel' \"\$f\" | sort -u || true; done; done; } 2>&1 | mail -s 'iris nocap twin summary' $MAIL_TO" >/dev/null
echo "run dir: $run | averaging dir: $avg_dir | mail: arm smoke, arm train, one summary at the end"
