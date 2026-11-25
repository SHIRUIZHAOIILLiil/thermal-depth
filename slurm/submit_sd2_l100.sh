#!/bin/bash
# The SD2-start post-hoc chain without tmax1 (docs/SD2_L100_POSTHOC_V1.md), for both arms of the pair:
#   caption arm  sd2_cap_pct_logtn_img_l0100_s43   (trained 2026-10, job 8214242; checkpoints on disk)
#   empty arm    sd2_nocap_pct_logtn_img_l0100_s43 (new: smoke -> train -> full val via submit_arm.sh)
# Each: checkpoint average (not_worse) -> decoder smoke -> decoder train + full val
#   -> reproduce val for start / selected -> test day/night/rain for both.
# Mail: the empty arm's smoke and training on their own, then one summary per arm.
set -euo pipefail
shopt -s inherit_errexit
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
export IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
MAIL_TO="${MAIL_TO:?set MAIL_TO}"
CAP=sd2_cap_pct_logtn_img_l0100_s43
NOCAP=sd2_nocap_pct_logtn_img_l0100_s43
base="$IRIS_RUNS/outputs/lotus_line_v2"

for f in "$base/newcrf_val_ssi_log_v1/job_9488948/summary.json" \
         "$base/iris_pair_rmse_val_v1/job_9167854/newcrf_moments.npz" \
         "$base/iris_far_pairs_l050_frame4k_v1/job_9163317/summary.json" \
         "$IRIS_RUNS/iris_ms2/$CAP/converted/step16000_weights.pt"; do
    [[ -e "$f" ]] || { echo "!! missing $f" >&2; exit 1; }
done
for s in 12000 13000 14000 15000 17000 18000 19000 20000; do
    [[ -d "$IRIS_RUNS/iris_ms2/$CAP/checkpoint-$s" ]] || { echo "!! missing $IRIS_RUNS/iris_ms2/$CAP/checkpoint-$s" >&2; exit 1; }
done
[[ ! -e "$IRIS_RUNS/iris_ms2/$NOCAP" ]] || { echo "!! $IRIS_RUNS/iris_ms2/$NOCAP already exists" >&2; exit 1; }
matches=()
for file in "$base"/iris_vae_decoder_only_v1/run_*/jobs.txt; do
    if [[ -f "$file" ]] && grep -qx 'val reproduction -> 9213840' "$file"; then
        matches+=("${file%/jobs.txt}")
    fi
done
[[ ${#matches[@]} == 1 ]] || { echo '!! expected exactly one q98 decoder run reproduced by 9213840' >&2; exit 1; }
comparison=${matches[0]}
python3 tools/check_contract_code.py "$comparison/contract.json"

mailafter() {
    sbatch --dependency=afterany:$1 -J mail_$1 -t 00:10:00 --mem=1G -c 1 -o /dev/null \
        --wrap="bash $IRIS_REPO/slurm/mail_summary.sh $MAIL_TO 'iris job $1' $1" >/dev/null
}

# posthoc <arm> <averaging output base> <decoder experiment> <job prefix> <dependency or "">
# Prints the job ids as "avg smoke train rs rd ts td" and writes <run dir>/jobs.txt.
posthoc() {
    local arm=$1 avg_base=$2 experiment=$3 prefix=$4 after=$5 avg smoke train rs rd ts td run
    avg=$(sbatch --parsable --kill-on-invalid-dep=yes -J iris_ckpt_avg --time=03:00:00 ${after:+--dependency=afterok:$after} \
        --export=ALL,SOURCE_RUN=$arm,SAVE_RULE=not_worse slurm/diagnose_checkpoint_average.sbatch)
    local avg_dir="$base/$avg_base/job_$avg"
    mkdir -p "$base/$experiment"
    run=$(mktemp -d "$base/$experiment/run_XXXXXXXX")
    local D="--kill-on-invalid-dep=yes --export=ALL,SOURCE_RUN=$arm"
    smoke=$(sbatch --parsable $D -J ${prefix}_sm --time=03:00:00 --dependency=afterok:$avg \
        slurm/iris_synth_decoder.sbatch smoke "$run" "$avg_dir" "$comparison")
    train=$(sbatch --parsable $D -J ${prefix}_tr --time=12:00:00 --dependency=afterok:$smoke \
        slurm/iris_synth_decoder.sbatch train "$run" "$avg_dir" "$comparison")
    local E="--kill-on-invalid-dep=yes --dependency=afterok:$train"
    rs=$(sbatch --parsable $E -J ${prefix}_rs --time=02:00:00 slurm/eval_stage_decoder.sbatch reproduce "$run" start)
    rd=$(sbatch --parsable $E -J ${prefix}_rd --time=02:00:00 slurm/eval_stage_decoder.sbatch reproduce "$run" selected)
    ts=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$rs -J ${prefix}_ts --time=03:00:00 --array=0-2 \
        -o "%x-%A_%a.out" -e "%x-%A_%a.err" slurm/eval_stage_decoder.sbatch test "$run" start)
    td=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$rd -J ${prefix}_td --time=03:00:00 --array=0-2 \
        -o "%x-%A_%a.out" -e "%x-%A_%a.err" slurm/eval_stage_decoder.sbatch test "$run" selected)
    { echo "arm $arm | run dir $run | averaging dir $avg_dir"; echo "average -> $avg"; echo "decoder smoke -> $smoke";
      echo "decoder train+val -> $train"; echo "reproduce start -> $rs | selected -> $rd"; echo "test start -> $ts | selected -> $td"; } > "$run/jobs.txt"
    cat "$run/jobs.txt" >&2
    echo "$avg $smoke $train $rs $rd $ts $td"
}

summary() {  # summary <subject> <job ids...>
    local subject=$1; shift
    local deps; deps=$(IFS=:; echo "$*")
    sbatch --dependency=afterany:$deps -J mail_l100_summary -t 00:10:00 --mem=1G -c 1 -o /dev/null \
        --wrap="bash $IRIS_REPO/slurm/mail_summary.sh $MAIL_TO '$subject' $*" >/dev/null
}

# 1. Caption arm: its checkpoints already exist, so the chain starts now.
cap_jobs=$(posthoc $CAP iris_checkpoint_average_l100_v1 iris_l100_avg_decoder_v1 iris_l1ad "")
summary 'iris l100 caption summary' $cap_jobs

# 2. Empty arm: same l100 recipe, captions off. --truncnorm_max 0.98 restates the l100 default
#    (1 - truncnorm_min) because submit_arm.sh needs a non-empty argument; it changes nothing.
out=$(NO_CAPTIONS=1 bash slurm/submit_arm.sh "$NOCAP" "--truncnorm_max 0.98")
echo "$out"
SM=$(echo "$out" | sed -n 's/^smoke -> \([0-9]*\).*/\1/p')
TR=$(echo "$out" | sed -n 's/^train -> \([0-9]*\).*/\1/p')
VA=$(echo "$out" | sed -n 's/^val   -> \([0-9]*\).*/\1/p')
[[ "$SM" =~ ^[0-9]+$ && "$TR" =~ ^[0-9]+$ && "$VA" =~ ^[0-9]+$ ]] || { echo '!! could not read the arm job ids' >&2; exit 1; }
mailafter $SM
mailafter $TR
nocap_jobs=$(posthoc $NOCAP iris_checkpoint_average_l100_nocap_v1 iris_l100_nocap_avg_decoder_v1 iris_n0ad "$VA")
summary 'iris l100 nocap summary' $VA $nocap_jobs
echo "caption arm jobs: $cap_jobs"
echo "empty arm: smoke $SM train $TR val $VA | post-hoc $nocap_jobs"
echo "mail: empty-arm smoke, empty-arm train, one summary per arm"
