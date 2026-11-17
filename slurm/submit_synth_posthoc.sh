#!/bin/bash
# Synth arm post-hoc chain (docs/SYNTH_POSTHOC_V1.md):
#   checkpoint average (12k-20k, 96+96 frames) -> decoder smoke (full val start) -> decoder train + full val.
# Every job gets an afterany mail to $MAIL_TO. No U-Net training, no test jobs.
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
export IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
MAIL_TO="${MAIL_TO:?set MAIL_TO}"
RUN=sd2_cap_pct_logtn_img_l0100_tmax1_synth_s43
SRC="$IRIS_RUNS/iris_ms2/$RUN"
base="$IRIS_RUNS/outputs/lotus_line_v2"

# Everything the chain reads, checked before anything is queued.
need=("$SRC/converted/step16000_weights.pt"
      "$base/newcrf_val_ssi_log_v1/job_9488948/summary.json"
      "$base/iris_pair_rmse_val_v1/job_9167854/newcrf_moments.npz"
      "$base/iris_far_pairs_l050_frame4k_v1/job_9163317/summary.json")
for s in 12000 13000 14000 15000 17000 18000 19000 20000; do need+=("$SRC/checkpoint-$s"); done
for f in "${need[@]}"; do [[ -e "$f" ]] || { echo "!! missing $f" >&2; exit 1; }; done
matches=()
for file in "$base"/iris_vae_decoder_only_v1/run_*/jobs.txt; do
    if [[ -f "$file" ]] && grep -qx 'val reproduction -> 9213840' "$file"; then
        matches+=("${file%/jobs.txt}")
    fi
done
[[ ${#matches[@]} == 1 ]] || { echo '!! expected exactly one q98 decoder run reproduced by 9213840' >&2; exit 1; }
comparison=${matches[0]}

mailafter() {
    sbatch --dependency=afterany:$1 -J mail_$1 -t 00:10:00 --mem=1G -c 1 -o /dev/null --wrap="{ sacct -X -j $1 -o JobID,JobName%30,State,Elapsed,ExitCode; for f in $IRIS_REPO/*-$1.out $IRIS_REPO/*-$1.err; do echo \"--- \$f\"; grep -h -E '^\[(control|row|gate|use_average|reproduced|selected|verdict|vs-q98-decoder)\]' \"\$f\" || true; tail -c 4000 \"\$f\" | tr '\r' '\n' | tail -n 25; done; } 2>&1 | mail -s \"iris job $1\" $MAIL_TO" >/dev/null
}

avg=$(sbatch --parsable --kill-on-invalid-dep=yes -J iris_ckpt_avg --time=03:00:00 \
    --export=ALL,SOURCE_RUN=$RUN slurm/diagnose_checkpoint_average.sbatch)
avg_dir="$base/iris_checkpoint_average_synth_v1/job_$avg"
mkdir -p "$base/iris_synth_decoder_v1"
run=$(mktemp -d "$base/iris_synth_decoder_v1/run_XXXXXXXX")
smoke=$(sbatch --parsable --kill-on-invalid-dep=yes -J iris_syvd_sm --time=03:00:00 \
    --dependency=afterok:$avg slurm/iris_synth_decoder.sbatch smoke "$run" "$avg_dir" "$comparison")
train=$(sbatch --parsable --kill-on-invalid-dep=yes -J iris_syvd_tr --time=12:00:00 \
    --dependency=afterok:$smoke slurm/iris_synth_decoder.sbatch train "$run" "$avg_dir" "$comparison")
{ echo "average -> $avg"; echo "decoder smoke -> $smoke (afterok:$avg)"; echo "decoder train+val -> $train (afterok:$smoke)"; } | tee "$run/jobs.txt"
for j in $avg $smoke $train; do mailafter $j; done
echo "run dir: $run | averaging dir: $avg_dir | comparison: $comparison | mail after all three"
