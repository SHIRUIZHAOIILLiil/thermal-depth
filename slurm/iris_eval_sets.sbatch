#!/bin/bash
#SBATCH --job-name=iris_evalsets
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# Iris's five zero-shot depth test sets: data, captions, and Iris's own evaluation
# (docs/IRIS_SYNTH_PRETRAIN_V1.md, "复现 Iris 零样本结果").
#   STEP=probe             HEAD the five Marigold tars (size, status)
#   STEP=download          fetch them (resumable), then check every listed image is in its tar
#   STEP=caption           (GPU, array, COUNT=<shards>) tools/caption_eval_sets.py shard
#   STEP=caption-finalize  COUNT=<shards>; tests, then the five <set>_depth_descriptions.json
#   STEP=eval              (GPU) lotus/eval_rng.py as eval-depth-g.sh calls it, plus
#                          --strict_text_mode; CKPT=<pipeline dir or repo>, TAG=<output name>
set -euo pipefail
F="${SCRATCH:?}/from_sc23sz"
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
EV="${EVAL_ROOT:-$SCRATCH/data/iris_eval}"
CAP="$EV/captions"
URL=https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset
TARS=(nyuv2/nyu_labeled_extracted.tar kitti/kitti_eigen_split_test.tar eth3d/eth3d.tar
      scannet/scannet_val_sampled_800_1.tar diode/diode_val.tar)

case "${STEP:?STEP=probe|download|caption|caption-finalize|eval}" in
probe)
  for t in "${TARS[@]}"; do
    h=$(curl -sIL "$URL/$t" | tr -d '\r')
    code=$(echo "$h" | awk '/^HTTP/{c=$2} END{print c}')
    size=$(echo "$h" | awk 'tolower($1)=="content-length:"{s=$2} END{print s+0}')
    echo "[probe] $t: HTTP $code, $(python -c "print(f'{$size/2**30:.2f}')") GiB"
  done
  ;;
download)
  for t in "${TARS[@]}"; do
    mkdir -p "$EV/depth/$(dirname "$t")"
    curl -fsSL --retry 10 --retry-delay 30 -C - -o "$EV/depth/$t" "$URL/$t"
    echo "[download] $t $(du -h "$EV/depth/$t" | cut -f1)"
  done
  python -u tools/caption_eval_sets.py check --data "$EV"
  ;;
caption)
  conda activate caption
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  for attempt in $(seq 1 20); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
    (( free >= 20000 )) && break
    echo "[wait] GPU free ${free} MiB"; sleep 30
  done
  (( free >= 20000 )) || { echo "GPU free ${free} MiB after 10 min"; exit 1; }
  python -u tools/caption_eval_sets.py shard --data "$EV" --out "$CAP" \
    --count "${COUNT:?}" --index "${SLURM_ARRAY_TASK_ID:-0}"
  ;;
caption-finalize)
  python tests/test_caption_eval_sets.py
  python -u tools/caption_eval_sets.py finalize --data "$EV" --out "$CAP" --count "${COUNT:?}"
  ;;
eval)
  CKPT="${CKPT:?pipeline directory or HF repo}"
  OUT="$IRIS_RUNS/iris_zero_shot/${TAG:?}"
  mkdir -p "$OUT"
  echo "ckpt=$CKPT  out=$OUT"
  cd lotus
  git -C "$IRIS_REPO" log --oneline -1
  # CAPTIONS=0: empty prompts, how Lotus-G itself is evaluated -- a check of this
  # pipeline against Lotus's published numbers.
  TEXT=()
  if [[ "${CAPTIONS:-1}" == "1" ]]; then
    for s in nyu kitti scannet eth3d diode; do
      [[ -s "$CAP/${s}_depth_descriptions.json" ]] || { echo "missing $CAP/${s}_depth_descriptions.json"; exit 1; }
    done
    TEXT=(--nyuv2_text_descriptions_path="$CAP/nyu_depth_descriptions.json"
          --kitti_text_descriptions_path="$CAP/kitti_depth_descriptions.json"
          --scannet_text_descriptions_path="$CAP/scannet_depth_descriptions.json"
          --eth3d_text_descriptions_path="$CAP/eth3d_depth_descriptions.json"
          --diode_text_descriptions_path="$CAP/diode_depth_descriptions.json"
          --strict_text_mode)
  fi
  # RNG_STATE=<lotus/rng_states/*.pkl>: the noise draws Lotus ships to reproduce its paper
  # numbers exactly (Lotus README, "Optional"). Only meaningful for Lotus's own weights.
  [[ -n "${RNG_STATE:-}" ]] && TEXT+=(--rng_state_path="$IRIS_REPO/lotus/$RNG_STATE")
  # EVAL_DATASETS="eth3d": rerun only those sets into the same output directory (the other
  # sets' eval_metrics files stay). ETH3D's 6048x4032 frames OOM-killed both 32G runs.
  [[ -n "${EVAL_DATASETS:-}" ]] && TEXT+=(--eval_datasets $EVAL_DATASETS)
  echo "captions=${CAPTIONS:-1}  rng_state=${RNG_STATE:-<none>}  datasets=${EVAL_DATASETS:-all}"
  # Without captions eval_rng.py prints one "No text description..." line per image.
  python -u eval_rng.py \
    --pretrained_model_name_or_path="$CKPT" \
    --output_dir="$OUT" \
    --seed=42 \
    --mode=generation \
    --task_name=depth \
    "${TEXT[@]}" \
    --base_test_data_dir="$EV" \
    --eval_config_dir="$IRIS_REPO/lotus/datasets/eval/depth" \
    --half_precision \
    --disparity > "$OUT/eval.log" 2>&1 || { tr '\r' '\n' < "$OUT/eval.log" | grep -v "No text description" | tail -n 40; exit 1; }
  tr '\r' '\n' < "$OUT/eval.log" | grep -E "abs_relative_difference|Text Description|images without" || true
  for f in "$OUT"/depth/*/eval_metrics*.txt; do echo "--- $f"; cat "$f"; done
  ;;
*) echo "unknown STEP=$STEP"; exit 1 ;;
esac
