#!/bin/bash
# Attach only test jobs. Optional: FRAME_TRAIN_JOB_ID EMPTY_TRAIN_JOB_ID.
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
export IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
[[ $# == 0 || $# == 2 ]] || { echo "Usage: bash $0 [frame_train_id empty_train_id]"; exit 2; }
run="$IRIS_RUNS/outputs/lotus_line_v2/iris_local_gate_lora_kv_r4_v1"
# Prefer this experiment's submission receipt over potentially ambiguous names.
if [[ $# == 0 && -r "$run/submission.txt" ]]; then
  f=$(awk '$1=="frame" {print $2}' "$run/submission.txt")
  e=$(awk '$1=="empty" {print $2}' "$run/submission.txt")
  [[ "$f" =~ ^[0-9]+$ && "$e" =~ ^[0-9]+$ && "$f" != "$e" ]] || {
    echo "Incomplete submission receipt; inspect $run/submission.txt"; exit 1;
  }
  set -- "$f" "$e"
fi
owner=$(id -un)
queue=$(squeue -h -u "$owner" -o '%i|%j')
resolve() {
  local name="iris_ll_$1" explicit="$2" ids records
  ids=$(awk -F'|' -v n="$name" -v id="$explicit" '$2==n && ($1==id || id=="") {print $1}' <<< "$queue")
  if [[ -z "$ids" ]]; then
    if [[ -n "$explicit" ]]; then
      records=$(sacct -nXP -u "$owner" -j "$explicit" -o JobIDRaw,JobName%48,State)
    else
      records=$(sacct -nXP -u "$owner" -S now-7days --name="$name" -o JobIDRaw,JobName%48,State)
    fi
    ids=$(awk -F'|' -v n="$name" -v id="$explicit" '$2==n && ($1==id || id=="") && $3=="COMPLETED" {print $1}' <<< "$records")
  fi
  [[ "$ids" =~ ^[0-9]+$ ]] || { echo "Need exactly one $name job for $owner; supply both training IDs." >&2; return 1; }
  echo "$ids"
}
if [[ $# == 2 ]]; then
  [[ "$1" =~ ^[0-9]+$ && "$2" =~ ^[0-9]+$ && "$1" != "$2" ]] || exit 2
fi
frame=$(resolve frame "${1:-}")
empty=$(resolve empty "${2:-}")
dest="$run/test_selected_v1/submissions"
mkdir -p "$dest"
exec 9>"$dest/attach.lock"
flock -n 9 || { echo 'Another attachment is in progress'; exit 1; }
for arm in frame empty; do
  parent=${!arm}
  receipt="$dest/${arm}_$parent.id"
  if [[ -s "$receipt" ]]; then
    echo "$arm test already submitted -> $(cat "$receipt") (afterok:$parent)"
    continue
  fi
  job=$(sbatch --parsable -J "iris_ll_test_$arm" --dependency="afterok:$parent" --kill-on-invalid-dep=yes \
    --export="ALL,ARM=$arm" slurm/iris_local_gate_lora_test.sbatch)
  job=${job%%;*}
  [[ "$job" =~ ^[0-9]+$ ]]
  echo "$job" > "$receipt"
  echo "$arm test -> $job (afterok:$parent)"
done
