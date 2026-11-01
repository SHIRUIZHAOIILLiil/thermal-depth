#!/bin/bash
set -euo pipefail
cd "${IRIS_REPO:-$HOME/Iris}"
export IRIS_REPO="$PWD"
export IRIS_RUNS="${IRIS_RUNS:-${SCRATCH:?}/runs}"
root="$IRIS_RUNS/outputs/lotus_line_v2"
parent="$root/iris_visual_gate_matched_v1"
out="$root/iris_local_gate_lora_kv_r4_v1"
test -r "$root/da2_iris_fusion_matched_pair_v2/empty_s43/step006000.pt"
test -r "${DA2_CAPTION_ASSETS:-$IRIS_RUNS/da2_caption_assets/assets.json}"
for arm in frame empty; do
  for file in summary.json contract.json full.csv; do test -r "$parent/$arm/$file"; done
done
test -r "$parent/frame/step002000.pt"
test -r "$parent/empty/step006000.pt"
mkdir -p "$out"
exec 9>"$out/submit.lock"
flock -n 9 || { echo "Another submission is in progress"; exit 1; }
# A receipt prevents accidentally queuing a second pair while the first is pending.
[[ ! -e "$out/submission.txt" && ! -e "$out/frame" && ! -e "$out/empty" ]] || {
  echo "Existing run/submission: $out; inspect receipt before explicit recovery"; exit 1;
}
printf 'started %s\n' "$(date -u +%FT%TZ)" > "$out/submission.txt"
smoke=$(sbatch --parsable -J iris_ll_smoke --time=02:00:00 --export=ALL,ACTION=smoke slurm/iris_local_gate_lora.sbatch)
smoke=${smoke%%;*}
[[ "$smoke" =~ ^[0-9]+$ ]]
echo "smoke $smoke" >> "$out/submission.txt"
echo "paired smoke -> $smoke"
export SMOKE="$out/smoke_$smoke"
for arm in frame empty; do
  id=$(sbatch --parsable -J "iris_ll_$arm" --dependency="afterok:$smoke" --kill-on-invalid-dep=yes \
    --export="ALL,ACTION=train,ARM=$arm" slurm/iris_local_gate_lora.sbatch)
  id=${id%%;*}
  [[ "$id" =~ ^[0-9]+$ ]]
  echo "$arm $id afterok:$smoke" >> "$out/submission.txt"
  echo "$arm train+val -> $id (afterok:$smoke)"
done
echo "Outputs: $out"
echo "No test evaluation submitted."
