#!/bin/bash
#SBATCH --job-name=capshow
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Caption 展示图的原料：每个条件里「加 caption 后单帧 AbsRel 降得最多」的前 8 帧
# （官方子集，tools/caption_showcase_rank.py 按测试 CSV 排，不推理），两个臂各推一遍。
#
#   cap    sd2_cap_pct_logtn_s43    step18000  correct prompt
#   nocap  sd2_nocap_pct_logtn_s43  step18000  empty prompt
#
# 两个臂都走主表那条路（train_route_suite，ssi_log，percentile，posterior mode），
# 清单按条件补空行，让每帧保留它在完整测试里的行号 = 噪声种子。
# ⭐ 自检：两个臂每帧的 AbsRel 都要和当初测试 CSV 一致，否则图上画的不是表里的模型。
#
# ⚠️ 这是「最佳情形」的例子。图注必须写明选帧规则和平均增益（rank_*.txt 第一行）。
#
#   sbatch ~/Iris/slurm/caption_showcase.sbatch

F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
set +e

S=/mnt/scratch/sc23sz/runs
OUT="${OUT:-$IRIS_RUNS/analysis/caption_showcase}"
export IRIS_THERMAL_STRETCH=percentile
export IRIS_DECODE_CLAMP=0

mkdir -p "$OUT"
cd "$IRIS_REPO"
git log --oneline -1

echo "=== 1. 排序选帧 ==="
python tools/caption_showcase_rank.py --ms2-root "$IRIS_MS2_ROOT" \
  --manifest-dir "$IRIS_MANIFEST_DIR" --out "$OUT" || { echo "!! 排序失败，停"; exit 1; }

for spec in "cap correct" "nocap empty"; do
  arm=${spec% *}; prompt=${spec#* }
  ckpt="$S/iris_ms2/sd2_${arm}_pct_logtn_s43/converted/step18000_weights.pt"
  head -c 1 "$ckpt" > /dev/null 2>&1 || { echo "!! 读不了 $ckpt"; continue; }
  mkdir -p "$OUT/$arm/raw_predictions"
  for cond in day night rainy; do
    echo; echo "=== 2. $arm  $cond ==="
    man="$OUT/frames_ours_$cond.jsonl"
    python tools/train_route_suite.py \
      --route b_thermal_unet --output-dir "$OUT/run_${arm}_$cond" --ms2-root "$IRIS_MS2_ROOT" \
      --train-manifest "$man" --val-manifest "$man" --eval-checkpoint "$ckpt" \
      --val-caption-mode "$prompt" --align-mode ssi_log --num-inference-steps 1 \
      --condition-latent mode --val-stride 1 --save-raw-pred --local-files-only \
    && cp "$OUT/run_${arm}_$cond"/raw_predictions/*.npy "$OUT/$arm/raw_predictions/" \
    || echo "!! $arm $cond 失败"
  done
done

echo; echo "=== 3. 自检：每帧 AbsRel，本次 vs 当初测试 ==="
python - "$OUT" <<'PY' | tee "$OUT/selfcheck.txt"
import csv, glob, sys
from pathlib import Path
out = Path(sys.argv[1])
fps = {"day": "f50d4506", "night": "273310de", "rainy": "5a63743f"}
runs = {"cap": ("sd2_cap_pct_logtn_s43", "correct"), "nocap": ("sd2_nocap_pct_logtn_s43", "empty")}
def read(paths):
    got = {}
    for p in paths:
        with open(p, newline="", encoding="utf-8") as h:
            for r in csv.DictReader(h):
                got[r["id"]] = float(r["abs_rel"])
    return got
bad = 0
for arm, (run, prompt) in runs.items():
    for cond, fp in fps.items():
        ref = read(glob.glob(f"/mnt/scratch/sc23sz/runs/eval/{run}_test_{fp}_s18000_{prompt}/eval_eval_affine_invariant_log_space_per_sample.csv"))
        new = read(glob.glob(f"{out}/run_{arm}_{cond}/eval_*per_sample.csv"))
        if not new:
            print(f"{arm:5s} {cond:5s} 缺本次结果 ⚠️"); bad += 1; continue
        for fid, b in sorted(new.items()):
            a = ref.get(fid)
            ok = a is not None and abs(a - b) < 5e-4
            bad += not ok
            print(f"{arm:5s} {cond:5s} {fid:28s} {a if a is not None else float('nan'):9.5f} {b:9.5f}  {'✅' if ok else '⚠️'}")
print(f"\n{'全部对上' if bad == 0 else f'{bad} 处对不上 ⚠️'}")
PY

echo; echo "=== 4. 打包 ==="
cd "$OUT/.."
B="$(basename "$OUT")"
tar czf "$B.tar.gz" "$B/inputs" "$B/frames.jsonl" "$B"/rank_*.txt "$B/selfcheck.txt" \
  "$B/cap/raw_predictions" "$B/nocap/raw_predictions"
echo "[完成] $(pwd)/$B.tar.gz  ($(du -h "$B.tar.gz" | cut -f1))"
