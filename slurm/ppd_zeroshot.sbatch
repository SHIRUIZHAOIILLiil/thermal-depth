#!/bin/bash
#SBATCH --job-name=ppd0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 已发布的 Pixel-Perfect Depth 权重，直接推热像，**不做任何适配**。
#
# 补的是主表「没做热像适配」那一组里缺的第三条线。Depth Anything V2 和 Lotus-G
# 都有零样本行，PPD 只有适配后的，缺了它那一组读起来是瘸的。
#
# ⚠️ PPD 约 0.63 秒一帧，day 全量 2.3 万帧要四小时以上。所以默认只跑官方子集
#    那 2331 帧 —— 主表用的就是子集，全量再折算一次是白花卡。
#    做法和 res_probe 一样：先裁子清单，推理和打分都喂它。
#
# ⚠️ 对齐用 ssi_log —— PPD 的原生输出是 log 深度（2026-09-24 实测，作业 7941688）。
#    用 ssi_disparity 会把近远弄反，而且不报错。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   COND     day | night | rainy，默认 day
#   N        帧数，默认 0 ＝ 子清单全部（即官方子集那 2331/2292/2503 帧）
#   PPD_CKPT 已发布权重，默认 $PPD_REPO/checkpoints/ppd.pth
#   OUT_TAG  输出目录名
#
#   COND=day sbatch --export=ALL,COND ~/Iris/slurm/ppd_zeroshot.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

COND="${COND:-day}"
N="${N:-0}"
OUT_TAG="${OUT_TAG:-ppd_zeroshot_$COND}"
OUT="$IRIS_RUNS/analysis/$OUT_TAG"

export PPD_REPO="${PPD_REPO:-$SCRATCH/ppd_repo}"
PPD_CKPT="${PPD_CKPT:-$PPD_REPO/checkpoints/ppd.pth}"
PPD_SEM="${PPD_SEM:-$PPD_REPO/checkpoints/depth_anything_v2_vitl.pth}"
PY="${PPD_PY:-$HOME/.conda/envs/ppd/bin/python}"

case "$COND" in
  day)   MANIFEST="$IRIS_MANIFEST_DIR/ms2_test_day3_common_thermalcap_20260821.jsonl"; ENV=test_day ;;
  night) MANIFEST="$IRIS_MANIFEST_DIR/ms2_test_night3_thermalcap_20260821.jsonl";      ENV=test_night ;;
  rainy) MANIFEST="$IRIS_MANIFEST_DIR/ms2_test_rainy3_thermalcap_20260821.jsonl";      ENV=test_rain ;;
  *) echo "!! COND 只支持 day | night | rainy"; exit 1 ;;
esac
[[ -f "$MANIFEST" ]] || { echo "!! 找不到 $MANIFEST"; exit 1; }
[[ -x "$PY" ]]       || { echo "!! 找不到可执行的 $PY"; exit 1; }
[[ -e "$PPD_CKPT" ]] || { echo "!! 找不到已发布权重 $PPD_CKPT"; ls -1 "$PPD_REPO/checkpoints" 2>/dev/null | sed 's/^/     /'; exit 1; }
[[ -e "$PPD_SEM" ]]  || { echo "!! 找不到语义权重 $PPD_SEM"; exit 1; }

# 与适配线、与我方线一致。错了量到的是预处理差异。
export IRIS_THERMAL_STRETCH="${IRIS_THERMAL_STRETCH:-percentile}"

mkdir -p "$OUT"
cd "$IRIS_REPO"
git log --oneline -1
echo "=== 配置 ==="
echo "条件    = $COND   ($ENV)"
echo "权重    = $PPD_CKPT   **已发布，无任何适配**"
echo "语义    = $PPD_SEM"
echo "对齐    = ssi_log   (PPD 的原生输出空间，实测)"
echo "thermal = $IRIS_THERMAL_STRETCH"
echo "python  = $PY"
echo "============"

# 官方子集 ＝ 每 10 帧抽 1，省掉 90% 的卡时。
# ⚠️ 不能对清单直接 rows[::10]：官方那个列表是按 MS2 的目录清单建候选再取 stride，
#    候选数（day 23317）和清单行数（23311）本来就不一样，顺序也未必一致。
#    所以这里调用权威实现 official_frames，再按 id 过滤清单 —— 和表里其余行
#    用的是同一批帧，不是「大约每十帧」。
SUBSET="$OUT/subset.jsonl"
python - "$MANIFEST" "$SUBSET" "$IRIS_MS2_ROOT" "$ENV" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")   # 上面已经 cd 到 $IRIS_REPO
from tools.run_ms2_supdepth_baselines import official_frames

src, dst, root, env = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
want = {row["id"] for row in official_frames(root, env, 10)}
rows = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
keep = [line for line in rows if json.loads(line).get("id") in want]
missing = len(want) - len(keep)
dst.write_text("\n".join(keep) + "\n", encoding="utf-8")
print(f"[子清单] 官方 {len(want)} 帧，清单里有 {len(keep)} 帧"
      + (f"，缺 {missing} 帧（清单本来就没有）" if missing else ""))
PY
FRAMES="$(wc -l < "$SUBSET")"
echo "[子清单] $FRAMES 帧 -> $SUBSET"

INFER=(--model pixel_perfect_depth --manifest "$SUBSET" --data-root "$IRIS_MS2_ROOT"
       --out-dir "$OUT/raw_predictions" --stretch "$IRIS_THERMAL_STRETCH"
       --ppd-checkpoint "$PPD_CKPT" --ppd-semantics "$PPD_SEM")
[[ "$N" != "0" ]] && INFER+=(--limit "$N")
"$PY" tools/run_external_depth_inference.py "${INFER[@]}"

echo
echo "=== 官方协议打分 ==="
SCORE=(--manifest "$SUBSET" --data-root "$IRIS_MS2_ROOT"
       --prediction-dir "$OUT/raw_predictions" --route iris-lotus
       --align ssi_log --output-dir "$OUT/official")
[[ "$N" != "0" ]] && SCORE+=(--limit "$N")
python tools/run_official_ms2_evaluation.py "${SCORE[@]}"

echo
echo "[完成] $OUT"
echo "⚠️ 这已经是官方子集口径，不要再过 subset_to_official.py 折算一次。"
echo "⚠️ 逐帧 .npy 打完分就可以删：rm -rf $OUT/raw_predictions"
