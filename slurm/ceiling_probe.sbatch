#!/bin/bash
#SBATCH --job-name=ceiling
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# log 目标下的 VAE 往返天花板，以及模型在同一批帧上的成绩。
#
# 问的是一件事：如果 U-Net 把目标 latent 预测得完美无缺，能得几分？答案就是
# 目标过一遍冻结 VAE 的往返，因为目标只以这个形式被看见。两行贴在一起 ＝ U-Net
# 相对这个表示已经到顶；差得远 ＝ 还有余量。
#
# ⚠️ 两边必须是同一批帧。2026-09-01 那次「2.996 对 2.996」是拿训练帧的上界
# 对 test 的成绩，当时没标出来，那个三位小数的巧合因此没有它读起来那么硬。
# 这个脚本先切一份子清单，再把同一份喂给两边，没有第二种选法。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   RUN     必填。训练输出目录名，如 sd2_cap_pct_logtn
#   STEP    必填。checkpoint 步数，须已转换
#           （$IRIS_RUNS/iris_ms2/$RUN/converted/step${STEP}_weights.pt）
#   FRAMES  抽多少帧，默认 200。冒烟用 20
#   PROMPT  empty | correct | permuted，默认 correct。须与该臂训练时一致
#   OUT_TAG 输出目录名，默认 ceiling_${RUN}_s${STEP}
#
# ── 用法 ────────────────────────────────────────────────────────────
#   冒烟（20 帧，确认两边都真的出数）：
#     RUN=sd2_cap_pct_logtn STEP=16000 FRAMES=20 \
#       sbatch --time=00:40:00 --export=ALL,RUN,STEP,FRAMES ~/Iris/slurm/ceiling_probe.sbatch
#
#   正式（200 帧）：
#     RUN=sd2_cap_pct_logtn STEP=16000 \
#       sbatch --export=ALL,RUN,STEP ~/Iris/slurm/ceiling_probe.sbatch
#
# 不用 `sbatch --wrap`：common.sh 里有 module load，--wrap 的非交互 shell 里
# module 不是已定义的函数，配上 set -euo pipefail 就是两秒退出、退出码 2。

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

: "${RUN:?必须指定 RUN（训练输出目录名）}"
: "${STEP:?必须指定 STEP（checkpoint 步数）}"
FRAMES="${FRAMES:-200}"
PROMPT="${PROMPT:-correct}"
OUT_TAG="${OUT_TAG:-ceiling_${RUN}_s${STEP}}"

# ⚠️ 这一行不能省。iris_ms2_pipeline.sbatch 把它默认成 minmax，而 logtn 这条线
# 是 percentile 训的。错了不报错，只让数变差 —— 然后我们会把那笔差记在天花板头上。
export IRIS_THERMAL_STRETCH="${IRIS_THERMAL_STRETCH:-percentile}"
export IRIS_TEXT_PADDING="${IRIS_TEXT_PADDING:-legacy}"
export IRIS_DECODE_CLAMP="${IRIS_DECODE_CLAMP:-0}"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl}"
PSEUDO_DIR="${PSEUDO_DIR:-$IRIS_RUNS/pseudo_gt/train_full/calibrated_pseudo_depth}"
CKPT="$IRIS_RUNS/iris_ms2/$RUN/converted/step${STEP}_weights.pt"
OUT="$IRIS_RUNS/analysis/$OUT_TAG"
SUB="$OUT/frames.jsonl"

if [[ ! -f "$CKPT" ]]; then
  echo "!! 找不到转换后的权重 $CKPT"
  echo "   $IRIS_RUNS/iris_ms2 下现有的臂："
  ls -1 "$IRIS_RUNS/iris_ms2" 2>/dev/null | sed 's/^/     /'
  echo "   该臂已转换的步数："
  ls -1 "$IRIS_RUNS/iris_ms2/$RUN/converted" 2>/dev/null | sed 's/^/     /'
  exit 1
fi
[[ -f "$TRAIN_MANIFEST" ]] || { echo "!! 找不到清单 $TRAIN_MANIFEST"; exit 1; }
[[ -d "$PSEUDO_DIR" ]]     || { echo "!! 找不到伪深度 $PSEUDO_DIR"; exit 1; }

mkdir -p "$OUT"
cd "$IRIS_REPO"
git log --oneline -1

echo "=== 配置 ==="
echo "臂        = $RUN  step $STEP"
echo "帧数      = $FRAMES"
echo "prompt    = $PROMPT   (须与训练时一致)"
echo "thermal   = $IRIS_THERMAL_STRETCH   (须与训练时一致)"
echo "对齐      = ssi_log   (log_truncnorm 只能在 log 空间拟合)"
echo "输出      = $OUT"
echo "============"

# --- 0. 切子清单：两边共用，唯一的一份 -------------------------------------
# 按行等距抽，不随机 —— 同样的 FRAMES 重跑得到同样的帧，结果可以对比。
TOTAL=$(wc -l < "$TRAIN_MANIFEST")
EVERY=$(( TOTAL / FRAMES ))
[[ "$EVERY" -lt 1 ]] && EVERY=1
awk -v n="$EVERY" 'NR % n == 1' "$TRAIN_MANIFEST" | head -n "$FRAMES" > "$SUB"
GOT=$(wc -l < "$SUB")
echo "[子清单] $TOTAL 帧里每 $EVERY 抽 1，得到 $GOT 帧 -> $SUB"
[[ "$GOT" -ge 1 ]] || { echo "!! 子清单是空的"; exit 1; }

# --- 1. 上界：目标过一遍 VAE，不训练任何东西 --------------------------------
# trunc_disparity 并排跑，是为了确认这次测量能重现已知的那一行；没有它，log
# 那一行没有可信的参照。--stride 1 配一个大 limit ＝ 子清单里的每一帧都要。
echo
echo "=== 1/2 往返上界 ==="
python tools/probe_vae_roundtrip_ceiling.py \
  --manifest "$SUB" \
  --ms2-root "$IRIS_MS2_ROOT" \
  --pseudo-dir "$PSEUDO_DIR" \
  --output-dir "$OUT/probe" \
  --norm-types log_truncnorm trunc_disparity \
  --lotus-model-path "${SD2_REPO:-Manojb/stable-diffusion-2-1-base}" \
  --stride 1 --limit 100000 \
  --local-files-only

# --- 2. 模型：同一批帧 ------------------------------------------------------
# ⚠️ 这是训练帧，模型见过。出来的数会比汇报里的 test 数好，不许跟 0.07333 并排。
# 要比的是它和上面那一行 —— 同一批帧，同一个对齐。
echo
echo "=== 2/2 模型在同一批帧上 ==="
python tools/train_route_suite.py \
  --route "${ROUTE:-b_thermal_unet}" \
  --output-dir "$OUT/model" \
  --ms2-root "$IRIS_MS2_ROOT" \
  --train-manifest "$TRAIN_MANIFEST" \
  --eval-checkpoint "$CKPT" \
  --val-manifest "$SUB" \
  --val-stride 1 \
  --val-caption-mode "$PROMPT" \
  --num-inference-steps 1 \
  --align-mode ssi_log \
  --condition-latent mode \
  --local-files-only

# --- 3. 并排 ---------------------------------------------------------------
echo
echo "=== 对照 ==="
python - "$OUT" <<'PY'
import csv, json, sys
from pathlib import Path

out = Path(sys.argv[1])
probe = json.loads((out / "probe" / "vae_ceiling.json").read_text(encoding="utf-8"))
by_norm = probe["summary"]["by_norm"]

# The probe prints AbsRel as a percentage in its own second table; everything
# here is the fraction, the way the report writes it. Mixing the two once cost
# an afternoon.
rows = []
for norm, s in by_norm.items():
    rows.append((f"上界 {norm}", s["abs_rel_mean"], s["rmse_mean"], s["delta1_mean"]))

# Aggregate the model the same way the probe does: per frame, then mean over
# frames. Reading a headline out of the eval json instead would risk a
# different aggregation, and the whole comparison lives in that difference.
csvs = sorted((out / "model").rglob("*per_sample*.csv"))
if not csvs:
    raise SystemExit(f"找不到 per_sample csv，看 {out/'model'} 里有什么")
table = list(csv.DictReader(csvs[-1].open(encoding="utf-8", newline="")))


def column(*names):
    for name in names:
        if name in table[0]:
            return [float(r[name]) for r in table if r[name] not in ("", None)]
    raise SystemExit(f"csv 里没有这些列：{names}；实际有 {list(table[0])}")


n = len(table)
rows.append(("模型 " + csvs[-1].parent.name,
             sum(column("abs_rel")) / n,
             sum(column("rmse")) / n,
             sum(column("a1", "delta1")) / n))

print(f"\n{'':28s}{'AbsRel':>10s}{'RMSE (m)':>12s}{'δ1':>10s}")
for name, a, r, d in rows:
    print(f"{name:28s}{a:>10.5f}{r:>12.3f}{d:>10.5f}")
print(f"\n帧数：上界 {probe['summary']['frames']}，模型 {n}")
if probe["summary"]["frames"] != n:
    print("⚠️ 两边帧数不一致，这组对照作废 —— 查子清单是不是被两边各自抽稀了")

lg = by_norm.get("log_truncnorm")
if lg:
    print(f"\n[接缝] 往返@激光像素 {lg['round_trip_rel_lidar_p50']:.4f}　"
          f"往返@伪深度像素 {lg['round_trip_rel_pseudo_p50']:.4f}　"
          f"比值 {lg['lidar_over_pseudo']:.2f}")
    print("比值明显大于 1 ＝ 真实回波那些孤立尖峰是 VAE 扛不住的，问题在覆写方式，")
    print("不在「VAE 编不了深度图」。那是另一条路，别混为一谈。")
PY

echo
echo "[完成] $OUT"
