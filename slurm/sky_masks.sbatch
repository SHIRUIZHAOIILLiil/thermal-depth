#!/bin/bash
#SBATCH --job-name=skymask
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 天空掩码：Mask2Former（Cityscapes）跑在**热像**上，为 Jasmine 附录 C.3 的
# sky loss 提供 M_sky。
#
# ⚠️ Mask2Former 从没见过热像 —— 这条能不能用是个实验问题，所以**先抽样验收**：
#   MAX_SAMPLES=100（默认）跑 100 帧出质量报告，判据不是肉眼，是物理事实：
#
#     **真天空不返回激光。** 所以被判成天空的像素里带 GT 的比例应该 ≈ 0。
#     带 GT 就说明掩码抓到了楼、树或路面 —— 抓错了。
#
#   报告里 verdict=PASS 需要同时满足：
#     · 天空像素里带 GT 的 < 1%
#     · 落在地平线（第 152 行，来自 calib.npy 的 K_thrL cy=151.63）以下的 < 2%
#   不 PASS 就别往训练里接，改用几何掩码（无回波 且 在该列最高有效像素之上）。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   MANIFEST     默认训练集 manifest（掩码要给训练用，所以默认对训练集出）
#   TAG          输出目录名，默认 skymask_<max_samples>
#   MAX_SAMPLES  默认 100（验收）；0 = 全量（出正式掩码时用）
#   SAVE_MASKS   1 = 落盘 masks/<id>.png。验收阶段不用；全量时必须开
#
# ── 用法 ────────────────────────────────────────────────────────────
#   验收（100 帧，不落盘）：
#     cd $SCRATCH/logs && sbatch -J skymask_probe ~/Iris/slurm/sky_masks.sbatch
#
#   验收通过后出全量掩码（训练集 19,949 帧，落盘）：
#     MAX_SAMPLES=0 SAVE_MASKS=1 TAG=skymask_train_full \
#       sbatch -J skymask_full --time=06:00:00 \
#       --export=ALL,MAX_SAMPLES,SAVE_MASKS,TAG ~/Iris/slurm/sky_masks.sbatch
#
#   ⚠️ 权重要先预取：compute 节点没有外网。
#     cd $SCRATCH/logs && sbatch ~/Iris/slurm/prefetch_models.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

MANIFEST="${MANIFEST:-$IRIS_MANIFEST_DIR/ms2_train_day2seq_clip75_20260728.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
TAG="${TAG:-skymask_${MAX_SAMPLES}}"
OUT="$IRIS_RUNS/sky_masks/$TAG"

[[ -f "$MANIFEST" ]] || { echo "!! 找不到 manifest: $MANIFEST"; exit 1; }
mkdir -p "$OUT"

ARGS=(
  --manifest "$MANIFEST"
  --ms2-root "$IRIS_MS2_ROOT"
  --output-dir "$OUT"
  --max-samples "$MAX_SAMPLES"
)
[[ "${SAVE_MASKS:-0}" == "1" ]] && ARGS+=(--save-masks)
[[ "${LOCAL_FILES_ONLY:-1}" == "1" ]] && ARGS+=(--local-files-only)

cd "$IRIS_REPO"
echo "=== python tools/build_sky_masks.py ${ARGS[*]} ==="
python tools/build_sky_masks.py "${ARGS[@]}"
rc=$?

echo "=== 退出码 $rc ==="
echo "=== 验收看这里（verdict 与 GATE 那一行） ==="
[[ -f "$OUT/sky_mask_report.json" ]] && python - "$OUT/sky_mask_report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
s = r["summary"]
print(f'  verdict = {r["verdict"]}  ({r["frames"]} 帧, sky class {r["sky_class"]})')
print(f'  天空占画面      {s["sky_fraction_mean"]*100:.2f}%')
print(f'  天空里带 GT 的  {s["sky_with_gt_fraction_mean"]*100:.3f}%  (max {s["sky_with_gt_fraction_max"]*100:.2f}%)  <- 要 ~0')
print(f'  在地平线以下的  {s["sky_below_horizon_fraction_mean"]*100:.3f}%  <- 要 ~0')
print(f'  无监督区被覆盖  {s["unsupervised_covered_fraction_mean"]*100:.2f}%')
print(f'  完全没有天空的帧 {s["frames_without_sky"]}')
PY
ls -la "$OUT"
exit $rc
