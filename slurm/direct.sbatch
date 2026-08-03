#!/bin/bash
#SBATCH --job-name=direct
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00  # 2,543 帧单步扩散实测约 15 分钟，留一倍余量
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 任务 3 的第三个参照点：**零训练直推**。全链路没有一个可训练参数，
# 所以它回答「不训练能到哪」，六线的成绩减去它就是训练买来的东西。
#
# ⚠️ 「零训练直推」有两条，条件通路不同，答的不是同一个问题。选错了数字会差 3 倍：
#
#   KIND=thermal_vae （默认，主表用这条）
#     热像 → VAE 编码器 → 冻结 Lotus-G。给冻结 UNet 的是它训练时见过的那种 latent，
#     所以它是 b 线「U-Net 不训练」的受控对照：b 的 0.0884 减去它 ＝ 训 U-Net 买来的。
#     历史参考：val 5810 帧 0.1291（route_c_thermal_frozen_val_full）。
#
#   KIND=anythermal
#     AnyThermal 特征 → 零参数 bridge → 冻结 Lotus-G。特征金字塔的格式是冻结 UNet
#     从没见过的，所以这条本来就废：test 实测 0.4697，比 §7 把 condition 打乱
#     （0.2971）还差 —— 「乱但同分布」好过「不同分布」。
#     它不是主表那一行，而是 d 系的辅证：AnyThermal 特征不经 adapter 用不了，
#     与 §7 的 donor-swap 互补（donor-swap 证明 adapter 在用特征，这条证明少了它不行）。
#
# ⚠️ 另外两个必须显式覆盖的默认值：
#   1. 脚本自身 `--max-samples` 默认 **8**（调试值），这里默认传 0 ＝ 全量。
#   2. 脚本内置的 lotus evaluator 走的是上游 `least_square_disparity`，
#      **不是官方 BMSD 协议**，数字与主表不可比。所以本作业出完 raw 预测后
#      一律再用 `run_official_ms2_evaluation.py` 重算一遍，主表只引用重算值。
#
# 对齐：直推的输出与六线同一个量（Lotus-G 解码出的 [0,1] 视差），
# 所以可比的是 **ssi_disparity**。作业同时跑一遍 `ssi` 作对照 ——
# 这是 §1 那条坑的自查（用错空间会得到几倍大的数），CPU 上几十秒，白拿。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   KIND          thermal_vae（默认）| anythermal —— 见上面那段，选错差 3 倍
#   MANIFEST      考卷；默认独立 test 集 16-08-46
#   TAG           输出目录名；默认 direct_<kind>_<dtype>_<caption>
#   DTYPE         fp32（默认）| fp16
#                 直推没有训练过的权重去吸收量化误差，历史上 fp16 解码明显更差
#                 （本地 V1 线的 "fp32dec" 就是为此而来），所以基线取 fp32；
#                 要和六线的 fp16 推理精度严格对齐时再跑一遍 fp16 臂。
#   CAPTION_MODE  empty（默认）| correct
#   MAX_SAMPLES   默认 0 ＝ 全量
#
# ── 用法 ────────────────────────────────────────────────────────────
#   主臂（主表那一行）：
#     sbatch -J direct ~/Iris/slurm/direct.sbatch
#
#   d 系辅证臂：
#     KIND=anythermal sbatch -J direct_at --export=ALL,KIND ~/Iris/slurm/direct.sbatch
#
#   fp16 对照臂：
#     DTYPE=fp16 sbatch -J direct16 --export=ALL,DTYPE ~/Iris/slurm/direct.sbatch
#
#   出完之后接分层（CPU 作业）：
#     PREDS="$SCRATCH/runs/direct/direct_thermal_vae_fp32_empty/raw_predictions:direct" \
#     MANIFEST=$SCRATCH/manifests/sequence_level_internvl3_8b/ms2_test_16-08-46_rgb_depth_v1_clip75_20260728.jsonl \
#     TAG=regions_direct ALIGN=ssi_disparity \
#       sbatch -J regions_direct --export=ALL,PREDS,MANIFEST,TAG,ALIGN ~/Iris/slurm/analyze.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

KIND="${KIND:-thermal_vae}"
case "$KIND" in
  thermal_vae) SCRIPT=tools/run_ms2_lotus_thermal_vae_official.py ;;
  anythermal)  SCRIPT=tools/run_ms2_lotus_direct_official.py ;;
  *) echo "!! KIND 只能是 thermal_vae 或 anythermal，收到 '$KIND'"; exit 1 ;;
esac

DTYPE="${DTYPE:-fp32}"
CAPTION_MODE="${CAPTION_MODE:-empty}"
MANIFEST="${MANIFEST:-$IRIS_MANIFEST_DIR/ms2_test_16-08-46_rgb_depth_v1_clip75_20260728.jsonl}"
TAG="${TAG:-direct_${KIND}_${DTYPE}_${CAPTION_MODE}}"
OUT="$IRIS_RUNS/direct/$TAG"

# 计算节点没有外网，权重必须走 $HF_HOME 里预取好的那份（prefetch_models.sbatch）
LFO_FLAG=""
[[ "${LOCAL_FILES_ONLY:-1}" == "1" ]] && LFO_FLAG="--local-files-only"

[[ -f "$MANIFEST" ]] || { echo "!! 找不到 manifest: $MANIFEST"; exit 1; }

# 两个脚本都拒绝非空输出目录，重跑要么换 TAG 要么先清
if [[ -d "$OUT" ]] && [[ -n "$(ls -A "$OUT" 2>/dev/null)" ]]; then
  echo "!! $OUT 非空 —— 换个 TAG，或先 rm -rf 它"
  exit 1
fi
mkdir -p "$OUT"

cd "$IRIS_REPO"

echo "=== 第一步：出预测（KIND=$KIND / $DTYPE / caption=$CAPTION_MODE） ==="
echo "===   $SCRIPT ==="
python "$SCRIPT" \
  --manifest "$MANIFEST" \
  --ms2-root "$IRIS_MS2_ROOT" \
  --output-dir "$OUT" \
  --max-samples "${MAX_SAMPLES:-0}" \
  --dtype "$DTYPE" \
  --caption-mode "$CAPTION_MODE" \
  --save-raw-pred \
  $LFO_FLAG \
  || { echo "=== 出预测失败 ==="; exit 1; }

# set -e + pipefail 下 `ls | wc -l` 在零匹配时会直接终止脚本（错误信息反而印不出来），
# 所以用 find 数
npy=$(find "$OUT/raw_predictions" -name '*.npy' 2>/dev/null | wc -l || true)
echo "=== raw_predictions: $npy 个 .npy ==="
[[ "$npy" -gt 0 ]] || { echo "!! 没有 raw 预测，后面的重算无从谈起"; exit 1; }

# 官方协议重算。--route 只决定 auto 对齐和记录里的标签，这里两个空间都显式给，
# 所以选哪个 route 名不影响数字。
for align in ssi_disparity ssi; do
  echo "=== 第二步：官方协议重算（align=$align） ==="
  python tools/run_official_ms2_evaluation.py \
    --manifest "$OUT/selected_manifest.jsonl" \
    --data-root "$IRIS_MS2_ROOT" \
    --prediction-dir "$OUT/raw_predictions" \
    --route iris-lotus \
    --align "$align" \
    --gt-view thermal \
    --output-dir "$OUT/official_$align" \
    || echo "!! align=$align 重算失败（另一个空间的结果仍然有效）"
done

echo
echo "=== 汇报用的数字在这里（ssi_disparity 那个才与主表可比） ==="
for align in ssi_disparity ssi; do
  f="$OUT/official_$align/metrics/summary.json"
  [[ -f "$f" ]] || { echo "  $align  （无结果）"; continue; }
  python - "$f" "$align" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
st = s["image_wise"]["statistics"]
print(f'  {sys.argv[2]:>14}  n={s["sample_count"]}  '
      f'abs_rel={st["abs_rel"]["mean"]:.4f}  rmse={st["rmse"]["mean"]:.3f}  a1={st["a1"]["mean"]:.4f}')
PY
done

echo "=== 输出目录 $OUT ==="
ls -la "$OUT"
