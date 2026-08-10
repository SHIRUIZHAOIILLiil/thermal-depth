#!/bin/bash
#SBATCH --job-name=vae_ceiling
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 「换成 Iris 那套 latent 目标训练」之前的门禁：**不训练，只过一次 VAE**。
#
# 那套目标要把 GT 深度编码成 latent 再加噪，所以它要求 GT 稠密——稀疏 LiDAR 给不了，
# 补全后的稠密伪深度可以。但目标只能通过 VAE 被看见，所以**它的往返误差就是上界**：
# 即使 U-Net 把 latent 预测得完全正确，解码出来也只能到这个水平。
#
# 归一化按在用的 checkpoint（lotus-depth-g-v2-1-disparity）走：取倒数成视差、
# 逐图 min-max 到 [-1,1]，与 HypersimImageDepthNormalTransform 的 norm_type=disparity
# 一致。这一步错了，测到的是归一化不匹配，却会被当成 VAE 的极限。
#
# ── 判据（事先定，别看完再改）────────────────────────────────────────
#   往返 AbsRel < 0.05      → 值得往下做，写 latent 目标的 trainer
#   往返 AbsRel > 0.0849    → 直接否掉：上界还不如现有的像素空间目标
#   0.05 – 0.0849 之间      → 要权衡，看它换来的多步能力值不值
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   MANIFEST   默认训练集全量（只抽样，不全跑）
#   PSEUDO_DIR 默认 pseudo_gt/train_full/calibrated_pseudo_depth
#   LIMIT      抽样帧数，默认 200（在全集上均匀铺开）
#   TAG        输出目录名，默认 vae_ceiling
#
#   sbatch ~/Iris/slurm/vae_ceiling.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

MANIFEST="${MANIFEST:-$IRIS_MANIFEST_DIR/ms2_train_day2seq_clip75_20260728.jsonl}"
PSEUDO_DIR="${PSEUDO_DIR:-$IRIS_RUNS/pseudo_gt/train_full/calibrated_pseudo_depth}"
OUT="$IRIS_RUNS/analysis/${TAG:-vae_ceiling}"

case "$(basename "$MANIFEST")" in
  *test*) echo "!! 拒绝 test manifest —— 这是训练侧的可行性门禁"; exit 1 ;;
esac
[[ -f "$MANIFEST" ]]   || { echo "!! 找不到 manifest: $MANIFEST"; exit 1; }
[[ -d "$PSEUDO_DIR" ]] || { echo "!! 找不到伪 GT 目录: $PSEUDO_DIR"; exit 1; }
echo "伪 GT 帧数: $(ls "$PSEUDO_DIR" | wc -l)"

cd "$IRIS_REPO"
python tools/probe_vae_roundtrip_ceiling.py \
  --manifest "$MANIFEST" --ms2-root "$IRIS_MS2_ROOT" \
  --pseudo-dir "$PSEUDO_DIR" --output-dir "$OUT" \
  --limit "${LIMIT:-200}" --local-files-only
rc=$?

echo "=== 退出码 $rc ==="
echo "结果: $OUT/vae_ceiling.json"
exit $rc
