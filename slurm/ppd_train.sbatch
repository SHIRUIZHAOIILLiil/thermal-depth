#!/bin/bash
#SBATCH --job-name=ppd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Pixel-Perfect Depth 在 MS2 热像上微调。冒烟和正式共用这一个脚本 —— 因为
# 下面那个守卫在正式那次才真正值钱。
#
# ⚠️⚠️ 这条线有一个静默失败，2026-09-21 已经中过一次：
#
#   他们的 find_last_ckpt_path 只 glob "$output_dir/checkpoints/*.ckpt"，
#   **找不到就从随机初始化开始训，不报错**。而 dirpath 是 ${output_dir}/checkpoints/，
#   所以**每开一条新臂那个目录都是空的** —— 新臂默认就是随机初始化。
#   7927066 那次跑了 5 步，COMPLETED、退出 0，日志里只有一行
#   "[INFO] No checkpoint found, set model_path to None" 泄了底。
#
#   更阴的是第二次：那次跑完会往同一个目录写 last.ckpt，于是**下次 glob 找得到了**，
#   连那句话都不会再出现 —— 从一个随机初始化的模型接着训，全程无声。
#
# 所以这个脚本启动前查、启动后再查一遍日志，两道都不通过就退出。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   TAG       输出目录名（$SCRATCH/runs/ppd_ms2/$TAG），默认 smoke
#   BATCHES   pl_trainer.limit_train_batches，默认 5。正式跑传 0 ＝ 不限
#   EPOCHS    pl_trainer.max_epochs，默认 1
#   SEED_CKPT 起步权重。默认 $SCRATCH/runs/ppd_ms2/v1/checkpoints/e000-s000000.ckpt
#             （integrations/ppd/prepare_ms2.py 从 ppd.pth 转出来的那个）
#   ALLOW_ANY_CKPT=1  放行 dirpath 下的其它 .ckpt（真要断点续训时才用）
#
# ── 用法 ────────────────────────────────────────────────────────────
#   冒烟（5 批，确认真的从已发布权重起步）：
#     sbatch --export=ALL ~/Iris/slurm/ppd_train.sbatch
#
#   正式：
#     TAG=v1 BATCHES=0 EPOCHS=20 \
#       sbatch -J ppd_v1 --time=24:00:00 --export=ALL,TAG,BATCHES,EPOCHS \
#       ~/Iris/slurm/ppd_train.sbatch
#
# 不 source common.sh：PPD 有自己的 conda 环境（$HOME/.conda/envs/ppd），
# 走绝对路径的 python，不需要 iris 那套 module load。

set -euo pipefail

: "${SCRATCH:?SCRATCH 未定义 —— 确认已在 Aire 上运行}"
IRIS_REPO="${IRIS_REPO:-$HOME/Iris}"
PPD_REPO="${PPD_REPO:-$SCRATCH/ppd_repo}"
PPD_PY="${PPD_PY:-$HOME/.conda/envs/ppd/bin/python}"

TAG="${TAG:-smoke}"
BATCHES="${BATCHES:-5}"
EPOCHS="${EPOCHS:-1}"
OUT_DIR="$SCRATCH/runs/ppd_ms2/$TAG"
CKPT_DIR="$OUT_DIR/checkpoints"
SEED_CKPT="${SEED_CKPT:-$SCRATCH/runs/ppd_ms2/v1/checkpoints/e000-s000000.ckpt}"
SPLIT="${SPLIT:-$SCRATCH/runs/ppd_ms2/splits/train.json}"

export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"

for path in "$PPD_REPO" "$PPD_PY" "$SPLIT" "$SEED_CKPT"; do
  [[ -e "$path" ]] || { echo "!! 找不到 $path"; exit 1; }
done

# 我们的两个文件住在 Iris 仓库里，跑之前拷进克隆树 —— 放在这里而不是提交命令里，
# 是为了让「作业跑的版本」永远等于「仓库里的版本」。
for src in integrations/ppd/train_ms2_thermal.yaml integrations/ppd/ms2_thermal.py; do
  [[ -f "$IRIS_REPO/$src" ]] || { echo "!! 找不到 $IRIS_REPO/$src"; exit 1; }
done
cp "$IRIS_REPO/integrations/ppd/train_ms2_thermal.yaml" "$PPD_REPO/ppd/configs/"
cp "$IRIS_REPO/integrations/ppd/ms2_thermal.py"        "$PPD_REPO/ppd/data/"

# --- 守卫一：起步权重必须在，而且不能有别的 .ckpt 抢 glob ------------------
mkdir -p "$CKPT_DIR"
shopt -s nullglob
present=("$CKPT_DIR"/*.ckpt)
shopt -u nullglob

if [[ ! -f "$CKPT_DIR/$(basename "$SEED_CKPT")" ]]; then
  echo "!! $CKPT_DIR 下没有起步权重 $(basename "$SEED_CKPT")"
  echo "   现有的："; printf '     %s\n' "${present[@]:-（空）}"
  echo "   拷过去："
  echo "     cp $SEED_CKPT $CKPT_DIR/"
  echo "   ⚠️ 不拷就跑 ＝ 从随机初始化开始，而且不会报错。"
  exit 1
fi

if [[ "${ALLOW_ANY_CKPT:-0}" != "1" && "${#present[@]}" -gt 1 ]]; then
  echo "!! $CKPT_DIR 下有不止一个 .ckpt，glob 会挑错："
  printf '     %s\n' "${present[@]}"
  echo "   多出来的通常是上一次跑留下的 last.ckpt —— 那是**上次的产物**，"
  echo "   不是已发布权重。挪走它，或者 ALLOW_ANY_CKPT=1 明确表示要续训。"
  exit 1
fi

echo "=== 配置 ==="
echo "臂        = $TAG"
echo "输出      = $OUT_DIR"
echo "起步权重  = $(ls -la "$CKPT_DIR"/*.ckpt)"
echo "batches   = $BATCHES   epochs = $EPOCHS"
echo "split     = $SPLIT"
echo "============"
git -C "$IRIS_REPO" log --oneline -1

# --- 训练 -------------------------------------------------------------------
LOG="$OUT_DIR/train.log"
cd "$PPD_REPO"
args=(--cfg_file ppd/configs/train_ms2_thermal.yaml
      "output_dir=$OUT_DIR"
      "ms2_root=$SCRATCH/data/ms2"
      "ms2_split=$SPLIT"
      semantics_pth=checkpoints/depth_anything_v2_vitl.pth
      "pl_trainer.max_epochs=$EPOCHS")
[[ "$BATCHES" != "0" ]] && args+=("pl_trainer.limit_train_batches=$BATCHES")

set +e
"$PPD_PY" main.py "${args[@]}" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e

# --- 守卫二：日志自己招供了没有 --------------------------------------------
echo
echo "=== 起步权重核对 ==="
grep -nE "No checkpoint found|checkpoint|Trainable params|Non-trainable params" "$LOG" \
  | head -12 | sed 's/^/  /'

if grep -q "No checkpoint found" "$LOG"; then
  echo
  echo "⛔ 日志里出现 \"No checkpoint found\" —— 这一跑是从随机初始化开始的，作废。"
  echo "   守卫一放行了却还是这样，说明 config 的 dirpath 不是 $CKPT_DIR。"
  exit 1
fi
[[ "$status" -eq 0 ]] || { echo "!! 训练退出码 $status"; exit "$status"; }

# 下一次的地雷：这次写出来的 last.ckpt 会被 glob 找到，而且找到之后
# "No checkpoint found" 不会再出现 —— 守卫二就哑了，只剩守卫一挡着。
if [[ -f "$CKPT_DIR/last.ckpt" ]]; then
  echo
  echo "⚠️ 本次写出了 $CKPT_DIR/last.ckpt。"
  echo "   下次用同一个 TAG 跑，守卫一会拦下它（这是故意的）。"
  echo "   要重跑一次干净的：rm $CKPT_DIR/last.ckpt"
fi
echo
echo "[完成] $OUT_DIR"
