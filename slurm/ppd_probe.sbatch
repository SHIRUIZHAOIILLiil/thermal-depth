#!/bin/bash
#SBATCH --job-name=ppd_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 实测 PPD 的原生输出空间，并确认评估时加载的确实是我们训出来的权重。
#
# 为什么要有这一步：PPD 的输出空间**未声明** —— 论文说训练目标是 log 深度，
# 而 run.py 又对输出做了 min-max 上色，两边都没说 infer_image 到底返回什么。
# 猜错了整条评估链都是跨族打分，这个项目实测过差 30 倍。
#
# ⚠️ 顺带验一件同样要紧的事：run_external_depth_inference.py 用
# load_state_dict(..., strict=False)，一个键都对不上也不报错。它现在会打印
# 「匹配 N/M 个参数」并在低于一半时拒绝 —— 那一行是我们唯一能确认
# 「评估的是训练出来的模型」而不是随机初始化的证据。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   CKPT   要探的 checkpoint，默认 $SCRATCH/runs/ppd_ms2/v1/checkpoints/last.ckpt
#   FRAMES 探多少帧，默认 50
#
#   sbatch --export=ALL ~/Iris/slurm/ppd_probe.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

export PPD_REPO="${PPD_REPO:-$SCRATCH/ppd_repo}"
PPD_PY="${PPD_PY:-$HOME/.conda/envs/ppd/bin/python}"
[[ -x "$PPD_PY" ]] || { echo "!! 找不到可执行的 $PPD_PY"; exit 1; }
CKPT="${CKPT:-$SCRATCH/runs/ppd_ms2/v1/checkpoints/last.ckpt}"
FRAMES="${FRAMES:-50}"
SEM="${SEM:-$PPD_REPO/checkpoints/depth_anything_v2_vitl.pth}"
MANIFEST="${MANIFEST:-$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl}"

for path in "$PPD_REPO" "$CKPT" "$SEM" "$MANIFEST"; do
  [[ -e "$path" ]] || { echo "!! 找不到 $path"; exit 1; }
done

# 与训练一致。错了不报错，只让判定偏掉。
export IRIS_THERMAL_STRETCH="${IRIS_THERMAL_STRETCH:-percentile}"

cd "$IRIS_REPO"
git log --oneline -1
echo "=== 配置 ==="
echo "ckpt     = $CKPT"
echo "语义权重 = $SEM"
echo "帧数     = $FRAMES"
echo "thermal  = $IRIS_THERMAL_STRETCH"
echo "python   = $PPD_PY   (不是 iris 环境 —— 要 import 他们的依赖)"
echo "============"

"$PPD_PY" tools/run_external_depth_inference.py \
  --model pixel_perfect_depth \
  --manifest "$MANIFEST" \
  --data-root "$IRIS_MS2_ROOT" \
  --out-dir "$SCRATCH/runs/analysis/ppd_probe/raw_predictions" \
  --ppd-checkpoint "$CKPT" \
  --ppd-semantics "$SEM" \
  --stretch "$IRIS_THERMAL_STRETCH" \
  --probe-output-space \
  --limit "$FRAMES"

echo
echo "⭐ 上面两行是这次要的："
echo "   [weights] ... 匹配 N/M 个参数   —— 确认评估的是训练出来的权重"
echo "   判定：<空间> -> 评估时用 --align <档位>  —— 后面 val 扫描要用它"
