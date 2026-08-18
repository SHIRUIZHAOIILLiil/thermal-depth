#!/bin/bash
#SBATCH --job-name=iris_caption
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#
# 用 InternVL3-8B 重新生成 MS2 热像 caption（prompt 版本默认 thermal_depth_v3_1）。
#
# 为什么要上集群：本机 RTX 5090 实测 1.62 s/帧，28,302 帧约 12.6 小时，
# 而且用户的居住环境不适合长时间跑本机。分片后每片约 1.3 小时。
#
#   # ① 先预取权重（只需一次，不占 GPU）：
#   cd $SCRATCH/logs && sbatch $HOME/Iris/slurm/prefetch_models.sbatch
#
#   # ② 三个 split 各投一个 array（%12 是 QoS gpulimits 的 15 卡上限留了余量）：
#   sbatch --array=0-11%12 --export=ALL,SPLIT=train $HOME/Iris/slurm/caption_ms2.sbatch
#   sbatch --array=0-3%4   --export=ALL,SPLIT=val   $HOME/Iris/slurm/caption_ms2.sbatch
#   sbatch --array=0-1%2   --export=ALL,SPLIT=test  $HOME/Iris/slurm/caption_ms2.sbatch
#
# 分片数直接取自 SLURM_ARRAY_TASK_COUNT，所以改 --array 就等于改分片，
# 不需要同步改脚本里的任何常数。
#
# 超时或被抢占后**直接重投同一条命令即可**：--resume 是默认开启的，
# 每片有自己的输出文件，已成功的行会被跳过。

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

# common.sh 激活的是 iris 环境（transformers 钉在 4.40.1，Lotus 管线的要求）。
# InternVL3-8B 要 transformers>=4.49，两者不能共存 —— 见 environment_caption.yaml。
conda activate caption

SPLIT="${SPLIT:?SPLIT 未设置 —— 取值 train / val / test}"
PROMPT_VERSION="${PROMPT_VERSION:-thermal_depth_v3_1}"
CAPTION_REPO="${CAPTION_REPO:-$SCRATCH/captioning}"
CAPTION_SCRIPT="$CAPTION_REPO/scripts/generate_captions.py"
INPUT_MANIFEST="${INPUT_MANIFEST:-$SCRATCH/manifests/capinput/ms2_${SPLIT}_capinput_20260818.jsonl}"
OUT_ROOT="${OUT_ROOT:-$SCRATCH/captions/${PROMPT_VERSION}/${SPLIT}}"

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"

[[ -f "$CAPTION_SCRIPT"  ]] || { echo "!! 找不到 captioner：$CAPTION_SCRIPT"; exit 1; }
[[ -f "$INPUT_MANIFEST"  ]] || { echo "!! 找不到输入清单：$INPUT_MANIFEST"; exit 1; }

# 权重必须已在缓存里。让每个 array 任务自己去下载会并发写坏缓存，
# 所以这里宁可直接失败，也不要沉默地触发十几个并行下载。
MODEL_DIR="$HF_HOME/hub/models--OpenGVLab--InternVL3-8B"
[[ -d "$MODEL_DIR" ]] || {
  echo "!! InternVL3-8B 不在 $HF_HOME —— 先跑 prefetch_models.sbatch"; exit 1; }

mkdir -p "$OUT_ROOT" "$SCRATCH/shards"
SHARD_MANIFEST="$SCRATCH/shards/${SPLIT}_${PROMPT_VERSION}_${NSHARDS}x_${SHARD}.jsonl"

# 清单里存的是相对 MS2 根的路径，而 captioner 按 **清单自己所在目录** 解析。
# 分片清单不在 MS2 根下，所以这里必须改写成绝对路径 —— 否则每一帧都会
# 报成读取失败，而不是报路径错。（这个坑在 bakeoff prepare 里也踩过。）
python - "$INPUT_MANIFEST" "$SHARD_MANIFEST" "$SHARD" "$NSHARDS" "$IRIS_MS2_ROOT" <<'PY'
import json, sys
from pathlib import Path
src, dst, shard, nshards, root = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
rows = [json.loads(l) for l in open(src, encoding="utf-8")]
# 跨步取而非切块：连续帧几乎是同一张图，按块分会让各片的场景分布差很远，
# 某一片失败时补跑的代价也集中在一小段路上。
picked = rows[shard::nshards]
with open(dst, "w", encoding="utf-8") as fh:
    for r in picked:
        for key in ("thermal_path", "rgb_path"):
            v = r.get(key)
            if v and not v.startswith("/"):
                r[key] = (root / v).as_posix()
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[shard] {shard}/{nshards}: {len(picked)}/{len(rows)} 帧 -> {dst}", flush=True)
probe = Path(picked[0]["thermal_path"])
if not probe.is_file():
    raise SystemExit(f"!! 首帧不存在：{probe}\n   检查 IRIS_MS2_ROOT={root}")
print(f"[shard] 首帧已验证：{probe}", flush=True)
PY

echo "=== split=$SPLIT prompt=$PROMPT_VERSION shard=$SHARD/$NSHARDS ==="
echo "=== 输出 -> $OUT_ROOT ==="

srun python "$CAPTION_SCRIPT" \
  --manifest "$SHARD_MANIFEST" \
  --output-dir "$OUT_ROOT" \
  --input-modality thermal \
  --thermal-render grayscale \
  --models internvl3_8b \
  --prompt-type "$PROMPT_VERSION" \
  --source-tag "${SPLIT}_shard${SHARD}of${NSHARDS}"

echo "=== shard $SHARD 完成 ==="
