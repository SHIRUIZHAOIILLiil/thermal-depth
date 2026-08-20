#!/bin/bash
#SBATCH --job-name=iris_caption
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
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
# 首次提交时分片数取自 SLURM_ARRAY_TASK_COUNT，所以整段投的话改 --array 就等于
# 改分片。但**只补投其中几片时必须显式带 NSHARDS**，理由见下面 NSHARDS 处的注释。
#
# 超时或被抢占后**重投整段同一条命令最省心**：--resume 是默认开启的，
# 每片有自己的输出文件，已完成的片会在几秒内跳过退出。
#
# ⚠️ --time 是**跟着每片帧数走的**，默认值只对 1,600 帧左右的分片成立，换数据集
# 或换分片数时必须重算，别直接沿用。
#
#   Aire 的 L40S 实测 **2.0-2.36 s/帧**（18 片 × 1,663 帧，41:54 - 1:05:23）。
#   本机 RTX 5090 是 1.62 s/帧 —— **别拿本机速率去估集群时限**，会短算三成。
#   估法：每片帧数 × 2.4 s，再留 40% 余量。
#
#   1,663 帧/片（day2seq 19,949 ÷ 12） -> 约 56 分钟 -> --time=01:30:00（本文件默认）
#   3,190 帧/片（官方 train 76,547 ÷ 24） -> 约 2:05  -> --time=03:00:00
#
# 也别一律往大了填：Slurm backfill 让短作业插进大作业的空档先跑，时限报大等于
# 主动放弃插空机会（slurm/README.md 也写着这条）。
#
# 内存同理：sacct 实测主机峰值 1.42-4.01 GB，16G 留了四倍余量。
# GPU 是 L40S ×3/节点，全集群满载时只能等 —— 那种情况下少要 CPU 也换不来卡。

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
# ⚠️ 分片总数必须是"这个 split 一共切成几片"，不是"这次投了几个任务"。
# SLURM_ARRAY_TASK_COUNT 等于本次 --array 的任务数，所以只补投失败的几片时
# （例如 --array=5-11）它会变成 7 而不是 12，于是 rows[5::7] 和 rows[5::12]
# 是完全不同的帧集合 —— 补跑会既重复又漏帧，而 --resume 按帧去重、看不出异常。
# 因此补投时必须显式带上 NSHARDS，保持与首次提交一致：
#   sbatch --array=5-11 --export=ALL,SPLIT=train,NSHARDS=12 caption_ms2.sbatch
NSHARDS="${NSHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
if [[ "${SLURM_ARRAY_TASK_COUNT:-0}" -gt 0 && "$NSHARDS" -ne "$SLURM_ARRAY_TASK_COUNT" ]]; then
  echo "note: NSHARDS=$NSHARDS 与本次 array 任务数 ${SLURM_ARRAY_TASK_COUNT} 不同 —— 按补投处理"
fi
(( SHARD < NSHARDS )) || { echo "!! SHARD=$SHARD 超出 NSHARDS=$NSHARDS"; exit 1; }

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

# CAPTION_EXTRA_ARGS 用于补跑那种一次性场景，最典型的是
# --no-fail-on-quality-error：质量检查拦下的帧会写成 error 行，而
# build_thermal_caption_manifest.py 过滤掉 error 行之后那帧就"没有 caption"，
# 于是它在训练时等同于空 prompt —— 那是另一条实验臂，混进来就污染了。
# 宁可要一句啰嗦的 caption，也不要一个静默变成空 prompt 的样本。
srun python "$CAPTION_SCRIPT" \
  --manifest "$SHARD_MANIFEST" \
  --output-dir "$OUT_ROOT" \
  --input-modality thermal \
  --thermal-render grayscale \
  --models internvl3_8b \
  --prompt-type "$PROMPT_VERSION" \
  --source-tag "${SPLIT}_shard${SHARD}of${NSHARDS}" \
  ${CAPTION_EXTRA_ARGS:-}

echo "=== shard $SHARD 完成 ==="
