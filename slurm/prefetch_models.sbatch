#!/bin/bash
#SBATCH --job-name=iris_prefetch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 把 HuggingFace 权重预取到 $SCRATCH/hf_cache。
# 不申请 GPU —— 下载用不上，而且 GPU 是全集群最稀缺的资源。
#
#   cd $SCRATCH/logs && sbatch $HOME/Iris/slurm/prefetch_models.sbatch
#
# 必须在投六线之前跑完：六个作业同时启动会并发写同一个缓存目录，
# 容易把权重下坏，然后六个一起失败。

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

echo "=== HF_HOME=$HF_HOME ==="

python - <<'PY'
import os
from huggingface_hub import snapshot_download

REPOS = [
    "jingheya/lotus-depth-g-v2-1-disparity",  # 六条线都要
    "theairlabcmu/AnyThermal",                # 只有 d1 / d2 要
    # 天空掩码（Jasmine 附录 C.3 的 sky loss 要它）。取 Cityscapes 版：
    # 它的标签集里有 sky，域也是行车场景。约 1.6 GB。
    "facebook/mask2former-swin-large-cityscapes-semantic",
]

for repo in REPOS:
    print(f"\n=== 拉取 {repo} ===", flush=True)
    path = snapshot_download(repo)
    total = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(path)
        for f in fs
    )
    print(f"{repo} -> {path}  ({total / 1024**3:.2f} GB)", flush=True)
PY

echo
echo "=== 缓存占用 ==="
du -sh "$HF_HOME"
echo "=== scratch 配额 ==="
lfs quota -h -u "$USER" /scratch
