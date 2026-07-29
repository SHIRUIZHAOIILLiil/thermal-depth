#!/bin/bash
#SBATCH --job-name=iris_build_env
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 建 conda 环境。不申请 GPU —— 装包不需要，而且 GPU 是全集群最稀缺的资源，
# 拿它来跑 pip 是浪费。不指定 --partition，走默认的标准分区（52 个节点，基本秒进）。

set -x
module load miniforge/24.7.1 cuda

echo "=== 开始建环境 $(date) ==="
conda env create -f "$HOME/Iris/slurm/environment.yaml" -y || \
  conda env update -f "$HOME/Iris/slurm/environment.yaml" --prune

echo "=== 验证 $(date) ==="
conda activate iris
python -c "import torch, diffusers, transformers, bitsandbytes; print('torch', torch.__version__, 'cuda_built', torch.version.cuda); print('diffusers', diffusers.__version__); print('transformers', transformers.__version__)"

echo "=== home 配额 ==="
quota -s
echo "=== 完成 $(date) ==="
