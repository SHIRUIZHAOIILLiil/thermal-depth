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
#
# 默认建 iris（训练用）。caption 生成要的是另一套：InternVL3-8B 需要
# transformers>=4.49，而 iris 钉在 4.40.1（diffusers 0.28.0 / Lotus 管线的要求），
# 升级会打断训练线 —— 详见 environment_caption.yaml 里的注释。
#
#   sbatch $HOME/Iris/slurm/build_env.sbatch                                # iris
#   sbatch --export=ALL,ENV_YAML=environment_caption.yaml,ENV_NAME=caption \
#          $HOME/Iris/slurm/build_env.sbatch                                # caption

set -x
module load miniforge/24.7.1 cuda

ENV_YAML="${ENV_YAML:-environment.yaml}"
ENV_NAME="${ENV_NAME:-iris}"

echo "=== 开始建环境 $ENV_NAME（来自 $ENV_YAML） $(date) ==="
conda env create -f "$HOME/Iris/slurm/$ENV_YAML" -y || \
  conda env update -f "$HOME/Iris/slurm/$ENV_YAML" --prune

echo "=== 验证 $(date) ==="
conda activate "$ENV_NAME"
if [[ "$ENV_NAME" == "caption" ]]; then
  # 这套环境里没有 diffusers / bitsandbytes，只验 caption 真正要用的
  python -c "import torch, transformers, PIL, timm; print('torch', torch.__version__, 'cuda_built', torch.version.cuda); print('transformers', transformers.__version__)"
else
  python -c "import torch, diffusers, transformers, bitsandbytes; print('torch', torch.__version__, 'cuda_built', torch.version.cuda); print('diffusers', diffusers.__version__); print('transformers', transformers.__version__)"
fi

echo "=== home 配额 ==="
quota -s
echo "=== 完成 $(date) ==="
