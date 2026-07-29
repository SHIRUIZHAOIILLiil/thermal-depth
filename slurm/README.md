# 在 Aire（利兹大学 HPC）上跑 Iris

## 集群约束速查

| 项 | 值 |
|---|---|
| 登录 | `ssh -J sc23sz@rash.leeds.ac.uk sc23sz@aire.leeds.ac.uk`（**不支持公钥**，密码 + MFA） |
| 调度器 | Slurm |
| `gpu` 分区 | 28 节点 × 3 × L40S 48GB = 84 卡；每节点 **24 核 / 250GB** |
| 最长运行 | **48h**（`--time` 无默认值，必填） |
| 每人上限 | **15 GPU + 120 CPU**（QoS `gpulimits`），作业数不限 |
| 每卡配额 | **8 核**（120/15），多要就压低并发上限 |
| `$HOME` | 65GB / 150 万 inode —— 只放代码和 conda 环境 |
| `$SCRATCH` | `/mnt/scratch/<user>`，1TB —— 数据、输出、缓存全放这 |

> 注意：ARC 文档写的 `/mnt/scratch/users/$USER` 与实际不符，以 `$SCRATCH` 变量为准。

## 一次性设置

```bash
# 1. 缓存重定向（HF 会拉 5-10GB 的 SD 权重，不改必爆 home）
cat >> ~/.bashrc <<'EOF'
export HF_HOME=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache
export PIP_CACHE_DIR=$SCRATCH/pip_cache
export CONDA_PKGS_DIRS=$SCRATCH/conda_pkgs
EOF
source ~/.bashrc

# 2. 目录
mkdir -p $SCRATCH/{data,runs,logs,hf_cache,torch_cache,pip_cache,conda_pkgs}

# 3. 代码
git clone https://github.com/SHIRUIZHAOIILLiil/thermal-depth.git ~/Iris

# 4. conda 环境（环境本身必须在 home，包缓存已被上面导到 scratch）
module load miniforge/24.7.1 cuda
conda env create -f ~/Iris/slurm/environment.yaml
```

## MS2 数据传输

本地数据分在两处（下载时形成的，不是设计）：`E:\dataset\ms2`（5 个序列）和
`E:\dataset\ms2_partial`（3 个序列）。manifest 里的路径全是相对的
（`sync_data/<seq>/...`、`proj_depth/<seq>/...`），两边序列名不重叠，
所以**远端合并成单一根目录** `$SCRATCH/data/ms2`，所有 manifest 共用一个 `--ms2-root`。

### 只传用得到的模态

扫过全部 27 个 manifest，被引用的路径只有 5 个：

```
sync_data/<seq>/thr/img_left          热成像输入
sync_data/<seq>/rgb/img_left          RGB 输入
proj_depth/<seq>/thr/depth_filtered   热成像深度 GT
proj_depth/<seq>/thr/depth
proj_depth/<seq>/rgb/depth_filtered   RGB 深度（RGB-teacher 线）
```

`nir/`、`lidar/`、`gps_imu/`、所有 `img_right/`（右目）、`depth_multi/`、
`intensity*/` 从未被任何 manifest 引用，代码里也没有任何地方读它们。丢掉后：

| | 体积 | 文件数 | 占 scratch inode |
|---|---|---|---|
| 全部模态 | 249.2 GB | 1,467,464 | 97.8% ⚠️ |
| **只保留被引用的** | **56.9 GB** | **305,700** | **20.4%** ✅ |

8 个序列（train/val/test + day/night/rainy 全部测试集）一次传完即可，
不需要申请配额提升。将来要右目/NIR/LiDAR：`KEEP_ALL=1 bash slurm/transfer_ms2.sh data`。

不传：`ms2/sync_data/*.tar.bz2`（217 GB，无 manifest 引用）、
`ms2_partial/*.tar` 与 `*.tar.bz2`（~85 GB，与解压目录重复）。
caption 已内联在 jsonl 里，`E:\project\captioning\` 也不用传
（`caption_path` 仅为溯源信息，代码不读）。

### 跑法

在 **WSL** 里跑（不是在 HPC 上），开 tmux 以防断线：

```bash
bash slurm/transfer_ms2.sh manifests && bash slurm/transfer_ms2.sh data
```

核对：

```bash
bash slurm/transfer_ms2.sh verify
```

脚本按序列拆成 8 段，中断只需重跑，已完整的会自动跳过。
用 tar 流式传输而非 rsync —— 几十万个小文件，rsync 的每文件往返开销在跳板机上不可接受。

## 提交作业

**从 `$SCRATCH/logs` 提交**，`%x-%j.out` 才会落在 scratch 而不是 home：

```bash
cd $SCRATCH/logs
sbatch $HOME/Iris/slurm/smoke_lotus_d.sbatch      # 先冒烟，20 步
sbatch $HOME/Iris/slurm/train_lotus_d.sbatch      # 冒烟过了再全量
```

常用查询：

```bash
squeue -u $USER                    # 我的队列
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ReqTRES%40   # 实际用量，用来校准下次申请
scancel <id>
sinfo -p gpu -O NodeList:16,Gres:20,GresUsed:20,StateLong:12   # 全集群空卡
```

## 排队策略

集群 GPU 长期接近饱和（84 张常有 80+ 在跑），所以：

- **`--time` 报准**，别一律填 48h。Slurm backfill 会让短作业插进大作业的空档先跑。
- 六线对比**一次全投**，作业数不限，让 Slurm 去抢卡，比手工串行快得多。
- 不要挂着 `srun --pty` 空转 —— 那是在占全集群最稀缺的资源。
- 48h 上限靠 `--resume_from_checkpoint=latest` 分段续跑；同一 `RUN_TAG` 重投即自动接上。

## 已知待办

- ⚠️ `smoke_lotus_d.sbatch` / `train_lotus_d.sbatch` 挂的是**上游 baseline 线**
  （`lotus/train_scripts/train_iris_d_depth.sh` + hypersim/vkitti），**不是**当前
  MS2 六线对比。真正的入口是 `tools/train_route_suite.py --route <name> --epochs 20`，
  对应的 sbatch 尚未写。
- `tools/train_route_suite.py` 的 `--ms2-root` 默认值硬编码为 `/mnt/e/dataset/ms2`，
  caption manifest 又指向 `E:\project\thermal-depth\...`，在集群上都会失效，
  需要改成读环境变量（`$SCRATCH/data/ms2`、`$SCRATCH/manifests/...`）。
- `e2e` / `marigold` / `internVL` 各有自己的依赖（`marigold/environment.yaml` 等），
  与 `iris` 环境冲突，需要单独建环境后再补对应的 `.sbatch`。
- `train_iris_d_depth.sh` 里 `--main_process_port` 写死为 13324。单卡（`distributed_type: NO`）
  不受影响；将来上多卡需要改成按 `$SLURM_JOB_ID` 取值，否则同节点两个作业会撞端口。
- `--use_8bit_adam` 依赖 `bitsandbytes`，原 `lotus/requirements.txt` 漏了，已在
  `slurm/environment.yaml` 补上。
