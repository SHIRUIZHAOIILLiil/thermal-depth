# Aire 超算迁移状态（2026-07-29）

把 MS2 六线对比（`tools/train_route_suite.py`，20 epoch 套件）搬到利兹大学 Aire HPC 的进度记录。
写给另一个对话核对用——**每一条状态都标注了是「已验证」还是「未验证」**。

---

## 1. 目标任务

`docs/ROUTE_SUITE_20EPOCH_PLAN_20260725.md` 定义的六条线，即 `tools/train_route_suite.py`
里 `ROUTES` 字典的六项：

| 线 | 输入 | Condition 来源 | 训练模块 | 可训练参数 |
|---|---|---|---|---:|
| `a_rgb_unet` | RGB | 冻结 VAE latent | U-Net | 867.57 M |
| `b_thermal_unet` | Thermal | 冻结 VAE latent | U-Net | 867.57 M |
| `c1_vae_adapter` | Thermal | VAE latent + Adapter | Adapter | 7.11 M |
| `c2_vae_adapter_unet` | Thermal | VAE latent + Adapter | Adapter + U-Net | 874.67 M |
| `d1_anythermal_adapter` | Thermal | AnyThermal 特征 + Adapter | Adapter | 9.41 M |
| `d2_anythermal_adapter_unet` | Thermal | AnyThermal 特征 + Adapter | Adapter + U-Net | 876.98 M |

> ⚠️ **不是** `tools/run_six_route_caption_ablation.sh` 里的那套 a–f（route_a…route_f）。
> 那套是推理评测流程，两套的 `b` 含义不同：本套 `b` 是 thermal + U-Net，
> 旧套 `b` 是 RGB + U-Net（≈ 本套的 `a`）。本文档只涉及新套件。

---

## 2. 集群环境（已验证）

| 项 | 值 |
|---|---|
| 登录 | `ssh -J sc23sz@rash.leeds.ac.uk sc23sz@aire.leeds.ac.uk` |
| 认证 | 密码 + **Duo 双因素**；**不支持 SSH 公钥** |
| 调度器 | Slurm（不是 ARC4 的 SGE） |
| `gpu` 分区 | 28 节点 × 3 × NVIDIA L40S 48GB = 84 卡 |
| GPU 节点规格 | **24 核 / 250GB / 3 卡** → 每卡配 8 核 |
| 最长运行 | **48h**；`DefaultTime=00:00:00`，`--time` 必填 |
| QoS `gpulimits` | 每人 **15 GPU + 120 CPU**，作业数/提交数不限 |
| 实测负载 | 84 张卡常年 80+ 在跑，排队是常态 |
| `$HOME` | 65 GB / 150 万 inode |
| `$SCRATCH` | **`/mnt/scratch/sc23sz`**（⚠️ 不是官方文档写的 `/mnt/scratch/users/$USER`），1 TB / 150 万 inode |
| `$TMP_SHARED` | 未开通（变量为空） |
| 计算节点外网 | **可直连**（conda 建环境时从 PyPI 拉 torch 成功验证） |

---

## 3. 已搬运的内容

### 3.1 代码 — ✅ 已验证

- 分支 **`aire-hpc-setup`**，已推送到 `origin`（`github.com/SHIRUIZHAOIILLiil/thermal-depth`）
- 集群上位置：`~/Iris`（`/users/sc23sz/Iris`），由 `git clone --branch aire-hpc-setup --single-branch` 拉取
- `main` **未改动**，仍停在 `1ba59a96c`

提交记录：

| commit | 内容 |
|---|---|
| `ce5901280` | MS2 研究代码，178 文件 / 37,865 行 |
| `a11c8f169` | Aire 集群脚本 `slurm/`，9 文件 / 609 行 |
| `d9277f186` | 修 smoke/overfit 输出目录命名 |
| `516a99831` | 新增权重预取作业；修 `common.sh` 的 CPU 节点兼容 |

`ce5901280` 纳入的目录：`tools/`（82 py + 15 sh）、`models/`（13 py，含 `anythermal_*`）、
`ms2_eval/`（16 py）、`tests/`（23 py）、`configs/`、`docs/`、`AGENTS.md`、
`lotus/evaluation/dataset_depth/ms2_thermal_dataset.py`、`lotus/audit_lotus_d_shapes.py`。

**推送体积仅 85.4 MB**。`.git/objects` 本地有 55 GB，但那些是早期提交后被 reset 的
**不可达对象**（含多个 3.3 GB 的 `checkpoint_final.pt`），不进推送。
本地可用 `git gc --prune=now --aggressive` 回收约 54 GB（未执行）。

### 3.2 数据 — ✅ 已验证

位置：`$SCRATCH/data/ms2`。本地的 `E:\dataset\ms2`（5 序列）和 `E:\dataset\ms2_partial`
（3 序列）**合并成单一根目录**——manifest 里路径全是相对的且序列名不重叠，
所以所有 manifest 共用一个 `--ms2-root` 即可解析。

**只传 manifest 实际引用的 5 个子目录**：

```
sync_data/<seq>/thr/img_left          热成像输入
sync_data/<seq>/rgb/img_left          RGB 输入
proj_depth/<seq>/thr/depth_filtered   热成像深度 GT
proj_depth/<seq>/thr/depth
proj_depth/<seq>/rgb/depth_filtered   RGB 深度（RGB-teacher 线）
```

`nir/`、`lidar/`、`gps_imu/`、所有 `img_right/`、`depth_multi/`、`intensity*/` **未传**——
扫过全部 27 个 manifest 从未出现，`grep` 代码也确认无处读取。

| 序列 | 本地来源 | 用途 | 文件数（本地=远端） |
|---|---|---|---:|
| `_2021-08-06-10-59-33` | ms2 | caption train / day2seq | 52,210 |
| `_2021-08-06-11-23-45` | ms2 | caption val / day_inference | 29,050 |
| `_2021-08-06-11-37-46` | ms2 | caption test / day2seq | 47,575 |
| `_2021-08-06-16-59-13` | ms2_partial | test，`official_split=test_rainy` | 32,415 |
| `_2021-08-13-16-08-46` | ms2_partial | test，`official_split=val`，day_clear | 12,715 |
| `_2021-08-13-21-58-13` | ms2_partial | test，`official_split=val`，night | 12,695 |
| `_2021-08-06-16-19-00` | ms2 | test_rainy | 59,340 |
| `_2021-08-13-21-36-10` | ms2 | `ms2_manifest.jsonl` 全局 | 59,700 |
| | | **合计** | **305,700** |

裁剪效果：全模态 249.2 GB / 1,467,464 文件（占 inode 配额 97.8%）
→ 实传 **57.66 GB / 305,830 inode（20.4%）**。因此**不需要申请配额提升**。

`slurm/transfer_ms2.sh verify` 逐序列比对本地/远端文件数，**全部一致**。

### 3.3 Manifest — ✅ 已验证

| 位置 | 内容 |
|---|---|
| `$SCRATCH/data/ms2/*.jsonl` | 8 个，13.9 MB（相对路径从此处起算） |
| `$SCRATCH/manifests/sequence_level_internvl3_8b/` | 19 个，193 MB（caption manifest） |

caption **已内联在 jsonl 里**（抽查 2000 条全部非空），`caption_path` 字段仅为溯源信息、
代码中无任何地方读取（已 grep 确认）。因此 `E:\project\captioning\` **无需搬运**。

### 3.4 conda 环境 — ✅ 已验证

环境名 `iris`，由 `slurm/build_env.sbatch` 在计算节点建成（02:03 提交，02:14 完成）。
`$HOME` 占用 11.9 GB / 51,299 inode。

```
torch 2.3.1+cu121   cuda_built 12.1   （L40S 是 Ada sm_89，cu121 支持）
diffusers 0.28.0
transformers 4.40.1
bitsandbytes        （同一条 import 未报错）
```

> `bitsandbytes` 是 `lotus/requirements.txt` 漏掉的依赖（`--use_8bit_adam` 运行时才 import），
> 已补进 `slurm/environment.yaml`。

---

## 4. 有意排除的内容

| 内容 | 体积 | 原因 |
|---|---|---|
| `outputs/` | 188 GB | 实验产物；已加入 `.gitignore` |
| `archive/` | 87.6 GB | 历史产物；已加入 `.gitignore` |
| `logs/`、`.codex/` | 47 MB | 日志与工具缓存 |
| MS2 未引用模态 | 192.3 GB | `nir/` `lidar/` `gps_imu/` `img_right/` `depth_multi/` `intensity*/` |
| `ms2/sync_data/*.tar.bz2` | 217 GB | 15 个未解压序列，无任何 manifest 引用 |
| `ms2_partial/*.tar(.bz2)` | ~85 GB | 与已解压目录完全重复 |
| 本地 checkpoint | — | 新套件从头训练，不依赖既有 checkpoint |

---

## 5. 代码改动

| 文件 | 改动 | 对本地的影响 |
|---|---|---|
| `tools/train_route_suite.py` | `MANIFEST_DIR` 读 `IRIS_MANIFEST_DIR`；`--ms2-root` 默认读 `IRIS_MS2_ROOT` | **无**——两者的 fallback 就是原来的本地硬编码路径 |
| `.gitignore` | 新增 `outputs/` `archive/` `logs/` `.codex/` `__pycache__/` `*.pt` `*.pth` `*.ckpt` `*.npy` `*.npz` | 原有的 `/output`（单数）挡不住实际在用的 `outputs/` |
| `.gitattributes` | 新建，`*.sh`/`*.sbatch`/`*.yaml` 强制 LF | 防止 Windows 检出成 CRLF 导致 bash 报 `$'\r': command not found` |

> **本地训练行为未被改变**：环境变量不设时，两个路径的默认值与改动前完全一致。

---

## 6. 新增脚本（`slurm/`）

| 文件 | 用途 | 状态 |
|---|---|---|
| `README.md` | 集群约束速查、一次性设置、排队策略 | — |
| `environment.yaml` | conda 环境定义 | ✅ 已用于建环境 |
| `build_env.sbatch` | CPU 节点建 conda 环境 | ✅ 已成功执行 |
| `common.sh` | 公共环境：module、conda、路径导出、数据存在性检查 | ✅ 已验证 |
| `transfer_ms2.sh` | 数据/manifest 传输与校验（`check`/`manifests`/`data`/`verify`） | ✅ 已验证 |
| `overnight.sh` | 过夜编排：manifest → 建环境作业 → 数据 → 校验 | ✅ 已执行 |
| `prefetch_models.sbatch` | 预取 HuggingFace 权重 | ⏳ **未执行** |
| `route_suite.sbatch` | 六线训练作业（SMOKE / OVERFIT / 全量三种模式） | ⏳ **未验证通过** |
| `smoke_lotus_d.sbatch`、`train_lotus_d.sbatch` | ⚠️ 上游 **baseline 线**（hypersim/vkitti），**与六线无关**，历史遗留 | 未使用 |

---

## 7. 当前就绪状态

| 环节 | 状态 |
|---|---|
| 集群账号与登录 | ✅ |
| 代码在集群上 | ✅ `~/Iris`，分支 `aire-hpc-setup` |
| 数据在集群上 | ✅ 8 序列 / 305,700 文件，逐序列校验通过 |
| manifest 在集群上 | ✅ 27 个 |
| conda 环境 | ✅ `iris`，torch/CUDA 已验证 |
| Slurm 资源配比 | ✅ 1 GPU + 8 核 + 80G，已实际分配到 L40S |
| 路径解析（环境变量覆盖） | ✅ 作业日志确认 `ms2=` / `mfst=` 指向 scratch |
| **HuggingFace 权重** | ❌ **未拉取** |
| **冒烟测试** | ❌ **未通过** |
| **六线全量** | ❌ **未启动** |

### 唯一一次实跑：job 6913886（失败）

```
=== job 6913886 on gpu008 ===
NVIDIA L40S, 46068 MiB
ms2 =/mnt/scratch/sc23sz/data/ms2
mfst=/mnt/scratch/sc23sz/manifests/sequence_level_internvl3_8b
...
ValueError: --smoke-updates requires an output dir name containing 'smoke'.
```

失败点是 `validate_args()` 的护栏（防止冒烟结果覆盖正式训练输出目录），
不是环境问题——**GPU 分配、conda 激活、数据与 manifest 路径、存在性检查全部通过**。
已由 `d9277f186` 修复（输出目录按模式加 `smoke_` / `overfit_` 前缀），**修复后尚未重跑**。

---

## 8. 待办（按顺序）

**① 预取 HF 权重**（必须在投六线之前，否则六个作业并发写同一缓存目录会下坏）

```bash
cd ~/Iris && git pull && cd $SCRATCH/logs && sbatch ~/Iris/slurm/prefetch_models.sbatch
```

需要的两个仓库：

| 模型 | 用途 | 谁需要 |
|---|---|---|
| `jingheya/lotus-depth-g-v2-1-disparity` | Lotus 主干 | 六条线全部 |
| `theairlabcmu/AnyThermal` | 热成像编码器（`trust_remote_code`） | 仅 `d1` / `d2` |

**② 冒烟**（1 卡 / 30 分钟墙钟，backfill 易插队）

```bash
cd $SCRATCH/logs && ROUTE=b_thermal_unet SMOKE=1 sbatch -J smoke_b --time=00:30:00 --export=ALL,ROUTE,SMOKE ~/Iris/slurm/route_suite.sbatch
```

**③ 六线全投**（QoS 允许 15 卡并发、作业数不限）

```bash
cd $SCRATCH/logs && for r in a_rgb_unet b_thermal_unet c1_vae_adapter c2_vae_adapter_unet d1_anythermal_adapter d2_anythermal_adapter_unet; do ROUTE=$r sbatch -J "route_$r" --export=ALL,ROUTE ~/Iris/slurm/route_suite.sbatch; done
```

输出落在 `$SCRATCH/runs/route_suite/<route>/`，日志在 `$SCRATCH/logs/`。
`route_suite.sbatch` 报 24h（分区上限 48h），跑不完时重投同一 `ROUTE` 会自动
`--resume latest.pt` 续跑。

---

## 9. 已知问题与风险

1. **`route_suite.sbatch` 尚未跑通任何一次训练**。`validate_args()` 里还有其他约束
   （`--micro-batch-size` 必须为 1、`--freeze-adapter` 与 route 的组合限制等），
   后续可能再撞到。
2. **排队是主要瓶颈**，不是算力。84 卡常年 80+ 占用。`--time` 报准能吃到 backfill 红利。
3. **Windows 侧 SSH 无连接复用**（Windows OpenSSH 不支持 ControlMaster），每次连接
   要两次 Duo。WSL 侧已配 `ControlPersist 8h`，迭代循环建议走 WSL。
4. **本地 `.git` 有 55 GB 不可达对象**，不影响推送，但占磁盘。
5. `origin/main` 与本地 `main` **无共同祖先**（`git merge-base` 为空），
   所以走的是新分支 `aire-hpc-setup` 而非推 `main`。这条历史分叉尚未解决。
6. 六线全量的**实际耗时未知**——本地 20 epoch 的耗时未换算到 L40S，
   24h 是否够跑完 20 epoch 需要第一个作业跑起来才知道。

---

## 10. 请另一个对话核对的点

1. 六条线是否确为 `train_route_suite.py` 的 `ROUTES`（a / b / c1 / c2 / d1 / d2），
   而非 `run_six_route_caption_ablation.sh` 的 route_a–route_f？
2. 训练/验证 manifest 的默认值是否正确：
   - train `ms2_train_day2seq_20260725.jsonl`
   - val `ms2_fixed_sequence_internvl3_8b_filtered_caption_val_rgb_depth_v1_clip75_rerun_20260714.jsonl`
   - 记忆里提到「待决：1m 去冗余重跑」，是否应改用 `ms2_train_day2seq_1m_20260725.jsonl`？
3. 六线全量的超参是否沿用 `train_route_suite.py` 默认（20 epoch、
   `--snapshot-epochs 1,2,5,10,15`、micro-batch 1、GAS 4）？
4. 未传的 `nir/` `lidar/` `img_right/` 等模态，后续实验是否确实用不到？
   （需要时：`KEEP_ALL=1 bash slurm/transfer_ms2.sh data`）
5. `16-19-00`、`21-36-10` 两个序列只被 `ms2_test_rainy_*` 和全局 manifest 引用，
   六线训练是否会用到？
