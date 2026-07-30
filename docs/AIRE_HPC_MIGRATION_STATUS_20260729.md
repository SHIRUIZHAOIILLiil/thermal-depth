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
| `prefetch_models.sbatch` | 预取 HuggingFace 权重 | ✅ 已执行（6914389） |
| `route_suite.sbatch` | 六线训练（SMOKE / OVERFIT / 全量），支持 CAPTION、INIT_FROM、FREEZE_ADAPTER | ✅ 冒烟已通过 |
| `eval.sbatch` | `--eval-checkpoint` 全量评估，吃 ROUTE/CKPT/TAG/VAL_MANIFEST/CAPTION_MODE | ✅ 已跑通（6914834） |
| `transfer_checkpoints.sh` | 传 checkpoint 与外部权重：`calib`/`stage1`/`caption`/`midas`/`bmsd` | ✅ calib、stage1 已传 |
| `anythermal_midas.sbatch` | 任务 4：原版 AnyThermal MiDaS 出预测 | ⏳ 待 midas/bmsd 传上去 |
| `analyze.sbatch` | 九分层区域分析，**不占 GPU**（纯读 npy，走标准分区） | ⏳ 待 raw 预测 |
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
| **HuggingFace 权重** | ✅ **已预取** 6.7 GB（job 6914389） |
| **冒烟测试** | ✅ **已通过**（job 6914833，1 分 34 秒） |
| **评估链路** | ✅ **已跑通**（job 6914834，全量 5,810 帧 8.3 分钟） |
| **跨机器标定** | ✅ **已通过**，差 0.000009 |
| 已上集群的 checkpoint | ✅ `calibration/epoch05_weights.pt`、`c1_stage1/best_weights.pt` |
| **四条线全量**（a / c2 / d1 / d2） | ❌ **未启动** |
| MiDaS + BMSD（任务 4） | ❌ **未传** |
| b+caption e3（任务 1a / 2） | ❌ **未传** |
| raw 预测导出（任务 3） | ❌ **代码未提交**，见 §14 |

### 作业记录

| Job | 内容 | 结果 |
|---|---|---|
| 6913886 | 冒烟（旧脚本） | FAILED — 目录命名护栏，`d9277f186` 已修 |
| 6913999 | 冒烟（pull 前提交） | FAILED — `sbatch` 在提交时快照脚本，跑的仍是旧版 |
| 6914389 | 预取 HF 权重 | ✅ Lotus 4.02 GB / AnyThermal 2.65 GB，缓存 6.7 GB |
| 6914833 | 冒烟 `b_thermal_unet` | ✅ COMPLETED 0:0，1 分 34 秒 |
| 6914834 | 标定 `b` e05 全量 val | ✅ COMPLETED 0:0，9 分 12 秒 |

**教训**：`sbatch` 在提交那一刻就把脚本快照进 spool，之后 `git pull` 不影响已排队的作业。
改完脚本必须先 pull 再提交。

### 跨机器标定结果（§12.6 的前置，已完成）

```
集群 abs_rel = 0.07749105      本地 abs_rel = 0.0775      差 0.000009
```

阈值 0.0005，实测小 55 倍。**集群结果可与本地 b（0.0775）、c1（0.1127）、
b+caption（e3）并进同一张表**，脚注注明双机即可，不必自成一组。

旁证：`val_samples 5810` 一致；`val_caption_mode empty` 与该 checkpoint 训练模式一致；
协议 `official BMSD ssi_disparity, min_depth 1e-3, max_depth 80` 同口径；
环境 `torch 2.3.1+cu121 / L40S / capability 8.9 / 44.4 GB`。

**副产物**：全量 5,810 帧评估 = **8.3 分钟**。`eval.sbatch` 的 `--time` 已据此从 4 小时
下调到 30 分钟，backfill 更易插队。

---

## 8. 待办（按顺序）

§7 的前置全部完成——权重已预取、冒烟已过、标定已过、c1 adapter 已在集群。
以下是**当前真正剩下的**。

### ① 立刻可投，无任何依赖

d2 是六条线里唯一从未冒烟过的，先单独验一次：

```bash
cd ~/Iris && git pull && cd $SCRATCH/logs && ROUTE=d2_anythermal_adapter_unet SMOKE=1 sbatch -J smoke_d2 --time=00:30:00 --export=ALL,ROUTE,SMOKE ~/Iris/slurm/route_suite.sbatch
```

c2 阶段二与 d1 可以同时投（`INIT_FROM` 要的 c1 adapter 已在集群）：

```bash
cd $SCRATCH/logs && ROUTE=c2_vae_adapter_unet RUN_TAG=c2_stage2 FREEZE_ADAPTER=1 INIT_FROM=$SCRATCH/checkpoints/c1_stage1/best_weights.pt sbatch -J route_c2_stage2 --time=2-00:00:00 --export=ALL,ROUTE,RUN_TAG,FREEZE_ADAPTER,INIT_FROM ~/Iris/slurm/route_suite.sbatch && ROUTE=d1_anythermal_adapter sbatch -J route_d1 --time=1-00:00:00 --export=ALL,ROUTE ~/Iris/slurm/route_suite.sbatch
```

d2 冒烟过后再投它的全量（`--time=2-00:00:00`）。
**b 与 c1 本地已完结且标定证明可比，不要在集群重跑。**

### ② 还需要传（WSL 侧，与训练并行）

```bash
cd /mnt/e/project/Iris && bash slurm/transfer_checkpoints.sh caption && bash slurm/transfer_checkpoints.sh midas && bash slurm/transfer_checkpoints.sh bmsd
```

| 内容 | 体积 | 给谁 |
|---|---|---|
| b+caption `best_weights.pt`（= **epoch 3**，已由 `checkpoint_epoch` 确认） | 3.2 GB | 任务 1a、2 |
| AnyThermal MiDaS 三个变体 | 903 MB | 任务 4 |
| BMSD 脚手架 | — | 任务 4（重建器依赖） |

### ③ 任务 2 现在就能开跑

b(e05) 和 c1 的 checkpoint 已在集群，test manifest 也在，不必等训练：

```bash
cd $SCRATCH/logs && ROUTE=b_thermal_unet TAG=b_e05_on_test CKPT=$SCRATCH/checkpoints/calibration/epoch05_weights.pt VAL_MANIFEST=$SCRATCH/manifests/sequence_level_internvl3_8b/ms2_test_16-08-46_rgb_depth_v1_clip75_20260728.jsonl sbatch -J eval_b_test --export=ALL,ROUTE,TAG,CKPT,VAL_MANIFEST ~/Iris/slurm/eval.sbatch
```

caption 臂评估记得加 `CAPTION_MODE=correct`——必须与该 checkpoint 训练时的输入模式一致。

### ④ 被代码卡住的

任务 3 与任务 1a 的分层，见 §14。

---

## 9. 已知问题与风险

1. ~~`route_suite.sbatch` 尚未跑通~~ **已通过**（6914833）。但只验了 `b_thermal_unet`
   这一条，`--freeze-adapter` + `--init-from`（c2 阶段二）那条路径仍未实跑。
2. **排队是主要瓶颈**，不是算力。84 卡常年 80+ 占用。`--time` 报准能吃到 backfill 红利。
3. **Windows 侧 SSH 无连接复用**（Windows OpenSSH 不支持 ControlMaster），每次连接
   要两次 Duo。WSL 侧已配 `ControlPersist 8h`，迭代循环建议走 WSL。
4. **本地 `.git` 有 55 GB 不可达对象**，不影响推送，但占磁盘。
5. `origin/main` 与本地 `main` **无共同祖先**（`git merge-base` 为空），
   所以走的是新分支 `aire-hpc-setup` 而非推 `main`。这条历史分叉尚未解决。
6. 六线全量的**实际耗时仍未知**。已知的只有推理侧：全量 5,810 帧评估 8.3 分钟。
   训练吞吐没有可比数，`--time` 只能沿用本地外推值。第一条线跑完才能校准。
7. **`sbatch` 在提交时快照脚本**，`git pull` 不影响已排队作业（6913999 因此白跑）。
   改完脚本必须先 pull 再提交。

---

## 10. 核对结果（2026-07-29 已回答）

**1. 六条线** ✅ 确为 `train_route_suite.py` 的 `ROUTES`。参数量与运行产出的
`frozen_config.json → parameter_audit` 一致（b 线 unet 867,568,324；c1 adapter 7,106,308）。
route_a–route_f 命名撞车的警告保留，那套是推理评测流程。

**2. Manifest** — val 默认值正确；**train 要换成 `ms2_train_day2seq_clip75_20260728.jsonl`**
（sha256 `1e56fa1e028c990684a55b0ee7e97940a09c4c716bb8d82770b435616dc1ff0b`，19,949 行）。
旧的 `ms2_train_day2seq_20260725.jsonl` 里 11-37-46 的 9,508 行仍是旧 v1 caption
（实测最长 125 词，会被 CLIP 77 token 截断）；新版为 clip75 口径，全部 ≤75 token 且
**逐帧同序同路径**，因此换用后仍能与既有 empty 数字逐帧配对。无 caption 的线用哪份都行，
统一用新版省心。

**1m 去冗余：不换。** 本地已完成的 b（0.0775@e5）与 c1（0.1127）都是全帧跑的，
换 1m 后集群结果无法与其并表；集群瓶颈是排队而非算力，省这点步数不值得付治理代价。

**3. 超参**沿用默认，但三处必改：
- caption 臂必须同时给 `--caption-mode correct --val-caption-mode correct`。
  `--val-caption-mode` 默认 `empty`，会用空 prompt 给 caption 模型打分，而
  `best_weights.pt` 正是按这条曲线自动选的 —— 等于用它没训过的输入模式挑冠军。
- `--snapshot-epochs` 改成 `1,2,3,5,10,15`。本地 caption 臂的最好点落在 **e3**，
  而原设定没存 e3。
- `--time` 按线分别设，见 §12.5。

**4. 未传模态**：六线训练确实只读 `rgb_path` / `thermal_path` 与对应视角 GT。
⚠️ 但原 grep 漏了一处：**`build_ms2_multiseq_manifest.py --min-spacing-m` 会读
`sync_data/<seq>/gps_imu/data/*.txt`**（该文件第 133 行），目录缺失直接 `SystemExit`。
只要不在集群上重建"按里程去冗余"的 manifest 就不受影响 —— 1m/2m 两份已生成，
需要时传 jsonl 即可，不必传 `gps_imu/`。

**5. `16-19-00` / `21-36-10`**：六线训练都不用。`16-19-00` 是雨天二次验证的考卷
（评估要用，勿清）；`21-36-10` 目前只被全局 manifest 引用，留作夜间备用序列。

---

## 11. 本周任务（导师 2026-07-28 布置）在集群上的执行清单

本文档原先只覆盖「六线 × 20 epoch 无 caption」。本周的任务单是另一组，六线是它的底座
而不是它本身。逐条对应如下 —— **耗时列是本地实测或本地外推，L40S 未标定**（见 §12.6）。

| 任务 | 内容 | 集群上要跑什么 | 状态 / 预算 |
|---|---|---|---|
| **1a** | Thermal + 解冻 U-Net + caption，看远端 | `b_thermal_unet --caption-mode correct --val-caption-mode correct` | **本地已跑完**（22.42 h，99,760 步，best e3 = 0.07773）。集群不必重跑，只剩全量评估与分层 |
| **1b** | 先训 Adapter、再训后面的 U-Net，看天空 | 阶段一 = 已完成的 `c1`；阶段二 = `c2_vae_adapter_unet --init-from <c1 best> --freeze-adapter` | 阶段二约 32 h。若天空无改善，再加 caption 臂（再 32 h） |
| **1c** | AnyThermal Adapter，补齐未做的实验 | `d1_anythermal_adapter`（约 11 h）、`d2_anythermal_adapter_unet`（约 32 h） | d1 的 overfit gate 已过（`gt_ssi_l1` 降 72.2%）；**d2 从未 smoke，先跑 5 步** |
| **2** | 独立 Test 集 | 所有 checkpoint 用 `--eval-checkpoint --val-manifest <test>` 各评一次 | 数据已在集群（见下）；caption manifest 待补传 |
| **3** | 与 AnyThermal、零训练直推的对比可视化 | 导出 raw 预测 + `build_route_vis_figures.py` | 需要 vis 导出（本地的已在清盘时删光） |
| **4** | 原版 AnyThermal 是否也错判天空 | `run_ms2_anythermal_midas.py` 出预测 → `analyze_prediction_regions.py` 分层 | **checkpoint 未传，见 §12.1** |

### 11.1 独立 Test 集（任务 2）

`2021-08-13-16-08-46`，官方 **val** split、白天、**2,543 帧**，我们从未接触过，
AnyThermal 也只在官方 train 上训过 → 双方都没见过，对比公平。数据已随 §3.2 传上集群。

治理约束：**11-23-45 继续当 val（选 epoch 用），16-08-46 只在定稿时评一次**。
此前 11-23-45 同时充当选择集与汇报集（`best_weights.pt` 就是按它选的），
本周之后所有对外数字都应带 test 列。

备用考卷（数据都已在集群）：`21-58-13`（官方 val，夜间，2,539 帧）、
`16-59-13`（官方 test_rainy，6,483 帧）。夜间那条的 caption 由 RGB 生成、
夜景 RGB 接近全黑，caption 质量本身会成为混淆变量，结论需单独讲。

---

## 12. 补充的欠缺与风险（2026-07-29 追加）

### 12.1 还需要传上集群的东西

| 内容 | 体积 | 给谁用 | 目的地 |
|---|---:|---|---|
| `outputs/route_suite/c1_vae_adapter_20ep/best_weights.pt` | 28 MB | 任务 1b 的 `--init-from` | `$SCRATCH/checkpoints/c1_stage1/` |
| `outputs/route_suite/b_thermal_unet_20ep/epoch05_weights.pt` | 3.47 GB | 机器差标定（§12.6） | `$SCRATCH/checkpoints/calibration/` |
| AnyThermal MiDaS 权重 `E:/project/AnyThermal/_download/pretrained_checkpoints/depth/{Midas_anythermal,Midas_dinov2,Midas_small}` | — | 任务 4 | `$SCRATCH/models/anythermal_midas/` |
| BMSD 脚手架 `E:/project/AnyThermal/baselines/depth/BridgeMultiSpectralDepth` | — | 任务 4（重建器依赖） | `$SCRATCH/code/` 或并入仓库 |
| ~~`ms2_test_16-08-46_rgb_depth_v1_clip75_20260728.jsonl`~~ | — | — | ✅ **已在集群**。`transfer_ms2.sh manifests` 是整目录打包（`tar -C $CAPTION_SRC .`），19 个文件全部传了，`ms2_train_day2seq_clip75_20260728.jsonl` 同理 |

⚠️ **AnyThermal 的 depth checkpoint 不在 HuggingFace 上**，`prefetch_models.sbatch` 拉不到 ——
那三个模型是从官方 zip 用 `tools/build_anythermal_midas.py` 重建的（`strict 231/231` 张量
加载通过）。任务 4 没有它就做不了。

（`theairlabcmu/AnyThermal` 那个 HF 仓库是 **热成像编码器**，`d1`/`d2` 用；与这里说的
depth checkpoint 是两码事。）

### 12.2 只有训练、没有评估与回传

`--eval-checkpoint` 的全量评估（5,810 帧 × 多个 checkpoint，比训练更吃卡）、
`compare_route_evals.py`、`analyze_route_regions.py`、`analyze_prediction_regions.py`
都没有 sbatch 模板，产物如何回传本地做 slides 也没写。建议补一个
`eval.sbatch`，吃 `ROUTE` / `CKPT` / `TAG` / `VAL_MANIFEST` / `CAPTION_MODE` 五个变量。

### 12.3 输出配额没算过

scratch 1 TB。每条 U-Net 线 20 epoch 写：`latest.pt` 10.4 GB + best 3.47 + end 3.47
+ 5 个快照 × 3.47 ≈ **35 GB**。a/b/c2/d2 四条 ≈ 140 GB；加 caption 臂翻倍 ≈ 280 GB；
再加 raw predictions（本地历史上攒到 172 GB）就危险。建议 sbatch 结尾调
`tools/slim_checkpoints.py` 自动清中间档，三道闸照旧：①目录有 `summary.json`
②至少一个 `_end.pt` 幸存 ③只删非 `_end.pt`。

### 12.4 `HF_HOME` 必须显式指定，并配 `--local-files-only`

`$HOME` 仅 65 GB 且已被 conda 环境占 11.9 GB。更要紧的是本地踩过的坑：HF_HOME 不一致
会让训练启动时**静默重下 4 GB 权重**，表现为「任务没起来」（实为卡在 `Fetching 13 files`）。

- `common.sh` 里 `export HF_HOME=$SCRATCH/hf_cache`；
- 预取完成后，**所有训练/评估作业加 `--local-files-only`** —— 权重缺失会立即报错，
  而不是六个作业并发去写同一个缓存目录（§8 担心的正是这个，这个 flag 就是那道保险）。

### 12.5 墙钟算术：`--time` 别统一 24 h

`latest.pt` 在**每个 epoch 末尾**才写（`train_route_suite.py:1097`），被砍最多丢一个 epoch。
按本地实测外推：

| 线 | 20 epoch 本地耗时 | 24 h 报时够不够 |
|---|---:|---|
| `b_thermal_unet` | 21.73 h | **贴边**，很可能最后一个 epoch 差一点，白排一次队 |
| `b` + caption | 22.42 h | 同上 |
| `c1_vae_adapter` | 19.06 h | 够 |
| `a_rgb_unet` | 约 88 h（外推） | 要续 4 次 |

分区上限 48 h，贵的线直接报 48 h 更划算。

### 12.6 跨机器可比性 —— 投六线之前必须先做的一步

集群 torch 2.3.1+cu121 / L40S，本地 torch 2.7.0+cu128 / 另一张卡。本地已完成的
b（0.0775@e5）、c1（0.1127@best）、b+caption（0.07773@e3）若要与集群跑出的
a/c2/d1/d2 并进同一张表，中间隔着一个未标定的机器差。

**标定方法（1 卡 × 约 30 分钟）**：把 `b_thermal_unet_20ep/epoch05_weights.pt` 传上去，
在集群用 `--eval-checkpoint` 跑同一份 val，与本地的全量 **0.0775** 比对。
差值 <0.0005 即可并表并在脚注写明；更大则集群结果只能自成一组。
**这一步要在投六线之前做**，否则跑完才发现不可比就晚了。

### 12.7 代码状态（2026-07-29 更新）

本周新增的三处改动与两个工具**已在 `HEAD` 且与 `origin/aire-hpc-setup` 同步**，
集群 `git pull` 即可拿到：

- `train_route_suite.py`：`--init-from`（跨 route 搬权重、不带优化器状态）、
  `--freeze-adapter`（任务 1b 的阶段二）、**冻结的 adapter 现在也会写进 checkpoint**
  （否则阶段二的档只有 U-Net，评估时 adapter 会静默退回随机初始化）；
- `analyze_route_regions.py`：`--caption-mode`，可**逐 checkpoint 指定**
  （`empty,correct` → 两臂各自在自己训练时的输入模式下评估，同帧同序，配对仍合法）；
  并修了「两臂检查点同名（都叫 `epoch05_weights.pt`）会静默丢一条」的 bug；
- `analyze_prediction_regions.py`（新）：吃 manifest + `raw_predictions/*.npy`，
  给外部模型（AnyThermal）做同样的九分层；对齐镜像逐帧与 `evaluate_sample` 自校验；
- `build_ms2_sequence_manifest.py`（新）：为盘上任何一条序列生成全字段 manifest。

**新 manifest 是数据不是代码**，仍需走 `transfer_ms2.sh manifests`。

---

## 13. 提交前检查（2026-07-30，投 a / c2 / d1 / d2 之前）

> ⚠️ **本节的「仍缺」清单写于预取和标定完成之前，第 1、2、4、6 项均已完成。**
> 当前状态以 §7 为准，当前待办以 §8 为准。保留原文以便对照。

### ✅ 已核实

| 项 | 结果 |
|---|---|
| clip75 训练 manifest 在集群 | `ms2_train_day2seq_clip75_20260728.jsonl` **28,675,498 字节 = 本地同字节** |
| `slurm/` 全部脚本语法 | 12 个 `.sbatch`/`.sh` 全部 `bash -n` 通过 |
| 集群脚本的 python 依赖 | 只用 `train_route_suite.py`、`analyze_prediction_regions.py`、`run_ms2_anythermal_midas.py`，**三个都在 HEAD 且已推送** |
| 集群脚本是否依赖未推送的旗标 | **不依赖**（grep 过 `permuted` / `gt-sparsify` / 新探针，无引用）→ 不必等本地推送就能投 |
| `route_suite.sbatch` 已实现 | clip75 默认 manifest、`LOCAL_FILES_ONLY=1`、CAPTION 同时设 train+val、snapshot 含 e3、`INIT_FROM` 与续跑互斥、输出目录按模式加前缀 |

集群侧再核一次内容（不只是大小）：

```bash
sha256sum /mnt/scratch/sc23sz/manifests/sequence_level_internvl3_8b/ms2_train_day2seq_clip75_20260728.jsonl
```

应为 `1e56fa1e028c990684a55b0ee7e97940a09c4c716bb8d82770b435616dc1ff0b`。

### ❌ 仍缺，必须按此顺序

1. ✅ **已完成**（job 6914389，6.7 GB）。~~HF 权重预取（从未执行）。~~`LOCAL_FILES_ONLY` 默认 1 → 权重不在 `$SCRATCH/hf_cache` 时**所有作业立即失败**。
   `cd $SCRATCH/logs && sbatch ~/Iris/slurm/prefetch_models.sbatch`，完成后确认两个仓库都在。
2. ✅ **已完成**（job 6914833，`b_thermal_unet`，1 分 34 秒）。~~一条冒烟。至今没跑通任何一次训练（唯一一次死在命名护栏，修完未重跑）。
   `ROUTE=c1_vae_adapter SMOKE=1 sbatch -J smoke_c1 --time=00:30:00 --export=ALL,ROUTE,SMOKE ~/Iris/slurm/route_suite.sbatch`
3. ❌ **仍缺** — 投四条（b 与 c1 本地已完结，勿重跑）：c2 `--time=2-00:00:00`、
   d1 `1-00:00:00`、d2 `2-00:00:00`、a `2-00:00:00`（a 外推 88 h，需续投两次）。
   ⚠️ **d2 从未冒烟**（§11 自己写了），应先单独冒烟再投全量。
4. ✅ **已完成**：`c1_stage1/best_weights.pt`（28,442,797 字节）已在集群，省掉重跑 c1 的 19 GPU·h。
5. ❌ **仍缺** — 任务 4 前置：`transfer_checkpoints.sh midas` 和 `bmsd` 分两次跑
   （脚本一次只吃一个子命令）。AnyThermal 的 **depth** checkpoint 不在 HuggingFace 上，预取拉不到。
6b. ❌ **仍缺** — 任务 1a/2 前置：`transfer_checkpoints.sh caption`，b+caption 的
   `best_weights.pt`（= epoch 3，3.2 GB）。
6. ✅ **已完成且通过**（job 6914834）：集群 `0.07749105` vs 本地 `0.0775`，**差 0.000009**，
   阈值 0.0005。集群结果可与本地并表。

### ⚠️ 两条提醒

- **`CAPTION_MODE=permuted` 现在会失败**：HEAD 的 `--val-caption-mode` 只有 `empty/correct/shuffled`，`permuted` 还在本地未推送。
- **半长旋转的 `shuffled` 是被污染的对照**：实测旋转后的 caption 仍能解释接收帧 R² 0.32（MS2）/ 0.46（RGBDT500）的中位深度方差，而随机置换约 −0.1（`tools/probe_caption_scale_information.py`）。集群上做 caption 对照请等 `permuted` 推送后再用，或明确标注该局限。

---

## 14. Checkpoint 格式验证：任务 3 的拦路虎（2026-07-30）

任务 3（可视化对比）和任务 1a 的九分层都要 `raw_predictions/*.npy`。
`train_route_suite.py` **不能导出 raw 预测**，只有旧的 `run_ms2_lotus_*_official.py`
支持 `--save-raw-pred`。所以问题变成：route suite 训出的 checkpoint 能不能喂给旧脚本？

### 实测两边格式（从 `.pt` 文件直接读出，非推测）

route suite 写出的：

```
顶层键: format, route, epoch, caption_mode, manifest_sha256, val_metrics, state_dicts
b epoch05  → state_dicts: {'unet':    690 个张量, 'conv_in.weight' ...}
c1 best    → state_dicts: {'adapter':  54 个张量, 'blocks.0.norm1.weight' ...}
```

旧脚本要的：`checkpoint["lotus_unet_state_dict"]`（顶层扁平，`strict=True`）、
`checkpoint["adapter_state_dict"]` + `adapter_hidden_channels` + `adapter_blocks`。

### 逐条线结论

| 线 | 训练什么 | 能否转换 |
|---|---|---|
| `a` / `b` | 只有 unet | ✅ 张量键名就是 `UNet2DConditionModel` 原生键名，改外层 key 即可 |
| `c1` | 只有 adapter | ✅ 映射到 `--latent-adapter-checkpoint`，但要补 route suite 未记录的架构元数据 |
| `c2` | adapter + unet | ❌ **无路径** |
| `d1` / `d2` | anythermal adapter (+unet) | ⚠️ 需 `adapter_architecture` / `train_mode` / `settings`，route suite 一个都没存 |

`c2` 是硬阻塞——旧脚本明写 `--latent-adapter-checkpoint` 与 `--unet-checkpoint` 互斥
（"Line e keeps the U-Net frozen"）。旧脚本是为 a–f 那套设计的，每条线只训一样东西；
新套件的 `c2`/`d2` 联合训练 adapter 与 unet，旧脚本在架构上就没这个组合。

### 结论：不写转换器，给评估路径加 `--save-raw-pred`

转换器要分四种情况处理、补造未记录的元数据，且 `c2` 依然无解。
改 `train_route_suite.py` 的评估分支更直接：它已经在跑完整前向和官方协议，
落一份 `.npy` 是增量改动。六条线全部可用，不存在格式漂移。

**改动（已写好，未提交，见下）**：

1. 新参数 `--save-raw-pred`（`store_true`）
2. `run_validation()` 加 `raw_dir` 参数，默认 `None`；给了就在 resize 到 GT 之前存
   `raw_dir/<id>.npy`，float32、**原生分辨率**
3. `run_evaluation()` 建目录并传下去，metrics 记 `raw_predictions_saved`

原生分辨率这一点两边都核对过：旧脚本 `run_ms2_lotus_thermal_vae_official.py:376`
存的就是 resize 前的返回值，`analyze_prediction_regions.py:171` 自己做
`resize_dense_prediction(raw, gt.shape)`。文件名与 dtype 也一致，所以分层脚本
可以无差别地吃两种来源。

`raw_dir` 默认 `None` 意味着**训练期每 epoch 的验证不受影响**。
开销：每帧约 0.5 MB + 1 个 inode，全量 5,810 帧约 2.9 GB。

### ⚠️ 未提交，原因

改动落在 `tools/train_route_suite.py`，而该文件同时有另一个会话未提交的
`permuted` caption 模式改动，无法只提交其中一半。当前工作区未提交清单：

```
 M docs/AIRE_HPC_MIGRATION_STATUS_20260729.md
 M tools/export_qualitative.py
 M tools/train_ms2_thermal_vae_unet_gt.py
 M tools/train_route_suite.py        ← --save-raw-pred + permuted 混在一起
?? tools/probe_caption_predicted_alignment.py
?? tools/probe_caption_scale_information.py
?? tools/run_rgbdt500_sparsity_eval_queue.sh
```

需要两个会话协调后统一提交。**在此之前任务 3 无法在集群上开工。**

### 附带观察

`permuted` 与既有的 `shuffled` 都只接在 `run_evaluation()` 里，训练期的
`run_validation()` 走不到。训练时传 `--val-caption-mode permuted` 不会打乱 caption，
实际等同 `correct` 且不报错。看起来是有意为之（纯评估对照），但值得确认。

---

## 15. 仍然没人做的

| 项 | 说明 |
|---|---|
| 产物回传本地 | 做 slides 要把 vis 和 metrics 从 scratch 拉回来，没有脚本 |
| `slim_checkpoints.py` 未接入 sbatch | §12.3 提的输出配额自动清理。当前 scratch 64 GB/1 TB，四条线约 140 GB、加 caption 臂 280 GB、再加 raw predictions 约 510 GB（51%），有余量但仍该清 |
| `build_route_vis_figures.py` 无作业模板 | 它只吃 `--frames --which --style`，大概率在本地跑，需确认 |
| `compare_route_evals.py` / `analyze_route_regions.py` | 纯 CPU 分析，可复用 `analyze.sbatch` 的模式，尚未写 |
| `a_rgb_unet` 是否本周投 | 外推 88 h，不在本周任务单（1a/1b/1c/2/3/4）里，需决定 |

---

## 16. 跨会话协调结论（2026-07-30）

### 已解除：§14 的提交阻塞

统一提交完成并推送：**`7843153b8`**（`aire-hpc-setup`）。`--save-raw-pred` 与
`permuted` 现在都在 origin 上，集群 `git pull` 即可，**任务 3 不再被挡**。

### §14 的「附带观察」是真 bug，已修

训练期的 `run_validation()` 确实不做供体重排，`--val-caption-mode shuffled/permuted`
在训练时会**静默等同 `correct`**，而曲线标签仍写 shuffled。不是有意为之。
`validate_args()` 现在显式拒绝（实测已拦下）：这两个模式只允许配 `--eval-checkpoint`。

### §15 需订正一条

`analyze_route_regions.py` **不是纯 CPU**：它构造 `RouteModel` 自己跑前向，是 GPU 作业。
只有 `compare_route_evals.py` 是纯 CPU。所以它的模板不能照 `analyze.sbatch`
（那个吃现成 `.npy`），要按 GPU 作业写。

### §15 没覆盖、会卡任务 2 的两件

1. **test caption manifest 已生成但未传**：
   `ms2_test_16-08-46_rgb_depth_v1_clip75_20260728.jsonl` —— 2,543 行、`caption_status`
   全 ok、token 37–75、零超限、零空串。集群上没有它，任务 2 的 caption 评估无法进行。
   `bash slurm/transfer_ms2.sh manifests` 补传。§12.1 列的 `stage1` / `calib` /
   `midas` / `bmsd` 四项 checkpoint 传输同样待确认。
2. **谁进 test 集，没人定**。任务 2 的治理点是「11-23-45 只用于选 epoch，16-08-46
   只评一次」，但没有任何地方列出参评清单。建议定死为：b-e5（本地）、b+caption-e3
   （本地）、c1-best（本地）、c2/d1/d2-best（集群）、AnyThermal MiDaS —— **每个一次，
   且必须在 val 上选定 epoch 之后**。谁先跑完谁评，评完立即写回。

### 再补三项

3. **集群上要不要跑 caption 臂，没定**。§11 记 1a 已在本地完结，但 c2/d1/d2 的
   caption 臂尚无排期（六线×2 约再 95 GPU·h）。建议只给 **d2** 加：历史上 f 线
   （AnyThermal + Adapter）是唯一测出正 caption 效应的线（+0.0013\*），d2 是它的
   20-epoch 版本，先验最强。
4. **结论回写没人负责**：集群产出的所有数字最终要进
   `LOTUS_LINE_V2_ROUTE_AND_CAPTION_FREEZE_20260705.md`，未指定章节与责任人。
5. **机器差标定必须早于并表**（§12.6）。c2/d1/d2 出结果之后再做就晚了——那时若发现
   差异 >0.0005，集群数字只能自成一组，跨线阶梯（`a→b→c2→d2`）会断在本地/集群交界处。

### 数据质量脚注

三份 clip75 manifest 都有约 4.5–6.2% 的帧含相邻重复词（如 "driving driving"，
InternVL 的生成瑕疵）：test 16-08-46 6.2%、train 11-37-46 4.5%、val 11-23-45 4.7%。
比例在三份之间接近，且 caption 臂与 empty 臂读的是同一批文本，**不构成配对混淆**，
记录备查即可。
