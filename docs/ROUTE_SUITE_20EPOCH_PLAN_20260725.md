# 六线 × 20 epoch 实验计划（2026-07-25 立）

导师提出的问题：**只训练一个 epoch，网络能记住什么？** 本轮把四条路线全部训到
20 epoch，统一数据划分、损失函数与评估方式，并给出每条线的完整网络结构与参数量。

本文件是本轮的执行合同。数字全部可溯源；跑完后结论写回
`LOTUS_LINE_V2_ROUTE_AND_CAPTION_FREEZE_20260705.md`。

---

## 1. 六条线

任务单给了四条线；用户 2026-07-25 决定 c/d 各跑「仅 Adapter」与「Adapter+U-Net
联合」两版，用来把「Adapter 有没有用」和「U-Net 训没训」这两个变量分开。

| 线 | 输入 | Condition | 可训练 | 说明 |
|---|---|---|---|---|
| `a_rgb_unet` | RGB | 冻结 VAE latent | U-Net | Baseline（RGB 视角 GT） |
| `b_thermal_unet` | Thermal | 冻结 VAE latent | U-Net | Baseline（热像视角 GT） |
| `c1_vae_adapter` | Thermal | VAE latent + Adapter | Adapter | U-Net 冻结 |
| `c2_vae_adapter_unet` | Thermal | VAE latent + Adapter | Adapter + U-Net | |
| `d1_anythermal_adapter` | Thermal | AnyThermal 特征 + Adapter | Adapter | U-Net 冻结 |
| `d2_anythermal_adapter_unet` | Thermal | AnyThermal 特征 + Adapter | Adapter + U-Net | |

阶梯读法：`a→b` 是模态，`b→c2` 是 VAE-latent Adapter，`c2→d2` 是 AnyThermal
特征源，`c1 vs c2` / `d1 vs d2` 是 U-Net 可塑性。

⚠️ **a 线与 b/c/d 线不可直接比数字**：a 用 RGB 视角 GT，其余用热像视角 GT
（结论 15）。表里必须并列标注视角，不得写成「RGB 比 Thermal 好/差」。

---

## 2. 统一的三件事

### 2.1 数据划分

训练集 = **官方 MS2 train split 里两条已解压的白天序列**，合并 manifest
`ms2_train_day2seq_20260725.jsonl`（19,949 帧）：

| 序列 | 场景 | 帧数 | 轨迹 | caption |
|---|---|---:|---|---|
| 2021-08-06-10-59-33 | campus | 10,441 | 6.95 km | clip75 ✓ |
| 2021-08-06-11-37-46 | urban | 9,508 | 5.28 km | 旧 v1（未 clip75） |

两条轨迹**不重叠**（GPS 实测：一条上 0.0% 的点落在另一条 50 m 内，100 m 内
仅 2.5%），因此这是两条不同的路线而非同一条路开两遍。这一点很重要：1 epoch
时代的单序列训练集在 20 epoch 下会被直接背下来（已实测，见 §5）。

验证/报告集沿用 **11-23-45（官方 test_day，5810 帧）**，与冻结文档里的全部
历史数字同卷可比。每 epoch 用 stride 4 的子集（约 1452 帧）画曲线，选定 epoch
后跑全量。二次验证留 **16-19-00（官方 test_rainy）**。

生成命令：

```bash
python tools/build_ms2_multiseq_manifest.py --output /mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/ms2_train_day2seq_20260725.jsonl --verify
```

### 2.2 损失函数

**纯 GT 监督，不引入任何 teacher-student**（用户 2026-07-25 明确指令）：

```
loss = 5.0 * masked_ssi_l1(decode(U-Net(condition, noise)), LiDAR 视差, valid_mask)
```

即在有效像素上、视差空间做闭式 scale+shift 对齐（detached）后的 L1。**关掉**
了此前所有带 teacher 味道的项：冻结网络响应一致性（响应锚）、condition 蒸馏
（模仿 VAE latent）、稠密深度 teacher。本轮 trainer 里根本不实例化第二个网络
去模仿。

Lotus / AnyThermal 的预训练权重仍然使用——那是**初始化与冻结特征提取器**，
不是 teacher-student 关系。

GT 解码走 fp32 VAE 副本（`--gt-decode-fp32`，默认开），修的是 fp16 反传把 GT
梯度下溢成零的老问题；冻结 U-Net 的两条线（c1/d1）同样用 fp32 U-Net 副本前向，
因为 Adapter 的梯度必须穿过它回来。

优化：micro-batch 1 × 累积 4（等效 batch 4），AdamW，U-Net lr 1e-6，Adapter lr
3e-4，权重衰减 1e-4，梯度裁剪 1.0，**200 步线性 warmup + 余弦退火到 5%**
（20 epoch 用 1-epoch 的恒定 lr 不是公平的收敛测试）。

### 2.3 评估方式

官方 BMSD 协议 `ssi_disparity`（`ms2_eval/official_protocol.py`），逐图 2 参数
仿射对齐、clamp [1e-3, 80]、macro 平均，与冻结文档里所有数字同尺。每 epoch 内联
跑一次，写进 `epoch_metrics.jsonl`；`best_weights.pt` 按 val abs_rel 自动保存。

---

## 3. 网络结构与参数量（任务单第 3 条）

纯 CPU，不占 GPU：

```bash
python tools/dump_route_architecture.py --output docs/ROUTE_ARCHITECTURES_20EPOCH.md --local-files-only
```

输出六张表：每条线按前向顺序列出模块、类名、参数量、**训练 / 冻结**、作用，
外加一张跨线总览（可训练参数 / 冻结参数 / 可训练占比）和可训练模块的内部结构。
每次训练启动时 `frozen_config.json` 也会记一份同样的 `parameter_audit`，两者
必须一致。

---

## 4. 执行顺序与命令

纪律照旧：**smoke（5 步）→ overfit-32 gate → 全量**，gate 不过不烧全量。
所有命令单行，WSL 里 `conda activate wsl-pytorch`、工作目录 `/mnt/e/project/Iris`。

### 4.1 Smoke（每条线各 5 步，几分钟）

```bash
python tools/train_route_suite.py --route b_thermal_unet --output-dir outputs/route_suite/smoke_b --smoke-updates 5 --skip-val
```

把 `--route` 依次换成 `a_rgb_unet` / `c1_vae_adapter` / `c2_vae_adapter_unet` /
`d1_anythermal_adapter` / `d2_anythermal_adapter_unet`，`--output-dir` 里的名字
保持含 `smoke`。

### 4.2 Overfit-32 gate（32 帧，300 步）

```bash
python tools/train_route_suite.py --route d1_anythermal_adapter --output-dir outputs/route_suite/overfit_d1 --overfit-steps 300 --overfit-samples 32 --skip-val
```

判决规则**先立**：`gt_ssi_l1` 后 50 步均值必须显著低于前 50 步（降幅 ≥30%）。
d1 是风险最高的一条（结构上等同于此前崩到 AbsRel 0.4375 的 Arm 7，纯 GT、
无值域锚），gate 不过就如实记为「该配置学不动」，不偷偷加 teacher 回来。

### 4.3 全量 20 epoch（先跑 b 探路，再铺开）

```bash
python tools/train_route_suite.py --route b_thermal_unet --epochs 20 --output-dir outputs/route_suite/b_thermal_unet_20ep --snapshot-epochs 1,2,5,10,15
```

b 线最便宜（约 30 GPU·h），先用它看 val 曲线什么时候拐头。曲线形状决定后面
五条要不要照跑 20 epoch，尤其决定要不要把 89 GPU·h 投进 a 线。

其余五条同款命令，只改 `--route` 与 `--output-dir`。

断点续训：

```bash
python tools/train_route_suite.py --route b_thermal_unet --epochs 20 --output-dir outputs/route_suite/b_thermal_unet_20ep --resume outputs/route_suite/b_thermal_unet_20ep/latest.pt
```

---

## 5. 预算与风险

按 1-epoch 实测耗时按帧数线性外推（19,949 帧 ≈ 现在的 1.91 倍）：

| 线 | 1 epoch | 20 epoch |
|---|---:|---:|
| a_rgb_unet（原生 1224×384） | 4.43 h | **88.7 h** |
| b_thermal_unet | 1.51 h | 30.2 h |
| c1_vae_adapter | 1.18 h | 23.7 h |
| c2_vae_adapter_unet | 1.62 h | 32.5 h |
| d1_anythermal_adapter | 0.53 h | 10.7 h |
| d2_anythermal_adapter_unet | 1.59 h | 31.7 h |
| 每 epoch 验证（1452 帧 × 6 线 × 20） | | ~7 h |
| **无 caption 合计** | | **~224 h ≈ 9.3 天** |

a 线独占 40%，是唯一的大杠杆（降分辨率可省约 60 h，但用户选择保持原生以给
RGB 最强成色）。

**磁盘**：U-Net 线每次运行约 17 GB（`latest.pt` 含优化器约 10 GB + best + end
各 3.4 GB），四条 U-Net 线共约 68 GB；Adapter-only 线可忽略。盘上剩 588 GB，
够用。中间 epoch 快照默认不存，只在 `--snapshot-epochs` 指定时存。

**已知风险**：
1. **d1 可能学不动**（见 4.2）。这是结果，不是故障。
2. **U-Net lr 1e-6 是为 1 epoch 调的**。20 epoch 下总位移放大 20 倍，余弦退火
   是对冲；若 b 线曲线在前几个 epoch 就崩，改小 lr 重跑（改一次，记在案）。
3. **2 序列仍可能不够**。1 epoch 时代单序列在第 2 epoch 就全面退化
   （0.1260 vs 0.1226，冻结文档 3.9，且那是在 GT 梯度残废条件下取得的）。如果
   两条序列的曲线仍在第 3–5 epoch 拐头，就得扩到 6 条白天官方 train 序列
   （17-21-04 / 17-44-55 / 16-50-57 / 17-06-04），代价是重下这 4 条的
   `proj_depth` 深度 GT（Dropbox）+ 解压 87 GB sync_data + 补 caption。
4. **11-37-46 的 caption 是旧 v1 口径**（无 `caption_clip_tokens`，可能超 77
   token 被截断）。无 caption 阶段无影响；做 caption 臂之前必须用 clip75 口径
   重生成这 9508 条（约 3.3 GPU·h）。

---

## 6. caption 臂（第二阶段，待 6 条无 caption 出曲线后启动）

同样六条线，只把 `--caption-mode empty` 换成 `--caption-mode correct`
（训练侧 0.1 dropout），推理侧再分 empty / correct 两格，构成配对对照。
启动前必须先补齐 §5 风险 4 的 clip75 caption。
