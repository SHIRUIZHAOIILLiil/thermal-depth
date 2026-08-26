# 度量微调：把 SSI 视差输出变成米制深度（2026-08-26）

## 0. 为什么需要这一步

我们已训好的模型输出的是**归一化视差**：VAE 解码之后是 `[0,1]` 区间的一张图，
**没有单位**。基线（DORN / BTS / AdaBins / NeWCRF）输出的是**米制深度**。

这不是"对齐方式不同"，是**输出空间不同**。后果具体到操作上：

- 他们那一列的口径是**逐图中位数缩放**（1 参数，深度空间）。把一张 `[0,1]` 的视差
  图当成米去做中位数缩放，是错设——它不是"精度差一点"，是没有意义。
- 所以在微调之前，我们的模型**无法进入他们的那一列**，无论允许多少 GT 对齐。

微调这一步的目标是：**让模型自己输出米制逆深度，从而与基线在同一个输出空间里可比。**

⚠️ 一处需要修正的早期判断：2026-08-26 的协议审计发现"已发表的数字全部用了 test GT
对齐"，我据此认为度量微调对横向对比没有必要。**那个推论是错的。** 审计说的是对齐
方式，而问题在输出空间。两者不是一回事。

---

## 1. 微调改了什么

### 1.1 根源：目标在制作时就被除掉了尺度

原来的训练目标用 `trunc_disparity` 归一化：

```
disparity = 1 / depth
d_min = quantile(disparity[valid], 0.02)      ← 逐帧算
d_max = quantile(disparity[valid], 0.98)      ← 逐帧算
target = ((disparity - d_min) / (d_max - d_min) - 0.5) * 2       → [-1, 1]
```

**这两个分位数是每一帧各算各的。** 绝对尺度在**制作目标那一步**就被除掉了，不是在
评估时才丢的。所以模型的输出没有单位，是必然的，不是缺陷。

### 1.2 改动：把逐帧分位数换成两个冻结的常数

**函数形式完全不变**，只是那两个数不再随帧变化：

```
q     = 1 / D                              米制逆深度
u     = 2 (q − q_lo) / (q_hi − q_lo) − 1   VAE 编码器吃的 [-1,1] 目标
y     = (u + 1) / 2                        VAE 解码器给回的 [0,1]
q̂     = q_lo + y (q_hi − q_lo)             ← 上一行的精确逆
D̂     = 1 / q̂
```

`q_lo` / `q_hi` 只在**训练集**上估一次然后冻结。实现在
`tools/metric_depth_norm.py`，是这套约定的唯一出处。

⚠️ **两侧的约定是从代码里读出来的，不是假设的**：编码侧吃 `u`
（`ms2_thermal_dataset` 造 `[-1,1]` 后 `.repeat(3,1,1)` 直接给 `vae.encode`）；
解码侧给回 `y`（`decode_to_disparity` 结尾是 `decoded.mean(dim=1)/2 + 0.5`）。
`tests/test_metric_depth_norm.py` 断言两种编码算出的 `q` 一致，改坏一侧就会失败。

### 1.3 三条数据来源的优先级

| 来源 | 用途 | 说明 |
|---|---|---|
| 真实激光 | **度量锚点** | 图像空间 masked L1，只在官方有效像素上 |
| 稠密补全图 | 结构监督 | latent MSE，用**同一套**全局常数归一化 |
| 热像重建支路 | 原样保留 | 未改动 |

⚠️ 稀疏激光**从不 VAE 编码**。它的零是"洞"不是深度，而一个 latent cell 池化 8×8
像素——编码它会把洞摊到整张目标上，那是 AGENTS.md 里记录的 V1 确认 bug。所以度量项
留在图像空间。

---

## 2. 常数（TRAIN 上拟合，已冻结）

`tools/fit_metric_norm.py`，官方 8 条训练序列，**逐像素直方图**（对数空间 20 万格），
每个有效像素恰好数一次，不抽样，结果可复现到位。

```
q_lo = 0.017589748231560287  1/m      →  56.85 m
q_hi = 0.19845712700485040   1/m      →   5.04 m
Q0.02 / Q0.98，2,731,772,300 个有效像素 / 75,688 帧 / 8 序列
```

产物：`$IRIS_RUNS/metric_adapt/metric_norm_train.json`，含 quantile、有效范围、
`source_split`、像素数、序列名。⚠️ `MetricNorm` 在 `source_split != "train"` 时直接抛错。

### 2.1 训练集全局仿射基线

同一份 checkpoint、**不改网络**，只在 TRAIN 上拟合一个数据集级的
`q ≈ a·y + b`（3,785 帧 / 1.36 亿像素 / `correct` prompt）：

```
a = 0.21586326909931644      b = 0.013591162113036213
```

这条基线回答的是：**不动网络、只做一次训练集标定，能拿回多少度量性能。**
度量微调必须打赢它，否则这次微调买到的东西查表就能得到。

### 2.2 三条独立路径的交叉验证

| 来源 | a | b |
|---|---|---|
| 8 序列 TRAIN，3,785 帧，最小二乘 | 0.215863 | 0.013591 |
| 2 序列 TRAIN，200 帧，`empty` prompt（本地） | 0.216822 | 0.014101 |
| 官方 val 逐帧拟合取中位数，`correct` prompt | 0.21752 | 0.01723 |

**斜率三者相差 0.8% 以内**，而三条路径的划分、序列、prompt、拟合方式全不同。
截距散得多（最大 21%），与"逐帧截距变异 31%"一致——远场锚点本来就不稳。

---

## 3. 目标函数

```
L = λ_metric · L_metric + λ_dense · L_dense + λ_recon · L_I

L_metric = mean_{i∈M} |q̂_i − q_GT,i|     M = 官方有效激光掩码，图像空间
L_dense  = 原 anno_loss，稠密补全图的 latent MSE（目标改用全局常数归一化）
L_I      = 原 rgb_loss，热像重建支路，一个字没动
```

- 三个权重都可配；关闭 `--metric_adaptation` 时 `λ_dense`/`λ_recon` **只允许是 1.0**
  （硬性拒绝），保证基线配方逐项不变。
- `L_metric` 路径上**不做任何 clamp**——clamp 会把最错的那些像素的梯度清零。
  clamp 只在上报路径上，规则和命中数每次都打日志。

### 3.1 λ_metric 未定

实测每项单独反传的 U-Net 梯度范数（λ 都取 1 时）：

| 项 | loss | 梯度范数 |
|---|---|---|
| L_dense（latent MSE） | 0.6619 | 2.247e+00 |
| L_I（重建） | 0.000086 | 1.870e-02 |
| **L_metric（图像 L1，1/m）** | 0.0099 | **1.031e-01** |

比值 **21.8 : 1**。两个候选配置：

- **λ_metric = 1**：轻触。主要靠"目标换成全局归一化"本身带来尺度，L_metric 只是把
  锚点从伪深度拉回真激光。符合"学习率别给太大、拟合一下能横向对比就好"的指示。
- **λ_metric = 20**：与稠密项梯度量级持平，L_metric 真正主导绝对尺度。

**建议两臂都投**，选点走 val，让数据决定。

---

## 4. 训练配置

| | |
|---|---|
| 起点 | `iris_ms2_full8_thermalcap` 的 **step 20000**（seed 42，val 选出的冠军） |
| 可训 | 只有 U-Net |
| 冻结 | VAE、CLIP text encoder（实测 0 个可训参数、0 个带梯度） |
| lr | **1e-6**（基线是 3e-5，降 30 倍） |
| 步数 | 4000（基线是 20000） |
| caption 条件 | **不动**——这一阶段只隔离几何/尺度 |
| 其余 | batch 4×3、fp16、t=999、grad clip 1.0、8-bit Adam，全部沿用 |

⚠️ 度量项需要梯度穿过 VAE 解码器，而配方是 fp16。**fp16 反传会把梯度下溢到零**，
所以训练器额外持一份**只解码的 fp32 VAE 副本**（在 `.to(fp16)` 之前深拷贝，拿到的是
`from_pretrained` 加载的权重而不是 fp16 的往返）。这不是猜的，见 §5 门禁 C。

---

## 5. 冒烟门禁（本地全部通过）

`tools/smoke_metric_adaptation.py`，四道。

**A 归一化**：米 → 1/m → [-1,1] → 回来，往返最大误差 **1.49e-08 1/m**（1.14e-05 m）。
带外有效像素 1.6–3.8%，与 Q0.02/0.98 的预期一致。

**B 度量预测**（原 checkpoint，未适配）：

| 换算方式 | q 量程 (1/m) | D 量程 (m) | q≤0 | NaN/Inf | AbsRel |
|---|---|---|---|---|---|
| `raw_inverse`（y 直接当 1/m） | 0.0042–1.036 | 0.97–238.6 | 0% | 0 | 0.7315 |
| `global_norm` | 0.0185–0.208 | 4.81–54.1 | 0% | 0 | 0.0701 |
| `global_affine` | 0.0150–0.239 | 4.19–66.6 | 0% | 0 | 0.0974 |
| *逐帧 SSI（参照）* | — | — | — | — | *0.0551* |

解码出的 `y` 会略微越出 `[0,1]`（最大 1.036），已正确处理。

**C 梯度**：

```
fp32 解码（训练器用的）  U-Net 梯度范数 8.69e-02，690 个张量有梯度
fp16 解码（陷阱）        U-Net 梯度范数 0.00e+00，690 个全部恰好为零
VAE(pipeline) / VAE(fp32副本) / CLIP   0 个可训参数，0 个带梯度
U-Net                                  867,568,324 个可训参数
```

fp16 下溢不是引用别处的结论，是这次实测的。

**D 评估无拟合**（行为验证，不是读代码）：同一张预测对 GT 和 3×GT 打分，
**预测→深度的变换逐比特相同**；协议报告 `alignment scale=1.0 shift=0.0`。
对照：`ssi_disparity` 下 scale 随 GT 变（−0.00137 → 0.00048）——那正是度量路径没有的依赖。

**端到端**：`accelerate` 跑满 6 步，exit 0，`SL_A`/`SL_R`/`SL_M` 三项都在，
`[metric step N]` 的量程日志正常，无 NaN，pipeline 正常保存。

---

## 6. 评估路径

`tools/train_route_suite.py` 新增 `--align-mode` 与 `--metric-source`：

| 行 | `--align-mode` | `--metric-source` | 是否用 test GT 拟合 |
|---|---|---|---|
| 原 checkpoint，单位自检 | `none` | `raw_inverse` | 否 |
| 原 checkpoint，训练集标定 | `none` | `global_affine` | 否 |
| **适配后 checkpoint** | `none` | `global_norm` | 否 |
| 原 checkpoint，参照 | `ssi_disparity` | — | 是（仅作参照） |

- 三种 source 都是同一个仿射 `q = a·y + b`，只是 `(a,b)` 的来源不同，评估器把实际用的
  两个数和它们的出处写进结果 JSON。
- `affine_invariant` 和 `metric_no_test_alignment` **写不同的文件**，且写入前检查已有
  文件的 `evaluation_mode` 与 `metric_source`，不一致就拒绝覆盖。
- ⛔ 最后一行不是度量深度结果，不能和前三行并排排名。

---

## 7. 状态与下一步

| | |
|---|---|
| 代码 | ✅ commit `9fd93d15e` |
| 常数 + 训练集仿射 | ✅ 集群上已拟合（`metric_prep` 作业 7518919，26 分钟） |
| 冒烟门禁 A–D | ✅ 本地全过 |
| **训练** | ❌ **未投** |

投训练（两臂，λ_metric = 1 与 20）：

```bash
# 冒烟（20 步，验开关）
sbatch -J ma_smoke --time=00:40:00 --export=ALL,SMOKE=1,\
METRIC_NORM=$IRIS_RUNS/metric_adapt/metric_norm_train.json,\
INIT_CKPT=$IRIS_RUNS/iris_ms2/iris_ms2_full8_thermalcap/converted/step20000_weights.pt,\
TRAIN_MANIFEST=$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl \
  ~/Iris/slurm/metric_adapt.sbatch

# 全量：去掉 SMOKE，加 LAMBDA_METRIC=1 / LAMBDA_METRIC=20，各一臂
```

冒烟要看三样：`SL_A`/`SL_R`/`SL_M` 都在；`[metric step N]` 的量程行正常；
**稠密目标的真实 clip 比例**（本地用假伪 GT 量到约 20%，那个数不作数）。

### 7.1 事先写下的预期

逐帧仿射的 scale 变异约 **10%**（从官方 val 的 per-sample CSV 读出）。若几何正确、
只有尺度逐帧漂移，冻结一个全局仿射后 AbsRel 会额外吃进约 0.08。叠在现有的仿射不变
0.077 上，**预期 test AbsRel 落在 0.11–0.16**；12 帧 test 子集上实测 global_affine 是
**0.1379**，落在里面。

对照的靶子（我们自己跑出来的基线 `align=none` 列）：

```
最强 NeWCRF   0.0832 / 0.0830 / 0.1045   (day/night/rain)
最弱 DORN     0.1032 / 0.1028 / 0.1292
```

**这个预期在跑之前写下**，跑完再解释就没有解释力了。

---

## 8. 与横向对比的关系

- 微调之前：我们的输出没有单位，**无法进入基线那一列**。
- 微调之后：输出是米制逆深度，可以走 `align=none`（无 GT）或 `align=median`
  （1 参数，与 DORN 那四行同尺）。
- ⚠️ 主线的 caption 结论是在**未适配**的 checkpoint 上测的。适配会改变权重，
  所以适配后的模型是**另一个模型**——caption 的结论要么在它上面重测，要么明确说明
  那两张表来自不同的 checkpoint。这一点在论文里不能含糊。

相关：`docs/COMPARISON_PROTOCOL_20260826.md`
