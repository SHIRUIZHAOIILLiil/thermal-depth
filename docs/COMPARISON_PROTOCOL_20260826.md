# 横向对比协议（2026-08-26）

**主张**：caption 在热像深度估计上起作用。

**这份文档记的是支撑条件**：横向对比本身不是主张。"caption 让模型好了 0.002" 只有在
模型处在有竞争力的工作点上才说明问题——否则那 0.002 可能只是在补自己的短板。把我们的
臂放到已发表基线旁边，是为了证明这个效应发生在一个像样的位置上。

所以这份文档回答两个问题：**我们重新训练的是什么**，以及**怎样才算把我们的数和别人的
数放进了同一张表**。

---

## 0. 我们训练了什么

### 0.1 起点与可训部分

| | |
|---|---|
| 基座 | `jingheya/lotus-depth-g-v2-1-disparity`（Lotus-G，SD2 系） |
| 可训 | **只有 U-Net**（867.6 M 参数） |
| 冻结 | VAE、CLIP text encoder |
| `conv_in` | 已是 8 通道，**不做** 4→8 扩展（那是从 SD2-base 起训才需要的） |
| 训练脚本 | `lotus/train_iris_ms2_g.py` —— Iris 自己那份 trainer 的复制件，**只换数据集，超参一个没动** |

### 0.2 配方（`lotus/train_scripts/train_iris_ms2_g_depth.sh`）

```
batch 4 × 梯度累积 3 = 有效 12      lr 3e-5，constant，warmup 0
timestep 999（固定，单次前向）      max_train_steps 20000
mixed precision fp16                 grad clip 1.0
8-bit Adam                           gradient checkpointing
random_flip                          seed 42（第二个种子 43）
```

⚠️ `random_flip` 会翻转图像和深度图但**不翻转 caption 文本**，所以"左边有辆车"可能
描述的是翻转后的右边。这是 Iris 原本的行为，我们沿用而不是悄悄改掉。

### 0.3 目标：稠密补全深度图

```
每个像素：  有激光 → 用激光
            无激光 → 用标定后的 AnyThermal 伪深度
            全图 clip 到 [1e-3, 80] m
再逐帧 trunc_disparity 归一化（该帧 1/depth 的 Q2/Q98 映到 [-1,1]）
```

**为什么必须稠密**：Iris 把深度图当成一整张图 VAE 编码，latent 的有效掩码是无效掩码
做 8×8 max-pool——**一个坏像素废掉整个 latent cell**。MS2 激光只覆盖约 25% 的帧，
稀疏图会废掉几乎全部 cell。补全之后每个 cell 都存活，这正是这条线成立的前提。

### 0.4 目标函数：两支路 latent MSE

```
L = anno_loss + rgb_loss

anno_loss :  MSE(U-Net 的 x0 latent, VAE(稠密目标).sample())，只在有效 latent cell 上
rgb_loss  :  热像重建支路，同样在 latent 空间
```

**没有任何图像空间的损失，也没有任何 teacher。** 预训练权重只当初始化和冻结特征提取器用。

### 0.5 三条臂

官方 8 条训练序列，除 caption 外配方完全相同：

| 臂 | 帧数 | caption |
|---|---|---|
| `iris_ms2_full8_nocap` | 76,547 | 每帧空串（无文本对照） |
| `iris_ms2_full8_thermalcap` | 75,688 | InternVL3-8B，prompt `thermal_depth_v3_1`，**从热像生成** |
| `iris_ms2_full8_rgbcap` | 76,534 | InternVL3-8B，prompt `rgb_depth_v1`，**从 RGB 生成** |

caption 一律 ≤75 CLIP token（超长会被 tokenizer 截断，截掉的正是句尾那半句近远排序）。

⚠️ **三条臂的训练帧集不完全相同**，最多差 859 帧（1.1%），原因是 caption 生成在不同帧
上失败。明细见 `docs/data/EXCLUDED_FRAMES.md`。每一条跨臂结论都带着这个。

第二个种子（43）：`nocap_s43`、`thermalcap_s43` 已完成，`rgbcap_s43` 训练中。

### 0.6 选 checkpoint

官方 3 条验证序列，14,249 帧按 stride 4 抽成 3,563 帧，取 **val AbsRel 最小**。

⚠️ **prompt 与 caption 来源必须与该臂训练时一致**——用它没训过的文本分布挑权重，
挑出来的不是这条臂最好的那个。

选中的结果：

| 臂 | seed 42 | seed 43 |
|---|---|---|
| nocap | 12000 | 20000 |
| thermalcap | **20000** | 4000 |
| rgbcap | 20000 | 训练中 |

⚠️ **各臂选中的步数不同**，所以两条臂差的不只是 caption，还有训练长度。这是"每条臂用
自己的 val 曲线选冠军"这个规则的必然结果，是正规做法，但必须写进方法。

### 0.7 推理

单次 U-Net 前向，`t=999`，直接取 x0 作为答案；输入图的 latent 取后验的 **mode**。
这是本项目全部历史数字的口径。多步去噪是**双重分布外**（这条线只在 t=999 微调过，
它出发的 Lotus-G 权重也只在 t=999 训过），只能当排除性对照读。

### 0.8 基线：我们没有重训

DORN / BTS / AdaBins / NeWCRF 用的是 `UkcheolShin/SupDepth4Thermal` 发布的权重原样加载
（`depth_net.` 前缀剥掉，`strict=True`，张量数 660/736/973/513 逐个吻合）。我们只在自己
机器上重跑推理和评估，不碰它们的训练。

⚠️ **训练预算不对等**：论文自述训练集 26K 对，我们是 75,688 帧。这个差异在我们这一边，
必须披露。（他们发布的配置里 `train.sample_step: 1` 与 26K 这个数对不上，未解决。）

---

## 1. 已发表数字的真实协议（从源码和论文实证）

⛔ **没有一个已发表的 MS2 热像深度数字是"无对齐度量"的。** 全部用了 test GT。

| 方法族 | 原生输出 | test 时对齐 | 拟合空间 | 参数 | 出处 |
|---|---|---|---|---|---|
| DORN / BTS / AdaBins / NeWCRF | 度量深度 | 逐图**中位数缩放** `p·med(gt)/med(p)` | 深度 (m) | **1** | `SupDepth4Thermal/test_monodepth.py` 走 `compute_depth_errors(gt,pred)`，签名 `align=True` 默认 |
| AnyThermal Table IV（MiDaS 系） | 相对深度（与米差一个仿射） | 逐图 **scale+shift** 最小二乘 | 深度 (m) | **2** | 同一脚本另一分支：`inference_depth(img, gt)` 先拟合，再 `align=False` |
| **我们（Lotus-G 热像）** | 归一化视差 [0,1] | 逐图 **scale+shift** 最小二乘 | **视差 (1/m)** | **2** | `ms2_eval/official_protocol.py` 的 `ssi_disparity` |

**⚠️ 2 参数 ≠ 2 参数。** 深度空间仿射 `D = a·p + b` 与视差空间仿射 `D = 1/(a·p + b)`
是两个互不包含的函数族。同样的自由度、同样从 test GT 取信息，但可修正的误差形状不同。
每个模型必须在**自己的原生空间**里对齐——强行统一会把某一族拖垮
（实测：我们走 `ssi` → 0.37，AnyThermal 走 `ssi_disparity` → 0.2724）。

对 DORN 那四个，我们比他们多拿一个自由度。这一点必须在表里写明。

## 2. 官方评估帧集

**每 10 帧抽 1 帧，且抽帧作用在跨序列拼接后的整个列表上。**

- 序列顺序 = `test_<env>_list.txt` 的行序
- 每条序列的候选 = `sorted(sync_data/<seq>/thr/img_left/*.png)`，**全部帧**
- 拼接后 `sample_set[0:-1:10]`（`configs/Base/Base_Sup_Mono_Depth.yaml` 的 `test.sample_step: 10`）

得到 **day 2332 / night 2292 / rain 2503**，与论文自述的 "2.3K / 2.3K / 2.5K" 一致。

⚠️ 两个坑：
1. **相位跨序列延续**。按序列各抽各的会得到不同帧集（rain 上是 2504 而非 2503）。
2. **候选必须取自 `img_left` 目录，不能用我们的 manifest**。11-23-45 的 img_left 是 5,810
   而我们的清单是 5,805（5 帧没 caption），少了帧索引就整体错位。

实现：`tools/run_ms2_supdepth_baselines.py::official_frames`。

## 3. 共同项（已核实一致）

| | 值 |
|---|---|
| GT | `proj_depth/<seq>/thr/depth_filtered/`，`/256.0` |
| 有效掩码 | `gt > 1e-3 & gt < 80` |
| 对齐后 clamp | `[1e-3, 80]` |
| 聚合 | 逐图算指标 → 对图取**无权平均**（macro） |
| 输入分辨率 | 热像原生 640×256，不缩放 |
| 预测缩放到 GT | 双线性，`align_corners=False` |

**他们的输入预处理**（复现基线时必须照抄，`dataloaders/__init__.py::get_augmentations`）：

```
RescaleTo([256,640], INTER_LINEAR) → /2**14
  → 逐图 1–99 百分位裁剪并归一到 [0,1]      ← 这一步在 eval 阶段也生效
  → (x − 0.45) / 0.225 → repeat 到 3 通道
```

## 4. 复现门禁（已通过）

用他们发布的权重、他们的预处理、他们的帧集，在我们机器上跑一遍，`median` 那一列
必须复现论文 Table IV(c)。**12/12 格通过**：AbsRel 最大差 0.0005，RMSE 三位小数吻合，
δ1 同。这一步过了，我们才有资格说"同一台评估器、同一批帧、同一个 mask"。

```
newcrf day  0.0712 vs 已发表 0.071    RMSE 2.717 vs 2.717   δ1 0.9511 vs 0.951
dorn   rain 0.1273 vs 已发表 0.127    RMSE 4.377 vs 4.377   δ1 0.8417 vs 0.842
```

⚠️ 门禁不过时，`none` 和 `ssi` 两列一律不读。

## 5. 把我们的数搬到同一帧集

我们的评估默认跑全帧。`tools/subset_to_official.py` 从已有的 per-sample CSV 取官方子集
重算 macro 平均，**不用 GPU**。实测两个口径差 ≤0.00025：

| | 全帧 | 官方子集 |
|---|---|---|
| day | 0.07118 | 0.07141（2331/2332，缺的那帧没 caption） |
| night | 0.07950 | 0.07955 |
| rain | 0.09302 | 0.09321 |

**表用子集**（和基线可比），**消融统计用全帧**（基线不参与，没理由丢 90% 的帧）。
这一条要在论文里写明。

## 6. 工具链

| 用途 | 工具 |
|---|---|
| 复刻官方帧集 + 跑基线 | `tools/run_ms2_supdepth_baselines.py`、`slurm/baseline_bench.sbatch` |
| 把我们的数搬到官方子集 | `tools/subset_to_official.py`（`--write-subset` 可导出子集 CSV） |
| 逐帧配对 + 块 bootstrap | `tools/compare_route_evals.py --block-lengths 200 1` |
| 巡检所有评估结果 | `tools/survey_eval_runs.py` |
| 协议实现 | `ms2_eval/official_protocol.py`（`ssi` / `ssi_disparity` / `median` / `none`） |

## 7. 当前数字（官方子集，逐图 2 参数，各自原生空间）

| | day | night | rain |
|---|---|---|---|
| DORN | 0.0866 | 0.0868 | 0.1163 |
| AdaBins | 0.0795 | 0.0801 | 0.1046 |
| BTS | 0.0768 | 0.0797 | 0.1018 |
| NeWCRF | **0.0716** | **0.0735** | 0.0937 |
| 我们·无 caption | 0.07362 | 0.07944 | 0.09527 |
| 我们·RGB caption | 0.07362 | 0.08405 | 0.09600 |
| **我们·热像 caption（seed 42）** | **0.07141** | 0.07955 | **0.09321** |
| 我们·热像 caption（seed 43，全帧） | 0.07365 | 0.08153 | 0.09375 |

**读法**：seed 42 在 day/rain 与 NeWCRF 持平或微赢，night 落后；两种子均值略低于 NeWCRF。
RMSE 和 δ1 三个条件全面落后（RMSE 差 0.96–1.21 m），原因是视差表示在远场分辨率差而
AbsRel 权重压在近场。**三个指标必须一起报。**

## 8. 主张的证据（这才是文章要说的）

**caption 起作用**，两个种子：

| day AbsRel | 无 caption | 热像 caption | 差 |
|---|---|---|---|
| seed 42 | 0.07354 | 0.07118 | −0.0024 |
| seed 43 | 0.07698 | 0.07365 | −0.0033 |

night：seed 42 为零、seed 43 为 −0.0033，**两个种子不一致**，如实写。
rain：−0.0021 / −0.0055，两个种子都成立。

**热像 caption 优于 RGB caption**（全帧，块 bootstrap 200，九格里八格区间不含零，
唯一例外是 rain 的 δ1）：day −0.0024 / night −0.0047 / rain −0.0030。三条件三指标同向。

⚠️ 措辞：Iris 自己的结论是**只要描述的是这张图，文本就能起作用**，差别在程度。
所以只能说"热像 caption 比 RGB caption 更有用"，**不能说"文本必须来自热像"**。

## 9. 已知限制（必须进论文）

- 各臂用**自己的 val 曲线**选冠军，所以两条臂差的不只是 caption，还有训练步数
  （seed 42 是 20000 vs 12000，seed 43 是 4000 vs 20000）
- 种子噪声约 0.003（同配方换种子，day empty：0.07162 vs 0.07486），**大于我们与
  NeWCRF 的 margin**，所以"追平/超过"是种子相关的说法
- 我们比 DORN 那四个多一个拟合自由度
- 训练数据量：论文自述 train 26K，我们 75,688（配置里 `train.sample_step: 1` 与 26K 矛盾，未解决）
- day 子集缺 1 帧（2331/2332），RGB caption 线的 rain 缺 1 帧（2502/2503）

## 10. 相关记录

- `docs/data/EXCLUDED_FRAMES.md` — 各清单的帧数与掉帧原因
- `docs/RESULTS_20260822_FULL_DATA_LINE.md` — 全量线首批结果
- `docs/PAPER_EXPERIMENTS_V0_2_20260820.md` §Evaluation — 旧版方法段，⚠️ 其中
  "No method compared here predicts metric depth" 与本文档 §1 冲突，以本文档为准

## 11. 模型容量（2026-08-28 实测）

`tools/count_parameters.py`，CPU 数秒。**两边都不需要训练好的权重**：基线走
`build_network(..., pre_trained=False, ckpt_path=None)`，我们这边从 config 实例化，
所以这张表在任何 checkout 出代码的地方都能重跑。

| | backbone | 参数量 |
|---|---|---|
| AdaBins | EfficientNet-B5 + mini-ViT | 78.3M |
| DORN | ResNet-101 + 序数回归头 | 97.8M |
| BTS | ResNeXt-101 (`resnext101_bts`) | 112.8M |
| NeWCRF | Swin-L (`version="large07"`) | 270.4M |
| **我们** | U-Net 867.6M + VAE 83.7M + CLIP 文本塔 340.4M | **1291.6M** |

（VAE 拆开：encoder 34.2M / decoder 49.5M。两个都在推理路径上——encoder 编热像帧，
decoder 出深度。）

**我们是最大的**：NeWCRF 的 4.8 倍、AdaBins 的 16.5 倍。光 U-Net 一项就是 NeWCRF 的 3.2 倍。

⛔ **这一轴不能用来找胜利**，换任何说法都不行：可训练参数就是那 867.6M；
"只微调 500 步"是叠在 20000 步 + SD2.1 预训练之上的；Lotus 虽是单步、没有 50 步扩散
惩罚，但 1.29B 单步对 78M CNN 还是输。**正确用法是照 Iris Table 1 的 "Training Images"
列主动披露**，和"我们排第 4"放在一起，读者才知道那个第 4 是什么条件下拿到的。

## 12. 跨条件退化（2026-08-28 算完，负面结果）

以 §7 的官方子集 AbsRel 为准，看 day → night / rain 掉多少：

| | day→night 绝对 | day→night 相对 | day→rain 绝对 | day→rain 相对 |
|---|---|---|---|---|
| DORN | +0.0002 | **+0.2%** | +0.0297 | +34.3% |
| AdaBins | +0.0006 | **+0.8%** | +0.0251 | +31.6% |
| BTS | +0.0029 | +3.8% | +0.0250 | +32.6% |
| NeWCRF | +0.0019 | +2.7% | +0.0221 | +30.9% |
| 我们·无 caption | +0.0058 | +7.9% | +0.0216 | +29.4% |
| 我们·RGB caption | +0.0104 | **+14.2%** | +0.0224 | +30.4% |
| 我们·热像 caption s42 | +0.0081 | +11.4% | +0.0218 | +30.5% |
| 我们·热像 caption s43 | +0.0079 | +10.7% | +0.0201 | +27.3% |

**夜间：我们退化得最厉害**，8–14% 对基线的 0.2–3.8%，是最差基线的三到四倍。

**雨天不构成胜利**：我们两个种子 27.3% / 30.5% **跨在 NeWCRF 的 30.9% 两边**，
绝对差 0.0218 vs 0.0221 远小于 0.003 的种子噪声。⛔ 不要拿雨天鲁棒性说事。

**换到的东西（是限制，不是胜利）**：夜间是 caption 失效的地方 —— 无 caption 0.0794
对热像 caption 0.0795，效应归零；RGB caption 恶化到 0.0840，是全表最差退化。与 §8
"night seed 42 为零"一致，机制上的说法是：冻结的先验是 RGB 预训练的，夜间热像的强度
统计离它最远，描述这类帧的文本也最不可靠。

## 13. 结论：三条找胜利的路都走完了

AbsRel 排名（`align=none` 第 4/5）、参数量（最大，§11）、跨条件鲁棒性（夜间最差，§12）
—— **都不赢**。这不等于文章不成立：Iris 自己在 DIODE 上 δ1 72.5 对 Depth Anything v2
的 95.4、AbsRel 24.3 对 6.5，输得比我们难看得多，照样发表，因为它的主张住在**块内配对**
（同 backbone，加不加文本），上面那块只是量级参照。我们的主张同样在 §8，不在排名。
