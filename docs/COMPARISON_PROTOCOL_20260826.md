# 横向对比协议（2026-08-26）

**主张**：caption 在热像深度估计上起作用。

**这份文档记的是支撑条件**：横向对比本身不是主张。"caption 让模型好了 0.002" 只有在
模型处在有竞争力的工作点上才说明问题——否则那 0.002 可能只是在补自己的短板。把我们的
臂放到已发表基线旁边，是为了证明这个效应发生在一个像样的位置上。

所以这份文档回答一个问题：**怎样才算把我们的数和别人的数放在了同一张表里。**

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
