# IEEE IV 2027 稿：本轮改了什么、还差什么（2026-09-26）

工作目录 `E:\文献\paper\ieee4`（Overleaf 克隆）。本轮提交：`7e1cdf4` → `34aa5d0`。

投稿要求已联网核过,全部属实:6 页标准、最多 8 页(7–8 页收费)、页数含参考文献、
初稿双盲、US Letter PDF 可搜索字体内嵌不加密、PaperPlaza、短摘要 <200 词、视频也要匿名。
**截稿 2026-11-15**,通知 01-15,camera-ready 02-01。

---

## 一、本轮定下的两件事

1. **主表只放「从可见光模型迁到热像」这一类**。⛔ NeWCRFs、AnyThermal、
   Ours+image loss、BTS/AdaBins/DORN **一律不进主表**,全进补充表。
2. **整篇迁到 log 线**(`log_truncnorm` + `ssi_log`),**Lotus-D 砍掉**
   (它没有 log 线的 run)。

---

## 二、已完成

**表**

- `tab:main` 换成迁移设定,五行,分「没做热像适配 / 适配之后」两组：

  | | day | night | rain |
  |---|---|---|---|
  | Depth Anything V2(未适配) | 15.30 / 7.135 / 81.10 | 16.59 / 6.950 / 77.52 | 17.18 / 7.166 / 75.50 |
  | Lotus-G(未适配) | 14.73 / 6.057 / 77.53 | 18.22 / 6.466 / 69.61 | 18.34 / 7.116 / 69.65 |
  | PPD, adapted | 9.52 / 3.603 / 90.62 | 10.64 / 3.574 / 88.83 | 12.07 / 4.211 / 85.81 |
  | Depth Anything V2, adapted | 8.20 / 4.017 / 92.11 | 8.47 / 3.677 / 92.14 | 10.21 / 4.578 / 88.10 |
  | **Ours** | **7.37 / 2.924 / 94.53** | **7.63 / 2.639 / 94.82** | **9.65 / 3.538 / 90.99** |

  九格全赢。协议：逐图 2 参数仿射不变,各在自己的原生输出空间。
  DA2 零样本那一行本来是全帧口径,本轮已折算到官方子集,**现在每一行都是
  2331 / 2292 / 2503 帧**。
- `tab:supp` 新建：NeWCRFs / BTS / AdaBins / DORN(MS2 米制监督)、AnyThermal
  (热像专用)、Ours + image loss,三块各自注明为什么不在主表。
- `tab:caption` 换成 log 线,只剩 seed 42/43,cap 六格全胜。
- `tab:decomp` 重做,**只剩 Arm 和 Content 两行**(理由见下)。
- `references.bib` 加了 PPD：key 是 `xu2025ppd`(一作 Xu,NeurIPS 2025)。

**正文**

- 方法一节的归一化从 disparity 改成 log,新增公式 `eq:norm` 和理由
  (`d(log D)/dD = 1/D`,固定步长＝固定比例,而 AbsRel 本身是相对误差)。
- 删干净 Lotus-D：`\Phi_b` 的 cases 分支、管线图说明、「两个底座」的铺垫、
  两张消融表、Limitations 里的 "replication across backbone"(如实撤回)。
- Setup 的 **Alignment** 整段重写：五个模型都不出米制,各在自己空间拟合 2 参数
  是唯一可比的协议;补充表给基线的也是 2 参数,比它们自己发表的 1 参数中位数缩放更宽松。
- **Comparability** 段补上预算不对等(PPD 4.8 GPU-小时 vs 我方 12.6,且 e014 之后 val 翻头)。
- 「Comparison with supervised thermal depth」拆成
  **What adaptation buys** + **Beyond the transfer setting** 两节。
- 定性图 caption 从「nothing fitted to the test labels」改成「after the per-image affine fit」。

**⭐ 分解表的实质修正**

我一度把「cap 臂给空 prompt」当成「权重效应」写进表里(+0.11~+0.21,看起来是有害)。
**那是错的** —— cap 臂训练时每一步都有 caption,空串是它从没见过的条件,
量到的是**条件错配**。`docs/RESULTS_20260813_IRIS_MS2.md` §4 的 2×2 早就证过:
两个错配格 **+0.00200 / +0.00213,对称**,所以那是错配的属性不是文本的属性。
`Presence` 行同理脏(它的 empty 基线就是那个错配格)。

现在只剩两行,两边都保持训练条件不变：

| 效应 | seed | day | night | rain |
|---|---|---|---|---|
| **Arm**(各按训练时的 prompt 打分) | 42 | −0.09 | −0.14 | −0.41 |
| | 43 | −0.11 | −0.14 | −0.13 |
| **Content**(本帧 caption vs 随机置换) | 43 | −0.02 | −0.01 | −0.03 |

**内容效应只有臂效应的约十分之一。** 置换对照从「旋转集合」换成**随机置换**
(旋转在 MS2 上泄漏,路线会折返)。

---

## 三、⛔ 没做的,按优先级

### 1. 摘要和 Introduction 的核心主张没改 —— **必须先定**

`sections/00_abstract.tex:20`：

> "...which makes an offline recipe available --- caption the training set once
> and leave the captioner off the robot."

**这在 log 线上不成立。** 收益在推理端,captioner 必须在车上。

我没自作主张改。有一个数据支持的替代方向：**九成收益与 caption 的内容无关**
(别的帧的 caption 也行),所以需要的不是「正确的描述」而是「文本通路被激活」。
⚠️ 但「一句固定的通用 prompt 也行」**没测过** —— 我们只测到「别帧的 caption 行」。
要写进摘要得先补一个固定 prompt 的臂(一次推理,不用训练)。

`sections/01_introduction.tex` 第 60、78–90 行同一主张,连带要改。

### 2. 页数：现在 7 页,免费 6 页

`check_submission.py` 报数准确。要么砍一页要么付 page charge。两个候选：

- **`sec:metric`「Recovering Metric Scale Without the Test Labels」现在是孤儿** ——
  主表已经不用米制头了。砍掉它大概就够一页。⚠️ 但它是「零参数、完全不碰 test GT」
  那一档的方法描述,砍了就没法提那个结果。
- `tab:decomp` 的 Arm 行和 `tab:caption` 信息重复,可以合并成一张表。

### 3. 定性图没有按本轮定的方案重做

本轮定了可视化方案但**还没应用到 `figures/qualitative.png`**：

- 画法 **A**：深度空间 + `magma`,**近处深、远处浅**(和 Iris Figure 5 一致)。
  ⚠️ 我曾主张 `Spectral`(Lotus 代码里的),那是错的 —— Iris 论文的图不是用
  `colorize_depth_map` 画的。用户逐图核过。
- 展示帧换成 **`2021-08-06-11-23-45_000805`**(街道,两侧建筑 + 货车 + 立柱),
  不是原来那张空旷的林荫路。覆盖 31.6%、对齐后 AbsRel 0.0754,⚠️ 高于中位数,
  要把这两个数标在图上。
- **GT 那格的无数据底色要从黑改成中性灰** —— A 画法里近处也是黑的,两个黑会撞。

生成脚本在 `E:\project\Iris\docs\figures\`(`build_metric_chain.py` /
`build_slide_variants.py` 已经切到视差 + Spectral,**还没切到 A**)。

### 4. PPD 零样本在跑

补主表「没做热像适配」那组的第三行。作业 `slurm/ppd_zeroshot.sbatch`,
跑全帧再折算(和其余行一致),墙钟 8 小时,对齐 `ssi_log`。
出来之后加一行到 `tab:main`,并把 "What adaptation buys" 那节的
「两对」改成「三对」。

### 5. 其它 TODO

- `sections/05_conclusion.tex` 整节空的
- `sections/01_introduction.tex:74` 有个 `\papertodo`
- `sections/acknowledgment.tex` 有 `\papertodo`（双盲阶段本来就要删）
- `sections/00_abstract.tex:22` 的 TODO 注释

---

## 四、坑（本轮踩到的）

- ⛔ **改 .tex 不要用 bash heredoc 跑 python** —— 反斜杠会被吃掉,`\\begin` 变成
  `\begin` 然后匹配失败。用 Edit 工具。
- **Slurm 在提交那一刻就把脚本拷走了**,之后 `git pull` 不改变已排队作业跑的内容。
- 对齐族必须和训练时的 norm_type 配套,错了差 30 倍且**不报错**：
  `trunc_disparity→ssi_disparity`、`truncnorm→ssi`、`log_truncnorm→ssi_log`。
- `official_frames()`(在 `tools/run_ms2_supdepth_baselines.py`)才是官方子集的权威
  实现;对清单直接 `[::10]` 会选出另一批帧(候选 23317 vs 清单 23311,顺序也不保证)。
- `$IRIS_MANIFEST_DIR` / `$IRIS_MS2_ROOT` 在登录节点是空的,只有作业里才有。

---

## 五、数字来源

| 内容 | 文件 |
|---|---|
| 主表、补充表 | `docs/RESULTS_20260925_FULL_COMPARISON.md` 表一 |
| caption 两臂、内容效应 | `docs/RESULTS_20260921_LOG_LINE.md` §2 |
| 2×2 错配对称 | `docs/RESULTS_20260813_IRIS_MS2.md` §4 |
| DA2 / PPD 档案 | `docs/BASELINE_COMPARISON_LEDGER_20260921.md` |
| image loss 那组 | `docs/RESULTS_20260923_IMAGE_LOSS.md` |
| 「糊」的归因、前端方向关闭 | `docs/RESULTS_20260926_WHERE_THE_BLUR_IS.md` |
