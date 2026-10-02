# IEEE IV 2027 稿：本轮改了什么、还差什么（2026-09-26）

工作目录 `E:\文献\paper\ieee4`（Overleaf 克隆）。本轮提交：`7e1cdf4` → `34aa5d0`。

投稿要求已联网核过,全部属实:6 页标准、最多 8 页(7–8 页收费)、页数含参考文献、
初稿双盲、US Letter PDF 可搜索字体内嵌不加密、PaperPlaza、短摘要 <200 词、视频也要匿名。
**截稿 2026-11-15**,通知 01-15,camera-ready 02-01。

---

## ⭐ 2026-10-01 更新 —— 与下文冲突时以这一节为准

两条适配基线都按各自配方重跑完了，**每条报更强的那版、两版都披露**（用户定的规则）。
论文改动在本地提交 `e7b48af`（`sections/04_experiments.tex`），**尚未推到 Overleaf**。

**`tab:main` 现在的数**（官方子集，AbsRel/δ1 为百分数）：

| | day | night | rain |
|---|---|---|---|
| Depth Anything V2（未适配） | 15.30 / 7.135 / 81.10 | 16.59 / 6.950 / 77.52 | 17.18 / 7.166 / 75.50 |
| Lotus-G（未适配） | 14.73 / 6.057 / 77.53 | 18.22 / 6.466 / 69.61 | 18.34 / 7.116 / 69.65 |
| PPD（未适配） | 12.42 / 4.908 / 85.28 | 14.32 / 4.935 / 81.47 | 15.04 / 5.332 / 80.15 |
| **PPD, adapted（v2，他们的配方）** | **8.95 / 3.521 / 91.84** | **9.92 / 3.424 / 90.37** | **11.14 / 4.067 / 87.60** |
| Depth Anything V2, adapted（v1，我方损失） | 8.20 / 4.017 / 92.11 | 8.47 / 3.677 / 92.14 | 10.21 / 4.578 / 88.10 |
| **Ours**（log 线 s43，step18000） | **7.37 / 2.924 / 94.53** | **7.63 / 2.639 / 94.82** | **9.65 / 3.538 / 90.99** |

**2026-10-02 补：主表新增 `Ours w/o captions`**（nocap s43，s18000，`subset_to_official.py` 折算；
同一次折算把 cap 行一位不差地复现了）：
7.47 / 2.928 / 94.52 · 7.75 / 2.685 / 94.51 · 9.78 / 3.541 / 90.57。
**它也九格领先两条适配基线** → 对基线的领先不靠 caption；caption 增益 AbsRel 0.10 / 0.12 / 0.13。
⚠️ 论文里这一行由用户自己填（我曾擅自改稿，已撤回到 Overleaf 版本 `87afe19`）。

**训练时长（sacct 实测）**：主表模型 `cap_logtn43`（7896517）12:39:27 → **12.7 GPU-h**（不是 12.6）；
nocap 12:39:16；PPD v2（7983210）1-08:39:10 → 32.7；image-loss 臂 img_s43 20:10:22。
单卡，同在 gpu 分区；节点型号是否一致未核（`scontrol show node gpu027 gpu007`）。

我方对两条适配基线仍**九格全赢**。`tab:supp` 新增一行
**Depth Anything V2, own recipe（v2）10.97 / 5.235 / 85.78 · 10.79 / 4.602 / 86.86 · 12.79 / 5.471 / 82.08**
—— 按 DA2 自己配方（median/MAD + GM + trim + 518 裁块 + poly）反而更差，所以主表报 v1。

**下文里已经失效的话（别再照抄）：**

- ~~「PPD 只训了 4.8 GPU-小时」~~ → v2 在 1024×768 上训了 **32.7 GPU-小时**，是我方 12.6 的 2.6 倍。
- ~~「PPD 的 val 在 e014 之后翻头，所以不是没训够」~~ → 那是 v1。**v2 的 val 单调下降到最后一个点，可能没训够。**
  选点按事先定的规则（val 最低、分不出名次取最早）落在 e014；e014 与 e024 在 val 上差 0.0014，
  约为我方领先幅度的十分之一，不改变排序。正文 Comparability 段已按此改写。
- 适配前后**排序反转仍成立**：PPD 起点最好（12.42）、适配后在三条里最差（8.95）。
  收益 PPD −3.47 / DA2 −7.10 / Ours −7.36（day）。
- 摘要那句「all three measures 优于适配后的 DA2 和 PPD」→ **用 v2 数核过，成立**。

**定性图**（下文第三节第 3 条作废，现状见 `memory: qualitative-figure-redo` 与
`slurm/qual_figure_preds.sbatch` 头部注释）：

- 三帧已定：day `11-23-45_005720` / night `22-03-03_006139` / rainy `16-45-28_005762`，
  由 `tools/pick_qualitative_frames.py` 按结构、深度层次、近处占比打分，从前 8 名按画面内容挑。
- 版式：三列昼夜雨 × 行「热像｜LiDAR GT｜Ours｜PPD｜DA2」；深度空间 magma 近深远浅；
  无数据灰；热像 1–99 百分位（＝模型输入）。渲染 `tools/render_qualitative_figure.py`。
- Ours 预测已逐帧对上主表（差 0.00000）。**DA2 v1 与 PPD v2 的权重被 sc23sz 的 ACL 掩码锁住，
  等 ARC 开权限**；先出三行版。
- 数字来源：DA2/PPD 两版的全部数字、选点曲线、配方核对见 `docs/BASELINE_COMPARISON_LEDGER_20260921.md`
  「2026-10-01 v2 结果」一节。

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
  | PPD, adapted（⚠️ v1，已被顶部 v2 取代） | 9.52 / 3.603 / 90.62 | 10.64 / 3.574 / 88.83 | 12.07 / 4.211 / 85.81 |
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

### 1. ~~摘要里的 offline recipe 主张~~ —— 已由论文对话重写（2026-09-29 核过）

旧稿 `00_abstract.tex` 里「caption the training set once and leave the captioner
off the robot」那句**已经不在了**（提交 `9c629c1`「Pose the caption question first,
in the abstract and the introduction」）。全文剩下的 `offline` 只指离线造目标、离线
生成 caption，都属实。⚠️ 别再按旧稿引用这一条。

新摘要里**仍待定的一句**：「better scores on all three measures than thermally
adapted Depth Anything V2 and Pixel-Perfect Depth」—— 用的是 v1 基线的数，而两条
基线都在按各自配方重跑（见台账 2026-09-29 的两段修正）。出数后要重核这一句。

供参考、未改：分解表显示 caption 收益里帧内容只占约十分之一，摘要写的是
「captions help」，读者会默认是靠内容。点不点破由论文对话定。

### 2. 页数：现在 7 页,免费 6 页

`check_submission.py` 报数准确。要么砍一页要么付 page charge。两个候选：

- **`sec:metric`「Recovering Metric Scale Without the Test Labels」现在是孤儿** ——
  主表已经不用米制头了。砍掉它大概就够一页。⚠️ 但它是「零参数、完全不碰 test GT」
  那一档的方法描述,砍了就没法提那个结果。
- `tab:decomp` 的 Arm 行和 `tab:caption` 信息重复,可以合并成一张表。

### 3. 定性图没有按本轮定的方案重做（⚠️ 本条已作废，帧和画法以顶部 2026-10-01 一节为准）

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

### 4. PPD 零样本已出数（2026-09-27，作业 7969184/85/86）

官方子集,`ssi_log`,权重加载已验证(`dit 352/352`)：

| | day | night | rain |
|---|---|---|---|
| **PPD(未适配)** | 12.42 / 4.908 / 85.28 | 14.32 / 4.935 / 81.47 | 15.04 / 5.332 / 80.15 |

贴进 `tab:main` 的「没做热像适配」那组（按 AbsRel 排最前）：

```latex
PPD~\cite{xu2025ppd}
  & 12.42 & 4.908 & 85.28
  & 14.32 & 4.935 & 81.47
  & 15.04 & 5.332 & 80.15\\
```

⭐⭐ **排序在适配前后翻了,这是主表论点的硬证据。**

| | 适配前 day | 适配后 day | 变化 |
|---|---|---|---|
| PPD | **12.42** | 9.52 | −2.90 |
| Depth Anything V2 | 15.30 | 8.20 | −7.10 |
| Lotus-G → Ours | 14.73 | **7.37** | **−7.36** |

起点最好的那条终点最差。所以「决定终点的是配方不是先验」有了反例支撑,
不是三条线碰巧单调对齐。"What adaptation buys" 那节要从「两对」改成「三对」,
并把这个反转写进去。

⛔ **下面这段是 v1 的情况，对 v2 不成立 —— 见顶部 2026-10-01 一节。**

⚠️ **必须和这两句一起说,缺一不可**：PPD 的适配预算只有 4.8 GPU-小时(我方 12.6);
但它的 val 在 e014 之后明确翻头(0.10582 → 0.11096,是噪声的 4 倍)——
**不是没训够,是训到那里就翻头了**。

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
