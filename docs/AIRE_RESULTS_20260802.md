# Aire 集群实验结果汇总（2026-08-02）

两周任务在利兹 Aire 超算上的完整结果。**每个数字都注明了考卷、帧数、对齐方式**，
可直接用于汇报。迁移过程与集群约束见 `AIRE_HPC_MIGRATION_STATUS_20260729.md`。

---

## 0. 一句话

六条线全部完成 20 epoch 并在独立 test 集出数；**caption 的作用被拆成方向相反的两个效应**
（训练时有益、推理时有害），全项目最好的热像成绩来自"用 caption 训练、推理给空 prompt"
的组合（**0.0869**）；原版 AnyThermal 在远端几乎不退化（×1.13 vs 我方 ×1.90），
它与我方用的是**同一份稀疏 LiDAR 监督**，差别在架构；AnyThermal 特征分支经证伪检验
确认承载了几乎全部 condition 信号。

---

## 1. 口径（所有表格通用）

| 项 | 值 |
|---|---|
| **val** | `11-23-45`，5,810 帧，官方 BMSD 协议 |
| **test** | `16-08-46`，2,543 帧，官方 val split、白天、**训练与调参从未接触** |
| 对齐 | route suite 一律 `ssi_disparity`；原版 AnyThermal 用 `ssi`（官方表） |
| 协议 | `official BMSD ssi_disparity, min_depth 1e-3, max_depth 80` |
| 训练 manifest | `ms2_train_day2seq_clip75_20260728.jsonl`，19,949 帧 |
| 环境 | torch 2.3.1+cu121 / L40S 48GB |

> ⚠️ **两族模型的对齐方式不可混用。** 把 route suite 的预测按 `ssi` 对齐会得到 0.37
> （正确值 0.0884），把 AnyThermal 按 `ssi_disparity` 对齐会得到 0.2724（正确值 0.0821）。
> 分层工具的 `--align` 作用于所有预测目录，所以**不同对齐的模型必须分开跑、并排读**。

**跨机器标定已通过**：集群 b(e05) 全量 val `0.07749105` vs 本地 `0.0775`，差 **0.000009**
（阈值 0.0005）。本地与集群结果可并表。

---

## 2. 主表：六线 val + test

| 线 | 输入 | Condition | 可训练参数 | val | test | rmse | a1 |
|---|---|---|---:|---|---|---|---|
| **a** RGB+U-Net | RGB | 冻结 VAE latent | 867.57 M | 0.0791 | **0.0844** | 3.827 | 0.9269 |
| **b** Thermal+U-Net | Thermal | 冻结 VAE latent | 867.57 M | **0.0775** | 0.0884 | 3.956 | 0.9071 |
| **c1** VAE-adapter | Thermal | VAE latent + Adapter | 7.11 M | 0.1127 | 0.1217 | 5.465 | 0.8365 |
| **c2** VAE-adapter+U-Net | Thermal | VAE latent + Adapter | 874.67 M | 0.0932 | 0.1013 | 4.418 | 0.8826 |
| **d1** AnyTh-adapter | Thermal | AnyThermal 特征 + Adapter | 9.41 M | 0.0860 | 0.0973 | 4.432 | 0.8883 |
| **d2** AnyTh-adapter+U-Net | Thermal | AnyThermal 特征 + Adapter | 876.98 M | 0.0815 | 0.0903 | 3.855 | 0.9086 |

**读点**

- **热像几乎追平 RGB**：a 0.0844 vs b 0.0884，仅差 4.5%。
- **b 的跨序列泛化更差**：val→test 退化 +0.0109，a 只有 +0.0053。
- **d 系全面压过 c 系**：d1 > c1（0.0973 vs 0.1217）、d2 > c2（0.0903 vs 0.1013）。
  **AnyThermal 特征作为 condition 优于 VAE latent，两个参数量级上都成立。**
- **d1 用 9.41 M（1%）参数达到 0.0973**，逼近 867 M 的 b。

---

## 3. ⭐ caption 的两个效应（本轮核心发现）

以往把 caption 当成一个整体来评，结论总是"不显著"。**拆开后两个效应都显著，且方向相反。**

| 线 | 无 cap 训练 | cap 训练 / 空 prompt | cap 训练 / 真 caption |
|---|---|---|---|
| b Thermal+U-Net | 0.0884 | **0.0869** | 0.0882 |
| c2 VAE-adapter+U-Net | 0.1013 | 0.1030 | 0.1022 |
| d2 AnyTh-adapter+U-Net | 0.0903 | 0.0876 | 0.0881 |

| 线 | ① 权重效应（训练时用 caption） | ② 注入效应（推理时喂 caption） |
|---|---|---|
| b | **−0.00153** [−0.00188, −0.00118] 显著 | **+0.00130** [+0.00110, +0.00152] 显著 |
| c2 | **+0.00169** [+0.00146, +0.00194] 显著 | **−0.00082** [−0.00104, −0.00062] 显著 |
| d2 | **−0.00270** [−0.00332, −0.00206] 显著 | **+0.00046** [+0.00037, +0.00056] 显著 |

（负 = 更好。n=2,543 配对，bootstrap 3000 次。）

**四条结论**

1. **全项目最好的热像成绩是 0.0869** —— b 用 caption 训练、推理时给**空 prompt**。
   优于 d2 的 0.0876、优于朴素 b 的 0.0884。**caption 的价值在训练期，不在推理期。**
2. **注入税的大小由 condition 架构决定**：b +0.00130，d2 只有 +0.00046，**小 3 倍**。
   语义层级的 condition 不能让注入变得有益，但能把代价降到三分之一。
3. **d2 从 caption 训练中获益最多**（−0.0027，是 b 的 1.8 倍），只是起点低。
4. **c2 两个方向都反着来**，与它在所有其他指标上的劣势一致 —— VAE latent 这条路整体是错的。

**统计强度的区别（汇报时必须区分）**

- **② 注入效应无懈可击**：同一份权重、同一批帧，只有 prompt 不同。
- **① 权重效应带 seed 不确定性**：比较两次独立训练的 checkpoint，单 seed。
  三条线方向不一致（c2 反向）是有力旁证 —— 若只是"重训一次就变好"的假象，
  三条应同向 —— 但不等于替代。**建议标注"单 seed"。**

---

## 4. 远端 / 天空分层（`regions_5way_disp`，ssi_disparity，test 2,543 帧）

| 分层 | b | b+cap | d1 | c2 | **d2** |
|---|---|---|---|---|---|
| all | 0.0884 | 0.0882 | 0.0973 | 0.1013 | 0.0903 |
| **depth/far >30m** | 0.1680 | 0.1679 | 0.1726 | 0.1817 | **0.1524** |
| row/top | 0.1572 | 0.1598 | 0.1767 | 0.1858 | 0.1705 |
| structure/boundary | 0.1075 | 0.1089 | 0.1184 | 0.1242 | 0.1112 |
| structure/interior | 0.0862 | 0.0857 | 0.0946 | 0.0989 | 0.0878 |
| **锐利度 b/i** | 1.25 | 1.27 | 1.25 | 1.26 | 1.27 |
| **远端退化 far/all** | ×1.90 | ×1.90 | ×1.77 | ×1.79 | **×1.69** |

**读点**

- **d2 是唯一在远端改善的线**：0.1524 vs b 的 0.1680（**−9.3%**），退化倍数也最小。
  它用一点整体精度换来了远端的实质改善 —— 正是任务关心的区域。
- **受控归因**：c2 与 d2 结构完全相同，唯一差别是 condition 来源。
  `c2 far 0.1817 → d2 far 0.1524`，**−16%**。远端改善可归因到 AnyThermal 特征本身，
  不是 adapter 架构。
- **任务 1b 的答案：没修好。** c2（先训 adapter 再训 U-Net）远端 0.1817、row/top 0.1858，
  都是所有线里最差。
- **锐利度全线一致**（1.25–1.27）。caption 不改变锐利度，adapter 也不改变。

> ⚠️ **分层只统计有 GT 的像素。** MS2 每帧 `valid_pixels` 平均 43,424 / 163,840 = **26.5%**，
> 而天空恰恰没有 LiDAR 回波。所有 strata 都与 `valid` 求交，所以 `row/top` 统计的是
> 上三分之一里**有回波**的像素（楼顶、电线杆），**不是天空**。
> 现有工具无法测量真正的天空区域；若要测，需要不依赖 GT 的判据
> （如"无回波区域被预测为近处的比例"），该工具尚未实现。

---

## 5. 任务 4：原版 AnyThermal

| | all | far >30m | row/top | boundary | interior | b/i | **far/all** |
|---|---|---|---|---|---|---|---|
| b（我方最佳 baseline） | 0.0884 | 0.1680 | 0.1572 | 0.1075 | 0.0862 | 1.25 | **×1.90** |
| **原版 AnyThermal** | 0.0821 | 0.0930 | 0.1257 | 0.0973 | 0.0808 | 1.20 | **×1.13** |

（AnyThermal 用 `ssi` 对齐，独立作业 `regions_at_ssi`；两表分层定义共享，可并排读。）

**它没有天空/远端问题** —— 远端几乎不退化。所以这不是热像深度的通病，是我方方法特有的。

**复现验证**：论文 Table IV 报 AbsRel **0.0883** / RMSE 2.7432（day/night/rainy 平均），
我方实测 0.0821（仅白天 test），量级吻合。

### 它用了什么监督（论文 §III-B / §III-F）

| | 原版 AnyThermal | 我方六线 |
|---|---|---|
| 深度监督 | **MS2 自己的 train split + 稀疏 LiDAR GT** | 同左 |
| 骨干 | DINOv2 ViT-B/14，**CLS token 对比损失**蒸馏（自监督、无标注） | Stable Diffusion U-Net |
| 深度头 | MiDaS 架构，"其余部分不变" | 扩散单步 x0 预测 |
| **训练时骨干** | **冻结，只训头** | **整个 U-Net 解冻微调** |

> 论文 §VI-C 明确："segmentation and depth baselines are **trained on the evaluation
> datasets**"，并"evaluate on the MS² dataset using **sparse LiDAR ground truth**"。
> §III-B 那句 "MS² reserved for zero-shot evaluation" **只管 VPR**，不管深度。

**机制假设**：稀疏 LiDAR 在远端/天空无信号；全解冻的 U-Net 在无监督区自由漂移，
而冻结骨干保住了 DINOv2 的语义先验。d2 部分借到了这个好处（用 AnyThermal 特征），
但下游 U-Net 仍全解冻，所以只到 ×1.69。

**可借鉴的两条**：① 冻结更多；② 加 MiDaS 的多尺度梯度匹配损失（见 §7，已试，失败）。

---

## 6. 雨天：caption 注入效应的条件边界

**三个考卷，同一份 b+caption 权重，只有 prompt 内容不同**（②注入效应口径）。

| 考卷 | 帧数 | b 的 abs_rel | 注入效应 | 胜率 | caption 词表(等n) | 唯一率 |
|---|---|---|---|---|---|---|
| 晴 `16-08-46` | 2,543 | 0.0884 | **+0.00130** 显著（有害） | — | 810 | 92.7% |
| **轻雨 `16-19-00`** | 2,374 | 0.0930 | **−0.00169** 显著（**有帮助**） | 60.7% | **831** | 90.9% |
| 重雨 `16-59-13` | 2,161 | 0.1170 | **+0.00049** 显著（有害） | 45.0% | **709** | 87.2% |

**caption 质量验收**：三批都过 `validate_caption_tokens.py`（0 条超 75 token）；
token 中位 55/53、p95 74/73。词数中位 46/44/44，**长度持平但词表收缩**。
重雨样本最少却重复率最高（样本量偏差是反方向的），信号是真的且被低估。

**两因素解释**（三点全部吻合）

| 考卷 | 视觉通路 | caption 信息量 | 结果 |
|---|---|---|---|
| 晴 | 强 | 高（810） | 文本冗余 → 注入税 |
| **轻雨** | **被削弱** | **最高（831）** | **补位成功** |
| 重雨 | 严重削弱 | 最低（709） | 文本自己也失效 → 注入税 |

**两个条件必须同时满足**：视觉通路被削弱 **且** 文本仍携带有效信息。
这修正了原先的"文本价值 = 视觉通路有多弱"—— 那个说法在 caption 源与模型输入
同模态时看不出区别；本设置 caption 来自 RGB、模型吃热像，两个变量解耦后才露出真正的驱动因素。

**未复现的部分**：d2 上轻雨与重雨的注入效应几乎相同（−0.00059 / −0.00062），
**尽管词表差 15%**。所以"效应跟着 caption 信息量走"在 b 上成立、**在 d2 上不成立**。
可能是语义 condition 对文本质量不敏感，但这是事后解释，需单独检验。

**可证伪的下一步**：夜间 `21-58-13`（2,539 帧，RGB 近全黑，caption 全空待生成）。
预测：词表低于重雨、注入税大于 +0.0005。

---

## 7. 代码正确性验证（donor-swap 证伪对照）

担心："adapter 可能学会忽略 AnyThermal 特征，d2 退化成带额外参数的 b" —— 损失曲线看不出来。

**方法**：推理时把 condition 换成随机 donor 帧的（uniform，自配对已换走）。

| | abs_rel | rmse | a1 |
|---|---|---|---|
| d2 原始 | 0.0903 | 3.855 | 0.9086 |
| **d2 只打乱 AnyThermal 特征** | **0.2971** | 10.005 | **0.5186** |
| d2 打乱整个 condition | 0.2981 | 10.034 | 0.5171 |
| b 打乱整个 condition（标尺） | 0.2945 | 9.952 | 0.5206 |

**结论**：只换特征金字塔，误差翻 **3.3 倍**、a1 从 0.91 塌到 0.52。
且"只打乱特征"与"打乱全部"几乎相同（0.2971 vs 0.2981）——
**AnyThermal 特征承载了几乎全部 condition 信号**，adapter 的 `thermal` 张量输入不独立贡献。
崩塌幅度与 b 打乱 VAE latent 相当。**代码正确。**

`anythermal` 模式只换特征、保留正确的热像张量，所以零结果不能推给"adapter 丢了图像"——
这是该检验的设计关键。

---

## 8. 负面结果：MiDaS 多尺度梯度匹配损失

**动机**：逐点 L1 让"边界抹平"比"锐利但错位"便宜（合成阶跃上 0.0385 vs 0.0729），
而 AbsRel/RMSE 同为逐点平均，奖励同一种投机 —— 这解释了"数值好看但可视化模糊"。
MiDaS 的损失有第二项（梯度匹配），我方只有数据项。

**实现验证**：合成阶跃上，正确锐利预测两项均为 0；模糊预测梯度项 0.0262；
**26.5% 稀疏 mask 下为 0.026164 vs 稠密 0.026163** —— 多尺度掩码在 MS2 GT 密度下成立。

**权重标定**：合成数据比值 0.68，真实数据实测 `gmatch/gt_ssi_l1 = 0.087`（差 8 倍）。
按 `w ≈ 1.7 × (gt_ssi_l1/gmatch)` 得 20（2.5 会让梯度项只占 4.3%，等于没开）。

**结果：单调恶化，三条臂 3 epoch**

| 臂 | best epoch | val abs_rel | rmse | a1 |
|---|---|---|---|---|
| w=0 | 2 | **0.07974** | 3.464 | 0.9256 |
| w=5 | 3 | 0.08539 | 3.760 | 0.9136 |
| w=20 | 2 | 0.09284 | 4.277 | 0.9011 |

**失败机制**：`alignment_scale` 从 0.26 涨到 **3.3**（输出动态范围被压缩 12 倍），
`clamped_above` 从 5 涨到 **80**。scale 虽 detach 但出现在损失对预测的梯度里，
形成正反馈 —— `w=20` 的梯度范数常态触顶 `--max-grad-norm 1.0`。

**有剂量-反应关系、有机制、可写。** 按损失**值**之比定权重是错的方法，应按梯度量级。

---

## 9. 已知局限（汇报必须写）

1. **GT 覆盖率 26.5%**，天空无回波 → **AbsRel/RMSE/所有分层都测不到天空**。
   RMSE 逃不出这个问题（同一个 valid mask），它只是单位可读。
2. **权重效应（①）单 seed**，注入效应（②）无此问题。
3. **`a + caption` 未跑**（第一周任务 4 缺口，67 小时）。
4. **对比图未做**（第二周任务 3 的后半；前半「与 AnyThermal 对比」用的就是 §5 那份
   已有的预测，无需另跑 —— 见 §11 那条作废记录）。
5. 雨天两因素模型在 d2 上未复现（见 §6）。
6. 分层的 `row/top` 是"上三分之一有回波的像素"，**不是天空**，措辞不能混。

---

## 10. 集群上的产物（路径）

```
$SCRATCH/runs/route_suite/     a_rgb_unet, b_*, c1/c2_stage2, c2_stage2_caption,
                               d1_*, d2_*, d2_caption, grad_{off,mid,on}_e3
$SCRATCH/runs/eval/            每个评估一个目录，含 eval_eval.json +
                               eval_eval_per_sample.csv（+ raw_predictions/ 若 SAVE_RAW=1）
$SCRATCH/runs/regions/         regions_4way_test, regions_5way_disp, regions_at_ssi
$SCRATCH/runs/anythermal/      Midas_anythermal（原版预测 + raw_predictions）
$SCRATCH/checkpoints/          calibration, c1_stage1, b_caption_e03
$SCRATCH/data/ms2/             8 序列，57 GB / 306k inode
$SCRATCH/manifests/sequence_level_internvl3_8b/   27 个 manifest（含三批雨天 caption）
```

带 `raw_predictions/` 的评估（可用于可视化与分层）：
`a_raw, b_e05_raw, bcap_e03_raw, c2_raw, c2cap_raw, d1_raw, d2_raw, d2cap_raw,
grad_off_raw, grad_on_raw`，各 2,543 个 `.npy`。

**自查工具**：`bash ~/Iris/slurm/status.sh {overview|results|regions|runs|log|err|why|wait}`

---

## 11. 下一步（做 slides 用）

**必做 —— 工具已就绪，只差把作业投出去**

- ~~**零训练直推**~~ —— **这一项作废了，本节原来的写法把人带沟里。**

  「与 AnyThermal 的对比」和「零训练直推」在任务表里并列，读起来像两个东西，
  于是有人（2026-08-03）去跑了两条 Lotus 路线，都不是要的：

  | 试过的 | 条件通路 | 数 | 为什么不是 |
  |---|---|---|---|
  | `run_ms2_lotus_direct_official.py` | AnyThermal 特征 → 零参数 bridge → 冻结 Lotus-G | test **0.4697** | 冻结 U-Net 训练时见的 condition 是 VAE latent，特征金字塔对它是分布外输入 —— **比 §7 打乱 condition 的 0.2971 还差**，「乱但同分布」好过「不同分布」 |
  | `run_ms2_lotus_thermal_vae_official.py` | 热像 → VAE 编码器 → 冻结 Lotus-G | val 0.1291（历史） | 这是旧冻结文档里「零训练基线」的定义（`LOTUS_LINE_V2_..._FREEZE_20260705.md` §18/§40），但不是这次任务问的东西 |

  **要的其实是 §5 的原版 AnyThermal**：AnyThermal 自己的 MiDaS/DPT 深度网直接出深度，
  全程不碰 Lotus。它**早就跑完了** —— `$SCRATCH/runs/anythermal/Midas_anythermal/`，
  test 2,543 帧 **0.0821**（`ssi` 对齐）。所以这一项没有任何东西要跑，
  主表也不再单列「零训练直推」那一行（任务 4 那页已有完整分层表，重复无意义）。

  留下的东西：`slurm/direct.sbatch`（`KIND=thermal_vae|anythermal`）两条都还能跑，
  以后要拿零训练基线时不必重新考古。而 `anythermal` 那条的 0.4697 **有独立价值** ——
  它是 §7 donor-swap 的另一半：donor-swap 证明 adapter 在用 AnyThermal 特征
  （打乱就塌 3.3 倍），这条证明少了 adapter 这些特征根本用不了。两个方向合起来
  才说明 adapter 不是白挂的参数。已写进汇报。

- **对比可视化**：`tools/build_comparison_figure.py` + `slurm/vis.sbatch`（CPU）。
  沿用 `build_route_vis_figures.py` 的渲染（同一个 Spectral、同样两种上色、同样
  「按官方空间对齐 → 转深度 → clip」），替换掉它写死的三处：路线名、数据根目录、等距挑帧。

  两个关键点：
  - **每列可以有自己的对齐空间**（`目录:标签@ssi`）。`analyze.sbatch --align` 是一刀切的，
    所以原版 AnyThermal（`ssi`）和六线（`ssi_disparity`）进不了同一个作业；而图上必须并排。
    每列下方都印出它用的对齐。
  - **挑帧由分层扫描驱动**：`--pick gap --gap d2:b --stratum 'depth/far >30m'` 挑 d2 在远端
    领先 b 最多的帧；`--pick flip` 挑逐帧排序与全集排序相反的帧；`--pick far` 挑该分层
    误差最大的帧。`--min-separation` 防止排行榜前几名全是同一瞬间的连续帧。
    扫描复用 `ms2_eval.stratify`（本次从 `analyze_route_regions.py` 抽出来的共享模块），
    所以图上的「远端」与表里的「远端」是同一个定义，且每帧都与 `evaluate_sample` 对账。

      PREDS="$SCRATCH/runs/eval/b_e05_raw/raw_predictions:b,\
      $SCRATCH/runs/eval/c2_raw/raw_predictions:c2,\
      $SCRATCH/runs/eval/d2_raw/raw_predictions:d2" \
      MANIFEST=$SCRATCH/manifests/sequence_level_internvl3_8b/ms2_test_16-08-46_rgb_depth_v1_clip75_20260728.jsonl \
      TAG=far_gap PICK=gap GAP=d2:b \
        sbatch -J vis_far --export=ALL,PREDS,MANIFEST,TAG,PICK,GAP ~/Iris/slurm/vis.sbatch

  在集群上出图、只把 PNG 拉回来：一次对比要读几个 GB 的 npy，出来的图是几 MB。
  集群没有中文字体，工具会自动把图上文字降级成英文；要中文就把一个 CJK ttf 放到
  `$SCRATCH/fonts/msyh.ttc`。

**汇报 slides**：`tools/build_report8_slides.py`（14 页，本文档是它唯一的数字来源）。
主线 = 六线主表 → caption 双效应 → d2 远端优势（受控归因）→ 任务 4 外部参照 → 局限；
雨天与梯度匹配作补充。主表最后一行和可视化页留了标注出来的占位，等上面两个作业回来后填。

**可选**
- `a + caption`（67 小时，补齐第一周任务 4）
- `d2+caption` 第二 seed（27 小时，把 ① 从单次观测升级为可复现）
- 夜间 `21-58-13` 第四点（§6 的可证伪预测）
- 无 GT 的天空判据工具（§4 的窟窿，约 60 行）
