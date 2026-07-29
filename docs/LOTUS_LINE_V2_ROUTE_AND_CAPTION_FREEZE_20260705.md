# Lotus 线 V2：路线消融 + Caption 阶段冻结报告（2026-07-05，P1 补充于 2026-07-06）

本文档冻结路线选择阶段（Phase G）与 Caption 阶段（Phase H，路线选择级）的全部结论，
接续 `docs/ADAPTER_V2_HANDOFF_20260703.md`。下一阶段的方向决策见文末"待决事项"。

## 0. 口径声明

- 所有指标来自上游官方 Lotus evaluator（`lotus/evaluation/evaluation.py::evaluation_depth`），
  对齐方式 `least_square_disparity`，min/max depth 0.1/80 m。
- 数据：MS2 Val 全量 5810 张（manifest sha256 `c8b63d0a...`，同一文件贯穿所有运行）。
- **Test 集（9508 张）至今零次接触**，留待最终 `ms2_eval` 统一评估（Phase I）。
- 本表数字用于路线选择与阶段汇报；论文最终数字须过 `ms2_eval` 统一协议，两套口径不混用。

## 1. 四路训练消融（全量 Val）

| 路线 | AbsRel↓ | SqRel↓ | RMSE↓ | RMSElog↓ | δ1↑ | δ2↑ | δ3↑ | silog↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Thermal-VAE 直喂（零训练） | 0.1291 | 0.7636 | **4.9348** | 0.1883 | 0.8155 | 0.9544 | 0.9882 | 18.6350 |
| U-Net-only | 0.1698 | 1.2088 | 6.1197 | 0.2417 | 0.7298 | 0.9223 | 0.9757 | 23.9472 |
| Adapter-only | 0.1347 | 0.8147 | 5.1231 | 0.1956 | 0.8005 | 0.9503 | 0.9870 | 19.3574 |
| **Joint（胜出）** | **0.1246** | **0.7389** | 4.9554 | **0.1838** | **0.8258** | **0.9578** | **0.9893** | **18.2009** |

训练 checkpoint（均为 end-of-epoch，无挑点）：

- Adapter-only：`outputs/lotus_line_v2/full_train_epoch1_v2_4_response/adapter_end.pt`（sha256 `c1127277424f...`）
- U-Net-only：`outputs/lotus_line_v2/full_train_epoch1_unet_only_v2_2_energy/unet_end.pt`（`e9d02cbdcdff...`）
- Joint：`outputs/lotus_line_v2/full_train_epoch1_joint_adapter_unet_v2/joint_end.pt`（`9ea5d8cf3613...`）

统一训练协议：train manifest 10441 张、seed 20260703、同样本顺序、等效 batch 4、
1 epoch = 2611 步、checkpoint 节点 0/100/500/1000/2000/end、teacher =
冻结 Thermal-VAE mode 条件 + 冻结预训练 U-Net、无 GT/Val/Test/caption 参与训练。
配置细节见各输出目录 `frozen_config.json` / `summary.json`。

### 结论（冻结）

1. **条件对齐是主要矛盾**：Adapter-only（修输入，9.4M 参数）远好于
   U-Net-only（迁就输入，867M 参数）：AbsRel 0.135 vs 0.170。
2. **Adapter-only 按构造不超过其 teacher**（蒸馏目标即 Thermal-VAE 条件），
   0.135 vs 0.129 属预期；其价值在于提供可学习通路并支撑 Joint。
3. **Joint 是唯一超过零训练基线的路线**（0.1246 < 0.1291），逐样本对
   Adapter-only 胜率 70.4%（4092/5810），且可视化确认结构更锐、难样本兜底
  （见 `outputs/lotus_line_v2/route_comparison_montages/`，含 4×4 最差/最好样本对照）。
4. U-Net-only 输出系统性丢失高频结构（silog 23.9），其"模糊保守"仅在
   低纹理场景占优。

## 2. Caption 阶段（Phase H，路线选择级）

### 训练

`tools/train_ms2_joint_caption_v2.py`，Joint 协议不变，唯一新增：
manifest caption（InternVL3-8B，源自配对 RGB）经冻结 CLIP 文本编码器进
cross-attention，teacher 与 student 同文本；**冻结式 dropout 10%**
（规则 `random.Random(f"{seed}:caption_dropout:{manifest_index}").random() < 0.1`，
固定 1015/10441 张以空文本训练，训练计数与名单分毫不差）。
checkpoint：`outputs/lotus_line_v2/full_train_epoch1_joint_caption_v2/joint_caption_end.pt`
（sha256 `fa038b6f1bc4...`，最终 response cosine 0.9898）。

### 三方对照（同 checkpoint、同 seed、同样本，全量 Val）

| caption 模式 | AbsRel↓ | RMSE↓ | δ1↑ | silog↓ |
|---|---:|---:|---:|---:|
| Correct | 0.1267 | 4.9909 | 0.8217 | 18.4240 |
| Empty | 0.1264 | 4.9577 | 0.8210 | 18.3893 |
| Hard-wrong（seeded 错位映射，无固定点） | 0.1268 | 4.9835 | 0.8213 | 18.4376 |

逐样本配对：Correct vs Empty 胜率 44.9%；Correct vs Hard-wrong 胜率 50.7%
（= 抛硬币）；|Hard-wrong − Empty| 中位 0.009、最大 0.185（文本确实扰动预测）。
Caption 训练版整体亦不优于无 caption Joint（0.1267 vs 0.1246）。

### 机制诊断

零训练 Thermal-VAE 路线 × correct/empty（val32 配对，`thermal_vae_mode_official_val32[_correct]`）：
均值几乎不动（0.1352 vs 0.1358），但 32 对样本**无一对**预测相同
（|AbsRel 差| 中位 0.017、最大 0.091）。

### 结论（冻结）

5. **语言通路是活的**：预训练 teacher 与训练后模型都对文本有可测反应。
6. **caption 内容正确性对深度质量的贡献为零**：Correct ≈ Empty ≈ Hard-wrong，
   连对错都不区分；文本表现为与内容无关的扰动源（单样本可 0.285→0.119，
   也可 0.179→0.343，净效应归零；样例 `pred_002531` / `pred_001532`）。
7. 机制解释：蒸馏式目标从头到尾没有把"文本内容"与"贴近 GT 的深度"挂钩的
   信号——teacher 自身即"文本敏感但无向"，student 蒸馏后同样如此。
8. dropout 设计验证有效：Empty 为分布内输入（0.1264，未崩），对照合法。
9. 适用范围限定：CLIP 文本编码器 77-token 上限对训练与全部评估一致生效
   （约两三成 caption 被截尾，截掉的是句尾背景描述）。结论 6 的精确表述为
   "caption **前 77 token** 的内容正确性无增益"；截断行为训练/推理一致，
   不构成对照实验的伪影。若后续重生成短 caption（空间信息前置），此限定自动解除。

## 3. P1：masked GT 深度监督两臂（2026-07-06 补充）

方案 B（text-aware 底座）确认不可行：Iris 论文未发布模型权重。执行方案 A（P1）。

### 设计

`tools/train_ms2_joint_gt_v3.py`：在 Joint 协议上新增第三项损失——student
x0 经冻结 VAE 解码为视差图（逐行镜像官方评估器的解码路径），与 LiDAR GT
视差在有效像素上做闭式 scale-shift 对齐（detach）后取 masked L1。
gt_loss_weight=5.0（overfit-32 扫描 0.5/2.0/5.0 三档终点等效，GT 损失触及
数据地板 ~0.0096，取各项微优档）。蒸馏两项保留为正则。

- Arm 1（caption+dropout 10%）：`full_train_epoch1_gt_v3_caption/gt_v3_end.pt`
  （sha256 `98488ea4ddd1...`）
- Arm 2（全程空文本对照）：`full_train_epoch1_gt_v3_nocaption/gt_v3_end.pt`
  （sha256 `6142296689f4...`）

### 四轮判决（全量 Val）

| 评估 | AbsRel↓ | RMSE↓ | δ1↑ | silog↓ |
|---|---:|---:|---:|---:|
| Arm 1 × correct | 0.1240 | 4.9264 | 0.8286 | 18.0778 |
| Arm 1 × empty | 0.1227 | 4.8951 | 0.8310 | 17.9763 |
| Arm 1 × hard-wrong | 0.1241 | 4.9199 | 0.8284 | 18.0908 |
| **Arm 2 × empty** | **0.1226** | **4.8619** | **0.8311** | **17.8911** |

逐样本配对：correct vs hard-wrong 胜率 50.4%（内容仍不被区分）；
Arm 1 empty vs Arm 2 胜率 48.1%、中位差 0.0003（caption 训练零增量）；
correct vs Arm 2 胜率 43.3%（推理时加 caption 反而轻微有害）。

### 结论（冻结）

10. **GT 监督本身有效**：Arm 2 全面刷新项目纪录（AbsRel 0.1226 vs 无 GT
    Joint 0.1246），masked SSI 损失设计成立，且与蒸馏正则无冲突。
11. **GT 在环时语言依然无效**：三个判据全阴。语言无效的机制定性由
    "缺监督信号"修正为"**信息冗余**"——热成像输入 + LiDAR 监督已覆盖
    现有通用场景 caption 所能提供的信息。
12. **当前项目最佳模型 = Arm 2**（GT 监督、无 caption）。

## 3.5 Arm 6：纯 Iris 路线对照 + RGB teacher 排查（2026-07-09 补充）

导师两问的实验回应。

**Arm 6（正品 Thermal-VAE 条件 + U-Net 微调 + GT，`tools/train_ms2_thermal_vae_unet_gt.py`）**：
同 GT Arm 2 配方（masked SSI λ=5、response 正则、1 epoch、同 seed）。
全量 Val：AbsRel 0.1293 ≈ 零训练基线 0.1291（全指标千分位重合）。
overfit-32 诊断（rw 1.0 与 0.1 行为相同）：GT 损失开局即位于数据地板
（gt_ssi ~0.008，与 P1 λ 扫描地板一致），U-Net 侧无增量空间。
checkpoint `1fecad9ddee2...`；评估工具 `run_ms2_lotus_thermal_vae_official.py`
新增 `--unet-checkpoint`。

**结论 14（冻结）**：同配方下"纯 Iris 路线"训练后≈训练前（0.1293 vs 0.1291），
而 AnyThermal+Adapter 线 0.1226 全指标胜出。**深度改进空间在条件侧而非
U-Net 侧**；adapter 细节支路 + 联合适应构造出优于正品 VAE 编码的条件，
是 0.1226 的全部来源——与结论 1"条件对齐是主要矛盾"闭环。

**RGB teacher 排查**：当前冻结三序列全为 day_clear（"RGB 夜间退化"论点
对本数据不适用）；RGB(1224×384, AR 3.19) vs 热像(640×256, AR 2.50)
视场不同；边缘结构 NCC 仅 0.13–0.24，最优平移逐帧漂移 ±10–26 px
（视差签名，折合 latent 1–3 格）。**结论 15（冻结）**：跨视角逐像素
latent 蒸馏几何上非法；RGB 信息仅可走视角不敏感通道（caption 已证冗余）
或标定级重投影（工程重，未排期）。

**VAE 热像重建探针（2026-07-09，`tools/audit_vae_thermal_reconstruction.py`）**：
16 张抽样，热像重建 PSNR 39.9 dB / 高频保留 94.8%（对照 RGB：32.1 dB / 94.3%）。
**结论 16（冻结）**：RGB 预训练 VAE 对热像的编码近乎无损，"VAE 丢热像细节"
假设不成立，编码器微调不立项。条件侧优势的机制修正为：**可塑性而非信息量**
——adapter 条件可被联合训练与 GT 梯度持续重塑，冻结的 VAE 条件则已被
预训练 U-Net 榨干（与结论 14 的 Arm 6 饱和互证）。teacher 侧改造方向就此关闭。

**RGB 深度 teacher 线（2026-07-09，导师提议，四探针后关闭）**：
思路：RGB 走母语管线出稠密深度，经标定重投影补 LiDAR 盲区（70% 像素）监督。
探针①几何对账通过（双 GT 重投影中位差 3–4%，`calib.npy` 毫米制/NIR 中转
约定已核实）；探针②零样本成色 0.156（劣于学生 0.1226）；gate③ 视差 SSI
对远景失明（20% 误差在 40 m 处的损失贡献仅为 4 m 处的 10%，已实现
log-depth 损失修复，`--gt-loss-form log_depth`）；gate④ 修复后 lr 1e-6 横盘、
lr 1e-5 失稳（梯度尖峰→fp16 下溢归零）。
**修正（2026-07-09，分段/分辨率复审，`tools/probe_view_band_comparison.py`）**：
原探针②在原生 1216×384 运行低估了 teacher；且两视角考卷构成不同
（RGB 近景像素 23% vs 热像 33%，辛普森效应存在）。修正后零样本成色
0.1397（@768 长边）；分段：近 0.1322/中 0.1255/远 0.2466，中远景与热像
直喂（0.1064/0.1333/0.2553）相当或略优——目视"RGB 直出更好"与分段数字
一致。**结论 17（冻结，修正版）**：即便按分段公平计分，teacher 唯一的
正优势段（远景 +3%）恰被 3–4% 重投影损耗抵消，近景落后学生 24%，
无任何深度段存在扣除搬运费后的正利润；微调修复期望（本配方历史最大
增益 1.6%）与所需（15%+）差一个量级。RGB teacher 线关闭。几何重投影
工具链（探针①）保留供复用；原探针②总分 0.156 作废，以分段表为准。

**Arm 6-E：解冻 VAE 编码器的原始路线（2026-07-10，两 gate 后关闭）**：
针对"Arm 6 未给原始路线条件可塑性"的公平性质疑，
`train_ms2_thermal_vae_unet_gt.py --train-vae-encoder`（编码器+quant_conv
解冻转 fp32、独立 lr、解码器保持冻结、checkpoint 携带编码器权重、评估
工具同步支持）。overfit-32 gate：enc-lr 1e-5 下编码器梯度健康（1.2–3.4）
但 gt_absrel 横盘于地板（0.146→0.147）；enc-lr 1e-4 下前 200 步横盘、
后 100 步失稳（grad 29.7，absrel 爆炸）。依据冻结版 Arm 6 全量先例
（gate 地板横盘 ⇒ Val 不变），不跑全量。
**结论 18（冻结）**：原始路线在两种容量配置（仅 U-Net / 加编码器）下均
自起点即处于目标函数地板——**可塑性有用的前提是有坡可下**。冠军的优势
源于 adapter 从随机初始化到地板的真实学习行程（0.036→0.0096）及其泛化，
而非单纯"哪里有可训练参数"。消融棋盘至此全部格子已填或已判。

## 3.9 P3 空间显式 caption 终审 + 多 epoch 判决（2026-07-11）

**多 epoch**：2-epoch 全量训练（`full_train_2epoch_gt_v3_nocaption`）train loss
续降（0.0082→0.0063）但全量 Val 全面退化（0.1260 vs 0.1226）——教科书式
过拟合；val32 曾显示四项全升（0.1325 vs 0.1351），被全量推翻——小样本
第二次误导（第一次为 64 张远景假阳性）。**多 epoch 杠杆关闭；val32 仅可
做 sanity，换代决定只认全量。**

**P3**：新 caption（captioning 仓库 `rgb_depth_spatial_v2`：InternVL3-8B
纯看 RGB 输出 near/middle/far 分层短语，token 中位 28、零超限、零 GT 泄漏）
重训 caption 臂（协议=冠军，唯一变量为文本；ckpt `9727dacf...`）。
全量三连评：correct 0.1308 / empty 0.1238 / hard-wrong 0.1315。
- correct 胜 hard-wrong 52.7%（z≈4.2，**语言线首次检出内容信号**——模型
  确实在读文本）；
- correct 胜 empty 仅 33.1%，中位差 -0.0055——**caption 为净害**，且伤害
  大于旧 caption 时代（0.0070 vs 0.0013）。

**结论 19（冻结，语言线终审）**：caption 机制定性由"信息冗余"升级为
"**信息劣势**"——空间显式提示可被模型读取并区分对错，但 VLM 看图猜测的
粗深度先验劣于视觉通路自身的估计，注入即降质。自然散文无增量、显式提示
净有害，语言线在两端点全部闭环。冠军维持 Arm 2（无 caption，0.1226）。

## 3.10 语言线终审判词（结论 19 的机制细化，2026-07-11）

P3 证明模型对 caption 的掌握分三层，败在第三层，且该层在本数据上无解：

1. **会"听"**：correct 胜 hard-wrong 52.7%（5810 配对，z≈4.2）——六轮
   caption 实验中首次检出内容信号，通路与内容识别均无故障；
2. **会"照做"**：caption 为结构级指令——说有 signpost 即立杆、说远处是
   建筑即推深背景（样本级图证：`outputs/lotus_line_v2/p3_showcase/`
   `p3_best_help.png` 0.234→0.110 / `p3_worst_hurt.png` 0.170→0.396）；
3. **不会也无法"批判地听"**：模型照单全收 VLM 的布局指令（训练中"照做"
   确实降低训练损失——文本与图像在训练对上相关，构成一条过拟合通道），
   而 caption 源在本数据上信息严格劣于视觉通路——**"完美利用"的理论
   上限就是完全忽略（= empty 基线），正收益在白天 MS2 上不存在**。
   全 Val 总账：caption 画好 1926 张、画坏 3884 张（胜率 33.1%）。

语言产生正贡献的必要前提：文本携带视觉通路无法获取的信息。该前提在
day_clear 数据上不成立；唯一可能翻转前提的场景是夜间/恶劣天气序列
（视觉通路被削弱）——即"选项 C"的立论依据。

**区域级分析（结论 13）**、五把尺子、P3 三连评共同构成语言线的完整
证据闭环；本节为最终判词，语言线关闭。

## 3.11 P4：caption 成分分解与"注入税"（2026-07-11）

对结论 19"信息劣势"的机制解剖。工具：`run_ms2_lotus_trained_official.py`
新增 `--caption-transform`（`no-positions` 剥横向标签 (left/right/center)、
`objects-only` 连深度带 Near/Middle/Far 一起剥成纯物体清单）与
`--caption-mode fixed`（全样本同一句场景无关中性文本）；分解分析
`tools/analyze_p3_caption_decomposition.py`。P3 checkpoint（`9727dacf...`）
推理六连（同卷 5810、同 seed，empty=0.1238 基线）：

| 推理文本 | AbsRel | 对 empty 伤害 |
|---|---:|---:|
| empty | 0.1238 | — |
| correct（完整空间 caption） | 0.1308 | +0.0071 |
| no-positions（剥横向标签） | 0.1314 | +0.0076 |
| hard-wrong（他图完整 caption） | 0.1315 | +0.0077 |
| objects-only（纯物体清单） | 0.1317 | +0.0079 |
| fixed-neutral（"Music theory and abstract algebra..."） | **0.1455** | **+0.0217** |

关键配对（5810 对，符号检验）：correct 胜 no-positions 63.0%（z=+15.1）、
胜 objects-only 57.1%（z=+10.8）、胜 hard-wrong 52.7%（z=+4.2）；
objects-only 反输 hard-wrong（48.2%，z=−2.8）；neutral 输 empty 78.8%
（z=−43.9）。限定：剥离版 caption 为训练分布外格式（训练只见完整格式
或空文本），"伤害消失"为强证据、"伤害残留"为弱证据；同格式净证据以
correct 胜 hard-wrong 52.7% 为准。

**结论 20（冻结，取代结论 19 的机制表述）**：P3 伤害**不来自空间脚手架**
——假说 A 载体（深度带）边际 −0.0002、假说 B 载体（横向标签）边际
−0.0006，双双不显著且方向为负（正确的空间标注反而是唯一的正贡献成分，
也说明 cross-attention 接地至少部分可用）。伤害是**注入税**：文本通道
被任何非空文本激活即偏离 empty 工作点，税额随文本偏离训练分布单调放大
（贴分布场景文本 +0.007，场景无关文本 +0.022）；场景相关性可退回约 2/3
税额，内容正确性只再退 ~0.0008。层级：**沉默 < 说对的场景话 < 说错的
场景话 ≈ 只报物体 < 无关的话**。"完美利用"的上限仍是沉默（= empty），
语言线阴性结论维持，冠军无 caption 不变。

## 3.12 GT 梯度 fp16 下溢：发现、修复与新冠军（2026-07-11）

**发现链**：纯 GT 无蒸馏 overfit-32 gate 横盘（gt_ssi 0.024–0.029，地板
3 倍）→ 梯度审计发现 adapter 梯度 **284/300 步精确为零**（非零仅
5.8e-5–4.3e-3；同规格冠军配方 2.5–36.6；lr×10 复测 298/300 为零）→
定位：GT 损失是唯一反传穿过 VAE 解码器的项，而解码为 fp16 权重 +
fp16 autocast、全脚本无 GradScaler（`train_ms2_joint_gt_v3.py` 原 L237），
L1 摊到 ~1e5 像素的逐元素梯度 ~1e-5 量级在 fp16 链上下溢归零。

**追溯影响**：P1 λ 扫描"0.5/2/5 三档终点等效"的更简单解释是"λ 乘在
近零梯度项上"，"触及数据地板"旧解读作废；结论 10 的 GT 增益幅度
（0.0020）、结论 11、多 epoch 判决（3.9 节）、P3 caption 臂均在 GT 梯度
残废下得出，引用需携带此限定（方向性结论未被推翻，幅度类结论待复验）。
限定去除进度：语言线已由 P5 复验去除（3.14）；**Arm 6 已复验去除
（2026-07-12，`overfit_arm6_fp32dec`）**——fp32 解码、U-Net 梯度全程
健康（中位 7.8，零零值）条件下，gt_ssi 轨迹与旧 gate 重合（last-50
0.0083 vs 0.0088，step 10 即达 0.007–0.008 地板并横盘 290 步），
结论 14/18"原始路线无坡"在健康梯度下成立，无需重跑全量。
仍带限定的仅剩：多 epoch 判决（3.9，复验未排期）。

**修复**：`--gt-decode-fp32`（GT 路径独立 fp32 解码器副本，encoder 释放，
默认关闭保留旧行为）。三针验证：①smoke 梯度复活（0.62–2.04，零零值）；
②纯 GT overfit 破旧地板（last-50 SSI 均值 0.0067、最低 0.0046，但后段
absrel 尖峰 40/50——见 3.13）；③冠军配方+修复 overfit：last-50 SSI 均值
0.0096→**0.0079**、最低 0.0071→**0.0051**、absrel 中位 0.150→**0.125**，
尖峰未增（18 vs 22）。

**新冠军（全量重训，协议与 Arm 2 完全一致仅加修复）**：
`full_train_epoch1_gt_v3_nocaption_fp32dec/gt_v3_end.pt`（sha256
`be1d3adfd730...`，step 2611）。全量 Val（同卷 c8b63d0a）：

| | AbsRel↓ | SqRel↓ | RMSE↓ | RMSElog↓ | δ1↑ | δ2↑ | δ3↑ | silog↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧冠军 Arm 2 | 0.1226 | 0.7000 | 4.8619 | 0.1798 | 0.8311 | 0.9598 | 0.9899 | 17.8911 |
| **新冠军（修复版）** | **0.1172** | **0.6745** | **4.7447** | **0.1743** | **0.8427** | **0.9637** | **0.9911** | **17.2767** |

（旧冠军行摘自 3 节四轮判决表及其评估目录。）逐样本：新胜 **75.6%**
（z=39.0），中位 −0.0040；改善>0.02 者 391 张 vs 退步>0.02 者 33 张。

**结论 21（冻结）**：GT 监督的真实力量此前被 fp16 下溢扣压，修复后
带来项目最大单次提升（0.1226→0.1172，−0.0054）。**当前冠军 =
`full_train_epoch1_gt_v3_nocaption_fp32dec/gt_v3_end.pt`**，Phase I 应
以其为对象。台阶更新：零训练 0.1291 → Joint 蒸馏 0.1246 → +GT(残废)
0.1226 → +GT(真梯度) **0.1172**。

**结论 22（冻结，追溯限定条款）**：凡引用 gt_v3 时代结论（10/11、
3.9 多 epoch、P3 臂）须注明"GT 梯度残废条件下"；如需去除限定，须以
`--gt-decode-fp32` 重跑对应臂（caption 臂与 2-epoch 各约 2 h，未排期）。

## 3.13 Arm 7：纯 GT 无蒸馏全量（teacher 必要性终审，2026-07-11）

修复后纯 GT（`--condition-weight 0 --response-weight 0 --gt-decode-fp32`）
全量 1 epoch：训练流健康（新样本 gt_absrel 中期 0.074–0.10，absrel≥1
尖峰 486/2611 且集中在前期、末 300 步仅 1 次），但**全量 Val 崩溃：
AbsRel 0.4375**（中位 0.4355，p10–p90 0.358–0.523，>0.3 者 5697/5810
——全体均匀失效而非尾部爆炸），预测可视化呈无场景结构的"平均街道"
先验图（`gt_only_nodistill_fp32dec_official_val_full_empty/vis/`）。

机制（**主要嫌疑，待验**）：SSI 损失对 scale/shift 免疫 + 训练侧 GT 解码
不截断，而官方评估经 `image_processor.postprocess` 截断到 [0,1]——无
response 锚时 x0 值域自由漂出解码窗，训练指标看不见、评估全军覆没。
验证探针（未排期）：对该 checkpoint 做免截断解码的诊断评估，若 absrel
恢复正常则值域假说实锤。

**结论 23（冻结）**：teacher-student 结构**不可简单舍弃**（回应导师
第 1/4 条）：同架构同 GT 去蒸馏 → 0.4375 崩溃 vs 带蒸馏 0.1172。teacher
的角色三段修订定稿：非情报源（GT 可覆盖其信息，结论 11）、非开坡器
（fp16 假象，本节推翻 3.12 发现前的误判）、而是 **x0 值域/表征锚**——
把 student 输出钉在预训练 VAE 可解码的流形内。真正的"最小替身"试验
（纯 GT + 显式值域约束替代 teacher，可再验 teacher 是否还有残余贡献）
留待排期。

## 3.14 P5：GT 活体条件下的 caption 臂（语言线限定去除，2026-07-12）

背景：训练日志审计显示 P3 caption 臂的三项损失稳态占比为 condition 56% /
response 37% / GT 7%，且该臂训练于 fp16 修复前——GT 项梯度近零，文本
通路的唯一监督是"模仿 teacher（文本敏感但无向，结论 7）对同一文本的
响应"。即：**此前全部语言结论均来自 GT 从未对文本投票的模型**（teacher
虽非 text-blind，但垄断了文本监督）。P5 = P3 协议 + `--gt-decode-fp32`
重训（ckpt `fe92c05b63fa...`，caption 计数 9426/1015 与冻结规则一致），
首次让"文本是否帮助贴近 LiDAR"参与塑造 cross-attention。全量三连
（同卷 5810）：

| | 旧 P3（GT 残废） | P5（GT 活体） |
|---|---:|---:|
| empty | 0.1238 | 0.1197 |
| correct | 0.1308 | 0.1231 |
| hard-wrong | 0.1315 | 0.1235 |
| 注入税（correct−empty） | +0.0070 | **+0.0034** |
| correct 胜 empty | 33.1% | 40.1%（z=−15.0） |
| correct 胜 hard-wrong | 52.7%（z=+4.2） | 51.8%（z=+2.8） |
| 文本扰动幅度（hard-wrong−empty） | +0.0077 | +0.0038 |
| 本臂 empty vs 无 caption 冠军 | +0.0001（48.1%） | **+0.0025**（37.7%，z=−18.8） |

**结论 24（冻结，语言线在 GT 活体条件下的终审）**：
(a) 注入税**不是** teacher 垄断的伪影——GT 全程在环仍余 +0.0034（按
预登记阈值 ≥0.003 判阴性加固），但税额几乎减半：GT 梯度确实在把模型
推向"忽略文本"的理论最优方向，1 epoch 只走到半程；
(b) 内容信号未被 GT 放大（52.7%→51.8%）：给了 GT 完整一轮机会去教
"用文本贴近 LiDAR"，它没有教出来——day_clear + LiDAR 在环下文本
无可教的增量，与结论 20 的信息前提论互证；
(c) 新增账目：GT 活体时 caption 训练连本臂的 empty 模式也拖累 0.0025
（旧时代仅 0.0001）——文本通道占用的学习容量在 GT 有真梯度时有了
机会成本。语言全套方案对冠军的总账：0.1231 vs 0.1172（−0.0059）。
结论 22 对语言线的限定就此去除；冠军维持
`full_train_epoch1_gt_v3_nocaption_fp32dec`（0.1172）。

## 3.15 语言线区域级终审 + Iris 外部锚点（2026-07-12）

回答"caption 是否藏在细结构/远景等 LiDAR 稀疏区起作用"（"裁判失明"
假说）。两侧证据。

### 我方：P5 模型区域级复验（`region_effect_p5_val_full`）

同噪声同种子 correct vs empty 配对，五种掩码内分区记分（掩码仅由热像
输入与 GT 导出，不看预测/caption）。全量 5810 张（工具 global 区与官方
评估器逐位重合：correct 0.1230/empty 0.1197/胜率 40.3% ≈ 官方
0.1231/0.1197/40.1%，两条独立管线互证，排除掩码/对齐大 bug）：

| 区域 | correct 胜率 | 中位改善（empty−correct，正=帮忙） | 像素 |
|---|---:|---:|---:|
| depth_mid [10,30)m | 35.9% | −0.00423 | 25k |
| thermal_edge（细结构） | 37.2% | −0.00372 | 21k |
| global | 40.3% | −0.00279 | 43k |
| depth_near [0.1,10)m | 48.5% | −0.00041 | 14k |
| depth_far [30,80]m | 50.8% | +0.00034 | 4.4k |

预登记判据为"edge 显著优于 global"——实测 edge（37.2%）**劣于**
global，假说**证伪**：caption 伤害恰集中于细结构与中景，近景中性、
远景抛硬币。附：16 张冒烟曾显示 depth_far +0.0076/56.3%，全量塌为
+0.0003/50.8%——**本项目第三次"远景小样本假阳性"**（前两次 64 张、
val32），远景每张仅 ~4k 像素、方差大，小样本必现鬼影，判决只认全量。

### 他方：Iris 论文的 caption 增益量级与前提（arXiv:2411.16750v4）

Iris 在合成数据（HyperSim/vKITTI）训练、**零样本**评估真实数据。
Table 1，Lotus-G（同架构）AbsRel↓：

| 数据集 | 原版 Lotus-G | 重训 Lotus-G* | +Text（训练+推理） |
|---|---:|---:|---:|
| NYUv2 | 5.4 | 6.7 | 5.9 |
| KITTI | 8.5 | 8.9 | 8.6 |
| ETH3D | 5.9 | 9.2 | 9.0 |
| ScanNet | 5.9 | 7.6 | 6.4 |
| DIODE | 22.9 | 25.4 | 24.3 |

+Text 对**重训基线**赢 0.2–1.2 点，但对**原版 Lotus-G 五个数据集全部
更差**（重训基线本身较原版退化 6.7 vs 5.4；caption 只补回部分退化，
未及原版）。Table 2（仅 Marigold、仅 NYUv2）按物体面积细分 AbsRel：
整图 6.1→5.9（0.2）、小面积<5% 9.0→8.3（**0.7**）、<10% 8.4→7.9（0.5）、
<20% 7.3→6.8（0.5）——收益集中于小区域（0.5–0.7），是整图（0.2）的
2.5–3.5 倍。Iris 无近/中/远深度分段，细分轴是物体大小。

**结论 25（冻结，语言线的外部锚定）**：Iris 的 caption 收益天花板约
**0.8 AbsRel 点**（NYUv2，相对 ~12%），且集中于小物体、仅相对退化基线
成立。该收益赖以存在的前提是**稠密 GT**——训练用合成（100% 稠密）、
评估用 Kinect（NYUv2/ScanNet，可对小物体逐像素记分）。MS2 半稠密
LiDAR（热像视图 filtered GT 有效像素占比实测均值 **0.289**）恰在小物体/
细结构处失明，训练监督与评估记分同时缺席——正是那 0.8 点赖以存在的
前提。我方注入税（+0.0034）与他方 +0.8 点**符号相反但同为零点几的小钱，
方向由"稠密 GT 能否给小物体记分"这一前提决定**。语言线在 MS2 的阴性
由此从"我们未做出"抬升为"**Iris 的成立前提在车载热像上不满足**"。
补：VTD 等车载数据集 GT 同为 Velodyne-64 投影、半稠密，不补此前提；
稠密热像 GT 于真实驾驶场景基本不存在（LiDAR 物理决定），仅合成热像
或室内可提供。

## 4. 待决事项（2026-07-11 状态）

25 条冻结结论。当前状态：

1. **导师定稿方向**（三选一，素材已更厚）：A 阴性成果路线（语言失效
   机制链，现含注入税分解 + 中性句层级 + 区域级终审 + Iris 前提锚定，
   结论 20/24/25）/ B 架构成果路线（adapter 进化 + GT + fp16 修复，
   新冠军 0.1172，teacher=值域锚的三段论）/ C 换考卷或换考场——
   **修正（结论 25）**：让 caption 起作用需二选一前提，(i) 稠密 GT
   （给小物体记分，真实车载热像无解，仅合成/室内）或 (ii) 削弱热像
   视觉通路（**恶劣天气/雨，非夜间**——热像夜间不退化；且雨天须保持
   RGB-VLM captioner 不变以控变量，见会话推理）。数据集现状：VTD/
   KAIST/CATS 的 GT 均半稠密或样本过少，不补前提 (i)；前提 (ii) 待查
   MS2 本地是否存在白天雨序列（带热像 filtered GT + 可用配对 RGB）；
2. **Phase I 收官**：改用**新冠军**
   `full_train_epoch1_gt_v3_nocaption_fp32dec/gt_v3_end.pt`（0.1172）过
   `ms2_eval` 统一协议 → **Test（9508 张）唯一一次评估** → 报告定稿。
   Test 至今零接触（fp16 修复恰好赶在唯一一次评估之前）；
3. 可选复验（按导师优先级）：修复版 caption 臂重训（去除结论 11/20 的
   "GT 残废"限定）、修复版 2-epoch（复验多 epoch 判决）、Arm 7 值域
   假说诊断探针、"纯 GT + 值域约束"最小替身臂。

## 5. 补全 3×2 矩阵的说明

Adapter-only + caption / U-Net-only + caption 已论证为低信息量（前者
cross-attention 冻结、机制上不可能起效；后者被条件更优的 Joint+caption
覆盖），暂不执行；若需要，每臂约 1.5 h 可补。

## 6. 本阶段产物索引

- 训练脚本：`tools/train_ms2_unet_only_v2_2_energy.py`、
  `tools/train_ms2_joint_adapter_unet_v2.py`、`tools/train_ms2_joint_caption_v2.py`、
  `tools/train_ms2_joint_gt_v3.py`（P1，SSI 损失单测 `tests/test_gt_v3_ssi_loss.py`）
- 评估：`tools/run_ms2_lotus_trained_official.py`（本阶段新增 hard-wrong 模式）、
  `tools/run_ms2_lotus_thermal_vae_official.py`
- 可视化：`tools/build_four_route_extreme_montages.py` →
  `outputs/lotus_line_v2/route_comparison_montages/{worst,best}_of_each_route_4x4.png`
- 全部评估输出：`outputs/lotus_line_v2/full_epoch1_*_official_val_full/`、
  `outputs/lotus_line_v2/joint_caption_official_val_full_{correct,empty,hardwrong}/`、
  `outputs/lotus_line_v2/thermal_vae_mode_official_val_full/`
- 3.11–3.13 新增：`tools/analyze_p3_caption_decomposition.py`；
  `run_ms2_lotus_trained_official.py` 的 `--caption-transform`/`--caption-mode fixed`；
  `train_ms2_joint_gt_v3.py` 的 `--gt-decode-fp32`；评估输出
  `p3_official_val_full_{objectsonly,nopositions,fixedneutral}/`、
  `gt_v3_nocaption_fp32dec_official_val_full_empty/`（新冠军）、
  `gt_only_nodistill_fp32dec_official_val_full_empty/`（Arm 7 崩溃样本）；
  gate 记录 `overfit_gt_only_nodistill{,_lr10x,_fp32dec}/`、
  `overfit_gt_v3_fp32dec_lam5/`
