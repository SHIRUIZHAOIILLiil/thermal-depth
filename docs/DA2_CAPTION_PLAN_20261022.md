# DA V2 + caption：受控可行性实验（2026-10-22）

状态：已实现新训练器、CPU测试和冒烟脚本，尚未下载权重或投递。SD2.1 分段平衡臂不投；
保留 l100 和全部诊断。不是认定 SD2.1 达到能力上限，而是停止缺少明确依据的调参。

## 目标与边界

第一版按用户修正：相同稀疏真实激光监督、底座和预算下，本帧 caption 方案比纯热像
模型是否更好。两臂参数量不同，因此首轮不能把收益全部归因于文本语义。
能否超过 NeWCRF 是另一个问题，不拿更强底座的收益冒充文本收益。
完全不训练只能检查接口/初始预测；新建 cross-attention 的零样本输出无法证明文本有用。

本仓库已有 tools/finetune_depth_anything_thermal.py 和 slurm/da2_train.sbatch：
这是原相对深度比较基线，涉及伪 GT、裁剪、梯度匹配且缺少完整 resume。
新方法另建训练入口，不覆盖旧脚本、checkpoint 或表格，不能直接拿旧基线充当新消融。

## 1. 模型：一个尺寸、一种接法

- DA V2 metric outdoor **ViT-B**，官方 `depth_anything_v2_metric_vkitti_vitb.pth`。
  使用完整预训练编码器和 DPT head；不是只载入 encoder 后随机重建 head。
  原输出 sigmoid ×80 m 保留。没有逐帧 percentile 深度目标，没有深度 VAE。
  初始选择 Base 是预算决定，不宣称其优于 Large；本轮不扫模型尺寸。
- 四层特征中只改后两层（ViT-B 的 block 8、11，官方零基索引）：每层一个
  residual cross-attention。视觉 tokens 作 Q，文本 tokens 作 K/V。
  内部维度256、4 heads，独立投影/LayerNorm，attention dropout=0；输出投影到768维。
  每层标量门控 alpha 初始化0，投影正常随机初始化，不同时把所有分支参数置零。
  保留 class token，patch token 形状不变，原 DPT 接口不变。
- 复用当前已验证 SD2.1 文本编码器，冻结且 eval 模式；逐 token 输出投影到256维。
  固定长度77、显式 padding mask，缓存 train/val caption tokens/embeddings（记录权重hash、
  tokenizer、文本hash、截断率）。文本mask必须进入 cross-attention。
- 零门控时必须复现原 DA V2 输出；训练后可检查门控及梯度，但不把 attention 图当作深度证据。

## 2. 数据和监督

- Train：ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl，预期75688帧。
- Val：ms2_val_official3_thermalcap_20260821.jsonl，预期14249帧。
- 仅原始 thermal-view filtered LiDAR，valid=finite & 0.001<d<80 m。
  不读伪 GT 文件，不补零监督，不做稠密 gradient loss，不丢弃最大10%真实激光残差。
- caption 复用现有清单，不重生成。实现门禁必须核实生成输入来源；文件名 thermalcap
  不能替代证据。如果使用 RGB，则标明额外模态，不能称为纯热像推理。
- 热像保留现有 percentile 高位深转换，复制3通道后使用 DA 的 ImageNet normalize。
  保留整帧，不随机裁剪；本轮不开水平翻转，避免 caption 左右关系未同步变换。
- 为满足 patch14，输入仅向右/下 reflect pad 到14倍数，输出裁掉padding；GT不插值。
  若 native=256×640，则输入266×644。输入尺寸需在 manifest 数据审计中验证。
  这是我们的方法配置，不宣称逐字复现官方518像素训练配方。
- 唯一训练损失：每帧有效点的 |log(max(pred,1e-3))-log(gt)| 平均，再对非空帧平均。
  不做训练时GT对齐，不加米制L1、分段权重或其他辅助loss。无有效GT帧明确计数，
  全空batch报错。用真实GT的log误差是本方法设计，不声称是官方SiLog的同义实现。
  此loss未保证远处RMSE改善，必须由验证结果决定。

## 3. 最小训练对照

第一阶段只训两臂，初始参数逐张量一致：

| 臂 | 图像/模块 | 文本 |
|---|---|---|
| A no-caption | DA V2，无文本编码器和adapter | 无文本输入 |
| B caption | DA V2 + 深层adapter | 本帧caption |

A/B具有相同主干初始权重、训练图像顺序和有效监督；B新增adapter参数组。
A不是给文本分支喂空字符串，而是不建立该分支。固定文本容量控制推迟到有前景后，
本轮不训练它；推理时关闭adapter也不能替代从头训练的A。

固定预算：seed43；每卡microbatch4、accumulation3，有效batch12；不丢数据尾部的
处理在实现时固定并记录。AdamW wd=.01，encoder LR5e-6，DPT/adapter LR5e-5；
poly=(1-step/20000)^0.9；所有视觉参数可训练，文本冻结。BF16，必要时梯度检查点。
共同主干初始化/数据RNG与adapter构建RNG分离，避免固定seed却不同数据顺序。

## 4. 有限预算与停止条件

1. 工程冒烟：合成mask/形状/梯度/零门控测试，然后32步真实训练检查，不算科学结果。
   给出实测秒/步和显存后才提交双臂pilot，不先猜GPU耗时。
2. A/B先各到8000步。按有效batch12约96000图像呈现，即约1.27遍训练集，
   不是8000张。学习率从一开始使用20000步总计划，停止时可无损续训。
3. 固定step6000和8000在完整val评估；不按两条曲线分别挑最优点来判断caption。
4. 继续预算的预设条件：B在两个点的AbsRel都低于A；8000步至少降低0.0005，
   RMSE不恶化超过0.5%，a1不下降超过0.001，且至少2/3序列AbsRel方向改善。
   这是资源决策门槛，不是统计显著性标准，也不是原SD2 loss-sweep门槛。
5. 若满足，A/B从完整checkpoint继续到20000；否则停止本配置，保留结果。
   不把早停解释为caption永远无效，不默认增加层数/损失/模型尺寸。
6. 后续补固定文本容量控制，并在声称稳定贡献前用另一预设seed44复现A/B；这部分是条件预算，
   不在pilot前全部投递。预算不支持复现时只能称单种子探索结果。

必须保存模型、门控、optimizer、scheduler、AMP状态、RNG、sampler/数据游标；
resume必须通过同数据序列的连续训练对照测试。老DA2脚本的仅存权重方式不能复用。

## 5. 评估与caption证据

- 主比较沿用当前MS2官方mask、clamp和逐帧聚合；A/B均冻结为深度空间ssi。
  原l100仍按ssi_log、NeWCRF按ssi，不改变旧成绩；披露不同模型的表示和对齐。
  同时保存DA2的none无对齐指标，单独报告，不能把ssi成绩称为原生米制准确度。
- 报告7指标、3序列和已有深度分段；不新开大规模诊断。
- Pilot只用固定点；完整阶段按step2000..20000、stride4的val AbsRel选点，
  并列取更早，再对选中点跑完整val。A/B固定同step结果另保留用于对照。
- 后续B用同checkpoint比较正确/固定/空/同序列打乱caption。打乱映射固定、无自身配对，
  记录重复文本比例；这些推理扰动是辅助证据，不能替代训练对照。本轮先交付pilot。
- 不使用test做预算决策；新路线完成验证与方案冻结后，另行确认最终test计划。
  只能逐项报告相对NeWCRF的胜负，不用单指标赢宣称全面超过。

## 6. 实施前固定审计项（不额外研究实验）

精确源代码commit、预训练权重hash/许可、文本编码器hash、train/val清单hash与分割
无重叠、caption来源、热像min/max/std、实际尺寸、有效GT点数、输入/输出单位。
核对预训练公开数据来源；不能仅因官方checkpoint就声称绝无测试污染。
未满足这些工程门禁不启动大训练。输出用独立新目录，不覆盖旧DA2或Lotus结果。

## 官方参考与本方案的区别

- https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth
  官方提供metric encoder+DPT及室外VKITTI权重。
- https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/train.py
  官方使用SiLog、encoder/head差分LR、AdamW/poly；本方案借鉴优化器配置，但使用
  明确的masked log-L1、整帧热像和caption adapter，不冒称官方训练配方。
- https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/depth_anything_v2/dpt.py
  四层特征接口、patch14、metric sigmoid×max_depth是集成依据。
