# DA2–Iris 融合 v2：匹配权重/文本主对照与内容消融

本版取代 v1 作为下一次投递方案，但不修改 v1 的代码、提交脚本、缓存或结果。
新目录：`outputs/lotus_line_v2/da2_iris_fusion_matched_pair_v2`。

## 为什么调整

v1 两臂共享 caption 训练过的冻结 Iris，但 empty 给空文本。这个差异混合了
本帧文本的作用与预训练条件错配，不能把 frame>empty 直接解释为内容贡献。
历史证据参见 `PAPER_HANDOFF_20260926.md` 的“分解表的实质修正”及
`RESULTS_20260813_IRIS_MS2.md` §4。历史 +0.002 或5%不是本融合模型的已知效应。

## 主对照（两条正式训练）

| 项目 | frame | empty |
|---|---|---|
| DA2 | 同一个 `da2_caption_v1/no-caption_s43/step018000.pt` | 相同 |
| Iris | `sd2_cap_pct_logtn_s43/converted/step18000_weights.pt` | `sd2_nocap_pct_logtn_s43/converted/step18000_weights.pt` |
| Iris 文本 | 本帧 caption | 空字符串 |
| 新融合参数 | seed43，同一初始参数 | 相同 |
| 架构、参数数量、特征层、优化器、数据顺序、噪声 | 相同 | 相同 |

两份 Iris 载入各自原训练条件；不是把无文本臂的 Iris 或融合模块关掉。
权重文件名、run 名、converted payload 的 route/global_step/caption_mode 严格校验。
两份权重文件与参数状态必须不同；两份完整来源路径/hash都写进每臂 contract。
配对检查只允许 arm 和初始 Iris 参数hash不同，其余配方、DA2与融合初始状态一致。

文本 padding 是有意保留的偏离：两份旧权重按 legacy 训练（训练补到 batch 内最长、
推理不补），这里两臂统一补到 77 且不传 mask。nocap 权重训练时恒为 2 个 token，
偏离大于 cap 权重。融合模块在各自的 77-token 特征上训练，能适应分布偏移，
但冻结 U-Net 内 pad token 造成的信息损失无法恢复。因此 frame−empty 只报告为
“匹配权重+文本”的整体差，不归因为 caption 内容；内容效应只看 frame 臂内
正确 vs 置换（同模型、同 padding）。

完整结构/训练设置沿用 `DA2_IRIS_FUSION_RUN_20261030.md`：冻结DA2和Iris，
只训练两个空间融合模块；真实激光masked log-L1；20k updates，batch4×3，
lr5e-5 polynomial衰减；77 token和原mask规则；固定按图像ID生成噪声；
不增加teacher、伪GT、latent/reconstruction loss或解冻阶段。

这个主对照测量的是“caption训练与推理路线”的总效果。由于继承的Iris权重不同，
不是相同权重下的纯内容干预，也不能用结果减法严格分解出可加的因果成分。

## 同一 Caption 融合模型的随机置换验证

- 训练前固定规则：全量14,249张验证集内随机双射，seed43；拒绝自身配对。
  使用随机排列拒绝采样，不使用循环平移；不从train/test借文本。
- 文本及其77-token序列/metadata mask整体置换；保留同一文本集合与长度分布。
  重复文本、截断后相同token可能存在，报告 `identical_text_count` 和
  `identical_token_count`，不通过重采样把它们偷偷移除。
- 每臂保存 `permutation.json`，包含完整原帧/文本来源帧映射与hash。规则/映射hash
  进入contract，恢复或正式训练不可更换。正式训练的empty臂不会做完整置换推理。
- 只在frame臂按本帧caption验证曲线选好checkpoint后执行。该模型保持所有权重、
  图像、GT、逐帧噪声、推理与评估规则不变，替换文本；复用已经完成的本帧全量验证。
- 冒烟另跑8张置换验证，检查代码路径，不作为选择条件；不训练第三条臂。
- 结果：`frame_s43/caption_content.json`、
  `frame_s43/permuted_selected/val_stepXXXXXX.json`和逐帧CSV。
  `summary.json`的 `caption_content` 包含本帧/置换、差值、选中步数与权重hash。
- `delta_permuted_minus_correct`：误差指标为正表示本帧更好；a1/a2/a3方向相反。
- 置换指标不参与选点、调预算或选择臂。它检验正确图文配对的贡献，不是模型是否
  “理解语言”的充分证据。主对照赢而置换不退化，只能报告路线总收益，不能声称
  逐帧语义内容驱动了收益。

## 检查与作业链

继续执行原融合模块7项检查（初始精确B0、冻结参数、梯度、空间对齐/像素、噪声、
恢复、依赖）；新增5项检查：匹配权重/metadata、随机置换与重复文本计数、配对
contract允许/禁止的差异、同一模型和选中步数复用、两权重预检及新依赖链。

GPU paired smoke实际载入两份不同Iris权重，各32步，检查两臂初始FP32/BF16输出
精确复现B0、真实输入/噪声一致、融合层更新及冻结权重不变。通过后afterok放行：

1. `da2_if2_frame`：20k训练、每2k stride4曲线、选中与固定20k全帧验证、选中模型置换验证。
2. `da2_if2_empty`：相同训练与原条件验证，无全量置换。

不自动提交测试集评估或通知作业。选择最低SSI AbsRel、并列选较早步数，保持不变。
比较主对照、B0和NeWCRF时使用一致评估口径，不能混用旧SSI-log表。

## 一次性提交

```bash
(
  set -euo pipefail
  cd ~/Iris
  git pull --ff-only
  unset IRIS_MS2_ROOT IRIS_MANIFEST_DIR IRIS_DATA
  IRIS_RUNS=/mnt/scratch/dwdh0363/runs bash slurm/submit_da2_iris_fusion_v2.sh
)
```

脚本先寻找当前runs目录和`/mnt/scratch/sc23sz/runs`里的两份指定权重，缺失时
在登录节点退出、不排队。其他位置可分别设置`IRIS_CAP_FEATURE_CKPT`和
`IRIS_EMPTY_FEATURE_CKPT`；v1的`IRIS_FEATURE_CKPT`不会被复用。
v1冒烟、checkpoint、token cache不能用于v2。若此前已投v1，它不会被新脚本取消。
