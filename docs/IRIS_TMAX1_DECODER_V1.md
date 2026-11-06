# q02–q100 + 原 VAE 深度解码器微调

## 目的与固定起点

复用已经完成的 `sd2_cap_pct_logtn_img_l0100_tmax1_s43/converted/step16000_weights.pt`。
这是本帧 caption、log 深度、q02–q100、原图像 L1 权重 0.1 的已有 U-Net。
它的验证集起点约为 AbsRel 0.08137、RMSE 2.7571 m、δ1 0.93937；冒烟必须复现。
不再训练一遍 tmax1，不自动改用其他续训点，也不加载 q98 已微调的 decoder。

本方案分阶段：已有 q100 U-Net 提供实际推理 latent，新增阶段仅训练原 VAE 的深度
`decoder.*`，U-Net、热像 VAE、深度 VAE encoder / post_quant_conv、文本编码器全部冻结。
初始 decoder 从原始 fp32 深度 VAE 复制，必须逐帧复现该 tmax1 起点。
联合更新 U-Net 暂不采用，因为那会同时改变 latent 和解码映射，无法清楚判断新增阶段的作用。

q02–q100 取消的是有效目标范围内的远端百分位截断；近端 q02 和物理有效范围沿用原设置。
它不是无界深度，也不等于保证恢复全部远端细节。q100 原训练已改变远端信息的表示，
本次检验的是适配原解码器能否把这类 latent 更好地转换成深度，而非单独证明截断是唯一根因。

## 训练协议

直接复用 `tools/train_iris_decoder_only.py` 已运行的训练和验证函数，不修改旧文件。

| 项目 | 固定设置 |
| --- | --- |
| 输入 | 同一官方 8 段训练清单，75,688 帧，热像 percentile 输入与本帧 caption |
| 预测 latent | `target_latent=None` 的实际部署推理路径；GT 不进入此路径 |
| 深度目标 | 真实激光；原校准伪 GT 仅参与每帧 log 归一化边界的计算，不新增伪 GT loss |
| 归一化边界 | q02–q100，与该 U-Net 训练约定一致；不把激光远端截回 q98 |
| 损失 | `0.05 × 激光米制 L1 + 0.01 × 局部有符号点对差 + 0.0005 × 每帧远端平均偏差平方` |
| 损失定义 | 与上一轮 decoder-only 完全相同；点对与远端偏差沿用训练时的 ssi_log 对齐 |
| 不再计算的项 | 冻结 U-Net 下的 latent/reconstruction loss；不把伪 GT 编码成预测输入 |
| 优化 | 新 AdamW，lr=1e-5，weight_decay=0.01；梯度裁剪 1 |
| 预算 | 1,000 更新，batch=4，累积=3，有效 batch=12，seed=43 |
| 图像顺序 | 与上一轮 decoder-only 相同，image_cursor=48,000；这是配平样本顺序，不是恢复 tmax 优化器 |
| 数据增强 | 无翻转，与部署输入一致 |
| 输出 | 原 decode/denormalise 算术顺序，无额外 [0,1] 截断；官方评估范围保持原设置 |

原 tmax1 的图像 L1 权重是 0.1；**本次 decoder 阶段用 0.05**，明确与上一轮
decoder-only 的优化条件配平，不能把它写成整个训练过程中权重完全未变。
归一化边界和激光只进入监督/评估，不进入部署预测。数据集会生成兼容旧接口的目标字段，
但本训练不使用这些目标的 VAE 编码，也不使用 latent loss。

## 对照与选择

1. **主对照：同一个 q100 起点的 0 / 500 / 1000 步。**相同 U-Net、原 decoder 起点、
   同一 14,249 帧完整验证集和激光掩码；Iris 使用原 `ssi_log`。这回答 decoder 微调是否有效。
2. **历史性能参照：q98 + decoder。**固定读取由作业 `9213840` 复现过的历史 run，
   核对清单、推理配方、伪 GT 配置、预算、NeWCRF 参照、代码与权重哈希。
   只读验证 `reproduced.json`，不读取 day/night/rain 的 test 指标。
   分别报告 fixed1000 对 fixed1000、验证选中点对验证选中点，不能混用。
3. 历史 q98 来自 λ=0.05 + 点对/偏差续训的 caption4000，而 q100 来自 λ=0.1 原16k。
   **两者只比较最终性能，不是仅改变上界的控制变量消融。**本轮没有无 caption 臂，
   不能据此重新证明或保证保住 caption 增益。

选择沿用 decoder-only 的规则，参照已核验的 NeWCRF 验证缓存，NeWCRF 保持 `ssi`。
相对本轮自己的 0 步：AbsRel 增加不得超过 0.002，δ1 下降不得超过 0.003，
RMSE 至少降低 0.02 m，且对 NeWCRF 的三项正相对差距均值改善或三项都达到参照。
通过者按该均值、RMSE、AbsRel、步数排序；未通过则保留 0 步。
这是内部选点分数，不是论文新指标；三项验证达到参照不等于七项测试全面胜出。

报告全部七项指标和 0–20 / 20–40 / 40–60 / 60–80 m 分段 AbsRel。
历史 q98 decoder1000 的已复现验证为 AbsRel 0.07752990、RMSE 2.675439 m、δ1 0.94692823。
对它是全面改善、全面退步还是取舍，会单独列出，不会仅凭 RMSE 更低宣布整体超过。

## 检查与投递

CPU 必要测试复用上一轮的初始输出、梯度、冻结参数、GT 隔离、选择门槛测试；
新增 q100 远端保留/q02 不变、固定起点、历史结果来源、比较规则和模拟提交依赖检查。
GPU 冒烟先完整复现原 tmax1 验证，并验证改 GT 路径不影响预测，再做 4 次更新，
确认 decoder 有梯度且更新、所有冻结参数未变、实际远端监督非空。正式训练重新加载原 decoder，
不沿用冒烟更新。冒烟失败则正式训练不执行。

输出独立存放于 `outputs/lotus_line_v2/iris_tmax1_decoder_v1/run_*`。
只提交冒烟与依赖它的 decoder 训练/验证两个 GPU 作业，**不提交 U-Net 训练或 test**。
GPU 冒烟需全验证，不能按几秒单元测试估时；实际耗时以排队和节点为准。

```bash
(
  set -euo pipefail
  cd ~/Iris
  git pull --ff-only origin aire-hpc-setup
  unset IRIS_MS2_ROOT IRIS_MANIFEST_DIR IRIS_DATA
  export IRIS_RUNS=/mnt/scratch/dwdh0363/runs
  bash slurm/submit_iris_tmax1_decoder.sh
)
```

脚本自动定位原 tmax1 16k 权重与 `val reproduction -> 9213840` 对应的历史 run。
来源缺失或不唯一直接停止，不猜测“最新”权重。日志重点读 `[reproduced]`、`[validation]`、
`[selected]`、`[vs-q98-decoder]`。本地 CPU 检查不能代替 Aire 上的真实 checkpoint 冒烟。
