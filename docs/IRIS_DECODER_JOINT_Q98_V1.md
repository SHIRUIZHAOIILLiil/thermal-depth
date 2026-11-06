# q98 decoder1000 后的配对续训

## 固定问题与起点

比较“只延长 decoder”与“原 decoder + U-Net 末端卷积一起适配”各新增 1000 步。
两臂都用本帧 caption，不是有无 caption 消融，也不能据此证明 caption 收益不变。
不改旧实验，不用 q100，不新增 teacher、预测头或损失，不提交 test。

固定起点是 q98 caption4000 U-Net 与上一轮 `decoder_1000.pt`，后者由验证选择、
作业 9213840 经真实部署路径复现。脚本自动定位其 `jobs.txt`，检查父链、代码与权重哈希、
清单以及验证复现记录；不使用 test 场景结果选点。
起点完整验证约为 AbsRel 0.07752990、RMSE 2.675439 m、δ1 0.94692823。

## 两臂

| 设置 | decoder | joint |
| --- | --- | --- |
| 深度 VAE | 从已微调 1000 步开始，只更新 `decoder.*` | 完全相同 |
| decoder 学习率 | 1e-5 | 1e-5 |
| U-Net | 全部冻结 | 仅最后两个 up blocks 中 resnets/upsamplers 的 Conv2d 权重与偏置 |
| U-Net 学习率 | 无 | 1e-6 |
| 其他参数 | 冻结 | 冻结 |

明确冻结 U-Net 的其他层、注意力、归一化、输出卷积，热像 VAE，深度 VAE encoder /
post_quant_conv，文本编码器。U-Net 保持 eval，避免开启 dropout 等额外变量。
训练的原 decoder 开启梯度检查点，结构不变。两臂都使用新 AdamW、weight_decay=0.01，
不恢复上一阶段优化器状态；本轮的初始权重相同。

## 监督、数据与数值

沿用上一轮三个损失：`0.05 × 米制激光 L1 + 0.01 × 局部有符号点对差 +
0.0005 × 每帧 60–80 m 平均偏差平方`。没有 latent/reconstruction loss。
真实有效激光不做百分位截断；原校准伪 GT 与激光合成目标仍用于计算 q02–q98 的 log
归一化边界，只参与监督的米制转换，不作为预测输入。解码输出不限制在 [0,1]。

预测沿用实际部署 forward：热像、caption、原始 manifest index 决定的噪声，
`target_latent=None`。热像编码与文本编码在 no_grad 内；decoder 臂的 U-Net forward
也在 no_grad 内；joint 臂允许该 forward 建图，并去掉损失入口对 latent 的 detach。
GT、归一化边界与伪 GT 都在预测结束后才进入损失。

有效 batch 12（4 × 累积3），seed43，无翻转。数据游标从 60000 开始：上一轮 decoder
用过 [48000,60000)，本轮两臂用相同的后续 12000 个训练样本访问位置。
输入 trace 记录图像、caption、样本顺序及推理噪声种子；两臂必须一致。

U-Net 前向保持原生 FP16 精度，优化器维护选定卷积的 FP32 master weights，避免 1e-6
更新逐步被半精度舍入丢失。每次更新后转换回前向权重。两臂损失都固定乘 128 后反传，
更新前除回 128；非有限梯度直接失败，不静默跳步。decoder 与 U-Net 分别裁剪范数至 1，
使联合臂不会仅因增加另一组梯度而改变 decoder 的裁剪阈值。
checkpoint 保存实际前向权重、FP32 masters 与优化器状态。

## 验证与选择

完整同批 14,249 验证帧；Iris 仍用 ssi_log，NeWCRF 仍用已核验 ssi 缓存。
以本轮新增 0 / 500 / 1000 步记录曲线，0 步是旧 decoder1000，并非原 VAE。
冒烟完整复现一次公共 0 步，正式训练两臂从这个未经冒烟更新的起点重新加载。

按已有规则各自选点：相对公共起点 AbsRel 增加 ≤0.002、δ1 下降 ≤0.003、RMSE 至少
改善 0.02 m，对 NeWCRF 的三项正相对差距分数改善或三项已达到参照。通过者按差距分数、
RMSE、AbsRel、步数排序；否则保留 0。报告七项指标及各深度段 AbsRel。

依赖两条训练完成的 CPU 汇总分别比较 fixed1000 与验证选中点，报告 `joint - decoder`
及相对 NeWCRF 的差距。若 joint 不优于 decoder，不能把联合臂相比旧起点的改善归因于
解冻 U-Net。这是单种子路线实验，不是显著性检验；验证三项达到参照也不是 test 七项胜出。
选择回 0 步不会产生新权重，不启动测试。

## 必要检查与投递

CPU 测试覆盖实际 RouteModel forward 的初始数值复现、真实损失原语数值一致、
仅 joint latent 有梯度、精确可训练范围、冻结权重未变、FP32 master 累积小更新、
GT 隔离边界、数据游标、短程配对训练以及模拟 Slurm 依赖。
实际 checkpoint / CUDA 仍必须通过一次 paired smoke：先完整复现旧起点，再各更新4步，
检查初始输出相同、输入 trace 相同、指定参数确实改变、冻结参数哈希不变。
只有两臂冒烟都通过，两个正式训练作业才能运行。

独立目录：`outputs/lotus_line_v2/iris_decoder_joint_q98_v1/run_*`。
一次提交 1 个 GPU 冒烟、2 个依赖冒烟的 GPU 训练验证、1 个依赖两条训练的 CPU 汇总。
没有 test 作业。共享冒烟含完整验证；联合臂增加 backward 成本，耗时不能按旧 decoder-only 原样估计。

```bash
(
  set -euo pipefail
  cd ~/Iris
  git pull --ff-only origin aire-hpc-setup
  unset IRIS_MS2_ROOT IRIS_MANIFEST_DIR IRIS_DATA
  export IRIS_RUNS=/mnt/scratch/dwdh0363/runs
  bash slurm/submit_iris_decoder_joint.sh
)
```
