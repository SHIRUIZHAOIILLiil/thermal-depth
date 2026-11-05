# Iris 原 VAE 深度解码器短训 v1

本轮只检验：固定 caption 4000 模型实际推理出的 latent，原 VAE 解码器经过真实激光监督的短训，能否进一步缩小 NeWCRF 的 RMSE 差距。不是更换 VAE、增加深度头，也不重新训练 U-Net。不运行 test。

## 固定起点与参数

- 起点：`iris_far_bias_square_l050_frame_v1` 的作业 **9171487**，已选中 `conv_4000.pt`。读取并核对原 16k 权重、局部点对、偏差平方续训的完整来源链。
- 复制该模型现用的 fp32 深度 VAE，只训练 **`decoder.*`**。`post_quant_conv`、U-Net 全部参数、热像 VAE 编码器和 CLIP 全冻结。原热像／重建 VAE 不变。
- 本帧 caption，沿用该起点的文本规则、噪声种子、timestep、热像处理和官方 `ssi_log` 评估。
- 新 AdamW，学习率 `1e-5`，weight decay `0.01`，梯度裁剪 `1.0`。batch 4 × accumulation 3，1000 次更新。训练图像游标从 `4000×12` 继续；种子 43。
- 输出独立目录：`outputs/lotus_line_v2/iris_vae_decoder_only_v1/run_*`。

## 实际推理 latent 与监督边界

每帧调用原 `RouteModel.predict_disparity`，输入白名单仅包括热像路径、caption、原 manifest 索引；`target_latent=None`，不使用加噪 GT latent。单帧前向保留原部署的文本 padding 与随机噪声行为，随后把已 detach 的 latent 拼成解码器训练 batch。

latent 生成完毕后才加载监督。训练不做随机翻转／裁剪，避免监督与固定推理坐标错位。GT、伪 GT、归一化上下界都不进入预测函数。验证时也是先预测，再读取 GT 作官方对齐和评分。

不新增损失项，保留起点已有可影响解码器的三项：

`L = 0.05 × 米制 LiDAR L1 + 0.01 × 局部激光点对差 L1 + 0.0005 × 每帧 60–80m 平均偏差平方`。

L1 使用原 q02/q98 log bounds 解释输出尺度，目标仍为**未截断的真实激光**；解码输出不夹到 [0,1]。点对与偏差项沿用原可微 `ssi_log` 对齐、有效点和归一化规则。已有校准伪 GT 只用于复现训练尺度 bounds，不作为新的监督目标。U-Net 冻结后 latent loss 对解码器梯度为零，原重建任务也走冻结 VAE，因此两项不再计算。保留 bounds 并不意味着把激光目标重新截断。

这不能恢复已经丢失的所有信息；实验检验 latent 中尚存的热像／深度线索能否由适配后的解码器更准确地读出。不预先承诺保持 caption 收益或超过 NeWCRF。

## 冒烟与验证选点

冒烟先完整跑同一 14,249 帧 val：新旧解码输出逐像素复现（rtol/atol 1e-6），并复现父模型 AbsRel、RMSE、δ1（容差分别 0.00005、0.002 m、0.00005）。替换 GT 路径为不存在的路径后，实际 GPU 推理 latent 和输出必须逐元素不变。再做 4 次更新，确认解码器梯度／权重变化、所有冻结参数 hash 不变，且真实远端监督和点对均出现。

正式短训从**原始解码器重新开始**，不继承冒烟更新。核对冒烟合同、代码和初始权重，训练 1000 步后完整评估 500、1000 步，0 步使用已核对的冒烟全量基线。额外报告 0–20、20–40、40–60、60–80m 的逐帧平均 AbsRel，仅作解释，不新设门槛。

沿用冻结的 NeWCRF 导向规则，相对本轮 0 步：

- AbsRel 增加不超过 0.002，δ1 下降不超过 0.003；RMSE 至少降低 0.02 m。
- 相对已核对 NeWCRF val 参考的三项平均正向相对差距须缩小；按该差距、RMSE、AbsRel、步数排序。
- 无合格点则保留 0 步。选点只使用 val；三项 val 追平不等于 test 或全部七指标胜出。

提交仅产生两个作业：`iris_vd_smoke` → `afterok` → `iris_vd_train`（训练+验证）。冒烟失败不会启动短训。时间上限分别 3 h / 12 h，不是运行时间保证。主机小单测不替代真实权重的 GPU 冒烟。

## Aire 提交

```bash
(
  set -euo pipefail
  cd ~/Iris
  git pull --ff-only origin aire-hpc-setup
  unset IRIS_MS2_ROOT IRIS_MANIFEST_DIR IRIS_DATA
  export IRIS_RUNS=/mnt/scratch/dwdh0363/runs
  bash slurm/submit_iris_decoder_only.sh
)
```
