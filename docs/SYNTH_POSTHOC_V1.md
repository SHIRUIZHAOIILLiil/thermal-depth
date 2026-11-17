# synth 臂的后处理：权重平均 + 只训 decoder（2026-11-17 预登记）

## 想法

synth 臂（`sd2_cap_pct_logtn_img_l0100_tmax1_synth_s43`，全量 val 0.0763 / 2.651 / 0.9491）是目前最好的纯 Iris U-Net。
另外两种手段已经在别的父模型上量过，都没在 synth 臂上用过：

| 手段 | 量过的父模型 | AbsRel | RMSE | δ1 |
|---|---|---|---|---|
| 权重平均 12k–20k（job 9229303） | tmax1，96 帧 val | −0.0031 | −3.9 cm | +0.0056 |
| 只训 decoder，+1000 步（run_rUV8qVux） | tmax1 16k，全量 val | +0.0001 | −6.8 cm | +0.0026 |
| 只训 decoder，+1000 步（run_vDeUjSDM） | conv_4000，全量 val | −0.0015 | −6.6 cm | +0.0014 |

问的是：现有最好的 U-Net 叠上这两样，在同一对齐（`ssi_log`）下能不能三项都超过 NeWCRF
（全量 val，job 9488948：0.07286 / 2.5625 / 0.9504）。不训练 U-Net，不碰 test。

## 做法（一条链，`slurm/submit_synth_posthoc.sh`）

1. **权重平均**（`tools/diagnose_checkpoint_average.py`，`SOURCE_RUN=synth`）：synth 臂 12000–20000 共 9 个 checkpoint 算术平均，
   在 11-07 那批 96 + 96 帧上用 `ssi_log` 打分。
   - 对照：step 16000 的 val RMSE 须复现 2.67386（job 9488823），容差 2 mm。
   - 规则：平均后的 96 帧 val 三项都**不差于** step 16000，才用平均权重（`use_average`）；否则下一步用 step 16000。
   - 开跑前检查 9 个 checkpoint 都在，缺一个就失败退出，不进 GPU。
2. **decoder 冒烟**（`tools/train_iris_synth_decoder.py smoke`）：冻结第 1 步选中的 U-Net，原始 VAE decoder。
   - 用 step 16000 时：全量 val 起点须复现 synth 臂（AbsRel 0.0763、δ1 0.9491 容差 6e-5，覆盖文档四位舍入；RMSE 2.65101 容差 2 mm）。
   - 用平均权重时：没有现成的全量 val 数可复现，step 0 的全量 val 就是「只做平均」的结果，照记；权重 sha256 须与第 1 步一致。
   - GT 隔离检查、4 步训练，与 tmax1 decoder 完全相同。
3. **decoder 训练 + 全量 val**：`train_iris_tmax1_decoder.py` 的 decoder 阶段原样复用——1000 次更新、lr 1e-5、
   q02/q100 目标、像素 L1 0.05 + 原有点对与远段偏差项、0/500/1000 三点全量 val、冻结的 `newcrf_primary3_v2` 选点规则
   （对起点 RMSE 至少降 2 cm、AbsRel 增加 ≤ 0.002、δ1 下降 ≤ 0.003、差距分数变小）。只换了冻结的 U-Net。

`train_iris_tmax1_decoder.py` 改成可以接不同的起点（`Start`），tmax1 的默认行为不变（测试锁定）。

## 判定（看结果前定）

唯一算数的问题：**选中的模型在全量 val、`ssi_log` 下，AbsRel、RMSE、δ1 三项都严格优于 NeWCRF `ssi_log`**
（0.07286 / 2.5625 / 0.9504，读 job 9488948 的 summary 原值）。结果写在 `<run>/train/verdict.json` 的 `passed`。

同时报告、不参与判定：
- 对 NeWCRF 自己的对齐 `ssi`（0.08455 / 2.4710 / 0.9422）的三项差值；
- 对起点（step 16000 或平均）的三项差值，以及平均单独带来的变化（第 1 步 96 帧、第 2 步全量 step 0）；
- 对 q98 decoder（run_vDeUjSDM）的比较（工具原有输出）。

通过才做下一步：官方子集 test 三天气各跑一次，测试脚本另写，不在这条链里。没通过就记录差距，不改门槛、不换选点。

## 事先的预期

粗估（两样部分叠加，不是测量）：RMSE 2.58–2.60，AbsRel 0.073–0.075，δ1 0.952–0.955。
最可能的结果是 δ1 赢、AbsRel 接近打平、RMSE 仍差 2–4 cm，即 **`passed` 为否**。对 NeWCRF 自己的对齐（2.471），RMSE 仍会差 10 cm 以上。

## 已知限制

- decoder 阶段沿用原配方里的点对和远段偏差项（11-07 已知远段「偏近」主要是回归均值）。这里不改，是为了沿用两次都量出约 6.7 cm 的同一配方。
- 只训 decoder 不符合稿子「只训 latent」的协议，进不进论文由用户决定。
- 一个种子；只有 caption 臂，没有 empty 臂。
- 第 1 步的规则只看 96 帧 val，噪声比全量大；全量数字由第 2 步的 step 0 给出。

## 投递

预计：平均约 2 h，冒烟约 3 h（含一次全量 val），训练 + 两次全量 val 约 10 h。与 log 预训练那条链互不依赖。

```bash
(
  set -euo pipefail
  cd ~/Iris
  git pull --ff-only origin aire-hpc-setup
  unset IRIS_MS2_ROOT IRIS_MANIFEST_DIR IRIS_DATA
  export IRIS_RUNS=/mnt/scratch/dwdh0363/runs MAIL_TO=anmolidawn@gmail.com
  bash slurm/submit_synth_posthoc.sh
)
```

脚本先检查所有要读的文件（synth 臂 9 个 checkpoint、converted 16k、NeWCRF 两份存档、q98 decoder run），缺任何一个就不投。
三个作业各挂一封 `afterany` 邮件，带 `[control] [row] [use_average] [reproduced] [selected] [verdict]` 行。

## 结果

**权重平均（作业 9489609）：** 对照 2.67386 逐位复现。96 帧 val：step 16000 0.07678 / 2.6739 / 0.94918 →
平均 0.07540 / 2.6435 / 0.94946，三项都不差，`use_average` 为真。（tmax1 的 5 cm 旧门槛 `passed: false` 只记录。）

**decoder 冒烟（作业 9489610）：** 平均权重的全量 val（14249 帧，step 0，即「只做平均」）：

| 全量 val，`ssi_log` | AbsRel | RMSE (m) | δ1 |
|---|---|---|---|
| synth 16k | 0.0763 | 2.651 | 0.9491 |
| 平均 12k–20k | 0.07462 | 2.5950 | 0.95157 |
| NeWCRF `ssi_log`（job 9488948） | 0.07286 | 2.5625 | 0.95039 |

- 平均单独：AbsRel −0.0017、RMSE −5.6 cm、δ1 +0.0025（比 96 帧上大）。
- 对 NeWCRF：δ1 已经领先 0.0012；RMSE 还差 3.2 cm；AbsRel 还差 0.0018。
- GT 隔离、4 步训练、冻结检查全部通过。
