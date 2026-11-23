# SD2 起点的最终模型：tmax1 + 权重平均 + decoder，test 一次（2026-11-23 预登记）

## 为什么

按导师的消融结构（先比无 / 有 caption，再在 caption 模型上累加改动到最终模型），最终模型留在 **SD2 起点**：
caption 在这条线上有效（`RESULTS_20260921_LOG_LINE.md`；组件消融 `RESULTS_20261011_ABLATION_CAPTION_DATA.md` 第 4 行），
而合成起点上 caption 效应反向（`IRIS_SYNTH_PRETRAIN_V1.md`）。

消融表（test 官方子集，caption 臂，已有的行见上面两份文档）：

| 行 | 改动 | test |
|---|---|---|
| 1 | Iris 迁到 MS2，无 caption（视差目标） | 已有 |
| 2 | + caption | 已有臂，需按同一官方子集整理 |
| 3 | + percentile、log 深度目标 | 已有 |
| 4 | + 米制 L1 损失 | 已有 |
| 5 | + 远端上界 100%（tmax1） | 已有：day 7.03 / 2.707 / 94.90，night 7.38 / 2.522 / 95.08，rain 9.29 / 3.339 / 91.88 |
| 6 | + 权重平均 12k–20k | **本链** |
| 7 | + decoder 微调（最终模型） | **本链** |

## 做法（`slurm/submit_sd2_posthoc.sh`，一条链）

1. 权重平均（`diagnose_checkpoint_average.py --save-rule not_worse`，输出 `iris_checkpoint_average_tmax1_v2/`）：
   tmax1 12000–20000 共 9 个 checkpoint。对照：step 16000 的 96 帧 val RMSE 复现 2.79800（job 9229268）。
   96 帧 val 三项都不差于 step 16000 才用平均，否则用 step 16000。
   - 与 job 9229303 的差别：那次是 5 cm 门槛（只算「平均是否值得单独报」），没存权重。这次的用途是消融表的一行，
     规则改成与 synth 链相同的「不变差就用」。在看本链任何结果之前改。
2. decoder 阶段（`train_iris_synth_decoder.py --source-run sd2_cap_pct_logtn_img_l0100_tmax1_s43`）：与 synth 链
   （run_Pu1YrC5F）完全相同的 decoder 配方和冻结选点规则，只换冻结的 U-Net。若用的是 step 16000，起点须复现
   tmax1 全量 val 0.08137 / 2.7571 / 0.93937；若用平均，step 0 的全量 val 就是第 6 行的 val 数。
3. 评估（`tools/eval_stage_decoder.py`）：两个 stage 各自先复现全量 val（AbsRel / δ1 容差 5e-5，RMSE 2 mm），
   再各跑一次 test day / night / rain（官方子集，`ssi_log`）。
   - `start`：平均后的 U-Net + 原始 VAE decoder → 第 6 行；
   - `selected`：+ val 上选中的 decoder → 第 7 行。若选点规则保留起点（0 次更新），第 7 行等于第 6 行，`selected` 不跑。

## 判定与报告（看结果前定）

这是最终评估，不再在 test 上选任何东西，结果无论好坏都报告。三种天气 × 三个指标，报告对 NeWCRF 的比较：
- **主比较**：NeWCRF 用它发表的协议（逐图中位数缩放，1 参数；9 月复现 12/12 格），我们用原生 `ssi_log`（2 参数），
  遵循 BridgeMSD 的惯例，表中注明参数个数。NeWCRF：day 0.07116 / 2.717 / 0.95110，night 0.07390 / 2.544 / 0.95230，
  rain 0.09535 / 3.503 / 0.90943。数「赢了几格」。
- **同时报告**：NeWCRF 在 2 参数 `ssi` 下（day 7.16 / 2.430 / 95.56，night 7.35 / 2.223 / 95.78，rain 9.37 / 2.972 / 92.12）。
- 一个种子；只有 caption 臂，caption 的对照在消融第 1–2 行。

## 投递

```bash
(
  set -euo pipefail
  cd ~/Iris
  git pull --ff-only origin aire-hpc-setup
  unset IRIS_MS2_ROOT IRIS_MANIFEST_DIR IRIS_DATA
  export IRIS_RUNS=/mnt/scratch/dwdh0363/runs MAIL_TO=anmolidawn@gmail.com
  bash slurm/submit_sd2_posthoc.sh
)
```

脚本先检查 tmax1 的 9 个 checkpoint、converted 16k、NeWCRF 存档和 q98 decoder run，缺一个就不投。七个作业各挂一封邮件。
预计：平均约 1 h，冒烟约 1 h，decoder 训练 + 两次全量 val 约 3 h，复现各约 30 min，test 各约 10 min。
