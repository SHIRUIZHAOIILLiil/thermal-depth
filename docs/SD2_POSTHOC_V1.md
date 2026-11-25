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

## 结果

**权重平均（作业 9608903）：** 对照 2.79800 逐位复现。96 帧 val：step 16000 0.08332 / 2.798 / 0.93551 → 平均 0.08017 / 2.7589 / 0.94111，
三项都不差，`use_average` 为真。

**decoder 冒烟（作业 9608904）：** 平均权重的全量 val（14249 帧，step 0，即消融第 6 行的 val）：
0.07779 / 2.7030 / 0.94493；tmax1 16k 是 0.08137 / 2.7571 / 0.93937（AbsRel −0.0036、RMSE −5.4 cm、δ1 +0.0056）。
GT 隔离、4 步训练、冻结检查全部通过。

**第 6 行 test（作业 9608908，平均 + 原始 decoder，官方子集，`ssi_log`）：** AbsRel / RMSE / δ1

| test 官方子集 | 第 5 行 tmax1 | **第 6 行 + 平均** | NeWCRF（发表协议，中位数 1 参数） | NeWCRF（`ssi`，2 参数） |
|---|---|---|---|---|
| day | 0.0703 / 2.707 / 0.9490 | **0.06627 / 2.5888 / 0.95450** | 0.07116 / 2.717 / 0.95110 | 0.0716 / 2.430 / 0.9556 |
| night | 0.0738 / 2.522 / 0.9508 | **0.06980 / 2.4171 / 0.95717** | 0.07390 / 2.544 / 0.95230 | 0.0735 / 2.223 / 0.9578 |
| rain | 0.0929 / 3.339 / 0.9188 | **0.08842 / 3.2138 / 0.92541** | 0.09535 / 3.503 / 0.90943 | 0.0937 / 2.972 / 0.9212 |

- 对 NeWCRF 发表协议：**9 格全赢**。对 2 参数 `ssi`：AbsRel 3 格赢、雨天 δ1 赢，RMSE 三格输、day/night δ1 输。
- 对第 5 行：三种天气三项全部变好。

**第 7 行 test（作业 9608909，平均 + decoder +1000，最终模型）：** AbsRel / RMSE / δ1

| test 官方子集 | 第 6 行 + 平均 | **第 7 行 + decoder** | NeWCRF（发表协议，1 参数） | NeWCRF（`ssi`，2 参数） |
|---|---|---|---|---|
| day | 0.06627 / 2.5888 / 0.95450 | **0.06597 / 2.5703 / 0.95508** | 0.07116 / 2.717 / 0.95110 | 0.0716 / 2.430 / 0.9556 |
| night | 0.06980 / 2.4171 / 0.95717 | **0.06915 / 2.4037 / 0.95801** | 0.07390 / 2.544 / 0.95230 | 0.0735 / 2.223 / 0.9578 |
| rain | 0.08842 / 3.2138 / 0.92541 | **0.08784 / 3.1863 / 0.92649** | 0.09535 / 3.503 / 0.90943 | 0.0937 / 2.972 / 0.9212 |

- decoder 对第 6 行：三种天气三项全部变好。
- 对 NeWCRF 发表协议：**9 格全赢**。对 2 参数 `ssi`：AbsRel 3 格、δ1 夜间和雨天赢，RMSE 三格输、day δ1 差 0.0005。
- 一个种子；caption 臂；SD2 起点；无远处点对 / 偏差损失之外的额外数据。

## 空 caption 孪生臂（2026-11-24 预登记）

问题：最终模型去掉 caption 会怎样。按同一配方训练全程空 caption 的臂，再走同样的后处理和 test。

- 臂 `sd2_nocap_pct_logtn_img_l0100_tmax1_s43`：`NO_CAPTIONS=1 bash slurm/submit_arm.sh <臂> "--truncnorm_max 1.0"`，
  与 `sd2_cap_pct_logtn_img_l0100_tmax1_s43` 只差训练文本（空串）；val 用空 prompt。
- 后处理与 caption 臂完全相同（同一权重平均窗口与 not_worse 规则、同一 decoder 配方与选点规则、同一评估器），
  只是每一帧都用空 prompt：`tools/train_iris_tmax1_decoder.py` 的 `EMPTY_PROMPT` 标记在模型的 `encode_prompt` 上映射到空串（普通的空 caption 仍报错）；
  权重平均与 converted checkpoint 用 `caption_mode=empty`；评估器从合同的 `prompt_arm` 读取。
- 新臂没有事先登记的 96 帧数，平均作业的 control 记为「无」；若 decoder 起点用 step 16000，须复现该臂自己的全量 val。
- 一条命令：`bash slurm/submit_sd2_nocap.sh`（训练约 21 h，后处理与 test 约 6 h；三封邮件：臂的冒烟、臂的训练、全部结束后一封汇总）。

判读（看结果前定，只看均值）：第 7 行（最终模型）caption 臂对空 caption 臂，test 三种天气 × 三个指标：
- 9 格 caption 臂都更好：caption 在最终模型上有用；
- 9 格空 caption 臂都更好：caption 在最终模型上有害；
- 其余：混合，如实记录各格。
第 6 行同样报告。空 caption 臂对 NeWCRF 的比较也一并列出。

### 第一次投递（2026-11-24/25）

- 臂冒烟 9622934、训练 9622935、全量 val 9622936：完成。step 16000 全量 val（空 prompt）
  **0.0830 / 2.813 / 0.9363**；caption 臂同一位置 0.08137 / 2.7571 / 0.93937。
- 平均 9622937：完成，`use_average` 为真。96 帧上 step 16000 0.08577 / 2.86051 / 0.93107 → 平均 0.08292 / 2.85289 / 0.93473。
- decoder 冒烟 9622938 **失败**，其后 5 个作业被自动取消（run_wNYncDyl 只有 jobs.txt，弃用）。
  原因：`EMPTY_PROMPT` 当时写进了 `tools/iris_decoder_only.py`，而 q98 对照 run 的合同锁定了该文件的哈希，
  `eval_iris_decoder_only.selection` 拒绝运行。修正：该文件恢复为原字节（blob ae2ed0e57），标记改放在
  `train_iris_tmax1_decoder.py`；投递脚本先用 `tools/check_contract_code.py` 在登录节点核对合同锁定的代码哈希；
  汇总邮件改由 `slurm/mail_summary.sh` 发送，附失败作业 `.err` 的最后 20 行。
- 续跑：`AVG_JOB=9622937 bash slurm/submit_sd2_nocap.sh`，从 decoder 冒烟开始，臂与平均不重跑。判读规则不变。
