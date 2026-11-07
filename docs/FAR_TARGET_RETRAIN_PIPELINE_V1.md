# 换稠密目标来源的重训流水线 V1

接在 `tools/diagnose_far_target_source.py` 之后，无人值守跑到 val 判定。门槛在任何重训结果出来之前写定。

## 步骤

1. **选来源：** 只有 `clean_source_passed` 才继续，来源取诊断的 `chosen`（BTS 或 AdaBins）。
   `only_newcrf_passed_needs_user_decision`、`no_source_passed`、`control_failed` 都停下，不训练。
2. **全量目标：** 对 75,688 帧训练集推理，用全部 LiDAR 做逐帧仿射标定（与 AnyThermal 同一工具），写到
   `far_target_retrain_v1/run_*/target/calibrated_pseudo_depth`。推理可断点续跑。
3. **冒烟：** caption、empty 两臂各 20 步。
4. **训练：** tmax1 配方（`submit_arm.sh` 的 l100 配方加 `--truncnorm_max 1.0`），seed 43，训满 20000 步。
   两臂只差 `NO_CAPTIONS`。每 4000 步存一次以控制占盘，不影响训练，16000 和 20000 都会保存。
5. **val：** 与 `submit_arm.sh` 相同，固定评 16000 步。全量 14,249 帧，`ssi_log`，caption 臂用 `correct`，empty 臂用 `empty`。
6. **判定：** 见下一节，只读 val。
7. **test：** 判定通过才跑。两臂的 16000 步各评 day、night、rain，`ssi_log`，提示词与 val 相同。
8. **邮件：** 无论走到哪一步都发。内容是停止原因，或 val 判定和 test 官方子集结果（`tools/full_metric_table.py`）。

## 冻结判定

caption 臂 16k 对比 `sd2_cap_pct_logtn_img_l0100_tmax1_s43` 的 16k（val 8.137 / 2.7571 / 93.94）：

- RMSE 至少降 0.10 m；
- AbsRel 增加不超过 0.002；
- δ1 下降不超过 0.003。

empty 臂只报告，不参与判定。

## 不自动做的事

流水线不接续训或 decoder 微调。判定未通过时不跑 test。
本线 test 用掉之后，再对后续微调版本跑 test，就是第二次看 test，写结果时要说明。

```bash
(
  set -euo pipefail
  cd ~/Iris
  git pull --ff-only origin aire-hpc-setup
  unset IRIS_MS2_ROOT IRIS_MANIFEST_DIR IRIS_DATA
  export IRIS_RUNS=/mnt/scratch/dwdh0363/runs
  bash slurm/submit_far_target_pipeline.sh 9229162
)
```

诊断作业还在队列里就用 `afterok` 挂上；已结束则直接读它的 `summary.json`。
