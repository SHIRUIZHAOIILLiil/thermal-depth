# 稠密目标来源诊断：fit-only v2

本轮只推理、标定和诊断，不训练、不生成全量目标、不使用 test。
对照来源为 AnyThermal，候选 BTS、AdaBins，另报告 NeWCRF 作为需用户决定的备选。
固定原清单中的 train / val 各96帧，seed=20261017；原始预测在两个诊断分支间复用。

## 独立旧口径对照

仅在验证96帧上，对 AnyThermal 完整复刻 job9174706：全部有效 LiDAR 标定，
按原 build_config 的 dtype 保存伪 GT，使用原 MS2ThermalDataset 回填全部激光并做 q02–q98，
与旧诊断相同的 completed/pure 两张一批、fp32 VAE posterior mode 和 scale 算术，
对齐与评分都用全部有效 LiDAR。只取其中 source 和 completed 经 VAE 两个阶段作为控制。

记录标为 `mask=control_all, mode=ssi_log`，不进入来源选择。
60–80 m 的逐帧平均 bias 必须分别在 −8.10 / −12.86 m 的 ±1 m 内。
不通过即 `control_failed`，不选任何来源。该容差不变。

## 四来源留出比较

每帧固定抽约20%的有激光8×8网格作 heldout，四来源用完全相同的划分。
只用其余 fit 点标定和回填；由该稠密图计算 q98 / q100 边界并经过 VAE。
最终 ssi_log 的 scale/shift **也只由 fit 点拟合**，同一个变换用于整图，
再分别在 heldout 和 fit 点统计分段 bias、RMSE。该模式明确命名 `ssi_log_fit_only`，
不能与官方全帧对齐分数混称同一个评估。另保留不做 SSI 的 target_scale 记录。

8×8 留出避免这些网格内直接回填 heldout 激光，但不隔绝 VAE 的整个感受野。
如果来源模型之前训练时见过这些 train 帧，heldout 只表示本次标定/回填/对齐没有
使用这些标签，不能称为来源模型从未见过的标签。val 候选来源结果也完整报告。

## 冻结判定

判定只读 train、q100、VAE 后、heldout、ssi_log_fit_only：

- 60–80 m 的绝对平均 bias ≤ AnyThermal 的 0.7 倍；
- 0–40 m 的逐帧平均 RMSE 相对 AnyThermal 增加 ≤0.02 m。

比较只加 1e-12 的浮点边界容差，数值门槛不变。
BTS / AdaBins 通过则选绝对偏差较小者；只有 NeWCRF 通过时
`only_newcrf_passed_needs_user_decision` 且 `chosen=null`；全不通过则不选来源。
脚本在任何情况下都不会提交训练。远端 RMSE 随 bias 一起报告，偏差改善不等于误差全面改善。

## 检查与输出

必要测试包括：确定性整网格划分、改变 heldout GT 不影响目标及对齐、
独立旧口径对照与旧目标链计算一致、旧对照不得被 fit-only 行替代、门槛及来源选择、
拒绝 test 序列。真实权重与 VAE 的历史数值复现仍须在 Aire 实测。

新输出目录：`outputs/lotus_line_v2/iris_far_target_source_fitonly_v2/job_<ID>`。
主要文件：`summary.json`、`per_frame.csv`、固定子清单与原始预测。
主要日志：`[control]`、`[target]`、`[decision]`、`[completed]`。

```bash
(
  set -euo pipefail
  cd ~/Iris
  git pull --ff-only origin aire-hpc-setup
  unset IRIS_MS2_ROOT IRIS_MANIFEST_DIR IRIS_DATA
  export IRIS_RUNS=/mnt/scratch/dwdh0363/runs
  J=$(sbatch --parsable slurm/diagnose_far_target_source.sbatch)
  J=${J%%;*}
  echo "job=$J  log=$HOME/Iris/iris_far_source-$J.out"
)
```

后续只有诊断通过才考虑全量目标与 caption / empty 重训。同源两臂用于比较 caption 收益；
不同来源之间若要归因，必须匹配初始化、训练预算与监督设置。旧 AnyThermal 结果只在这些
设置相同时可复用，否则需要匹配基线，不能把历史训练差异全部归因于目标来源。
