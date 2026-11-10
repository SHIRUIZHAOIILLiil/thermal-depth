# 第二轮伪 GT（自训练）V1（2026-11-08，未投递）

## 想法

第一轮（tmax1）的训练目标 = 真 LiDAR + AnyThermal 补全的空白。第二轮把空白处换成 tmax1 16k
自己的预测（每帧在全部 LiDAR 上做 SSI-log 对齐，换成米），有 LiDAR 的地方照样写真值，
然后从 SD2 重新训练，配方与 tmax1 完全相同（caption、`--truncnorm_max 1.0`、20000 步、seed 43）。

不引入任何外部模型。

## 只在一种情况下投递（事先定下）

留出格子臂（`docs/TMAX1_HOLDOUT_ARM_V1.md`）判定为 `learns_beyond_target`：模型在训练时
没见过真值的格子上，比 AnyThermal 伪 GT 更接近真值。只有这样，用它的预测当题目才可能比
第一轮准；否则是拿自己的错教自己。判 `partial` 由用户决定；`copies_target` 或对照失败不投。

**结果（2026-11-09，作业 9237833）：`learns_beyond_target`，对照通过（1.12896）。**
0–40 m 留出格子：holdout 臂 1.1377，tmax1（见过）1.1290，AnyThermal 目标 1.5660，r = 0.020。
val 上 holdout 1.9847、tmax1 1.9920，而 AnyThermal 目标在 val 帧上是 1.904：模型在训练场景的
空白处远胜目标，在新场景上反而不如目标。MS2 是连续帧，被留出的格子多半在相邻帧里有 LiDAR，
所以前者可能部分是记住了场景。第二轮会让训练帧上的目标更准，能否转成 val 提升事先不知道，
由下面的门槛判定。

## 判定（与 strongaug、gamma 同一门槛）

全量 val，step 16000，对比 tmax1 16k（8.137 / 2.7571 / 93.94）：
RMSE 至少降 10 cm，AbsRel 增加不超过 0.002，δ1 下降不超过 0.003。不跑 test。

## 代价

- 造目标：tmax1 16k 推理 75688 帧，4 个分片并行，每片约 1 小时；约 50 GB
  （`$IRIS_RUNS/pseudo_gt/self_tmax1_16k`）。分片可续跑，已有文件跳过。
- finalize：逐帧检查存在、float32、有限，写 `build_config.json`（记录权重 sha256）。
- 训练约 22 小时，checkpoint 每 4000 步（约 25 GB），然后 val。

## 投递（只在判定允许后）

```bash
(
  set -euo pipefail
  cd ~/Iris
  git pull --ff-only origin aire-hpc-setup
  unset IRIS_MS2_ROOT IRIS_MANIFEST_DIR IRIS_DATA
  export IRIS_RUNS=/mnt/scratch/dwdh0363/runs
  P=$IRIS_RUNS/pseudo_gt/self_tmax1_16k
  [[ ! -e "$P" ]] || { echo "$P 已存在"; exit 1; }
  SH=$(sbatch --parsable --array=0-3 --export=ALL,MODE=shard,COUNT=4 slurm/build_self_pseudo_gt.sbatch)
  FN=$(sbatch --parsable --dependency=afterok:$SH --time=01:00:00 --export=ALL,MODE=finalize slurm/build_self_pseudo_gt.sbatch)
  echo "shards -> $SH  finalize -> $FN"
  RUN=sd2_cap_pct_logtn_img_l0100_tmax1_self2_s43
  out=$(AFTER=$FN PSEUDO_DIR="$P/calibrated_pseudo_depth" CKPT_STEP=4000 \
    bash slurm/submit_arm.sh "$RUN" "--truncnorm_max 1.0")
  echo "$out"
)
```

`submit_arm.sh` 最后打印的比较命令以 l100 为对照；把 `--control l100=...sd2_cap_pct_logtn_img_l0100_s43_val_...`
换成同目录下的 `sd2_cap_pct_logtn_img_l0100_tmax1_s43_val_...`，再按上面的门槛判。

`submit_arm.sh` 新增 `AFTER=<作业号>`：冒烟等该作业成功后才开始。不设时行为不变。

## 结果（2026-11-10，val 作业 9397616，全量 val 14249 帧，step 16000）

| 臂 | AbsRel | RMSE (m) | δ1 |
|---|---|---|---|
| tmax1 16k | 0.08137 | 2.7571 | 0.93937 |
| self2 16k | 0.0770 | 2.705 | 0.9454 |

- 三项都变好：AbsRel −0.0044，RMSE −5.2 cm，δ1 +0.006。
- 判定：**不通过**。门槛要求 RMSE 至少降 10 cm，实际 5.2 cm；AbsRel、δ1 两条满足。门槛不因结果改。
- 对 NeWCRF：tmax1 在全量 val 上的 RMSE 差距是三个序列合计约 28.6 cm（`docs/DIAGNOSIS_TO_STRONG_AUG_20261107.md` 第 4 节），
  这一轮约缩小 5 cm，差距仍在 20 cm 以上。
- 只有 val，一个种子；没跑 test。
