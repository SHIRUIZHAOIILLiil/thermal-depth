# tmax1 留出格子臂 V1（2026-11-08）

## 要回答的问题

在训练帧上，没有 LiDAR 的地方，训练目标只能是校准后的 AnyThermal 伪深度。
模型在那些地方学成的样子，是照抄了伪 GT 的错，还是比伪 GT 更好？

此前的数据说明不了这一点：训练用的伪 GT 是拿全部 LiDAR 补全的，模型训练时见过
每个评估点的真值，所以训练帧上的分数（tmax1 0–40 m 留出格子 RMSE 1.129，作业 9229261）
测的是“复现给过的值”，不是“点与点之间学成了什么”。

## 做法

配方与 `sd2_cap_pct_logtn_img_l0100_tmax1_s43` 完全相同（SD2 起训、caption、`--truncnorm_max 1.0`、
20000 步、seed 43），只换训练目标的构造（`lotus/utils/ms2_holdout.py`）：

- 每帧固定留出 20% 含 LiDAR 的 8 px 格子（= 一个 latent 像素），格子与作业 9229162 完全相同
  （seed 20261017，`tools/diagnose_far_target_source.holdout_mask`）。
- AnyThermal 只用另外 80% 的 LiDAR 重新做尺度/偏移校准，只把这 80% 写回目标。
- 图像空间 LiDAR 损失（λ=0.1）也只看这 80%。
- 留出格子里的目标因此只剩校准后的伪深度，和平时没有 LiDAR 的地方一样。

入口 `lotus/train_iris_ms2_holdout.py` 只替换 `MS2ThermalDataset`，两个带校验和的文件不动。
数据集初始化时先用全部 LiDAR 重建前 8 帧的伪深度，必须与磁盘上存的伪 GT 逐位相同，
否则拒绝训练（证明原始预测目录和校准流程对得上）。

checkpoint 每 4000 步存一次（`CKPT_STEP=4000`），只占约 25 GB。

## 判读规则（结果出来前定下，不改）

在 9229162 的 96 帧训练帧上，同一批留出格子，0–40 m，fit-only SSI-log 对齐：

- `target`：伪 GT 在留出格子上的误差（9229162，AnyThermal q100 过 VAE）= 1.566；
- `tmax1`：见过这些 LiDAR 的模型，在本作业里重测，必须与 9229261 相差不超过 2 mm（对照）；
- `holdout`：新臂 16k。

r = (holdout − tmax1) / (target − tmax1)：

| r | 结论 |
|---|---|
| ≥ 2/3 | `copies_target`：模型照抄伪 GT 在点与点之间的错 |
| ≤ 1/3 | `learns_beyond_target`：模型学到了比伪 GT 更好的规律 |
| 之间 | `partial` |

同时报告 40–60、60–80 m 和 val 帧的同口径数字，只作参考。val 全量指标由常规 val 作业给出，
不参与这个判定（少了 20% LiDAR，val 略差是预期内的）。不跑 test。

## 投递

`build_config.json` 记的原始预测目录是旧账号的
`/mnt/scratch/sc23sz/runs/pseudo_gt/official_train/raw_predictions_run/raw_predictions`
（float32，76547 帧，0 失败），迁移后在 `from_sc23sz/runs/pseudo_gt/` 下同一相对路径。
投递前检查文件数，冒烟时再逐位重建 8 帧验证。

投递（冒烟 → 训练 → val → 判读 → 邮件）：

```bash
(
  set -euo pipefail
  cd ~/Iris
  git pull --ff-only origin aire-hpc-setup
  unset IRIS_MS2_ROOT IRIS_MANIFEST_DIR IRIS_DATA
  export IRIS_RUNS=/mnt/scratch/dwdh0363/runs
  RAW=/mnt/scratch/dwdh0363/from_sc23sz/runs/pseudo_gt/official_train/raw_predictions_run/raw_predictions
  n=$(find "$RAW" -maxdepth 1 -name '*.npy' | wc -l)
  [[ "$n" == 76547 ]] || { echo "原始预测只有 $n 个，不投"; exit 1; }
  RUN=sd2_cap_pct_logtn_img_l0100_tmax1_holdout_s43
  out=$(TRAIN_ENTRY=train_iris_ms2_holdout.py IRIS_HOLDOUT_RAW_DIR="$RAW" CKPT_STEP=4000 \
    bash slurm/submit_arm.sh "$RUN" "--truncnorm_max 1.0")
  echo "$out"
  SM=$(echo "$out" | awk '$1=="smoke" && $2=="->"{print $3}')
  TR=$(echo "$out" | awk '$1=="train" && $2=="->"{print $3}')
  VA=$(echo "$out" | awk '$1=="val" && $2=="->"{print $3}')
  DG=$(sbatch --parsable --dependency=afterok:$VA --export=ALL slurm/diagnose_holdout_fit.sbatch)
  echo "diag  -> $DG"
  sbatch --dependency=afterany:$DG -J holdout_mail -t 00:10:00 --mem=1G -c 1 -o /dev/null \
    --wrap="{ sacct -X -j $SM,$TR,$VA,$DG -o JobID,JobName%22,State,Elapsed,ExitCode; grep -h -E '^\[(control|band|decision)\]' ~/Iris/iris_holdout_fit-$DG.out; } 2>&1 | mail -s 'holdout arm result' anmolidawn@gmail.com"
)
```

冒烟跑完先看两行：`[ms2-holdout] ... rebuilt 8 stored pseudo maps exactly; held LiDAR share 0.2xx`
和 `truncnorm_max 1.0`。没有 `[ms2-holdout]` 这一行就说明入口没换上，`scancel` 训练和后续作业。
