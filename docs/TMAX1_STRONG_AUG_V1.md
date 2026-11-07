# tmax1 强数据增强

## 问题

tmax1 16k 在 96 帧上 train RMSE 1.48、val 2.80，差距随步数变大，而训练只有翻转。
这一轮回答：在不接外部模型、SD2 起点、保留 caption 的前提下，数据增强能不能缩小这个差距。
不照抄 SupDepth4Thermal 的强度，直接测强版本。

## 改动

- 几何：翻转之后，两轴各放大 1–1.5 倍，随机偏移裁回 640×256。图像双线性，深度、稀疏 LiDAR 和掩码最近邻，深度值不缩放。
- 强度：分位拉伸之后做 gamma，对数均匀取 0.7–1.4。线性亮度和对比度在拉伸前做会被抵消，所以不用。

`lotus/utils/ms2_thermal_dataset.py` 和 `lotus/train_iris_ms2_g.py` 不改，它们在已有 run 的代码指纹里。
增强放在 `lotus/utils/ms2_train_augment.py`，由 `lotus/train_iris_ms2_aug.py` 替换 transform。
`train_iris_ms2_g_depth.sh` 新增 `TRAIN_ENTRY`，不设时命令不变。

其余配方与 `sd2_cap_pct_logtn_img_l0100_tmax1_s43` 相同：SD2 起点，seed 43，caption，训满 20000 步，比较点 16k。

## 冻结判定

全量 val，对比 tmax1 16k（8.137 / 2.7571 / 93.94）：

- RMSE 至少降 0.10 m；
- AbsRel 增加不超过 0.002；
- δ1 下降不超过 0.003。

另在 96 + 96 帧上报告 16k 的 val−train RMSE 差距（tmax1 是 1.31），只作诊断，不算通过。
差距明显缩小但门槛未过，再调强度；差距基本不变，就关掉这条路。这一步不跑 test。

```bash
(
  set -euo pipefail
  cd ~/Iris
  git pull --ff-only origin aire-hpc-setup
  unset IRIS_MS2_ROOT IRIS_MANIFEST_DIR IRIS_DATA
  export IRIS_RUNS=/mnt/scratch/dwdh0363/runs
  TRAIN_ENTRY=train_iris_ms2_aug.py IRIS_AUG_MAX_SCALE=1.5 IRIS_AUG_GAMMA=0.7,1.4 \
    bash slurm/submit_arm.sh sd2_cap_pct_logtn_img_l0100_tmax1_strongaug_s43 "--truncnorm_max 1.0"
)
```
