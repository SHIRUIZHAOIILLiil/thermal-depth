# 2026-08-11 结果：caption×天空损失 2×2 补齐、latent 目标线中途数

评测集统一为 test `2021-08-13-16-08-46`，2,543 帧，官方 BridgeMSD 协议、
`ssi_disparity` 对齐、min_depth 1e-3 / max_depth 80。三个主指标：AbsRel ↓ / RMSE ↓ / δ1 ↑。

---

## 1. caption × 天空损失的 2×2（`b_skyloss_caption_20ep` 补上最后一格）

新跑的这一格配置：`sky_loss_weight=0.5`、`sky_mask_dir=skymask_train_full/masks`、
`caption_mode=correct`、`pseudo_weight=0`、20 epoch。val best **e3 = 0.07642**。

**全部空 prompt**（比的是训练方式的效应）：

| | 无 caption | 有 caption |
|---|---|---|
| **无天空损失** | 0.08843 / 3.956 / 0.9071 | 0.08690 / 3.947 / 0.9077 |
| **有天空损失** | **0.08487 / 3.816 / 0.9142** | 0.08677 / 3.974 / 0.9102 |

天空损失单独用是四格中最好。在它之上再叠 caption 训练，三个指标同时退回：
AbsRel +0.0019、RMSE +0.158、δ1 −0.0040，约抵消一半收益。
b + 稠密补全线上 caption 是同一方向（0.08682 → 0.09098）。
**两次独立复现：caption 训练与这两种补洞手段不叠加，而是相抵。**

**`correct − empty`**（同一份权重只换 prompt）：

| checkpoint | empty | correct | 差 |
|---|---|---|---|
| e3（val best） | 0.08677 | 0.08800 | **+0.00123** |
| e5（val 平台） | 0.09036 | 0.09134 | +0.00098 |

第六条 caption 线，六条中没有一条为负。

**一处预判被推翻**：val 曲线上 e3（0.07642）比左右邻居低 0.0024/0.0035，
是平台之外的单点，我判断它在 test 上会缩水更多。实测 e3 退化 ×1.135、e5 ×1.145，
**e3 反而更好**，且绝对值全面领先。

---

## 2. latent 目标线（e3/20 中途数，两条线仍在跑）

`--latent-target` + 稠密伪 GT，`GT_LOSS_WEIGHT=0.3`（按梯度定标，见补全线记录）。
val best 都在 e3：无 caption 0.08393、有 caption 0.08352。

| 权重 | prompt | AbsRel | RMSE | δ1 |
|---|---|---|---|---|
| b_latent | empty | **0.09542** | 3.897 | 0.9034 |
| b_latent | correct | 0.11171 | 4.227 | 0.8740 |
| b_latent_caption | empty | 0.09584 | 3.971 | 0.9008 |
| b_latent_caption | correct | 0.09647 | 3.926 | 0.9019 |

- **latent 目标落后像素目标 +0.0070**（0.09542 vs b 线 0.08843），与 val 的 +0.006 一致。
  VAE 往返上界已测得 0.036，所以瓶颈不在表示能力，在 U-Net 于该目标下的拟合。
- **`b_latent` + correct 的 0.11171 不可作为 caption 效应引用**：该模型从未见过文本，
  推理时注入 caption 属分布外输入（b 线同款先例 0.08843 → 0.11751）。
- 可读的两个比较：caption 训练代价（两边 empty）+0.0004；
  `correct − empty` = **+0.00063**，仍为正。
- **val 的序在 test 上反转**：val 里 caption 臂每个 epoch 都略优（e3 0.08352 < 0.08393），
  test 上反过来（0.09647 > 0.09542）。原因是 val 把训练方式与推理 prompt 绑在一起
  （无 caption 臂用 empty 评、caption 臂用 correct 评），test 换 prompt 才拆得开。
- 唯一例外：caption 训练那条线上 correct 相对 empty 是 AbsRel 差、RMSE 好、δ1 好，
  六条线里第一次三指标不同向。差值极小且只到 e3/20，跑完复核。

---

## 3. 待测：AnyThermal 在本 test 上的天花板

现有唯一的数是 400 帧 **train** 子集上留出 LiDAR 点 AbsRel 0.0573（深度空间逐帧仿射），
而我方最好为 0.0849。两者**对齐空间不同**（`ssi` vs `ssi_disparity`），
是 BridgeMSD 对两类输出的并行规定，可并排引用但必须注明。

```
sbatch -J at_official --cpus-per-task=8 --mem=32G --time=01:00:00 --output=at_official-%j.out \
  --wrap="source ~/Iris/slurm/common.sh; cd \$IRIS_REPO; python tools/run_official_ms2_evaluation.py \
  --manifest \$IRIS_MANIFEST_DIR/ms2_test_16-08-46_rgb_depth_v1_clip75_20260728.jsonl \
  --data-root \$IRIS_MS2_ROOT \
  --prediction-dir \$SCRATCH/runs/anythermal/Midas_anythermal/raw_predictions \
  --route anythermal-midas --align ssi --gt-view thermal \
  --output-dir \$IRIS_RUNS/eval_official/anythermal_test_ssi"
```

这个数决定导师 2026-08-11 定的「完整伪 GT + Iris 训练方式」那条线是否有头部空间。
