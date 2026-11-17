# Iris 合成数据预训练线 V1（2026-11-09）

## 想法

Iris 从 SD2 出发，在合成 RGB 数据上训练：Hypersim（室内，90% 的步数）+ Virtual KITTI 2
（行车，10%），渲染的稠密 GT，InternVL3-8B 生成的 caption（`train_iris_g.py:1009-1011`）。

这条线先按 Iris 的配方在合成数据上训一遍，再用我们的 MS2 配方（tmax1）微调。理由：诊断显示
差距主要是新场景上的泛化方差（`docs/DIAGNOSIS_TO_STRONG_AUG_20261107.md`；留出格子臂：训练帧
空白处 1.138 远胜目标 1.566，val 上 1.985 反而不如目标 1.904）。合成数据带来大量不同场景。

**事先说明的不确定性**：Lotus-G 就是"SD2 + 同样两套数据、空文本"，8 月从它起步的旧配方臂 RMSE
没有占优（3.43/3.31/4.13，视差空间旧配方，不能直接和 tmax1 比）。所以合成预训练能不能救 RMSE，
现在没有答案。

## 第一段：数据和 caption

- `tools/synth_data.py`
  - Hypersim 不下 1.9 TB 整包：用 HTTP range 远程读每个场景的 zip，只取 `process_hypersim.py` 拷的三个文件
    （`tonemap.jpg`、`depth_meters.hdf5`、`normal_cam.hdf5`），公开 train 划分。
  - 深度含 NaN 的帧照 `process_hypersim.py` 丢掉；读不了的深度文件让作业失败，而不是静默丢帧。
  - vKITTI 下 rgb/depth/textgt 三个包，解开后检查加载器要读的每一对文件和 `intrinsic.txt`。
  - 法向用 Lotus 自己的 `utils/depth2normal.py` 生成。深度训练不用法向，但 `VKITTIDataset` 每帧都会打开它。
- `tools/caption_synth.py`：模型、prompt、切块和生成设置（`max_new_tokens=77, do_sample=True`）照抄
  Iris 的 `internVL/scripts/lotus_inference/inference_hypersim.py`（测试逐字核对）。Iris 没有公开 caption 文件，只能重生成。
  - 两处改动，都是为了不让任何一帧静默变成空 prompt：
    - key 用训练加载器自己构造的路径字符串（加载器查不到就给 `""`）；
    - finalize 要求每张图都有非空 caption，Iris 原脚本出错就跳过。
  - 采样按分片设了种子（Iris 没设）。
- 数据放在 `$SCRATCH/data/synth/{hypersim,vkitti,captions}`。路径里不能出现 rgb/depth/normal，
  因为加载器会整串替换这几个词，检查会拒绝这种路径。
- 测试：`tests/test_synth_data.py`（5 项）、`tests/test_caption_synth.py`（4 项，含与两个加载器的 key 逐一对照）。

## 第二段：合成预训练 → MS2 微调

- `slurm/iris_synth_pretrain.sbatch`：参数逐行照抄 `train_scripts/train_iris_g_depth.sh`
  （SD2、Hypersim 576 / vKITTI 375、P_HYPERSIM 0.9、trunc_disparity、bsz 4 × GAS 3、
  LR 3e-5 constant、20000 步、seed 42、fp16、8-bit Adam、带 caption）。和 Iris 的差别：
  - 不做训练中校验（他们评 NYU/KITTI 等，我们没有这些数据）；
  - 只留最近 2 个 checkpoint（省盘，步数不变）；
  - dataloader 6 个 worker（只影响读数据速度）；
  - SD2 用所有 sd2 臂共用的 2.1-base 镜像（Iris 是 2.0-base，已知偏差）。
- 墙钟没有实测。一个作业上限 2 天，所以挂两份同样的作业：第二份 `afterany` 第一份，
  第一份超时就从最近的 checkpoint 续，已经训完就立刻退出。
- MS2 微调：`submit_arm.sh` 原样（START=sd2，tmax1 配方），只多一个
  `--init_unet_from <预训练输出>`：U-Net 换成合成预训练的，VAE 和文本编码器本来就冻结且与 SD2 相同。
  臂名 `sd2_cap_pct_logtn_img_l0100_tmax1_synth_s43`。
- 判定（事先定好，不看结果再改）：全量 val step 16000，对比 tmax1
  （`sd2_cap_pct_logtn_img_l0100_tmax1_s43`）：RMSE 至少降 10 cm，AbsRel 增加不超过 0.002，δ1 下降不超过 0.003。

第二段投递。第一次（2026-11-09）挂在 caption 合并后面，预训练冒烟 9399130 一秒就失败：
脚本没设 `IRIS_MS2_ROOT`，`common.sh` 的前置检查找不到 MS2 目录。已修；下面是重投版本，
每个关键作业结束都发邮件：

```bash
(
  set -euo pipefail
  cd ~/Iris
  git pull --ff-only origin aire-hpc-setup
  scancel 9399131 9399132 9399133 9399134 9399135 9399518 || true
  unset IRIS_MS2_ROOT IRIS_MANIFEST_DIR IRIS_DATA
  export IRIS_RUNS=/mnt/scratch/dwdh0363/runs
  mailafter() { sbatch --dependency=afterany:$1 -J mail_$1 -t 00:10:00 --mem=1G -c 1 -o /dev/null --wrap="{ sacct -X -j $1 -o JobID,JobName%30,State,Elapsed,ExitCode; for f in ~/Iris/*-$1.out ~/Iris/*-$1.err; do echo \"--- \$f\"; tail -c 3000 \"\$f\" | tr '\r' '\n' | tail -n 12; done; } 2>&1 | mail -s \"iris job $1\" anmolidawn@gmail.com" >/dev/null; }
  PS=$(sbatch --parsable --time=01:30:00 --export=ALL,SMOKE=1 slurm/iris_synth_pretrain.sbatch)
  P1=$(sbatch --parsable --dependency=afterok:$PS --export=ALL,SMOKE=0 slurm/iris_synth_pretrain.sbatch)
  P2=$(sbatch --parsable --dependency=afterany:$P1 --export=ALL,SMOKE=0 slurm/iris_synth_pretrain.sbatch)
  out=$(AFTER=$P2 bash slurm/submit_arm.sh sd2_cap_pct_logtn_img_l0100_tmax1_synth_s43 "--truncnorm_max 1.0 --init_unet_from $IRIS_RUNS/synth_pretrain/iris_g_synth_sd2_s42")
  echo "$out"
  SM=$(echo "$out" | sed -n 's/^smoke -> \([0-9]*\).*/\1/p')
  VA=$(echo "$out" | sed -n 's/^val   -> \([0-9]*\).*/\1/p')
  for j in $PS $P1 $SM $VA; do mailafter $j; done
  echo "pretrain smoke $PS -> $P1 -> (timeout cover) $P2 | ms2 smoke $SM, val $VA | mail after $PS $P1 $SM $VA"
)
```

MS2 冒烟跑完先看 `grep -h "U-Net initialised from" ~/Iris/sm_sd2_cap_pct_logtn_img_l010-<冒烟作业号>.*`，
必须指向上面的预训练目录。

## 投递

先探针（CPU，约几分钟）：

```bash
cd ~/Iris && git pull --ff-only origin aire-hpc-setup && unset IRIS_MS2_ROOT IRIS_MANIFEST_DIR IRIS_DATA && sbatch --time=01:00:00 --export=ALL,STEP=probe slurm/synth_data.sbatch
```

看 `iris_synth-<作业号>_4294967294.out`：
- Hypersim 的帧数、每帧大小和整个划分的估计 GiB；
- vKITTI 三个包的大小和首个条目；
- caption 环境和 InternVL3-8B 是否在。

探针结果（2026-11-09，作业 9397647）：
- Hypersim 公开 train 划分 59543 帧 / 365 个场景；ai_001_001 98/98 帧无 NaN，103 s，0.91 MiB/帧，全划分约 53 GiB。
- vKITTI rgb.tar 7.0 GiB、depth.tar 7.6 GiB、textgt.tar.gz 很小；包内顶层直接是 `Scene01/...`。
- 学长账号上没有 caption 环境，也没有 InternVL3-8B 缓存，所以全链前面加了建环境（`build_env.sbatch`）和下载模型（`STEP=model`）。
- scratch 已用 1.38T / 3T，文件数 271 万 / 400 万，放得下。

全链：

```bash
(
  set -euo pipefail
  cd ~/Iris
  git pull --ff-only origin aire-hpc-setup
  unset IRIS_MS2_ROOT IRIS_MANIFEST_DIR IRIS_DATA
  EN=$(sbatch --parsable --export=ALL,ENV_YAML=environment_caption.yaml,ENV_NAME=caption slurm/build_env.sbatch)
  MD=$(sbatch --parsable --dependency=afterok:$EN --time=02:00:00 --export=ALL,STEP=model slurm/synth_data.sbatch)
  HY=$(sbatch --parsable --array=0-15%8 --export=ALL,STEP=hypersim,COUNT=16 slurm/synth_data.sbatch)
  HF=$(sbatch --parsable --dependency=afterok:$HY --time=01:00:00 --export=ALL,STEP=hypersim-finalize slurm/synth_data.sbatch)
  VK=$(sbatch --parsable --time=08:00:00 --export=ALL,STEP=vkitti slurm/synth_data.sbatch)
  VN=$(sbatch --parsable --dependency=afterok:$VK --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=04:00:00 --export=ALL,STEP=vkitti-normals slurm/synth_data.sbatch)
  CP=$(sbatch --parsable --dependency=afterok:$HF:$VN:$MD --array=0-15%10 --export=ALL,COUNT=16 slurm/synth_caption.sbatch)
  CF=$(sbatch --parsable --dependency=afterok:$CP --time=00:30:00 --export=ALL,STEP=caption-finalize,COUNT=16 slurm/synth_data.sbatch)
  sbatch --dependency=afterany:$CF -J synth_mail -t 00:10:00 --mem=1G -c 1 -o /dev/null --wrap="{ sacct -X -j $EN,$MD,$HY,$HF,$VK,$VN,$CP,$CF -o JobID,JobName%22,State,Elapsed,ExitCode; cat $SCRATCH/data/synth/hypersim/fetch_summary.json; tail -n 3 ~/Iris/iris_synth-${CF}_4294967294.out; } 2>&1 | mail -s 'synth data and captions' anmolidawn@gmail.com"
  echo "env $EN -> model $MD | hypersim $HY -> finalize $HF | vkitti $VK -> normals $VN | captions $CP -> $CF"
)
```

全部可续跑：任何一步超时或失败，原样重投同一条即可，已完成的场景、文件和 caption 会跳过。
caption 分片的 `--time` 按 L40S 2.4 s/帧估算（约 9.3 万张 / 16 片 ≈ 3.9 h），设 6 h。

## 复现 Iris 零样本结果（2026-11-13 加，只评估不训练）

目的：检验合成预训练本身复现得对不对。MS2 微调无论好坏，都要靠这个判断能不能归因到「这条路」而不是「复现有偏差」。

- 测试集：Iris Table 1 的 NYUv2 / KITTI / ETH3D / ScanNet / DIODE，用 Lotus README 指向的 Marigold 评估包
  （`share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/`）。
- caption：Iris 评估 Lotus 系模型时，给测试图也配了 caption，由 `internVL/scripts/lotus_inference/inference_{nyu,kitti,eth3d,scannet,diode}.py`
  生成：每个划分列表的第一列，从 tar 里读 `"./" + 路径`，模型、prompt 和生成设置都与训练集相同。他们没公开这些 JSON，
  这里用 `tools/caption_eval_sets.py` 重生成。key 就是评估加载器的 `rgb_relative_path`（同一列），finalize 不允许有缺，
  评估开 `--strict_text_mode`。
  - 仓库 `marigold/Marigold_test_text/` 里有 Iris 公开的 2,605 条测试 caption（nyuv2 654 / kitti 697 / scannet 800 / eth3d 454，没有 DIODE），
    但那是给 Marigold 评估用的 `.txt`。README 说 Lotus 和 E2E 的评估 caption 由 `lotus_inference` 脚本生成，所以这里不用它们。
- 划分文件：Iris 仓库里 `lotus/datasets/eval/depth/data_split/` 不全（NYU 缺失，KITTI / ETH3D / DIODE 是空文件，只有 ScanNet 完整）。
  从同一仓库的 `marigold/data_split/` 补齐；ScanNet 两份排序后逐行相同。共 3376 张：scannet 800、diode 771、kitti 697、nyu 654、eth3d 454。
- 评估：`lotus/eval_rng.py`，参数照 `eval_scripts/eval-depth-g.sh`（generation、seed 42、half precision、`--disparity`，
  即 least_square_disparity 对齐），评两个模型：
  - Lotus-G 官方权重、空 prompt：和 Lotus 论文发表的数字对，检验评估管线本身；
  - 我们的合成预训练、带 caption：和 Iris 论文 Table 1 对。
- 已知偏差：SD 2.1-base（Iris 是 2.0-base）、一个种子、caption 重采样。所以是看有没有明显偏差，不是逐位复现。

```bash
(
  set -euo pipefail
  cd ~/Iris
  git pull --ff-only origin aire-hpc-setup
  unset IRIS_MS2_ROOT IRIS_MANIFEST_DIR IRIS_DATA
  export IRIS_RUNS=/mnt/scratch/dwdh0363/runs
  mailafter() { sbatch --dependency=afterany:$1 -J mail_$1 -t 00:10:00 --mem=1G -c 1 -o /dev/null --wrap="{ sacct -X -j $1 -o JobID,JobName%30,State,Elapsed,ExitCode; for f in ~/Iris/*-$1.out ~/Iris/*-$1.err; do echo \"--- \$f\"; tail -c 4000 \"\$f\" | tr '\r' '\n' | tail -n 40; done; } 2>&1 | mail -s \"iris job $1\" anmolidawn@gmail.com" >/dev/null; }
  G="--partition=gpu --gres=gpu:1 --cpus-per-task=8"
  DL=$(sbatch --parsable --export=ALL,STEP=download slurm/iris_eval_sets.sbatch)
  CA=$(sbatch --parsable --dependency=afterok:$DL $G --mem=16G --time=02:00:00 --array=0-3 --export=ALL,STEP=caption,COUNT=4 slurm/iris_eval_sets.sbatch)
  CF=$(sbatch --parsable --dependency=afterok:$CA --time=00:30:00 --export=ALL,STEP=caption-finalize,COUNT=4 slurm/iris_eval_sets.sbatch)
  EL=$(sbatch --parsable --dependency=afterok:$DL $G --mem=32G --time=04:00:00 --export=ALL,STEP=eval,CAPTIONS=0,CKPT=jingheya/lotus-depth-g-v2-1-disparity,TAG=lotus_g_official_empty slurm/iris_eval_sets.sbatch)
  EO=$(sbatch --parsable --dependency=afterok:$CF:9431551 $G --mem=32G --time=04:00:00 --export=ALL,STEP=eval,CAPTIONS=1,CKPT=$IRIS_RUNS/synth_pretrain/iris_g_synth_sd2_s42,TAG=iris_g_synth_sd2_s42_cap slurm/iris_eval_sets.sbatch)
  for j in $DL $CA $CF $EL $EO; do mailafter $j; done
  echo "download $DL -> captions $CA -> finalize $CF | lotus-g eval $EL | ours eval $EO (after $CF and pretrain 9431551)"
)
```

## MS2 微调结果（2026-11-14，val 作业 9431554，全量 val 14249 帧，step 16000）

| 臂 | AbsRel | RMSE (m) | δ1 |
|---|---|---|---|
| tmax1 16k（对照） | 0.08137 | 2.7571 | 0.93937 |
| self2 16k（第二轮自训练） | 0.0770 | 2.705 | 0.9454 |
| synth 16k（合成预训练起点） | 0.0763 | 2.651 | 0.9491 |

- 对 tmax1：AbsRel −0.0051，RMSE −10.6 cm，δ1 +0.0097。三条门槛全部满足（RMSE 至少降 10 cm），**通过**。
- 有待确认：MS2 冒烟日志里 `U-Net initialised from` 指向 `synth_pretrain/iris_g_synth_sd2_s42`（邮件截断没看到）。
- 对 NeWCRF：tmax1 在全量 val 上的 RMSE 差距约 28.6 cm，这一臂补回约 10.6 cm，仍差约 18 cm。
- 只有 val，一个种子；没跑 test。零样本复现在 DIODE 上偏低（AbsRel 28.1 对 Iris-G 24.3），其余数据集待看。

## 叠加臂（2026-11-14 预登记）

`sd2_cap_pct_logtn_img_l0100_tmax1_synth_self2_s43`：合成预训练起点（`--init_unet_from`）+ 第二轮自训练目标
（`PSEUDO_DIR=self_tmax1_16k`，tmax1 16k 对齐后的预测，LiDAR 覆盖其上），其余同 tmax1 配方。

判定：全量 val step 16000，对比 **synth 臂**（0.0763 / 2.651 / 0.9491），不是 tmax1：
RMSE 至少再降 10 cm，AbsRel 增加不超过 0.002，δ1 下降不超过 0.003。

**结果（2026-11-15，训练 9454889，val 9454890，全量 val 14249 帧，step 16000）：**

| 臂 | AbsRel | RMSE (m) | δ1 |
|---|---|---|---|
| synth 16k（判定对照） | 0.0763 | 2.651 | 0.9491 |
| synth + self2 16k（叠加） | 0.0793 | 2.705 | 0.9433 |

- 对 synth：AbsRel +0.0030（超过 +0.002 的上限），RMSE +5.4 cm（要求 −10 cm），δ1 −0.0058（超过 −0.003 的上限）。三条门槛全部没过，**不通过**。
- 叠加之后三项都比单独用合成起点差，也比单独用自训练目标的 self2 差（AbsRel 0.0770，δ1 0.9454；RMSE 两者同为 2.705）。
- 两条改动不能叠加。往后的合成预训练臂沿用 tmax1 配方原本的训练目标，不再挂 self_tmax1_16k。

## 零样本复现结果（2026-11-14）

δ1 / AbsRel，百分数，`least_square_disparity` 对齐（Iris 协议）：

| 模型 | NYUv2 | KITTI | ETH3D | ScanNet | DIODE |
|---|---|---|---|---|---|
| Lotus-G，Lotus 原文（Iris Table 1 灰行） | 96.8 / 5.4 | 92.2 / 8.5 | 97.0 / 5.9 | 95.7 / 5.9 | 72.9 / 22.9 |
| Lotus-G 官方权重，我们评（空 prompt，官方 RNG） | 96.7 / 5.4 | 92.2 / 8.5 | 97.0 / 5.9 | 95.7 / 5.9 | 72.9 / 23.3 |
| Lotus-G*，Iris 重训 | 95.2 / 6.7 | 92.2 / 8.9 | 95.7 / 9.2 | 93.7 / 7.6 | 71.7 / 25.4 |
| Iris-G（Lotus-G + Text，Train & Infer） | 96.3 / 5.9 | 92.8 / 8.6 | 96.3 / 9.0 | 95.3 / 6.4 | 72.5 / 24.3 |
| 我们的合成预训练 + caption（SD 2.1-base，有效 batch 12） | 93.2 / 7.9 | 91.0 / 9.7 | 94.2 / 17.0 | 89.7 / 9.6 | 69.2 / 28.1 |

- 评估管线没问题：官方权重在五个集上与 Lotus 原文几乎逐位一致（只有 DIODE AbsRel 差 0.4）。
- 我们的预训练五个集都比 Iris-G 差（AbsRel +1.1 到 +8.0，ETH3D 最大），也比 Iris 不带文本的重训 Lotus-G* 差，室内集最明显。是系统性偏差。
- 已确认的配方差别：Iris 补充材料 Lotus 段写的是每步 4 张 × 累积 3 步 × 3 张 RTX 3090 = **有效 batch 36**，20000 步；
  `train_iris_g_depth.sh` 的默认（单卡）是 12，我们照脚本跑的就是 12，样本量只有他们的三分之一。

## 有效 batch 36 重训（2026-11-14 预登记）

- 预训练：`GAS=9`（单卡 4 × 9 = 36），其余不变，输出 `synth_pretrain/iris_g_synth_sd2_s42_bsz36`。
  按 3.05 s × 3 估约 51 小时，超过单作业 2 天，由第二份 `afterany` 作业从 checkpoint 续完。
- 复现检查：同一套零样本评估。只记录与 Iris-G 的差距，不作为门槛。
- MS2 微调臂 `sd2_cap_pct_logtn_img_l0100_tmax1_synth36_s43`：tmax1 配方 + `--init_unet_from` 新预训练。
  判定对比 **synth 臂**（0.0763 / 2.651 / 0.9491）：RMSE 至少再降 10 cm，AbsRel 增加不超过 0.002，δ1 下降不超过 0.003。
- MS2 微调本身的有效 batch 也是 12（同一份脚本），这里不改，保持与所有 MS2 臂可比。

## 配方审查与第二处修正（2026-11-14）

逐项对照 Iris 补充材料 Lotus 段（第 14–15 页）和 `train_iris_g_depth.sh`：

- 一致：Hypersim 39649 帧（文中约 39K）、NaN 过滤、576 / 375 分辨率、Hypersim 概率 0.9、trunc_disparity（truncnorm_min 默认 0.02）、
  随机翻转 0.5、LR 3e-5 constant 无 warmup、8-bit Adam、fp16、timestep 999、20000 步、每图一条 InternVL3-8B caption、
  caption 确实进入 batch（两个 collate 都带 `text_descriptions`）、评估管线（官方 Lotus-G 逐位复现 Lotus 原文）。
- 已修正 1：有效 batch 12 → 36（上一节）。
- 已修正 2：底座 SD 2.1-base → **SD 2.0-base**（Iris 脚本写的是 `stabilityai/stable-diffusion-2-base`）。官方仓库取不到；
  `Manojb/stable-diffusion-2-base`、`PeggyWang/...`、`luongphamit/...` 三个互不相关的镜像 U-Net safetensors sha256 相同
  （`c60b3631cc85…`），并带官方文件名 `512-base-ema.safetensors`；VAE 与文本编码器和 2.1-base 镜像逐字节相同，只有 U-Net 不同。
  `synth_data.sbatch STEP=sd20` 下载并校验 sha256。
- 不能补的：只有一个种子（Iris 自己重训 Lotus-G* 在 NYU 上就比 Lotus 原文差 1.3 AbsRel）；文中提到每 1000 步做校验，
  没说 Table 1 用最后一步还是校验最好的一步，我们用最后一步；vKITTI 按加载器是 33580 帧，文中写约 20000（只占 10% 的步数）。
- MS2 微调自身的有效 batch 也是 12，这是所有 MS2 臂共用的配方，不在这里改。

替换上一节的 batch 36 链：预训练 `iris_g_synth_sd20_s42_bsz36`（SD 2.0-base + 有效 batch 36），
MS2 臂 `sd2_cap_pct_logtn_img_l0100_tmax1_synth20b36_s43`，判定同上一节（对比 synth 臂 0.0763 / 2.651 / 0.9491，RMSE 至少再降 10 cm）。

## SD 2.0 + batch 36 结果（2026-11-17）

预训练 9455028（48 h 超时于 19068 步）+ 9455029（从 checkpoint-19000 续完）；零样本 9455030；
MS2 冒烟 9455031（`U-Net initialised from …/iris_g_synth_sd20_s42_bsz36`，target covers 100.0%，SL_A/SL_R 均在）；
训练 9455032；val 9455033。

零样本（δ1 / AbsRel，%）：

| 模型 | NYU | KITTI | ETH3D | ScanNet | DIODE |
|---|---|---|---|---|---|
| Iris-G（原文） | 96.3 / 5.9 | 92.8 / 8.6 | 96.3 / 9.0 | 95.3 / 6.4 | 72.5 / 24.3 |
| Lotus-G*（Iris 重训） | 95.2 / 6.7 | 92.2 / 8.9 | 95.7 / 9.2 | 93.7 / 7.6 | 71.7 / 25.4 |
| batch 12, SD2.1 | 93.2 / 7.9 | 91.0 / 9.7 | 94.2 / 17.0 | 89.7 / 9.6 | 69.2 / 28.1 |
| batch 36, SD2.0 | 96.0 / 6.0 | 92.5 / 8.6 | 95.9 / 13.3 | 94.4 / 6.7 | 72.1 / 24.9 |

五个集都比 batch 12 版明显变好；除 ETH3D 的 AbsRel（13.3 对 9.0）外都在 Iris-G 一个点以内，
KITTI AbsRel 与 Iris-G 相同。预训练算复现出来了。

MS2 全量 val（14249 帧，step 16000）：

| 臂 | AbsRel | RMSE (m) | δ1 |
|---|---|---|---|
| tmax1 16k | 0.08137 | 2.7571 | 0.93937 |
| synth（batch 12, SD2.1 起点，判定对照） | 0.0763 | 2.651 | 0.9491 |
| synth20b36（batch 36, SD2.0 起点） | 0.0770 | 2.686 | 0.9454 |

- 对 synth：AbsRel +0.0007，RMSE +3.5 cm（要求 −10 cm），δ1 −0.0037（超过 −0.003）。**不通过**。
- 对 tmax1 仍是三项都好（RMSE −7.1 cm），合成起点本身的作用还在。
- 零样本五个集（包括室外驾驶的 KITTI）都接近 Iris-G，MS2 却没有跟着变好：预训练质量的提升不转化为 MS2 收益，
  「室外数据占比不够」的解释也被 KITTI 排除。
- 两处改动（batch、底座版本）同时改，分不开；只有一个种子，两条 synth 臂之间 3.5 cm 的差是否超出种子波动未知。
  不影响门槛结论（要求再降 10 cm）。

## 合成起点有没有缩小 train→val 落差（2026-11-17 预登记，只推理）

背景：`docs/DIAGNOSIS_TO_STRONG_AUG_20261107.md` 的结论是泛化方差问题（96 + 96 帧，tmax1 16k
train RMSE 1.484、val 2.798，落差 1.31 m；训练帧上比 NeWCRF 准，val 上差）。两条 synth 臂只量过 val。

做法：`slurm/diagnose_train_val_gap.sbatch`，`CANDIDATE` 分别为 `..._tmax1_synth_s43` 与 `..._tmax1_synth20b36_s43`，
同一批 96 + 96 帧、`ssi_log`、step 16000；脚本先复现 tmax1 的 1.484 / 2.798（容差 2 mm）。
工具里的 `next` 字段是为增强臂写的，这里不用，只读落差。

判读（看结果前定）：
- 落差比 tmax1 小至少 0.20 m：合成起点的收益来自缩小泛化落差，泛化仍是要攻的问题。
- 落差缩小不到 0.20 m：train 和 val 一起变好，收益不是靠缩小落差得来的，「只是泛化问题」这个判断要重新审。

**结果（2026-11-17，作业 9488823 / 9488824）：** 两次对照都逐位复现 tmax1 的 1.48375 / 2.79800。

| 96 + 96 帧，`ssi_log`，step 16000 | train AbsRel / RMSE / δ1 | val AbsRel / RMSE / δ1 | val − train RMSE |
|---|---|---|---|
| tmax1 | 0.0431 / 1.4838 / 0.9905 | 0.0833 / 2.7980 / 0.9355 | 1.314 |
| synth（batch 12, SD2.1 起点） | 0.0432 / 1.4844 / 0.9911 | 0.0768 / 2.6739 / 0.9492 | 1.189 |
| synth20b36（batch 36, SD2.0 起点） | 0.0426 / 1.4995 / 0.9913 | 0.0785 / 2.7091 / 0.9424 | 1.210 |

- 按预登记：落差缩小 0.125 m / 0.105 m，都不到 0.20 m，归入第二档。
- 但第二档的文字描述（「train 和 val 一起变好」）与数据不符：train RMSE 几乎不变（1.4838 → 1.4844），
  收益全部在 val 上。门槛判定不改，机制按数据写：合成起点的收益全部来自缩小 train→val 落差，只是幅度不到 0.20 m。
- 预登记本身的缺陷：0.20 m 是从增强臂沿用的，而 synth 臂在全量 val 上的总收益只有 10.6 cm，
  train 不变时落差最多缩小约这么多，这条门槛事先就几乎不可能达到。以后按臂的预期幅度定门槛。
- 同一批 96 帧 val、同为 `ssi_log`（`docs/DIAGNOSIS_TO_STRONG_AUG_20261107.md` 第 7 步）：NeWCRF 0.0749 / 2.573 / 0.9474；
  synth 臂 0.0768 / 2.674 / 0.9492，RMSE 差约 10 cm，δ1 高 0.0018。只是 96 帧，全量 val 上的同对齐差距没量过。

## 全量 val 同一对齐下对 NeWCRF（2026-11-17，作业 9488948，CPU）

`tools/score_newcrf_val_ssi_log.py`：job 9167854 的 NeWCRF 缓存图，14249 帧，与 val 作业同一个 `evaluate_sample`。
对照：`ssi` RMSE 逐位复现存档 2.470983 m，通过。

| 全量 val 14249 帧 | AbsRel | RMSE (m) | δ1 |
|---|---|---|---|
| NeWCRF，`ssi`（原主表口径） | 0.08455 | 2.4710 | 0.9422 |
| NeWCRF，`ssi_log` | 0.07286 | 2.5625 | 0.9504 |
| synth 臂，`ssi_log` | 0.0763 | 2.651 | 0.9491 |
| tmax1，`ssi_log` | 0.08137 | 2.7571 | 0.93937 |

- 同一对齐（`ssi_log`）下 NeWCRF 三项都赢 synth 臂：AbsRel 0.0034，RMSE 8.9 cm，δ1 0.0013。
- 各用各的对齐时，RMSE 差 18 cm，而 AbsRel、δ1 是我们好；这个「优势」来自对齐不同（与 11-07 第 7 步 96 帧结论一致，现在全量确认）。
- 只有 val、一个种子；未跑 test。表里 NeWCRF 用哪种对齐、怎么披露，由用户决定。

## 剩下的 8.9 cm 在哪（2026-11-17 预登记，只读逐帧 CSV）

`tools/val_gap_breakdown.py`：tmax1、synth 臂的 val 逐帧 CSV 与 NeWCRF `ssi_log` 逐帧结果按帧配对，
均值须复现各作业打印值。判读（看结果前定，只看 RMSE，AbsRel/δ1 同样报告）：

- 中位数 / 均值 ≥ 0.5：差距是普遍的，大多数帧都差一截 → 修的是全局表示，候选是 log 空间合成预训练。
- 中位数 / 均值 < 0.5：差距由少数帧拉出来 → 先看这些帧是什么场景，再决定修什么，不先开训练臂。
- 任一序列的贡献占比超过它的帧数占比 1.5 倍：差距集中在该序列 → 先看该序列。
- 「最差 10% 帧占总差距的比例」只报告不判读：逐帧差的噪声大时这个比例本身就会很高，不能单独说明集中。

**结果（2026-11-17，登录节点）：** 三个均值都复现（tmax1 2.75713、synth 2.65101、NeWCRF 2.5625）。

| val 序列 | 帧数占比 | synth − NeWCRF RMSE（每帧均值） | 占总差距 | 贡献 / 帧数占比 | tmax1 → synth 收益 |
|---|---|---|---|---|---|
| 16-08-46 | 17.8% | +23.1 cm | 46.5% | 2.61 | 4.3 cm |
| 16-14-48 | 64.3% | +3.5 cm | 25.4% | 0.40 | 11.4 cm |
| 21-58-13 | 17.8% | +13.9 cm | 28.1% | 1.58 | 14.1 cm |

- 中位数 / 均值 = 1.03（≥ 0.5），帧内是普遍差距；但 16-08-46（2.61 倍）和 21-58-13（1.58 倍）都超过 1.5 倍，按规则**先看这两个序列**。
- 占帧数 64% 的 16-14-48 上 RMSE 只差 3.5 cm，δ1 每帧均值我们高 0.0026。AbsRel 同样集中：16-08-46 +0.0099、21-58-13 +0.0072、16-14-48 +0.0007。
- 合成起点在 16-08-46 上只拿到 4.3 cm，另两个序列 11–14 cm。

### 这两个序列差在哪（2026-11-17 预登记，CPU）

`tools/val_gap_by_frame_content.py`：逐帧读 GT（0.001–80 m 有效点）与原始 16 位热像，记录 GT 平均深度、
40 m / 60 m 以外占比、有效点数、热像 p1 / p99 及其跨度。每个统计量按全体帧分五分位，
每个序列在所有序列都至少有 30 帧的分位上等权重新平均 RMSE 差距（相当于让三个序列帧构成相同）。

判读（看结果前定）：以 16-14-48 为参照，看 16-08-46、21-58-13 与它的差距在重新加权后缩小多少。
- 按 GT 深度类统计量重新加权后，差距缩小至少一半：这两个序列差是因为深度构成不同（远处或近处更多）。
- 深度类不行、热像类缩小至少一半：差在热像输入的范围 / 对比度上。
- 都缩小不到一半：差异不在这些统计量里，下一步看这两个序列的实际画面内容。

**结果（2026-11-17，作业 9489032）：** 与 16-14-48 的差距在重新加权后缩小的比例：

| 加权依据 | 共同分位 | 16-08-46 | 21-58-13 |
|---|---|---|---|
| GT 平均深度 | 5/5 | 8% | −9% |
| GT 40 m 以外占比 | 5/5 | 0% | 2% |
| GT 60 m 以外占比 | 5/5 | 13% | 20% |
| GT 有效点数 | 5/5 | 0% | −31% |
| 热像 p1 | 2/5 | 91% | −82% |
| 热像 p99 | 2/5 | −36% | 49% |
| 热像跨度 | 2/5 | −46% | 8% |

- 深度构成：五个分位都有足够帧，四个深度统计量都缩小不到一半，**排除**。逐帧差距与各统计量的相关都弱（|r| ≤ 0.18）。
- 热像：按规则字面，16-08-46 在 p1 上缩小 91%，归入「热像」；21-58-13 都不到一半。
  但三个序列的热像范围只在 5 个分位里的 2 个上都有至少 30 帧，而且同为热像类的 p99、跨度把 16-08-46 的差距反而放大，
  三个热像统计量方向互相矛盾。只能记为「热像范围未排除也未确立」，不据此开臂。
- 序列均值：16-08-46 热像跨度最大（944，16-14-48 为 827），21-58-13（夜间）最小（567）；16-08-46 远处稍多（40 m 以外 8.1% 对 4.4%）。
- 下一步按规则看画面：`tools/val_gap_frames_sheet.py` 画每个序列逐帧差距随帧序的走势，标出最差三段和最好一段 200 帧，
  附热像缩略图（只作诊断，不进任何材料）。

**画面（2026-11-17，作业 9489071，图在 `diag/gap_sheet/`，只作诊断）：**
- 16-08-46：101 帧滑动均值几乎全程在 +0.2 到 +0.6 m，只有约 880–1060 帧和 1650–1780 帧两段降到 0 以下；是整条序列抬高，不是几处尖峰。
- 16-14-48：滑动均值在 0 上下来回，有大段我们更好（440–639 帧约 −0.4 m）。
- 21-58-13：大部分时间略高于 0，最差段是开阔路面、远处地平线的画面。
- 缩略图里没有看出一种能解释差距的单一场景类型：16-14-48 的最差段是近处有车的街道，16-08-46 的最好段同样是近处有车。
- 16-08-46 与 16-14-48 同一天、相隔约 6 分钟（序列名就是开始时间），天气和时段相同，只是路线不同；
  所以 16-08-46 的差距不是天气或时段造成的，而是这条路线的场景本身。

### 帧内哪里输（2026-11-17 预登记，GPU 推理，只用 val）

`tools/val_pixel_gap.py` / `slurm/val_pixel_gap.sbatch`：三个 val 序列每 10 帧取 1 帧（约 1425 帧），synth 臂实时推理，
NeWCRF 用 job 9167854 缓存图；两者都用 `ssi_log` 对齐、裁到 0.001–80 m。每个有效 LiDAR 点按
图像上 / 中 / 下三分之一 × GT 深度 0–10 / 10–20 / 20–40 / 40–80 m 分成 12 格。
对照：每帧 RMSE 须复现全量 val 的逐帧值（synth 容差 1 mm，NeWCRF 1e-6 m）。

判读（看结果前定）：
- 在 16-08-46 上，误差平方差的占比 ≥ 点数占比 1.5 倍的格子，就是「输在哪」。
- 这些格子在 16-14-48 上也是最大的几格：同一个弱点在 16-08-46 被放大；不是：这条路线特有。
- 在这些格子里，synth 的平均 log 偏差绝对值比 NeWCRF 大 0.05 以上（约 5%）：系统性偏远或偏近；否则是逐帧方差。
- 像素合并的误差按点数加权，和逐帧 RMSE 均值的权重不同，只用来定位，不替代主指标。

**结果（2026-11-17，作业 9489118，1426 帧）：** 对照逐位复现（synth、NeWCRF 每帧 RMSE 差均为 0）。
邮件截掉了每个序列最后两格（下方 20–40、40–80 m），显示的十格点数与差额占比合计都约 100%，那两格可忽略。

按预登记规则（16-08-46，差额占比 / 点数占比 ≥ 1.5）：

| 格 | 点数 | 差额占比 | 比值 | synth / NeWCRF RMSE | log 偏差 synth / NeWCRF |
|---|---|---|---|---|---|
| 上 40–80 m | 1.9% | 15.1% | 7.95 | 7.72 / 7.03 | −0.084 / −0.051 |
| 中 40–80 m | 5.0% | 22.8% | 4.56 | 6.59 / 6.13 | −0.051 / −0.043 |
| 上 0–10 m | 1.9% | 3.7% | 1.95 | 2.51 / 1.95 | +0.182 / +0.115 |

- 输在哪：两格远处（40–80 m）占差额 38%，加上方近处一格共 42%，来自 8.8% 的点。
- 16-14-48 上差额最大的同样是中、上 40–80 m（上方 0–10 m 在那里是负的）：远处是**共同弱点在 16-08-46 被放大**。
- 远处两格 log 偏差绝对值比 NeWCRF 只多 0.033 / 0.008（< 0.05）：**逐帧方差，不是系统偏差**；两个模型都偏近，同一种回归均值。
  上方 0–10 m 多 0.067（> 0.05）：系统性偏远，但只占差额 3.7%。

规则之外的观察（未预登记，只记录）：按相对误差看，规律不同。
- synth / NeWCRF 的 RMSE 比值，16-08-46 在 0–20 m 各格是 1.23–1.34，21-58-13 是 1.08–1.28，16-14-48 只有 0.93–1.12；
  远处 40–80 m 三个序列都只有 1.01–1.10。
- 也就是说，远处是两个模型都差（RMSE 6–10 m），我们只多 5–10%；近处 0–20 m 在两条落后的路线上我们比 NeWCRF 差 25–35%，
  在 16-14-48 上基本持平。平方误差按米计，所以远处在 RMSE 差额里占比大；AbsRel 按相对误差，近处点多，AbsRel 的差距主要来自近处。

### 近处短板在训练帧上有没有（2026-11-17 预登记）

`tools/train_val_pixel_gap.py`：11-07 那批 96 训练帧 + 96 val 帧（job 9229162 缓存了两边的 NeWCRF 图），同样的格子拆解，
主看 0–20 m 全部点合并后的 RMSE 比值 synth / NeWCRF。对照：synth 每组均值 1.48442 / 2.67386（9488823），
NeWCRF val `ssi` 2.457972（容差 2 mm）。

判读（看结果前定）：
- 训练帧比值 ≤ 1.05 且 val 比值 ≥ 1.15：近处短板只在新场景出现，是**路线覆盖 / 泛化**。
- 训练帧比值 ≥ 1.15：训练帧上近处也比 NeWCRF 差，是**精度上限**（训练目标或模型），不是覆盖问题。
- 其他情况：两者都有，按两个比值的大小如实记录，不归类。

**结果（2026-11-17，作业 9489458）：** 两道对照都复现（synth 1.48442 / 2.67386，NeWCRF `ssi` 2.457972）。

| 96 帧 | 0–20 m 合并比值 | 合并全部点 synth / NeWCRF |
|---|---|---|
| train | 1.021 | 1.557 / 1.655 |
| val | 1.042 | 2.683 / 2.590 |

- 按预登记：train 1.021 ≤ 1.05，但 val 只有 1.042（< 1.15），归入「其他情况」，不归类。
- 规则的缺陷：96 帧 val 是三个序列混合（大半是 16-14-48），近处比值被冲淡；而且 0–20 m 合并把行为不同的两段混在一起，
  10–20 m 中间行占了大部分点。门槛判定不改，下面按格子如实记录。

按格子（synth / NeWCRF RMSE 比值）：

| 段 | train 96 帧 | val 96 帧 | 16-08-46 | 16-14-48 | 21-58-13 |
|---|---|---|---|---|---|
| 下 0–10 m | 1.55 | 1.20 | 1.29 | 1.12 | 1.28 |
| 中 0–10 m | 1.27 | 1.02 | 1.33 | 1.01 | 1.24 |
| 上 0–10 m | 1.17 | 1.07 | 1.29 | 0.98 | 1.15 |
| 中 10–20 m | 0.98 | 1.08 | 1.23 | 1.00 | 1.08 |
| 中 20–40 m | 0.92 | 1.00 | 1.09 | 0.97 | 1.04 |
| 中 40–80 m | 0.95 | 1.06 | 1.07 | 1.07 | 1.03 |
| 上 40–80 m | 0.86 | 1.04 | 1.10 | 1.01 | 1.05 |

- 0–10 m：训练帧上也比 NeWCRF 差 17–55%（绝对值小，下方 0.31 对 0.20 m）。这一段是**精度上限**，不是覆盖问题。
- 10–80 m：训练帧上我们比 NeWCRF 好（0.86–0.98），到陌生路线变差（16-08-46 1.07–1.23）。这一段是**泛化**。
- RMSE 按米计，差额主要在 10–80 m，所以 RMSE 的差距是泛化问题；0–10 m 的精度上限对 RMSE 影响小，主要影响 AbsRel。

## log 空间合成预训练（2026-11-17 预登记）

想法：合成起点是唯一证实能修泛化的手段（val −12.4 cm，train 不动）。预训练目标是 Lotus 的 `trunc_disparity`
（视差，近处 = +1），MS2 微调是 `log_truncnorm`（log 深度，近处 = −1）：U-Net 进 MS2 时输出既要换空间又要反向。
这一臂让预训练直接用 MS2 的目标。没有直接证据，只是这条线上唯一还没对齐的地方。

改动（只此一处，其余与 synth20b36 相同：SD 2.0-base、有效 batch 36、20000 步、Iris 其余参数）：
- `lotus/utils/log_target.py`：与 `ms2_thermal_dataset.py` 的 `log_truncnorm` 逐行相同的公式；
  Hypersim 用全部有限正深度，vKITTI 用加载器自己的有效掩码（天空 ≥ 80 m 不在掩码里，裁到 +1 = 远）。
- `hypersim_dataset.py` / `vkitti_dataset.py` / `train_iris_g.py`：新增 `log_truncnorm` 选项和 `--truncnorm_max`（默认不变）。
- 预训练 `NORM_TYPE=log_truncnorm TRUNCNORM_MAX=1.0`，输出 `synth_pretrain/iris_g_synth_sd20_s42_bsz36_log`。
- `tests/test_synth_log_target.py`：与 MS2 公式一致、近处低远处高、视差方向相反、天空裁到 +1、默认上界不变；
  预训练作业开跑前执行，失败即退出。
- 不做零样本复现：Lotus 评估器按视差 / 深度对齐，log 输出不适用，也不是这一臂要回答的问题。

MS2 臂 `sd2_cap_pct_logtn_img_l0100_tmax1_synth20b36log_s43`，tmax1 配方 + `--init_unet_from` 上面的预训练。判定（全量 val，step 16000）：
- 归因：对 synth20b36（0.0770 / 2.686 / 0.9454）报告三项差值，回答 log 预训练有没有用。
- ~~进展：对 synth 臂（0.0763 / 2.651 / 0.9491），RMSE 至少降 10 cm~~（2026-11-17 投递前改，见下）。
- 进展（唯一算数的门槛，投递前定）：对 **synth20b36**（0.0770 / 2.686 / 0.9454），RMSE 至少降 **5 cm**，
  AbsRel 增加不超过 0.002，δ1 下降不超过 0.003。

门槛修改说明（2026-11-17，任何结果出来之前）：
- 原门槛要求对 synth 臂（SD2.1、batch 12）再降 10 cm，等于对本臂自己的底座 synth20b36 降约 13.5 cm，
  超过合成预训练的全部效果（10.6 cm），事先几乎不可能达到（与上文 0.20 m 落差门槛同一个缺陷）。
- 底座保持 SD2.0 + batch 36：忠实于 Iris 配方（零样本已复现），且 synth20b36 是同配方、只差目标的现成对照。
  SD2.1 + batch 36 没有对应的视差臂，结果无法归因；synth 与 synth20b36 之间 3.5 cm 是单种子、两处改动混在一起，不作为选底座的依据。
- 5 cm 大于两条 synth 臂之间 3.5 cm 的差；过线时 RMSE ≤ 2.636，也优于 synth 臂（2.651）。
- 本臂过线后，作为新的 U-Net 底座再走 `docs/SYNTH_POSTHOC_V1.md` 的权重平均 + decoder 链。

## 全程空 caption 的配对臂（2026-11-17 预登记）

问题：合成起点上，caption 的作用还在不在。稿子里的 caption 证据全部来自 SD2 起点的 `sd2_cap` / `sd2_nocap` 配对；
从 Lotus（空 prompt 训练的深度模型）起训时看不到 caption 效应，合成起点也是训练好的深度模型，所以要单独验证。

对照按 Iris 的做法：Iris 的无文本基线是 Lotus-G*（SD2、同样数据和步数、prompt 全空），对 Iris-G（预训练和推理都带 caption）。
这里配 log b36 那条（`iris_g_synth_sd20_s42_bsz36_log` → `sd2_cap_pct_logtn_img_l0100_tmax1_synth20b36log_s43`），只差 caption：

- 预训练 `iris_g_synth_sd20_s42_bsz36_log_nocap`：`NO_CAPTIONS=1`，不传两个 caption 文件，`train_iris_g.py` 每帧喂 `""`；其余同 log 臂
  （SD2.0-base、有效 batch 36、log_truncnorm、`--truncnorm_max 1.0`、20000 步、seed 42）。
- MS2 臂 `sd2_nocap_pct_logtn_img_l0100_tmax1_synth20b36log_s43`：tmax1 配方 + `--init_unet_from` 上面的预训练，
  `NO_CAPTIONS=1` 训练，val 用空 prompt 选点（`SEL_PROMPT=empty`）；caption 臂用真 caption。
- 防错：`iris_synth_pretrain.sbatch` 要求 tag 含 `nocap` 当且仅当 `NO_CAPTIONS=1`；`submit_arm.sh` 要求臂名以 `sd2_nocap_` 开头当且仅当 `NO_CAPTIONS=1`。

判定（全量 val，step 16000，各用自己的 prompt，`ssi_log`），报告 caption 臂 − 空 caption 臂的三项差值：
- 三项都是 caption 臂更好：caption 效应在合成起点上保留。
- 三项都是空 caption 臂更好：反向。
- 其余：混合，如实记录。
块配对的区间按稿子现有方式另算，不改上面的判定。不做打乱 caption 的对照。

一个种子；如果 log 臂没过 5 cm 门槛、最终改用 synth20b36（视差目标），还要另补视差版的空 caption 预训练。

投递（与 caption 链并行；端口错开，免得两条预训练落到同一节点时冲突）：

```bash
(
  set -euo pipefail
  cd ~/Iris
  git pull --ff-only origin aire-hpc-setup
  unset IRIS_MS2_ROOT IRIS_MANIFEST_DIR IRIS_DATA
  export IRIS_RUNS=/mnt/scratch/dwdh0363/runs
  export NO_CAPTIONS=1 MAIN_PROCESS_PORT=13225 MAIN_PORT=13226
  export SD2_REPO=Manojb/stable-diffusion-2-base GAS=9 NORM_TYPE=log_truncnorm TRUNCNORM_MAX=1.0 RUN_TAG=iris_g_synth_sd20_s42_bsz36_log_nocap
  mailafter() { sbatch --dependency=afterany:$1 -J mail_$1 -t 00:10:00 --mem=1G -c 1 -o /dev/null --wrap="{ sacct -X -j $1 -o JobID,JobName%30,State,Elapsed,ExitCode; for f in ~/Iris/*-$1.out ~/Iris/*-$1.err; do echo \"--- \$f\"; tail -c 4000 \"\$f\" | tr '\r' '\n' | tail -n 25; done; } 2>&1 | mail -s \"iris job $1\" anmolidawn@gmail.com" >/dev/null; }
  PS=$(sbatch --parsable --time=01:30:00 --export=ALL,SMOKE=1 slurm/iris_synth_pretrain.sbatch)
  P1=$(sbatch --parsable --dependency=afterok:$PS --export=ALL,SMOKE=0 slurm/iris_synth_pretrain.sbatch)
  P2=$(sbatch --parsable --dependency=afterany:$P1 --export=ALL,SMOKE=0 slurm/iris_synth_pretrain.sbatch)
  unset SD2_REPO GAS NORM_TYPE TRUNCNORM_MAX RUN_TAG
  out=$(AFTER=$P2 bash slurm/submit_arm.sh sd2_nocap_pct_logtn_img_l0100_tmax1_synth20b36log_s43 "--truncnorm_max 1.0 --init_unet_from $IRIS_RUNS/synth_pretrain/iris_g_synth_sd20_s42_bsz36_log_nocap")
  echo "$out"
  SM=$(echo "$out" | sed -n 's/^smoke -> \([0-9]*\).*/\1/p')
  TR=$(echo "$out" | sed -n 's/^train -> \([0-9]*\).*/\1/p')
  VA=$(echo "$out" | sed -n 's/^val   -> \([0-9]*\).*/\1/p')
  for j in $PS $P1 $P2 $SM $TR $VA; do mailafter $j; done
  echo "nocap pretrain smoke $PS -> $P1 -> $P2 | ms2 smoke $SM -> train $TR -> val $VA | mail after all six"
)
```

预训练冒烟邮件里要看到 `captions: off`；MS2 冒烟里 `captions  = off`，`U-Net initialised from` 指向 `_bsz36_log_nocap`。

## AbsRel 输在哪个距离段（2026-11-17 预登记，GPU 推理约 5 分钟，只用 val）

背景：上面「帧内哪里输」只拆了 RMSE（平方误差）；「AbsRel 的差距主要来自近处」是从 RMSE 比值和点数推的，没有量过。
11-07 的同对齐拆分里，0–40 m 的 RMSE 差距 69% 在 20–40 m，所以差距也可能在 10–40 m。

做法：`tools/val_pixel_gap.py` v2（`slurm/val_pixel_gap.sbatch`，输出 `val_pixel_gap_v2/`），同一批 1426 帧、同一 synth 16k、
同一 NeWCRF 缓存。AbsRel 和 δ1 都是「帧内对有效点平均、再对帧平均」，所以每个格子的贡献是
mean_f(Σ_{格内点} x / N_f)，各格严格相加等于序列的帧平均值，不是近似。
- 对照：每帧各格之和与评估器同一帧的 AbsRel / δ1 相差 ≤ 1e-9；每帧 RMSE、AbsRel、δ1 复现存档逐帧 CSV
  （synth：RMSE 1 mm、AbsRel 1e-4、δ1 2e-3；NeWCRF 1e-6）。`tests/test_val_pixel_gap_split.py` 在作业开头跑。
- 报告每个序列和全部帧（`all`）：各距离段（上中下三行合并）的点数占比、AbsRel 差值及占总差距比例、δ1 差值及占比。

判读（看结果前定，看 `all` 的 AbsRel 差距，三段只有一段 ≥ 50% 才归类）：
- 0–10 m ≥ 50%：近处精度（训练帧上也输的那一段）。
- 10–20 m + 20–40 m ≥ 50%：中段，即 10–40 m。
- 40–80 m ≥ 50%：远处。
- 都不到 50%：分散，如实记录各段；两段以上过 50%（有段我们领先、份额为负时可能出现）：记为 ambiguous。
δ1 同样报告，不判读。结果只用来决定下一步攻哪一段，不改任何门槛。

**结果（2026-11-17，作业 9489965，1426 帧）：** 对照全部精确（逐帧与存档差 0；各格之和对评估器 3e-16）。
抽样帧的 AbsRel 差 0.00343（全量 val 0.0034），有代表性。

| 全部帧 | 点数 | AbsRel 差 | 占比 | δ1 差（正 = 我们差） |
|---|---|---|---|---|
| 0–10 m | 29.9% | +0.00242 | 70.5% | +0.00134 |
| 10–20 m | 42.7% | +0.00093 | 27.2% | −0.00040 |
| 20–40 m | 22.0% | −0.00034 | −9.9% | −0.00150 |
| 40–80 m | 5.5% | +0.00042 | 12.1% | +0.00101 |

- 按规则：**`near_0_10`**。0–20 m 合计 98%；20–40 m 我们 AbsRel 和 δ1 都领先。
- 分序列：16-08-46 判 `mid_10_40`（10–20 m 49%、0–10 m 34%）；21-58-13 `spread`（0–10 m 44%、10–20 m 42%）；16-14-48 `near_0_10`（10–40 m 我们更好）。
- AbsRel 和 RMSE 输在不同地方：RMSE 差额主要在 10–80 m（泛化），AbsRel 在 0–20 m（训练帧上 0–10 m 也输，精度上限）。

## 近端 2% 截断的代价（2026-11-17 预登记，CPU，只用训练目标）

假设：`log_truncnorm` 的近端界是每帧 log 深度的 2% 分位，更近的像素在目标里都被压成同一个深度，学准了目标的模型
在这些点上偏远。远端已由 tmax1 改到 100%，近端从没动过；上方 0–10 m 的 log 偏差 synth +0.182 / NeWCRF +0.115，方向吻合。

做法：`tools/check_near_truncation.py`（`slurm/check_near_truncation.sbatch`），训练集 2000 帧（seed 20261014，同远端核对）。
用训练数据集类本身构建的目标（`depth_values`、`norm_bounds`）还原成米，在真实 LiDAR 点上算「目标本身的 AbsRel 代价」：
每帧对有效 LiDAR 点平均 |目标 − LiDAR| / LiDAR，再对帧平均，按 GT 段拆开（各段相加等于总值，单位与 AbsRel 差距相同）。
同一稠密图上另算 q = 0.01 / 0.005 / 0 的代价，用来给修法定量。
- 对照：本工具算的 q=0.02 下界等于数据集的 `norm_bounds[0]`（≤ 1e-5）；公式算的代价等于数据集自己截断后目标的代价（每帧 ≤ 1e-5）。

判读（看结果前定，看 q=0.02 的 0–10 m 代价；val 上 0–10 m 的 AbsRel 差距是 0.00242）：
- ≥ 0.0010：**material**，截断是近处差距的一个实质机制。下一步开一个近端分位臂：q 取 0.01、0.005 中较大的、
  且 0–10 m 代价不超过 q=0.02 的 20% 的那个（工具的 `suggested_truncnorm_min`）。q=0 只报告不选：下界会变成帧最小值，
  就是 instnorm 那种单个离群点定整帧刻度的失败。
- ≤ 0.0003：**excluded**，排除这个机制，不开臂。
- 之间：**partial**，如实记录，由用户决定。
限制：是训练帧上目标本身的代价，不是模型在 val 上的误差；模型可能学不到这么准，也可能在别处补回。
