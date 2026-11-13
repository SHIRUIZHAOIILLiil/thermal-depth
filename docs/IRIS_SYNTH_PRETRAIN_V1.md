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
