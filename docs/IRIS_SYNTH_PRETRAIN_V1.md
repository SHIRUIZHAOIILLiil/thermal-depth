# Iris 合成数据预训练线 V1（2026-11-09，第一段：数据和 caption）

## 想法

Iris 从 SD2 出发，在合成 RGB 数据上训练：Hypersim（室内，90% 的步数）+ Virtual KITTI 2
（行车，10%），渲染的稠密 GT，InternVL3-8B 生成的 caption（`train_iris_g.py:1009-1011`）。

这条线先按 Iris 的配方在合成数据上训一遍，再用我们的 MS2 配方（tmax1）微调。理由：诊断显示
差距主要是新场景上的泛化方差（`docs/DIAGNOSIS_TO_STRONG_AUG_20261107.md`；留出格子臂：训练帧
空白处 1.138 远胜目标 1.566，val 上 1.985 反而不如目标 1.904）。合成数据带来大量不同场景。

**事先说明的不确定性**：Lotus-G 就是"SD2 + 同样两套数据、空文本"，8 月从它起步的旧配方臂 RMSE
没有占优（3.43/3.31/4.13，视差空间旧配方，不能直接和 tmax1 比）。所以合成预训练能不能救 RMSE，
现在没有答案。

## 第一段（本文件）：数据和 caption

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

## 第二段（还没写，等数据到位）

- Iris 合成预训练作业：`train_iris_g.py` 的校验依赖 NYU/KITTI 等评估集（`BASE_TEST_DATA_DIR`），要么备齐，要么关掉。
- MS2 微调：tmax1 配方，起点换成合成预训练的 checkpoint，判定门槛与其他臂相同
  （全量 val step 16000，对比 tmax1：RMSE 至少降 10 cm，AbsRel 增加不超过 0.002，δ1 下降不超过 0.003）。

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
