# 推理与 caption 的计算开销（2026-10-04，本地实测）

## 一句话

同一张 RTX 5090，batch 1，256×640：**深度推理 68 ms/帧，生成一条 caption 1.33 s/帧**，
caption 约为深度推理的 20 倍。带 caption 的模型推理时每帧都要先生成 caption，达不到实时。

## 环境

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090（32 GB），驱动 581.80 |
| 环境 | WSL Ubuntu，conda `wsl-pytorch`：torch 2.7.0+cu128、transformers 4.40.1、diffusers 0.28.0 |
| 权重 | 本地 HF 缓存 `E:\AI_Cache\huggingface`（`HF_HUB_OFFLINE=1`） |
| 数据 | MS2 测试序列 `2021-08-06-11-23-45` 的 16 位热像，等间隔取帧 |

## 一、深度推理（`tools/bench_depth_inference.py`）

按评估时 `train_route_suite.RouteModel.predict_disparity` 的实际配置：
CLIP 文本编码与 VAE 编码 fp16（`--frozen-dtype fp16`，所有已报数字的默认值）、
U-Net fp32（评估时被拷成 fp32）、t=999 一步、带任务嵌入、VAE 解码 fp16 autocast、通道平均。
预热 10 帧，计时 100 帧，CUDA event 计时。

| 部件 | 均值 ms | 中位 ms |
|---|---|---|
| CLIP 文本编码 | 9.35 | 9.59 |
| VAE 编码 | 8.47 | 8.46 |
| U-Net（一步，fp32） | 27.70 | 27.69 |
| VAE 解码 | 22.36 | 22.35 |
| **GPU 合计** | **67.89** | **68.07**（14.7 帧/秒） |
| CPU 预处理（读 16 位图 + 百分位拉伸） | 25.99 | 25.96 |

显存峰值 4.59 GiB。

- U-Net 用公开的 Lotus-G 权重（`jingheya/lotus-depth-g-v2-1-disparity`）：与我们的 checkpoint
  结构、张量形状完全相同，只是参数值不同，所以计算量与耗时相同。
- 不带 caption 的模型用空 prompt，文本编码可以缓存，GPU 部分约 58 ms。
- 带固定 caption 时文本编码也可缓存；逐帧 caption 则每帧都要算。

## 二、caption 生成（`tools/bench_caption.py`）

直接调用生成那 28,302 条 v3_1 caption 的类
（`E:\project\captioning\scripts\generate_captions.py` 的 `InternVLCaptioner`），设置同集群：
bf16（dtype auto）、贪心解码、`max_new_tokens=160`、最多 12 块 448 切片 + 缩略图、
热像按灰度渲染（同一条 16 位转换）、prompt `thermal_depth_v3_1`。预热 3 帧，计时 30 帧。

| | |
|---|---|
| 每帧生成耗时 | **均值 1.33 s，中位 1.32 s**（最短 1.18，最长 1.72） |
| 生成长度 | 均值 47.8 token，中位 47 |
| 速度 | 35.9 token/s |
| 图像准备（读图 + 灰度渲染） | 29.3 ms |
| 显存峰值 | 18.04 GiB |
| 模型加载（一次性） | 73.9 s |

### 与集群上实际 caption 的一致性核查

- 帧 `11-23-45_000528`：本地生成的句子与清单里的 caption **前半句逐字相同**，本地版在
  「lining both sides」处结束，清单版多出「and a bicycle partially visible behind a pole」。
  贪心解码在句尾走了不同分支，原因可能是 transformers 版本（本地 4.40，集群 ≥ 4.49）或
  bf16 在两种 GPU（5090 / L40S）上的细微数值差异。
- 长度：同一批 30 帧，清单 caption 均值 **48.8 token**（InternVL 分词）；整个白天测试集
  每 50 帧抽一条（467 条）均值 **49.1**。本地 47.8，差约 1 token（≈ 0.03 s）→ 计时有代表性。

## 写进论文时的注意事项

1. **写明 GPU 型号**（RTX 5090）。Iris 论文的 3.6 s 是 LLaVA v1.6 在 RTX 3090 上测的，
   模型与显卡都不同，不能直接比。
2. **caption 是未优化的实现**：普通 HF `generate`，未开 flash-attention / vLLM，可写
   「unoptimised」。
3. **实时性限制**：带 caption 的模型每帧都要先生成 caption（1.3 s），远离实时；不带 caption
   的模型约 68 ms，代价是 AbsRel 差 0.10–0.13 个百分点（主表官方子集，seed 43）。
   适合和 Limitations 一起写。
4. 深度的 68 ms 只是 GPU 部分；CPU 预处理另 26 ms，可作脚注。

参考句（英文，供改写）：

> On one RTX 5090 at batch 1 and 256×640, depth inference takes 68 ms per frame
> (text encoding, VAE encoding, one U-Net step and VAE decoding), while generating a
> caption with InternVL3-8B takes 1.3 s with an unoptimised implementation (about 48
> tokens). Captioning therefore dominates the cost of the captioned model by a factor of
> about twenty and keeps it far from real time; the caption-free model avoids it at a
> cost of 0.10–0.13 AbsRel points.

## 复现（WSL）

```bash
cd /mnt/e/project/Iris && HF_HOME=/mnt/e/AI_Cache/huggingface HF_HUB_OFFLINE=1 /home/dawn/miniconda3/envs/wsl-pytorch/bin/python tools/bench_depth_inference.py --frames-dir /mnt/e/dataset/ms2/sync_data/_2021-08-06-11-23-45/thr/img_left
```

```bash
cd /mnt/e/project/Iris && HF_HOME=/mnt/e/AI_Cache/huggingface HF_HUB_OFFLINE=1 /home/dawn/miniconda3/envs/wsl-pytorch/bin/python tools/bench_caption.py --frames-dir /mnt/e/dataset/ms2/sync_data/_2021-08-06-11-23-45/thr/img_left
```

`bench_depth_inference.py` 另有 `--unet-dtype fp16`，只用来看半精度能省多少，不代表评估配置。
