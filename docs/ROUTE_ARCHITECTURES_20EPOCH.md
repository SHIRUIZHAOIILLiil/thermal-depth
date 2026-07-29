# 四条路线的完整网络结构与参数量

由 `tools/dump_route_architecture.py` 自动生成。参数量为 fp32 权重的元素个数，「状态」列即训练时 `requires_grad` 的取值。

## 0. 六条线总览

| 线 | 输入 | Condition 来源 | 可训练模块 | 可训练参数 | 冻结参数 | 可训练占比 |
|---|---|---|---|---:|---:|---:|
| a_rgb_unet | RGB | 冻结 VAE latent | U-Net | 867.57 M | 424.04 M | 67.2% |
| b_thermal_unet | Thermal | 冻结 VAE latent | U-Net | 867.57 M | 424.04 M | 67.2% |
| c1_vae_adapter | Thermal | VAE latent + Adapter | Adapter (VAE latent) | 7.11 M | 1.29 B | 0.5% |
| c2_vae_adapter_unet | Thermal | VAE latent + Adapter | Adapter (VAE latent) + U-Net | 874.67 M | 424.04 M | 67.3% |
| d1_anythermal_adapter | Thermal | AnyThermal 特征 + Adapter | Adapter (AnyThermal→latent) | 9.41 M | 1.34 B | 0.7% |
| d2_anythermal_adapter_unet | Thermal | AnyThermal 特征 + Adapter | Adapter (AnyThermal→latent) + U-Net | 876.98 M | 476.46 M | 64.8% |

## 1. 线 a — RGB 输入，只训 U-Net（Baseline）

### 前向顺序与冻结状态

| # | 模块 | 类 | 参数量 | 状态 | 作用 |
|---:|---|---|---:|---|---|
| 1 | VAE encoder | `Encoder` | 34,163,592 | 冻结 | 把输入图编码成 4 通道 latent，作为 U-Net 的 condition |
| 2 | VAE quant_conv | `Conv2d` | 72 | 冻结 | latent 量化卷积（encoder 的一部分） |
| 3 | Text encoder | `CLIPTextModel` | 340,387,840 | 冻结 | 把 prompt（无 caption 阶段恒为空串）编码成 cross-attention 条件 |
| 4 | U-Net | `UNet2DConditionModel` | 867,568,324 | **训练** | Lotus 主干，单步 x0 预测 |
| 5 | VAE post_quant_conv | `Conv2d` | 20 | 冻结 | latent 量化卷积（encoder 的一部分） |
| 6 | VAE decoder | `Decoder` | 49,490,179 | 冻结 | 把 U-Net 输出的 latent 解回视差图（GT 损失在这之后算） |
| | **合计** | | **1,291,610,027** | 训练 867.57 M / 冻结 424.04 M | |

### 可训练模块内部结构

**U-Net** (`UNet2DConditionModel`, 867.57 M)

| 子模块 | 类 | 参数量 |
|---|---|---:|
| `conv_in` | `Conv2d` | 23,360 |
| `time_embedding` | `TimestepEmbedding` | 2,050,560 |
| `class_embedding` | `TimestepEmbedding` | 1,646,080 |
| `down_blocks` | `ModuleList` | 251,906,240 |
| `up_blocks` | `ModuleList` | 514,236,480 |
| `mid_block` | `UNetMidBlock2DCrossAttn` | 97,693,440 |
| `conv_norm_out` | `GroupNorm` | 640 |
| `conv_out` | `Conv2d` | 11,524 |

## 2. 线 b — Thermal 输入，只训 U-Net（Baseline）

### 前向顺序与冻结状态

| # | 模块 | 类 | 参数量 | 状态 | 作用 |
|---:|---|---|---:|---|---|
| 1 | VAE encoder | `Encoder` | 34,163,592 | 冻结 | 把输入图编码成 4 通道 latent，作为 U-Net 的 condition |
| 2 | VAE quant_conv | `Conv2d` | 72 | 冻结 | latent 量化卷积（encoder 的一部分） |
| 3 | Text encoder | `CLIPTextModel` | 340,387,840 | 冻结 | 把 prompt（无 caption 阶段恒为空串）编码成 cross-attention 条件 |
| 4 | U-Net | `UNet2DConditionModel` | 867,568,324 | **训练** | Lotus 主干，单步 x0 预测 |
| 5 | VAE post_quant_conv | `Conv2d` | 20 | 冻结 | latent 量化卷积（encoder 的一部分） |
| 6 | VAE decoder | `Decoder` | 49,490,179 | 冻结 | 把 U-Net 输出的 latent 解回视差图（GT 损失在这之后算） |
| | **合计** | | **1,291,610,027** | 训练 867.57 M / 冻结 424.04 M | |

### 可训练模块内部结构

**U-Net** (`UNet2DConditionModel`, 867.57 M)

| 子模块 | 类 | 参数量 |
|---|---|---:|
| `conv_in` | `Conv2d` | 23,360 |
| `time_embedding` | `TimestepEmbedding` | 2,050,560 |
| `class_embedding` | `TimestepEmbedding` | 1,646,080 |
| `down_blocks` | `ModuleList` | 251,906,240 |
| `up_blocks` | `ModuleList` | 514,236,480 |
| `mid_block` | `UNetMidBlock2DCrossAttn` | 97,693,440 |
| `conv_norm_out` | `GroupNorm` | 640 |
| `conv_out` | `Conv2d` | 11,524 |

## 3. 线 c1 — Thermal，VAE 后接 Adapter，U-Net 冻结

### 前向顺序与冻结状态

| # | 模块 | 类 | 参数量 | 状态 | 作用 |
|---:|---|---|---:|---|---|
| 1 | VAE encoder | `Encoder` | 34,163,592 | 冻结 | 把输入图编码成 4 通道 latent，作为 U-Net 的 condition |
| 2 | VAE quant_conv | `Conv2d` | 72 | 冻结 | latent 量化卷积（encoder 的一部分） |
| 3 | Adapter (VAE latent) | `ThermalVAELatentAdapter` | 7,106,308 | **训练** | 在冻结 VAE latent 上的残差 CNN，零初始化 ⇒ 未训练时是恒等 |
| 4 | Text encoder | `CLIPTextModel` | 340,387,840 | 冻结 | 把 prompt（无 caption 阶段恒为空串）编码成 cross-attention 条件 |
| 5 | U-Net | `UNet2DConditionModel` | 867,568,324 | 冻结 | Lotus 主干，单步 x0 预测 |
| 6 | VAE post_quant_conv | `Conv2d` | 20 | 冻结 | latent 量化卷积（encoder 的一部分） |
| 7 | VAE decoder | `Decoder` | 49,490,179 | 冻结 | 把 U-Net 输出的 latent 解回视差图（GT 损失在这之后算） |
| | **合计** | | **1,298,716,335** | 训练 7.11 M / 冻结 1.29 B | |

### 可训练模块内部结构

**Adapter (VAE latent)** (`ThermalVAELatentAdapter`, 7.11 M)

| 子模块 | 类 | 参数量 |
|---|---|---:|
| `conv_in` | `Conv2d` | 9,472 |
| `blocks` | `ModuleList` | 7,087,104 |
| `norm_out` | `GroupNorm` | 512 |
| `conv_out` | `Conv2d` | 9,220 |

## 4. 线 c2 — Thermal，VAE 后接 Adapter，Adapter + U-Net 联合训练

### 前向顺序与冻结状态

| # | 模块 | 类 | 参数量 | 状态 | 作用 |
|---:|---|---|---:|---|---|
| 1 | VAE encoder | `Encoder` | 34,163,592 | 冻结 | 把输入图编码成 4 通道 latent，作为 U-Net 的 condition |
| 2 | VAE quant_conv | `Conv2d` | 72 | 冻结 | latent 量化卷积（encoder 的一部分） |
| 3 | Adapter (VAE latent) | `ThermalVAELatentAdapter` | 7,106,308 | **训练** | 在冻结 VAE latent 上的残差 CNN，零初始化 ⇒ 未训练时是恒等 |
| 4 | Text encoder | `CLIPTextModel` | 340,387,840 | 冻结 | 把 prompt（无 caption 阶段恒为空串）编码成 cross-attention 条件 |
| 5 | U-Net | `UNet2DConditionModel` | 867,568,324 | **训练** | Lotus 主干，单步 x0 预测 |
| 6 | VAE post_quant_conv | `Conv2d` | 20 | 冻结 | latent 量化卷积（encoder 的一部分） |
| 7 | VAE decoder | `Decoder` | 49,490,179 | 冻结 | 把 U-Net 输出的 latent 解回视差图（GT 损失在这之后算） |
| | **合计** | | **1,298,716,335** | 训练 874.67 M / 冻结 424.04 M | |

### 可训练模块内部结构

**Adapter (VAE latent)** (`ThermalVAELatentAdapter`, 7.11 M)

| 子模块 | 类 | 参数量 |
|---|---|---:|
| `conv_in` | `Conv2d` | 9,472 |
| `blocks` | `ModuleList` | 7,087,104 |
| `norm_out` | `GroupNorm` | 512 |
| `conv_out` | `Conv2d` | 9,220 |

**U-Net** (`UNet2DConditionModel`, 867.57 M)

| 子模块 | 类 | 参数量 |
|---|---|---:|
| `conv_in` | `Conv2d` | 23,360 |
| `time_embedding` | `TimestepEmbedding` | 2,050,560 |
| `class_embedding` | `TimestepEmbedding` | 1,646,080 |
| `down_blocks` | `ModuleList` | 251,906,240 |
| `up_blocks` | `ModuleList` | 514,236,480 |
| `mid_block` | `UNetMidBlock2DCrossAttn` | 97,693,440 |
| `conv_norm_out` | `GroupNorm` | 640 |
| `conv_out` | `Conv2d` | 11,524 |

## 5. 线 d1 — Thermal → AnyThermal → Adapter，U-Net 冻结

### 前向顺序与冻结状态

| # | 模块 | 类 | 参数量 | 状态 | 作用 |
|---:|---|---|---:|---|---|
| 1 | AnyThermal encoder | `Dinov2Model` | 86,580,480 | 冻结 | AnyThermal 热像基础模型（DINOv2 主干），提供热像特征 |
| 2 | Adapter (AnyThermal→latent) | `AnyThermalLotusAdapterV23` | 9,408,012 | **训练** | 把 AnyThermal token 转成 U-Net 可读的 4 通道 latent |
| 3 | Text encoder | `CLIPTextModel` | 340,387,840 | 冻结 | 把 prompt（无 caption 阶段恒为空串）编码成 cross-attention 条件 |
| 4 | U-Net | `UNet2DConditionModel` | 867,568,324 | 冻结 | Lotus 主干，单步 x0 预测 |
| 5 | VAE post_quant_conv | `Conv2d` | 20 | 冻结 | latent 量化卷积（encoder 的一部分） |
| 6 | VAE decoder | `Decoder` | 49,490,179 | 冻结 | 把 U-Net 输出的 latent 解回视差图（GT 损失在这之后算） |
| | **合计** | | **1,353,434,855** | 训练 9.41 M / 冻结 1.34 B | |

### 可训练模块内部结构

**Adapter (AnyThermal→latent)** (`AnyThermalLotusAdapterV23`, 9.41 M)

| 子模块 | 类 | 参数量 |
|---|---|---:|
| `lateral_projections` | `ModuleList` | 590,592 |
| `native_fusions` | `ModuleList` | 3,985,344 |
| `stage_one` | `ProgressiveSkipStage` | 1,328,448 |
| `stage_two` | `ProgressiveSkipStage` | 1,328,448 |
| `detail_encoder` | `Sequential` | 814,080 |
| `semantic_detail_fusion` | `Sequential` | 1,328,448 |
| `to_residual` | `Conv2d` | 6,916 |
| `affine_head` | `Sequential` | 25,736 |

## 6. 线 d2 — Thermal → AnyThermal → Adapter，Adapter + U-Net 联合训练

### 前向顺序与冻结状态

| # | 模块 | 类 | 参数量 | 状态 | 作用 |
|---:|---|---|---:|---|---|
| 1 | AnyThermal encoder | `Dinov2Model` | 86,580,480 | 冻结 | AnyThermal 热像基础模型（DINOv2 主干），提供热像特征 |
| 2 | Adapter (AnyThermal→latent) | `AnyThermalLotusAdapterV23` | 9,408,012 | **训练** | 把 AnyThermal token 转成 U-Net 可读的 4 通道 latent |
| 3 | Text encoder | `CLIPTextModel` | 340,387,840 | 冻结 | 把 prompt（无 caption 阶段恒为空串）编码成 cross-attention 条件 |
| 4 | U-Net | `UNet2DConditionModel` | 867,568,324 | **训练** | Lotus 主干，单步 x0 预测 |
| 5 | VAE post_quant_conv | `Conv2d` | 20 | 冻结 | latent 量化卷积（encoder 的一部分） |
| 6 | VAE decoder | `Decoder` | 49,490,179 | 冻结 | 把 U-Net 输出的 latent 解回视差图（GT 损失在这之后算） |
| | **合计** | | **1,353,434,855** | 训练 876.98 M / 冻结 476.46 M | |

### 可训练模块内部结构

**Adapter (AnyThermal→latent)** (`AnyThermalLotusAdapterV23`, 9.41 M)

| 子模块 | 类 | 参数量 |
|---|---|---:|
| `lateral_projections` | `ModuleList` | 590,592 |
| `native_fusions` | `ModuleList` | 3,985,344 |
| `stage_one` | `ProgressiveSkipStage` | 1,328,448 |
| `stage_two` | `ProgressiveSkipStage` | 1,328,448 |
| `detail_encoder` | `Sequential` | 814,080 |
| `semantic_detail_fusion` | `Sequential` | 1,328,448 |
| `to_residual` | `Conv2d` | 6,916 |
| `affine_head` | `Sequential` | 25,736 |

**U-Net** (`UNet2DConditionModel`, 867.57 M)

| 子模块 | 类 | 参数量 |
|---|---|---:|
| `conv_in` | `Conv2d` | 23,360 |
| `time_embedding` | `TimestepEmbedding` | 2,050,560 |
| `class_embedding` | `TimestepEmbedding` | 1,646,080 |
| `down_blocks` | `ModuleList` | 251,906,240 |
| `up_blocks` | `ModuleList` | 514,236,480 |
| `mid_block` | `UNetMidBlock2DCrossAttn` | 97,693,440 |
| `conv_norm_out` | `GroupNorm` | 640 |
| `conv_out` | `Conv2d` | 11,524 |

