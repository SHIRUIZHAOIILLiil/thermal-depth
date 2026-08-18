# Implementation audit: language-conditioned thermal monocular depth

**Audit date:** 2026-08-17  
**Scope:** the current Iris-on-MS2 line, with the older sparse-GT route suite and the separate final-paper evaluator documented only where needed to prevent protocol conflation.  
**Research question:** Does language/caption conditioning improve diffusion-based monocular depth estimation when the visual input is thermal rather than RGB?

## Executive finding

The current paper line is **not** an AnyThermal-adapter network. It is an Iris/Lotus-G latent-diffusion model whose visual condition is a three-channel replication of a per-frame-normalized thermal image. A frozen Stable-Diffusion/Lotus VAE encodes that condition; a frozen CLIP text encoder produces token embeddings; the full conditional U-Net is trained. The AnyThermal MiDaS model appears only offline, as a pseudo-depth teacher used to complete sparse MS2 LiDAR targets.

The training objective is the sum of two clean-latent MSE terms: one predicts the VAE latent of normalized completed disparity, conditioned on the frame caption, and one reconstructs the thermal input latent with an empty caption. The implemented diffusion target is `sample` (clean latent, or \(x_0\)), at one fixed timestep \(t=999\); timesteps are not sampled.

The repository's existing 2x2 caption study does **not** establish a material language-content gain. The caption-trained model improves when evaluated with the conditioning form it saw during training, but shuffled captions retain about 95% of the reported effect, and the matched caption model is essentially tied with the no-caption baseline on AbsRel and \(\delta_1\). This is the repository's own stated interpretation, not a new inference: `docs/RESULTS_20260813_IRIS_MS2.md`, section “完整 2×2”, lines 83-115. Numerically, matched caption vs no-caption is AbsRel 0.08530 vs 0.08551, RMSE 3.613 vs 3.637, and \(\delta_1\) 0.9231 vs 0.9227 (`docs/RESULTS_20260813_IRIS_MS2.md`, lines 65-75). These numbers show a small conditioning-consistency effect; they do not isolate semantic language understanding.

### Audit-status vocabulary

- **VERIFIED** means the conclusion follows directly from an identified implementation or serialized configuration.
- **UNKNOWN** means the repository does not freeze or record enough information. No replacement assumption is made.
- The “current paper line” is `lotus/train_iris_ms2_g.py`, launched through `slurm/iris_ms2.sbatch` and `lotus/train_scripts/train_iris_ms2_g_depth.sh`. The older `tools/train_route_suite.py` is a distinct sparse-GT/ablation line (`slurm/iris_ms2.sbatch`, lines 11-13; `docs/RESULTS_20260813_IRIS_MS2.md`, lines 63-72).

## 1. Complete thermal input pipeline

### 1.1 Current Iris-MS2 training and inference condition

| Stage | Implemented operation | Exact evidence |
|---|---|---|
| Manifest | Each JSONL row must provide `id`, `thermal_path`, a thermal-view `thermal_depth_path`/`depth_path`, and may provide `caption`. A row that appears to pair RGB-view GT with thermal input is rejected. | `lotus/utils/ms2_thermal_dataset.py::MS2ThermalDataset.__init__`, lines 64-123, especially 98-116. |
| Raw decode | The thermal file is opened and converted to a NumPy array before any RGB conversion. It must be finite and two-dimensional. | `models/anythermal_lotus_v2.py::_read_thermal_array`, lines 91-100; `thermal_to_lotus_input`, lines 102-130. |
| High-bit-depth conversion | `uint8` is unchanged. Other types are cast to float; a true `[0,1]` image is multiplied by 255; values outside `[0,255]` are min-max normalized per frame to `[0,255]`; values are clipped, rounded, and cast to `uint8`. Thus an MS2 `uint16` frame is normalized from its own raw minimum and maximum. | `models/anythermal_encoder.py::AnyThermalEncoder._array_to_uint8`, lines 400-420. |
| Saturation guard | Conversion fails for a constant raw frame and for an all-white converted frame; min/max/mean/std diagnostics are recorded. The dataset additionally rejects converted standard deviation \(\le 0\). | `models/anythermal_lotus_v2.py::thermal_to_lotus_input`, lines 117-130 and 150-172; `lotus/utils/ms2_thermal_dataset.py::MS2ThermalDataset.__getitem__`, lines 147-155. |
| Channel and numerical mapping | The single `uint8` channel is repeated three times, converted to float, and mapped by `x/127.5 - 1` to three channels in `[-1,1]`. There is no learned thermal encoder in this path. | `models/anythermal_lotus_v2.py::thermal_to_lotus_input`, lines 132-135. |
| Spatial resolution | Training calls `processing_res=0`, retaining native 640x256; the optional branch would bilinearly resize only when a positive resolution is requested. | `lotus/utils/ms2_thermal_dataset.py::MS2ThermalDataset.__getitem__`, lines 151-155; `models/anythermal_lotus_v2.py::thermal_to_lotus_input`, lines 137-148. |
| Augmentation | If enabled, image and completed depth are horizontally flipped together. The caption is not rewritten, so left/right statements can become false. The active launcher passes `--random_flip`. | `lotus/utils/ms2_thermal_dataset.py::MS2ThermalTransform`, lines 49-61; module note lines 25-28; `lotus/train_scripts/train_iris_ms2_g_depth.sh`, lines 46-55. |
| Batch | Images, completed/normalized depth, valid/sky masks and caption strings are stacked or collected without further visual normalization. | `lotus/utils/ms2_thermal_dataset.py::collate_fn_ms2`, lines 216-247. |
| VAE condition | The frozen `AutoencoderKL` encodes the replicated thermal tensor. Training draws `latent_dist.sample()` and multiplies by `vae.config.scaling_factor`. The depth and reconstruction halves receive duplicated condition latents. | `lotus/train_iris_ms2_g.py::main` training loop, lines 1041-1045. |
| Paper-number inference | The evaluation wrapper loads the same thermal tensor, but `--condition-latent mode` selects the posterior mode, not a VAE sample. | `tools/train_route_suite.py::load_input_tensor`, lines 754-773; `RouteModel._condition_latent`, lines 941-955; `slurm/iris_ms2_pipeline.sbatch::run_eval`, lines 78-92. |

This yields the actual online condition path

\[
I^{16}_{\mathrm{th}} \rightarrow \operatorname{minmax}_{\text{frame}}(I_{\mathrm{th}})\in[0,255]
\rightarrow \operatorname{repeat}_3 \rightarrow [-1,1]
\rightarrow E_{\mathrm{VAE}}(I_{\mathrm{th}})=z_I.
\]

**Caption provenance caveat.** The handoff identifies the active system as thermal input plus **RGB-derived captions** (`docs/ADAPTER_V2_HANDOFF_20260703.md`, lines 229-230), and the evaluation launcher explicitly labels its default manifests “RGB caption” (`slurm/iris_ms2_pipeline.sbatch`, lines 72-75). The repository documents InternVL3-8B as the captioner (`README.md`, lines 112-120), and the manifest directory/name also records `internvl3_8b`/`rgb_depth_v1_clip75` (`slurm/iris_ms2.sbatch`, lines 31-34 and 82-86). The training code, however, merely reads the manifest's `caption` string and does not validate those claims (`lotus/utils/ms2_thermal_dataset.py::MS2ThermalDataset.__init__`, lines 98-116). The exact InternVL checkpoint revision, prompt version, sampling seed/configuration, and source-image hash for each active manifest are therefore **UNKNOWN from the frozen training invocation**.

### 1.2 Offline AnyThermal pseudo-depth input path (not the online network)

The pseudo teacher has a different thermal preprocessing path:

1. OpenCV reads the image unchanged, it is cast to float, and the first channel is selected if necessary.
2. Native 256x640 is resized to 252x630 so both dimensions are multiples of ViT patch size 14.
3. Values are divided by `raw_divisor`, clipped at the 1st/99th percentiles, min-max normalized, standardized with configured mean/std, and repeated to three channels.
4. A frozen AnyThermal MiDaS/DPT network predicts one raw relative-depth map, which is saved as `.npy`.

Evidence: `tools/run_ms2_anythermal_midas.py::tensor_iwmm`, lines 104-113; `preprocess_thermal`, lines 116-135; `main`, lines 163-221. This preprocessing is **not** called by `lotus/train_iris_ms2_g.py`.

## 2. Network architecture actually used

### 2.1 Current paper model

The active default launch starts from `jingheya/lotus-depth-g-v2-1-disparity`, not SD2-base: `slurm/iris_ms2.sbatch`, lines 55-59. The shell recipe supports both, but the SLURM default is `START=ckpt`; `lotus/train_scripts/train_iris_ms2_g_depth.sh`, lines 1-16, describes the two alternatives.

The online model is:

1. **Frozen VAE:** Stable-Diffusion `AutoencoderKL`, four down/up stages with channels 128/256/512/512, four latent channels, scale factor 0.18215.
2. **Frozen text encoder:** CLIP transformer, vocabulary 49,408, maximum 77 positions, 23 layers, 16 attention heads, hidden width 1,024.
3. **Trainable conditional U-Net:** `UNet2DConditionModel`, eight input channels and four output channels. The input is `[z_I ; z_t]`: four thermal-condition latent channels concatenated with four noised target latent channels. Block widths are 320/640/1280/1280; there are two ResNet/attention layers per block; down blocks are three cross-attention blocks then one plain block; the middle is cross-attentional; up blocks are one plain block then three cross-attention blocks. `cross_attention_dim=1024`; there is one transformer layer per attention block. A four-dimensional projected class embedding carries the task identifier.
4. **Output:** four clean target-latent channels, decoded by the frozen VAE decoder to a three-channel image; channel mean is used as normalized disparity in route-suite evaluation.

Construction is verified in `lotus/train_iris_ms2_g.py::main`, lines 716-790. The U-Net, VAE, and CLIP serialized dimensions are in the active cached checkpoint:

- `E:/AI_Cache/huggingface/hub/models--jingheya--lotus-depth-g-v2-1-disparity/snapshots/5a6f234319ae90f4bdf1b2d326a575e2715eb40d/unet/config.json`, lines 9-20, 23-34, 42-53, 64-72;
- the sibling `text_encoder/config.json`, lines 11-19;
- the sibling `vae/config.json`, complete file;
- the sibling `scheduler/scheduler_config.json`, lines 2-18.

The repository's module audit reports approximately 867.57M trainable U-Net parameters and 424.04M frozen parameters (VAE encoder/quantizer, CLIP, and VAE decoder): `docs/ROUTE_ARCHITECTURES_20EPOCH.md`, lines 45-72. This is the same thermal-U-Net route used after Iris-MS2 checkpoint conversion; it contains no online `AnyThermalEncoder` and no adapter.

For scoring, `tools/convert_iris_ms2_checkpoint.py::main`, lines 157-195, exports only the U-Net state dictionary as route `b_thermal_unet`. Before export, `check_against_reference`, lines 118-143, verifies every key/shape against the Lotus checkpoint and requires an eight-channel input convolution. The evaluator reconstructs the frozen VAE and CLIP from the Lotus base and overlays that U-Net. Conversion does not adapt or rename weights (`tools/convert_iris_ms2_checkpoint.py`, lines 1-16).

When a four-channel SD2 U-Net is selected, its input convolution is expanded 4→8 by duplicating weights and multiplying them by 0.5. When the active Lotus checkpoint already exposes eight channels, expansion is skipped: `lotus/train_iris_ms2_g.py::main`, lines 756-790.

### 2.2 Offline pseudo teacher

The frozen teacher is DINOv2 ViT-B/14 plus a MiDaS/DPT head: taps at transformer blocks 2/5/8/11, feature widths 96/192/384/768, refinement/fusion blocks, and a one-channel output head. Its checkpoint is strict-loaded and the model is set to eval/frozen. Evidence: `tools/build_anythermal_midas.py::build_anythermal_midas`, lines 105-240, especially 118-127, 143-179, 181-217, and 219-240. It creates supervision but is not part of the trained or deployed Iris-MS2 graph.

## 3. Frozen and trainable modules

### Current Iris-MS2 model

| Module | State | Evidence |
|---|---|---|
| `AutoencoderKL` encoder, quantizer and decoder | Frozen; moved to fp16 under the active launcher | `lotus/train_iris_ms2_g.py::main`, lines 742-748, 792-795, 914-926. |
| `CLIPTextModel` | Frozen; moved to fp16 under the active launcher | Same function, lines 742-748, 792-795, 914-926. |
| Entire `UNet2DConditionModel`, including self-attention, cross-attention Q/K/V/output projections, convolutional blocks, timestep embedding and task/class projection | Trainable | Same function, lines 750-795; optimizer receives exactly `unet.parameters()` at lines 849-868. |
| Tokenizer and DDPMScheduler | No learnable parameters | Same function, lines 716-721. |
| AnyThermal MiDaS/DPT teacher | Frozen offline; absent from the online graph | `tools/build_anythermal_midas.py::build_anythermal_midas`, lines 219-240; no import or construction in `lotus/train_iris_ms2_g.py`. |

The active optimizer is 8-bit AdamW, LR \(3\times 10^{-5}\), constant schedule, no warmup, max gradient norm 1, fp16, batch 4, accumulation 3, seed 42, and nominally 20,000 steps: `lotus/train_scripts/train_iris_ms2_g_depth.sh`, lines 24-32 and 46-70. These facts affect reproducibility but not module ownership.

### Distinct older route-suite ownership

Do not describe these routes as the current Iris-MS2 model. `tools/train_route_suite.py::RouteModel.__init__`, lines 853-927, freezes the original Lotus VAE/text encoder/U-Net, makes a separate U-Net copy trainable only for U-Net routes, optionally creates a trainable adapter, and keeps AnyThermal frozen. `RouteModel.trainable_modules`, lines 1067-1075, returns only the route-selected adapter and/or copied U-Net. Route definitions are at lines 92-100.

## 4. Where and how captions enter

1. The manifest caption is returned as `text_description`; `--no_captions` substitutes `""`: `lotus/utils/ms2_thermal_dataset.py::MS2ThermalDataset.__getitem__`, lines 147-199.
2. Every physical batch is doubled into a **depth annotation half** and a **thermal reconstruction half**. The depth half receives its frame caption; the reconstruction half always receives the empty string: `lotus/train_iris_ms2_g.py::main` training loop, lines 1041-1060 and 1105-1120.
3. `CLIPTokenizer` pads/truncates to its model maximum; frozen `CLIPTextModel(...)[0]` returns the tokenwise last hidden state: same loop, lines 1121-1148. Under the active checkpoint this is shaped \(B\times77\times1024\).
4. Those token embeddings are passed as the U-Net positional argument `encoder_hidden_states`: same loop, lines 1162-1164.
5. Separately, `class_labels` receives a four-vector task code: depth uses `[sin(1), sin(0), cos(1), cos(0)]`; reconstruction uses `[sin(0), sin(1), cos(0), cos(1)]`. This is task conditioning, **not** language conditioning: same loop, lines 1155-1164.

There is no caption dropout, shuffled caption, or classifier-free guidance branch in `train_iris_ms2_g.py`. The only current training comparison is correct manifest captions versus all-empty captions, selected by `--no_captions` (`lotus/train_scripts/train_iris_ms2_g_depth.sh`, lines 43-54). Shuffled/correct/empty prompts are evaluation ablations in `tools/train_route_suite.py`, not alternate operations inside the current trainer.

## 5. Exact cross-attention/text-conditioning mechanism

### Verified mechanism

Let \(H\in\mathbb{R}^{B\times N\times d}\) be a flattened, normalized U-Net spatial feature and \(C\in\mathbb{R}^{B\times77\times1024}\) the frozen CLIP token states. Each Diffusers cross-attention (`attn2`) computes learned projections

\[
Q=HW_Q,\qquad K=CW_K,\qquad V=CW_V,
\]

followed per head by

\[
A=\operatorname{softmax}\!\left(QK^\top/\sqrt{d_h}\right),\qquad
H' = H + \operatorname{proj}(AV).
\]

The query therefore comes from U-Net visual/target features; keys and values come from caption token embeddings. The residual update then continues through the transformer feed-forward block.

Evidence chain:

- The trainer supplies CLIP token states as `encoder_hidden_states`: `lotus/train_iris_ms2_g.py::main`, lines 1121-1164.
- The checkpoint fixes `cross_attention_dim=1024` and cross-attention down/mid/up block placement: active `unet/config.json`, lines 23-34, 46-53, 64-72, path given in §2.
- Diffusers v0.28.0 constructs `BasicTransformerBlock.attn2` with `query_dim=dim` and `cross_attention_dim=cross_attention_dim`, then calls it with normalized spatial hidden states and `encoder_hidden_states`, adding the result residually: `diffusers/models/attention.py::BasicTransformerBlock.__init__/forward` in [Diffusers v0.28.0](https://github.com/huggingface/diffusers/blob/v0.28.0/src/diffusers/models/attention.py), `attn2` construction and cross-attention sections.
- Its `Attention` layer defines `to_q` from `query_dim`, `to_k`/`to_v` from `cross_attention_dim`, scale \(d_h^{-1/2}\), and output projection: `diffusers/models/attention_processor.py::Attention.__init__`, lines 115-133 and 176-199 in [Diffusers v0.28.0](https://github.com/huggingface/diffusers/blob/v0.28.0/src/diffusers/models/attention_processor.py).
- The PyTorch-2 processor explicitly obtains queries from `hidden_states`, keys/values from `encoder_hidden_states`, applies scaled dot-product attention, and projects the result: the same external file, `AttnProcessor2_0.__call__`, lines 1226-1303.

The loaded U-Net exposes 16 named `attn2` sites (six in the first three down blocks, one middle, nine in the final three up blocks), corroborated by `outputs/route_suite/b_thermal_unet_20ep/attn_probe/cross_attention_probe.json`, lines 5-110. These sites are all part of the trainable U-Net.

### UNKNOWN runtime detail

The exact low-level attention kernel used by a historical run is **UNKNOWN**. Diffusers v0.28 selects `AttnProcessor2_0` when the installed PyTorch exposes scaled-dot-product attention, otherwise `AttnProcessor`; it can also be replaced by xFormers (`attention_processor.py::Attention.__init__`, lines 191-199). The trainer does not record the processor class or force xFormers. The mathematical Q/K/V mechanism and learned parameters above are fixed; Flash/memory-efficient kernel selection is not.

## 6. Diffusion formulation actually implemented

### Training

| Item | Actual implementation | Exact evidence |
|---|---|---|
| Scheduler | `DDPMScheduler.from_pretrained(.../scheduler)`, then overwrite `prediction_type` from CLI. Active checkpoint: 1,000 training steps, `scaled_linear`, beta start 0.00085, beta end 0.012, `clip_sample=false`, `set_alpha_to_one=false`, `steps_offset=1`. | `lotus/train_iris_ms2_g.py::main`, lines 716-718; active `scheduler/scheduler_config.json`, lines 2-18. |
| Timestep sampling | **No sampling.** Every item receives the scalar CLI timestep, repeated across the doubled batch. Active recipe sets \(t=999\). | Training loop, lines 1088-1090; `lotus/train_scripts/train_iris_ms2_g_depth.sh`, lines 30-32 and 65-67. |
| Clean target latent | Frozen VAE posterior sample of the normalized completed disparity image for the annotation half and of the thermal image for reconstruction; multiply by VAE scale. | Training loop, lines 1041-1057. |
| Noise | `torch.randn_like(target_latents)`. Optional offset and input perturbation branches exist but both defaults are 0 and the active launch does not override them. | `lotus/train_iris_ms2_g.py::parse_args`, lines 341 and 620; training loop, lines 1077-1097; active launcher lines 46-70. |
| Forward/noising process | `noise_scheduler.add_noise(z_0, ε, t)`. For the DDPMScheduler this is \(z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon\). | Training loop, lines 1092-1097; scheduler configuration above. |
| Model input | Concatenate clean thermal condition latent and noised target latent: \([z_I;z_t]\in\mathbb{R}^{8\times H/8\times W/8}\). | Training loop, lines 1099-1102. |
| `prediction_type` | `sample`, i.e. predict the clean latent \(z_0\), not noise \(\epsilon\), velocity \(v\), or score. | `lotus/train_iris_ms2_g.py::parse_args`, lines 552-555; scheduler registration lines 716-718; active checkpoint scheduler config. |
| Prediction target | Exactly `target_latents`; loss compares U-Net output to the clean latent. | Training loop, lines 1152-1169. |

There is therefore no expectation over randomly sampled timesteps in the implemented objective. It is single-noise-level clean-latent regression at the last training index.

### Inference

The generic `LotusGPipeline.__call__`:

1. encodes the prompt (`lotus/pipeline.py::LotusGPipeline.__call__`, lines 1272-1279);
2. uses explicitly supplied timesteps (lines 1281-1282);
3. initializes the target latent as pure Gaussian noise (`DirectDiffusionPipeline.prepare_latents`, lines 633-652; call at 1284-1295);
4. samples the VAE condition posterior (lines 1297-1298);
5. concatenates condition and current latent, runs the conditional U-Net (lines 1307-1321);
6. for one step, directly assigns the predicted clean latent; only a multi-step request calls `scheduler.step` (lines 1323-1327);
7. VAE-decodes and resizes to input resolution (lines 1333-1355).

The actual paper-number path is more specific: `slurm/iris_ms2_pipeline.sbatch::run_eval`, lines 78-92, invokes `tools/train_route_suite.py` with `--num-inference-steps 1 --condition-latent mode`. `tools/train_route_suite.py::RouteModel.predict_disparity`, lines 1000-1053, uses deterministic Gaussian noise seeded by global seed plus manifest index, performs one U-Net call at \(t=999\), takes that \(x_0\) directly, VAE-decodes it, averages decoded channels, and maps `[-1,1]` to `[0,1]`. There is no iterative denoising and no classifier-free guidance in this reported path.

The conversion payload records the source step and U-Net but sets `manifest_sha256` to `None`; its `caption_mode` is provenance only and is not read by the evaluator (`tools/convert_iris_ms2_checkpoint.py::main`, lines 173-193; `parse_args`, lines 50-64). Consequently, the evaluation command/JSON—not the converted weight file alone—must be retained to establish which manifest and prompt mode produced a number.

**Train/evaluation mismatch:** training samples the thermal VAE posterior (`train_iris_ms2_g.py`, lines 1041-1045); reported evaluation uses its mode (`iris_ms2_pipeline.sbatch`, line 89; `train_route_suite.py`, lines 941-955). This mismatch is verified and should be disclosed or removed in a future controlled rerun.

## 7. Training losses and exact targets

### Current Iris-MS2 losses (active)

Let \(f_\theta\) be the U-Net and \(M_z\) the latent valid mask. The active loss is

\[
\mathcal L = \mathcal L_{\mathrm{depth}}+\mathcal L_{\mathrm{recon}},
\]

\[
\mathcal L_{\mathrm{depth}}
=\operatorname{MSE}_{M_z}\!left(
f_\theta([z_I;z_t^{D}],t,C(c),e_D),z_0^{D}
\right),
\]

\[
\mathcal L_{\mathrm{recon}}
=\operatorname{MSE}\!\left(
f_\theta([z_I;z_t^{I}],t,C(\varnothing),e_I),z_0^{I}
\right).
\]

- `depth` target: VAE posterior sample of the three-channel, per-image normalized completed disparity.
- `recon` target: VAE posterior sample of the three-channel thermal condition itself.
- Both weights are exactly 1; total loss is their sum.
- There is no separate sparse-LiDAR loss, pseudo-pixel weight, SSI loss, gradient loss, ranking loss, sky penalty, metric-depth loss, or caption contrastive loss in this trainer.

Evidence: `lotus/train_iris_ms2_g.py::main` training loop, target construction lines 1041-1057, masks 1062-1075, captions/task codes 1105-1164, and losses 1166-1184.

The depth mask is made by OR-ing valid and sky masks, inverting it, max-pooling invalidity over non-overlapping 8x8 cells, and retaining only cells with no invalid pixel; it is repeated over four latent channels (`train_iris_ms2_g.py`, lines 1062-1075). Since completion clips every pixel into `[d_min,d_max]`, every pixel is assigned to valid or sky in normal operation (`ms2_thermal_dataset.py::_completed_depth`, lines 128-145; `_get_valid_mask/_get_sky_mask`, lines 201-213). The loss is consequently dense in latent space, not sparse-LiDAR-only.

### Losses in the distinct route-suite baseline/ablation line

For completeness, `tools/train_route_suite.py` can optimize the following, but they are **not active in `train_iris_ms2_g.py`**:

| Loss | Exact target/alignment | Evidence |
|---|---|---|
| Default sparse `masked_ssi_l1` | Decode prediction to disparity; fit detached per-image affine \(s,b\) on real valid LiDAR so \(d^*_{gt}\approx s\hat d+b\); mean \(|s\hat d+b-d^*_{gt}|\) on real pixels. | `tools/train_ms2_joint_gt_v3.py::masked_ssi_l1`, lines 234-260; selected in `tools/train_route_suite.py`, lines 1665-1685. |
| Dense-completion SSI L1 | Fit \(s,b\) only on real LiDAR, then real L1 plus `pseudo_weight` times L1 on missing pixels filled by pseudo GT. | `tools/train_ms2_joint_gt_v3.py::masked_ssi_l1_dense_completion`, lines 263-317; route loop lines 1673-1684. |
| Latent MSE option | VAE-encode a dense normalized disparity target, noise it at fixed \(t\), and compare U-Net \(x_0\) directly to that latent. | `tools/train_route_suite.py::load_sample`, lines 816-833; `RouteModel.predict_disparity`, lines 1027-1035; training loop lines 1654-1671. |
| SSI gradient matching | After detached real-LiDAR SSI fit, multiscale L1 gradients of disparity residual over neighboring valid pairs, four scales by default. | `tools/train_ms2_joint_gt_v3.py::ssi_grad_matching`, lines 320-372; route loop lines 1687-1690. |
| Caption ranking | For the same frame/noise/timestep/mask, hinge \(\max(0,m+L_{correct}-L_{wrong/empty})\), summed over requested references. | `tools/train_route_suite.py`, lines 1691-1733. |
| Sky loss | After real-LiDAR SSI fit, L1 or hinge to disparity \(1/D_{max}\) on sky∩invalid pixels. | `tools/train_ms2_joint_gt_v3.py::ssi_sky_loss`, lines 375-417; route loop lines 1737-1742. |

The route-suite defaults leave pseudo, gradient, ranking, and sky weights at zero and use the sparse SSI term; see `tools/train_route_suite.py::parse_args`, lines 138-239 and 349-388. Results that use those optional losses must be labeled as route-suite ablations, not as the current Iris objective.

## 8. Sparse-GT and pseudo-GT pipelines

### Sparse thermal-view LiDAR

MS2 depth PNGs are read as float and divided by 256. A real pixel is valid only when finite and \(d_{min}<D<d_{max}\) (normally 0.001–80 m), then its disparity target is \(1/D\). Evidence: `tools/train_ms2_joint_gt_v3.py::load_gt_disparity`, lines 192-201. The route suite preserves this sparse mask and may train only on it (`tools/train_route_suite.py::load_sample`, lines 776-815).

### Pseudo-depth production and completion

1. Frozen AnyThermal MiDaS predicts raw relative depth/disparity from thermal input: `tools/run_ms2_anythermal_midas.py::main`, lines 163-221; architecture in `tools/build_anythermal_midas.py::build_anythermal_midas`, lines 105-240.
2. For **each frame**, resize raw prediction to LiDAR resolution and fit the closed-form affine relation
   \[
   D_{gt}\approx aP_{raw}+b
   \]
   on that frame's real valid LiDAR pixels. Apply it to the whole raw map. This is a depth-space, per-frame calibration; it is not a globally learned calibration. Evidence: `tools/build_anythermal_pseudo_gt.py::fit_affine_depth_space`, lines 106-115; `calibrate_pseudo_depth`, lines 118-153; `main`, lines 164-223.
3. Current Iris-MS2 training loads this `.npy`, checks shape, overwrites pseudo values wherever valid real LiDAR exists, and clips all output to `[d_min,d_max]`: `lotus/utils/ms2_thermal_dataset.py::MS2ThermalDataset._completed_depth`, lines 128-145.
4. It converts completed metric depth to disparity, estimates the 2nd and 98th disparity quantiles on real-range pixels, linearly maps to `[-1,1]`, clips, and repeats to three channels: `MS2ThermalDataset.__getitem__`, lines 161-196. Active values are `norm_type=trunc_disparity` and default quantiles 0.02/0.98: `lotus/train_iris_ms2_g.py::parse_args`, lines 439-468; active launcher line 22.

### Exact difference

| Property | Sparse-GT route suite | Current pseudo-completed Iris-MS2 |
|---|---|---|
| Pixels supervised | Real thermal-view LiDAR only by default | Entire completed frame in latent cells |
| Pseudo values | Optional, independently weighted only where LiDAR is absent | Mandatory; unweighted after LiDAR overwrites it |
| Prediction domain | VAE-decoded normalized disparity | VAE latent of per-image truncated normalized disparity |
| Scale handling in loss | Detached per-frame SSI fit on real LiDAR | No prediction/target SSI fit; target itself was pre-calibrated per frame |
| Primary loss | Sparse SSI L1 | Clean-latent MSE + thermal reconstruction MSE |

Because pseudo calibration uses the same frame's LiDAR, pseudo targets are framewise LiDAR-calibrated teacher completions, not independent dense ground truth. The paper should use “completed pseudo GT” or “LiDAR-overwritten, per-frame-calibrated pseudo depth,” not “dense GT.”

## 9. Scale/shift and SSI alignment

### During current training

- **Target preprocessing:** per-image disparity quantile normalization (2%/98%) to `[-1,1]`; this is a per-image affine normalization in disparity space (`ms2_thermal_dataset.py::__getitem__`, lines 167-196).
- **Pseudo production:** per-image affine fit in **depth space** against real LiDAR (`build_anythermal_pseudo_gt.py::calibrate_pseudo_depth`, lines 118-153).
- **Training loss:** no fitted scale/shift between prediction and target; direct latent MSE (`train_iris_ms2_g.py`, lines 1152-1169).

### During reported Iris-MS2 evaluation

`tools/train_route_suite.py::run_validation`, lines 1175-1228, resizes the decoded raw normalized-disparity prediction to GT shape and calls `ms2_eval.official_protocol.evaluate_sample(..., align="ssi_disparity")`. The evaluator:

1. creates the valid mask from finite thermal-view GT with \(0.001<D<80\) m;
2. computes \(D^{-1}\) on valid pixels;
3. least-squares fits per image \(D^{-1}\approx s\hat d+b\);
4. clips aligned disparity below at \(10^{-3}\), inverts to depth, then clamps depth to `[0.001,80]` for metrics.

Evidence: `ms2_eval/official_protocol.py::official_valid_mask`, lines 74-81; `fit_scale_shift`, lines 84-104; `evaluate_sample`, lines 161-208. The file explicitly states that `ssi_disparity` is a repository extension and **not** in upstream BridgeMultiSpectralDepth (`PROTOCOL_REFERENCE`, lines 36-54, especially 44-48).

### Upstream Lotus route-selection evaluator

`lotus/evaluation/evaluation.py::evaluation_depth`, lines 50-233, supports least-squares alignment in depth or disparity, with optional maximum alignment resolution. Disparity alignment converts GT depth with `depth2disparity`, fits scale/shift through `align_depth_least_square`, converts back, and clips to configured depth bounds (lines 182-219). Closed-form least squares is in `lotus/evaluation/util/alignment.py::align_depth_least_square`, lines 8-55; depth/disparity conversion is lines 59-73.

The mandatory handoff says route-selection comparisons must use this upstream evaluator end to end (`docs/ADAPTER_V2_HANDOFF_20260703.md`, evaluator checklist and route-selection constraints). The current Iris-MS2 results pipeline instead calls the ported `ms2_eval.official_protocol` through `train_route_suite.py`. Although the alignment intent and common metrics are close, these are not the same callable and must not be described as one protocol.

### Separate final-paper unified evaluator

`ms2_eval/core.py::evaluate_depth_pair`, lines 93-106, reports raw metric-depth metrics when applicable, affine-aligned depth metrics, and Pearson/Spearman correlations. Its valid-mask threshold and alignment domain differ from `official_protocol.py` (`ms2_eval/core.py::build_valid_mask`, lines 21-27; `align_affine`, lines 64-73). `docs/MS2_UNIFIED_EVALUATION_PROTOCOL_V1.md` governs this distinct phase. Its results must not be mixed with route-selection/Iris-MS2 tables.

## 10. Evaluation metrics and implementations

### A. Metrics used by the current Iris-MS2 result pipeline

`ms2_eval/official_protocol.py::official_depth_errors`, lines 126-158, computes per image after alignment and prediction clamp. For valid GT \(g_i\), aligned prediction \(p_i\), and \(N\) valid pixels:

| Name | Implementation |
|---|---|
| `abs_diff` | \(N^{-1}\sum_i |g_i-p_i|\) |
| `abs_rel` | \(N^{-1}\sum_i |g_i-p_i|/g_i\) |
| `sq_rel` | \(N^{-1}\sum_i (g_i-p_i)^2/g_i\) |
| `log10` | \(N^{-1}\sum_i |\log_{10}g_i-\log_{10}p_i|\) |
| `rmse` | \(\sqrt{N^{-1}\sum_i(g_i-p_i)^2}\), in metres because aligned values are metric depth |
| `rmse_log` | \(\sqrt{N^{-1}\sum_i(\ln g_i-\ln p_i)^2}\) |
| `a1`, `a2`, `a3` | fraction with \(\max(g_i/p_i,p_i/g_i)<1.25^k\), \(k=1,2,3\) |

`OFFICIAL_METRICS` is declared at line 31. Validation macro-averages each numeric per-image field, unweighted by pixel count: `tools/train_route_suite.py::run_validation`, lines 1217-1228; protocol aggregation is declared in `official_protocol.py`, lines 36-55. The headline tables select AbsRel, RMSE and \(\delta_1\), but all nine metric fields are implemented and written.

### B. Metrics in the required upstream Lotus route-selection evaluator

`lotus/evaluation/evaluation.py`, lines 34-47, registers:

- AbsRel: `lotus/evaluation/util/metric.py::abs_relative_difference`, lines 69-79;
- SqRel: `squared_relative_difference`, lines 82-94;
- RMSE: `rmse_linear`, lines 97-109;
- RMSE-log: `rmse_log`, lines 112-122;
- log10: `log10`, lines 125-132;
- \(\delta_1,\delta_2,\delta_3\): `threshold_percentage` and wrappers, lines 136-162;
- inverse-depth RMSE: `i_rmse`, lines 165-177;
- SILog RMSE: `silog_rmse`, lines 180-193.

`evaluation_depth` evaluates aligned, clipped per-image tensors and accumulates them through `MetricTracker` (`lotus/evaluation/evaluation.py`, lines 182-233; `lotus/evaluation/util/metric.py::MetricTracker`, lines 10-31).

### C. Metrics in the separate unified final-paper evaluator

`ms2_eval/core.py::compute_depth_metrics`, lines 50-60, implements raw/aligned AbsRel, SqRel, RMSE-m, RMSE-log and \(\delta_{1,2,3}\). `correlation`, lines 88-91, implements Pearson correlation directly and Spearman by rank-transforming first (`rankdata`, lines 76-86). `evaluate_depth_pair`, lines 93-106, reports both raw and affine-aligned metric sets plus correlations. The Lotus-specific bridge separately uses upstream Lotus metrics and unweighted imagewise means: `ms2_eval/lotus_official.py::align_lotus_disparity_to_ms2_depth`, lines 21-46; `lotus_official_metrics`, lines 49-60; `aggregate_imagewise`, lines 64-72.

These three metric families must remain labeled by evaluator. A number called “RMSE” is in metres only after a prediction has been converted/aligned into metric depth; a normalized-disparity or relative-space error is not metric RMSE.

## 11. Discrepancies from Iris/Lotus formulation

This comparison uses [Iris arXiv v4 (2025-11-18)](https://arxiv.org/html/2411.16750v4), which matches the “Integrating Language” paper represented by this repository, and the [Lotus ICLR 2025 paper](https://arxiv.org/abs/2409.18124). The repository does not pin a paper revision to its commit, so a different manuscript revision must be named explicitly if used.

| Topic | Iris/Lotus repository formulation | Current thermal implementation | Status/evidence |
|---|---|---|---|
| Visual modality | Original Iris recipe trains on HyperSim/VKITTI RGB images. | MS2 high-bit thermal is per-frame min-max normalized and triplicated. | Changed. Original launcher `lotus/train_scripts/train_iris_g_depth.sh`, lines 3-11 and 36-67; current dataset lines 147-155. |
| Dense target source | Original datasets supply rendered dense annotations. | Real sparse LiDAR overwrites per-frame calibrated AnyThermal pseudo depth. | Changed. `train_iris_ms2_g.py` module description, lines 17-42; `ms2_thermal_dataset.py`, lines 128-145. |
| Starting weights | Iris shell recipe names SD2-base. | Active SLURM default starts from trained Lotus disparity weights. | Changed. `train_iris_g_depth.sh`, line 3; `slurm/iris_ms2.sbatch`, lines 55-59. |
| Conditioning modality | Iris defines RGB \(x\), caption \(c\), dense depth \(y^*\), with the caption describing the same input image (Iris v4, lines 85-90 and 127-129). | Caption is generated from paired RGB while the network sees thermal. | Cross-modal change. Handoff lines 229-230; dataset lines 98-116. |
| Text path | Iris freezes CLIP and passes the caption encoding into the diffusion model while concatenating VAE image/depth latents (Iris v4, lines 107-115). | Same mechanism, with thermal replacing RGB. | Mechanistically consistent. Current trainer lines 1041-1057 and 1105-1164. |
| Diffusion objective | Iris v4's general formulation samples \(\epsilon,t\) and minimizes \(\|\epsilon-\epsilon_\theta(z_t,t,x,c)\|^2\) (lines 94-118). | Fixed \(t=999\), `prediction_type=sample`, target \(z_0\), direct latent MSE. | **Equation-level discrepancy with Iris v4.** Current trainer lines 1088-1169. It is, however, consistent with Lotus's direct-annotation objective and single-step design (Lotus abstract). |
| Reverse process | Iris v4 describes progressive noise removal from \(z_T\) to \(z_0\) (lines 119-124). Its Lotus implementation details separately state DDIM with one sampling step (lines 281-285). | Reported results use one step at 999 and take U-Net \(x_0\) directly. | Consistent with Iris's Lotus-specific implementation paragraph and Lotus; inconsistent with Iris's generic reverse-process equation if presented as the actual Lotus-G procedure. `train_route_suite.py`, lines 1000-1053; `iris_ms2_pipeline.sbatch`, lines 86-92. |
| Timestep | Lotus describes a fixed terminal timestep; Iris's Lotus paragraph retains a 1,000-step scheduler and one-step inference. | Fixed code index 999. | Consistent under zero-based code vs one-based paper indexing. Trainer lines 1088-1090; active launcher lines 30-32. |
| Detail-preserver branch | Iris doubles the batch and reconstructs RGB with empty text. | Structure is preserved, but the branch reconstructs thermal because `pixel_values` is thermal. | Target modality changed. `train_iris_ms2_g.py`, lines 17-42 and 1041-1169. |
| Normalization | Lotus disparity checkpoint expects `trunc_disparity`. | Current V2 uses 2%/98% truncated disparity. | Consistent. Dataset lines 185-195; launch line 22. V1 legacy direct-depth outputs are explicitly invalid for final claims: `docs/ADAPTER_V2_HANDOFF_20260703.md`, legacy warning. |
| Data and resolution | Iris Lotus uses HyperSim/VKITTI at 90/10, with 576 and 375 training resolutions (Iris v4, lines 281-285). | One real MS2 dataset at native 640x256; no mixture. | Changed. Current dataset lines 151-159; trainer lines 870-892 and 1028-1038. |
| Effective batch | Iris Lotus reports batch 36 over three GPUs (Iris v4, line 282). | Active job requests one GPU; batch 4 × accumulation 3 = 12. | Changed. `slurm/iris_ms2.sbatch`, lines 1-7; current launcher lines 24-28 and 46-59. |
| Caption generation | Iris Lotus reports one InternVL3-8B caption, depth-focused prompt, maximum 77 tokens (Iris v4, lines 284-285). | Repository labels active manifests InternVL3-8B/RGB/clip75, but the frozen training command does not bind a generator revision or prompt configuration. | Partly consistent; exact active generation provenance **UNKNOWN**. `README.md`, lines 112-120; `slurm/iris_ms2.sbatch`, lines 31-34 and 82-86; dataset loader lines 98-116. |
| Online AnyThermal adapter | Adapter experiments are separate route options. | Current Iris-MS2 paper model has neither adapter nor online AnyThermal encoder. | Important naming correction. `train_route_suite.py`, lines 92-100 and 853-927; current model construction lines 716-795. |
| Training/eval condition latent | Generic Iris/Lotus training and pipeline sample VAE posteriors. | Training samples, but reported route-suite evaluation uses posterior mode. | Discrepancy. `train_iris_ms2_g.py`, lines 1041-1045; `iris_ms2_pipeline.sbatch`, line 89. |
| Caption/flip consistency | Original Iris augmentation leaves caption unchanged. | Same behavior retained; spatial words may be incorrect after flip. | Preserved implementation, methodological caveat. `ms2_thermal_dataset.py`, lines 25-28 and 49-61. |
| Evaluation alignment domain | Iris v4 describes a per-image affine fit of relative **depth** and reports \(\delta_1\)/AbsRel (lines 291-300). | Current paper pipeline fits raw **disparity** to inverse GT, then inverts; headlines also include RMSE. | Discrepancy/extension. `official_protocol.py`, lines 191-208; results lines 65-72. |
| Evaluation callable | Handoff requires upstream `lotus/evaluation/evaluation.py::evaluation_depth` for route selection. | Iris-MS2 pipeline calls `ms2_eval.official_protocol.evaluate_sample` through route suite. | Protocol discrepancy. `train_route_suite.py`, lines 1175-1228; `iris_ms2_pipeline.sbatch`, lines 86-92; handoff evaluator rule. |
| BridgeMSD alignment | Upstream BridgeMSD relative path fits affine raw output to depth. | Current results use `ssi_disparity`, explicitly a local extension. | Protocol extension. `ms2_eval/official_protocol.py`, lines 36-54 and 161-208. |

Two statements should therefore not appear in the thermal paper without qualification:

1. “We optimize the Iris noise-prediction objective over sampled timesteps.” The code does not; it optimizes Lotus-style clean-latent prediction at one fixed timestep.
2. “Captions describe the observed thermal image.” The active captions are documented as descriptions of paired RGB, and the training loader cannot verify their provenance.

### Consequence for the central claim

The implementation verifies that captions have a trainable causal path into every U-Net cross-attention block. It does **not** by itself verify that the network uses caption semantics. The repository's controlled prompt results indicate:

- correct vs empty prompt within the caption-trained weights changes AbsRel by -0.00221, but
- shuffled vs correct changes AbsRel by only +0.00012, and
- matched caption training/inference vs the no-caption baseline changes AbsRel by -0.00021 with a confidence interval crossing zero.

Evidence: `docs/RESULTS_20260813_IRIS_MS2.md`, lines 83-115. The faithful paper conclusion is therefore: **language conditioning is implemented and conditioning consistency matters, but the present evidence does not show a substantial semantic-caption improvement over a no-language thermal baseline.** A semantic claim would require the same checkpoint-selection protocol and frozen evaluator with correct, shuffled, empty, and no-text controls, plus a predeclared paired test.

## 12. Mathematical notation faithful to the implementation

| Symbol | Implementation meaning | Shape/domain | Source |
|---|---|---|---|
| \(I^{16}_{th}\) | Raw high-bit-depth MS2 thermal array | \(H\times W\), usually `uint16` | `thermal_to_lotus_input`, lines 102-123 |
| \(I_{th}\) | Per-frame min-max converted, triplicated and `[-1,1]` thermal tensor | \(3\times H\times W\) | same, lines 123-148 |
| \(D_L\) | Sparse thermal-view LiDAR depth | metres, valid on \(M_L\) | `load_gt_disparity`, lines 192-201 |
| \(P\) | Frozen AnyThermal raw pseudo prediction | relative scalar map | `run_ms2_anythermal_midas.py`, lines 163-221 |
| \(a_P,b_P\) | Per-frame depth-space pseudo calibration fit on \(M_L\) | scalars | `calibrate_pseudo_depth`, lines 118-153 |
| \(D_P=a_PP+b_P\) | Calibrated pseudo-depth map | metres | same |
| \(D_C\) | Completed depth: \(D_L\) on valid LiDAR, otherwise \(D_P\), clipped to bounds | metres, dense | `_completed_depth`, lines 128-145 |
| \(q_{.02},q_{.98}\) | Valid-pixel disparity quantiles of \(1/D_C\) | scalars per frame | dataset `__getitem__`, lines 185-190 |
| \(Y_D\) | clipped affine normalization of \(1/D_C\) using those quantiles, repeated 3 channels | \([-1,1]^{3\times H\times W}\) | lines 185-196 |
| \(E,V\) | Frozen VAE encoder and decoder | image↔4-channel latent | trainer lines 742-748, 792-795 |
| \(z_I\) | scaled VAE latent of thermal input; posterior sample in training, mode in reported eval | \(4\times H/8\times W/8\) | trainer 1041-1045; route suite 941-955 |
| \(z_0^D\) | scaled VAE posterior sample of \(Y_D\) | same latent shape | trainer 1046-1057 |
| \(z_0^I\) | scaled VAE posterior sample of \(I_{th}\), reconstruction target | same latent shape | trainer 1046-1057 |
| \(t\) | fixed diffusion index | exactly 999 in active recipe | trainer 1088-1090; launcher 30-32 |
| \(\epsilon\) | IID Gaussian noise | same shape as target latent | trainer 1077-1097 |
| \(z_t\) | \(\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon\) | target latent shape | trainer 1092-1097 |
| \(c\) | Manifest caption; \(\varnothing\) is empty string | text | dataset line 198 |
| \(C(c)\) | Frozen CLIP last hidden state | \(77\times1024\) tokens/features | trainer 1121-1148 |
| \(e_D,e_I\) | four-dimensional sine/cosine task embeddings | \(\mathbb R^4\) | trainer 1155-1160 |
| \(f_\theta\) | Fully trainable eight-input/four-output conditional U-Net | latent→clean latent | trainer 750-795, 1162-1164 |
| \(M_z\) | 8x8-eroded valid+sky mask, repeated over four latent channels | Boolean latent mask | trainer 1062-1075 |
| \(\hat z_0^D\) | \(f_\theta([z_I;z_t^D],t,C(c),e_D)\) | four-channel latent | trainer 1099-1167 |
| \(\hat z_0^I\) | \(f_\theta([z_I;z_t^I],t,C(\varnothing),e_I)\) | four-channel latent | trainer 1105-1168 |
| \(\mathcal L_D\) | masked MSE between \(\hat z_0^D\) and \(z_0^D\) | scalar | trainer line 1167 |
| \(\mathcal L_I\) | dense MSE between \(\hat z_0^I\) and \(z_0^I\) | scalar | trainer line 1168 |
| \(\mathcal L\) | \(\mathcal L_D+\mathcal L_I\) | scalar | trainer line 1169 |
| \(\hat d\) | VAE-decoded channel-mean output mapped from `[-1,1]` to `[0,1]` | raw normalized disparity | `decode_to_disparity`, lines 420-434 |
| \(s_E,b_E\) | Per-image least-squares evaluation fit \(D_L^{-1}\approx s_E\hat d+b_E\) | scalars | `official_protocol.evaluate_sample`, lines 191-197 |
| \(\hat D\) | \(1/\max(s_E\hat d+b_E,10^{-3})\), then metric-range clamp for scoring | metres | same, lines 191-208 |

## Reproducibility unknowns that must not be guessed

1. Exact caption-generator revision/prompt/sampling seed and source-image hash for each active manifest: **UNKNOWN from the frozen training invocation**. The broader repository does document InternVL3-8B, paired RGB, and a 75/77-token cap.
2. Exact attention processor/kernel used by historical jobs: **UNKNOWN unless recovered from environment logs**.
3. Exact VAE posterior RNG sequence after distributed resume: controlled by Accelerate/runtime state, but not reconstructable from source alone without a preserved job state: **UNKNOWN for an arbitrary checkpoint**.
4. Whether every reported result used the exact cached checkpoint snapshot hash listed in §2: the launch default points to the repository name, but a result artifact-to-HF-snapshot content hash is not established by the code references above: **UNKNOWN unless confirmed by frozen run metadata**.
5. The repository commit is not mapped to a paper revision. This report uses Iris arXiv v4 and Lotus ICLR 2025; identity with another revision is **UNKNOWN**.

---

# 13. Work after the implementation review

**Added:** 2026-08-17. This section records the experiments that address §11's two
unqualified statements, one experiment now in flight, and the direction being pursued. It
keeps the same vocabulary: measured numbers are **VERIFIED** against the artifact named,
and anything not established is marked **UNKNOWN** rather than assumed.

## 13.1 Caption provenance was closed, and it was not the cause

§11 warned that "captions describe the observed thermal image" could not be asserted: the
active captions describe the **paired RGB camera** while the model is conditioned on the
**thermal** one. That gap has now been measured, removed, and tested.

**Size of the gap (VERIFIED).** 30.0% of the active RGB-derived captions contain a
chromatic colour word — "a white sedan", "a yellow bus" — naming an attribute that cannot
exist in a thermal frame, seen from a camera that is not the model's input
(`tools/bakeoff_thermal_captions.py`, scored over 40 evenly spaced training frames).

**Implementation defect found while closing it (VERIFIED, fixed).**
`scripts/generate_captions.py::load_image` called `PIL.Image.convert("RGB")` directly on
MS2's 16-bit thermal PNGs. Measured on a real frame the result is min 255 / max 255 /
std 0.00 — an entirely white image. A caption model handed that still writes a fluent
sentence, so the failure never announces itself and its output reads like data. This is
the exact operation `AGENTS.md` forbids. The path now applies the same full-range scaling
the depth model's own input receives (`AnyThermalEncoder._array_to_uint8`), so captioner
and model see one picture; three regression tests cover it.

**Captioner adequacy on thermal (VERIFIED by audit, with a stated limit).** InternVL3-8B
is RGB-trained, so its competence on thermal was checked rather than assumed. Over 40
frames the grayscale rendering gave 0.0% chromatic colour words, 100% unique sentences,
and 0.238 content-word agreement with the same frame's RGB caption. Frame-by-frame
reading confirmed correct specifics: a pedestrian, an archway that the RGB caption calls
"brick pillars", fences on both sides, a pedestrian crossing with an approaching vehicle.
One systematic error class survives — brightness read as visible-light appearance. A
time-of-day ban in the prompt removed "night" (45% of a daytime sequence, down to 0.4%
over 19,945 frames), but the same error re-emerged as material inference:
"snow-covered ground", 1.4% of train and **4.9% of test**. A false-colour (magma)
rendering was tested and rejected — worse on colour words, and it began describing the
palette rather than the scene.

**Result (VERIFIED).** `iris_ms2_thermalcap`, identical to the active line in every
respect but caption source, checkpoint selected on validation at step 2000:

| test 16-08-46, 2543 frames | AbsRel | RMSE | \(\delta_1\) |
|---|---|---|---|
| empty prompt | 0.0882 | 3.753 | 0.9180 |
| correct caption | 0.0875 | 3.736 | 0.9193 |
| shuffled caption | 0.0873 | 3.724 | 0.9196 |

Paired over 2543 frames, 10,000 bootstrap resamples:

- `shuffled − correct` = **−0.00022**, CI [−0.00029, −0.00015], shuffled wins 57.6%;
  RMSE −0.01249, \(\delta_1\) +0.00024 — all three significant and agreeing in sign.
- `shuffled − empty` = −0.00097, CI [−0.00113, −0.00081], 61.1%.

**Reading.** Grounding the caption in the model's own input did not produce a semantic
gain. It produced a small, significant effect in the **opposite** direction: the frame's
own caption is worse than another frame's. Text presence still helps; text content does
not, and mildly hurts.

**Attribution check (VERIFIED, with a stated limit).** Splitting the test set by whether
the caption carries a detectable hallucination word gives `shuffled − correct` of
−0.00041 over 128 flagged frames and −0.00021 over 2415 clean frames, shuffled winning
60.2% and 57.5%. The harm is present in the clean 95%, so detectable caption inaccuracy
does not account for it. **UNKNOWN:** the detector is a keyword list. It catches
brightness-as-material errors and cannot detect a wrong object count or a mirrored
left/right claim, so "clean" means "no flagged word", not "accurate".

**Consequence for §11.** Captions describing the observed thermal frame have now been
produced, trained on and evaluated, and the semantic claim does not survive that change
either. The RGB-provenance caveat remains true of `iris_ms2_ckpt`, but it is no longer a
candidate explanation for the absent semantic effect.

## 13.2 Ground-truth-derived caption ceiling (designed, submitted, blocked)

**Question.** Every caption tested so far came from a model that had to infer depth from
an image. If text carrying depth information that is correct by construction still does
not help, then no captioner, prompt or thermal-specific VLM can, and the language line
closes on a bounded negative rather than an open one.

**Design.** `tools/build_gt_caption_manifest.py` computes captions from the frame's own
depth: nearest surface, scene extent, per-column typical distance, near-to-far ordering,
and a ratio clause. Three properties are deliberate:

1. **Templated wording** — only the geometry varies between frames, so the
   correct-versus-shuffled contrast isolates content from phrasing.
2. **Ratios and ordering rather than absolute metres** — the evaluator fits a per-image
   scale and shift, so a global scale stated in text is removed before the metric sees
   it; only relative structure survives.
3. **The sky band is excluded from the statistics** — it is a third of the frame and
   uniformly far, and including it made column medians describe the sky rather than the
   structure depth varies over, inverting the near-to-far ordering on inspection.

**⚠️ This is an oracle, not a method.** There is no ground truth at inference. A system
captioned this way cannot be deployed and its score must not be quoted beside the other
lines as if it were one. Rows carry `caption_is_oracle`.

**Source selection (VERIFIED by measurement).** Both candidate sources were built and
compared over the 19,949 training frames before committing GPU time:

| caption source | unique sentences | `extends to 80 m` | three columns equal | column spread, median |
|---|---|---|---|---|
| real sparse LiDAR | 31.0% | 0.0% | 2.6% | 5 m |
| completed pseudo GT | 24.1% | 0.1% | 5.5% | 3 m |

The two sets share 0.0% of captions verbatim, so they are genuinely different inputs. The
pseudo-GT version is measurably less discriminative: its column medians are dominated by
the road surface and the filled-in region, which are near-symmetric left to right, giving
40% less lateral spread and 7 points fewer unique sentences. Its original motivation —
that a LiDAR-derived caption would contradict the dense training target over the sky — no
longer applies once the sky is excluded from the statistics, because the two sources agree
wherever LiDAR returns exist. The LiDAR-derived source was therefore chosen: more
informative, and matching the standard the metric is computed on. **UNKNOWN:** whether a
different per-column statistic — a low percentile emphasising the nearest structure rather
than the road — would make the pseudo-GT source competitive.

**Status.** All three manifests are built and verified: `train_gtcap_lidar_v2.jsonl`
19,949 rows, `val_gtcap_lidar_v2.jsonl` 5,810, `test_gtcap_lidar_v2.jsonl` 2,543. Training
could not be submitted: Aire's Slurm controller intermittently fails every submission with
`Batch job submission failed: I/O error writing script/environment to file`, including a
minimal `--wrap="hostname"` that had succeeded minutes earlier (job 7266581), while
`scontrol ping` reports the controller UP. This is a cluster-side fault; a retry loop is
queued to submit as soon as a working window appears.

**Predeclared decision rule** on `shuffled − correct` for this arm: significantly positive
means a ceiling exists and caption quality is worth pursuing; crossing zero or negative
means correct geometry stated in text is unusable by this model, and the line closes.

## 13.3 Direction under investigation

Recorded as the researcher's stated position, not as a finding.

1. **Iris does appear to read caption content; the gap is in converting content into
   geometry.** Iris's Table 5 separates the two effects at comparable magnitude — blank
   6.2, "An image" 6.1, generic template 6.0–6.1, real caption 5.9 — so a conditioning
   effect and a content effect both exist in their data. Only the first reproduces here.
   The working hypothesis is that this system has the text channel but not the mapping
   from described content to a correct geometric correction. This is consistent with the
   July cross-attention probe: text moves the feature map by 113%, content accounts for
   0.279 of that, and the content signal survives to the output layer — it enters, it is
   distinguishable, it propagates, and its direction is uncorrelated with truth. A
   per-frame oracle over correct/shuffled bounds what that channel could deliver at about
   +0.0018 AbsRel.
   **⚠️ Symmetric skepticism:** Iris's content effect rests on a 0.1–0.2 point difference
   from a single run over 454–697 images with no error bars. By this repository's own
   standard that is not established either.
2. **Metric distance in captions** — never requested by any prompt version to date.
   Absolute metres cannot help under a scale-and-shift-invariant evaluator; only ratios
   and ordering survive, which is why §13.2's templates state those. A near-to-far ordered
   listing was tried once in the older RGB era (`rgb_depth_spatial_v2`, the P3 format) and
   judged an information disadvantage; that verdict predates the fixed GT-gradient
   pipeline and should not be carried over without a rerun.
3. **An A/B on caption source** was requested and partially executed; §13.2 records why
   the pseudo-GT arm was set aside on measurement rather than on argument.

## 13.4 Open items

| Item | State |
|---|---|
| GT-caption ceiling training and evaluation | Blocked on the Slurm fault above |
| Pseudo-GT caption arm | Set aside; would need a different per-column statistic to justify a run |
| Pseudo GT for val | Not buildable — no AnyThermal predictions exist for 11-23-45 (test has 2,543) |
| Generic-template prompt rung (Iris Table 5's middle level) | Never tested on the thermal line |
| Seed replication | **None.** Every arm is a single run, and between-arm differences are the same order as the within-arm prompt effects, so no arm has been shown better than another |
| `E:\project\captioning` | Not under version control; the 16-bit fix and the prompt versions exist only on disk |


---

# 14. Scene diversity of the training split (2026-08-18)

## 14.1 VERIFIED — the training set is two drives, and they repeat themselves

`ms2_train_day2seq_clip75_20260728.jsonl` holds 19,949 frames. Counting them by
sequence:

| sequence | frames | frame numbers | adjacent stride |
|---|---|---|---|
| `2021-08-06-10-59-33` | 10,441 | 0..10441 | 1 (10,439 of them) |
| `2021-08-06-11-37-46` | 9,508 | 0..9514 | 1 (9,500 of them) |

Two continuous drives at full rate. That alone says nothing about how much the
picture changes, so it was measured on the pixels: each frame reduced to a
32x16 signature under the same per-image min-max the encoder applies, then the
mean distance between frames plotted against how far apart in time they are.
The reference is the mean distance between two frames drawn at random, 0.2011.

| lag (frames) | distance | share of reference |
|---|---|---|
| 1 | 0.0294 | 15% |
| 10 | 0.0937 | 47% |
| 50 | 0.1395 | 69% |
| 150 | 0.1668 | 83% |
| 500 | 0.1900 | 94% |
| 800 | 0.1967 | 98% |

Two frames need to be ~500 apart before they are as unlike each other as two
frames picked at random. Dividing the frame count by that lag:

**Effectively independent scenes: ~39 (at the 90% threshold), ~132 (at 80%).**

Order 10^2, not 10^4. A contact sheet at 664-frame spacing confirms the content
does change -- car parks, tree-lined avenues, an overpass, an underpass, city
traffic -- so the frames are not one place revisited. What repeats is the
*layout*: ego-forward view, road surface below, structures at the sides, sky
above, which is the level the signature measures and also the level a depth
model works at.

⚠️ Limits of this measure: a 32x16 signature sees coarse layout and gross
brightness, not object identity. Two different streets with the same layout
register as similar, so the count is a lower bound on semantic diversity and a
fair estimate of geometric diversity.

## 14.2 What this invalidates

The 2026-08-18 ceiling result (§ the GT-caption line) was first written up as
"content is worth 7.8% of the text effect, so the ceiling on language is here."
The argument offered against a data-volume explanation was that too little data
would weaken *both* the presence effect and the content effect, whereas the
measured gap is 12x.

**That argument does not survive ~39 independent scenes.** Learning a binary
distinction (is there text or not) needs far fewer distinct examples than
learning a mapping from stated numbers to geometry, so the two effects are not
expected to degrade together. The confound cannot be ruled out with the data in
hand.

The result should therefore be stated with its condition attached:

> On a training set containing on the order of 40-130 effectively independent
> scenes, captions computed from the frame's own lidar contribute 7.8% of the
> total text effect.

Not: "the ceiling on language is here."

## 14.3 What would settle it

`E:\dataset\ms2\sync_data\` holds 16 further sequences, still as `.tar.bz2`
(~230 GB). Training currently uses 2 of them. Rebuilding the training split
from 6-8 sequences across different times and routes and rerunning the ceiling
experiment separates the two explanations: a content share that stays near 7.8%
is a property of the method; one that rises was a property of the data.

Cost: data preparation, one training run, one evaluation pipeline. Not started
-- worth raising before it is spent.


---

# 15. Retraction: the GT-caption run is not an upper bound (2026-08-18, after the meeting)

§13 and §14 call the GT-caption arm a ceiling experiment. It is not one, and
the write-up should not have used the word.

**An upper bound has to be at least as good as what it bounds.** The arm scores
AbsRel 0.08758. The ordinary RGB-caption arm, trained on captions a VLM guessed
from a different camera, scores 0.08530. The thing being bounded beats the
bound, so nothing is bounded.

The run is also degraded from the start, which the absolute numbers show
plainly:

| arm | empty prompt | its own caption |
|---|---|---|
| no caption training | **0.08551** | 0.08764 |
| RGB caption training | 0.08752 | **0.08530** |
| GT geometric caption training | **0.09104** | 0.08758 |

Training on captions computed from the ground truth cost 0.0055 AbsRel at the
empty prompt relative to the no-caption baseline. The model learned to lean on
a signal that does not exist at inference, and its visual pathway got worse for
it. A content share measured inside a run that starts 6% behind says something
about that run's internals, not about what language can contribute.

## 15.1 What survives

The within-run comparison stands: holding weights fixed, the frame's own
geometric sentence beats another frame's on all three metrics. That arm is
weakly but genuinely sensitive to caption content.

## 15.2 What does not survive

- "No captioner, prompt or thermal VLM will change this" — unsupported.
- "The ceiling on language is here" — retracted twice now: first for the scene
  diversity confound (§14.2), now because the arm is not a bound at all.
- The 7.8% figure may be quoted only as "the content share inside the
  GT-caption run", never as a limit on language.

## 15.3 What a real bound would need

At minimum the oracle arm has to reach the no-text baseline (0.08551) before
its internal decomposition means anything. The cheapest design that could:
keep training captions as they are on the arm that already works (RGB caption,
0.08530) and append the GT geometric clause to them, so the oracle information
is *added* to a model that already uses text rather than replacing the text
distribution it was trained on. If that arm does not beat 0.08530, the claim is
about language; if it does, the earlier arms were limited by caption content
after all.
