# DA2 + frozen Iris spatial-feature fusion, frame vs empty

New experiment: `outputs/lotus_line_v2/da2_iris_fusion_pair_v1`.
No old experiment source, weights, output or pending job is modified. No training
or test-set evaluation is launched from the development machine.

## Fixed recipe

- B0: `runs/da2_caption_v1/no-caption_s43/step018000.pt`, validation-selected.
  Strictly load the existing ViT-B metric-depth core, 128-channel DPT, max depth 80m.
- Iris: `iris_ms2/sd2_cap_pct_logtn_s43/converted/step18000_weights.pt`.
  This is the old log-target caption run WITHOUT the later image-L1 addition.
  Sources: `RESULTS_20261011_ABLATION_CAPTION_DATA.md` section 3 and
  `slurm/caption_showcase.sbatch`. The record has a same-checkpoint prompt swap.
  It also shows fixed templates capture part of the improvement; it does not
  prove that all improvement comes from frame-specific semantics.
- Submission first checks the exact run under current `IRIS_RUNS`, then the
  documented `/mnt/scratch/sc23sz/runs` location. `IRIS_FEATURE_CKPT` can select
  a copy with the same run/converted/filename layout. No fallback to l100 or dual.
- Original SD2.1 cached VAE/CLIP and U-Net configuration, audited by the existing
  SD2.1 asset helper. The converted Iris U-Net is loaded strictly. Original,
  random or partly matched U-Net weights must never silently substitute for it.
- Both arms use identical B0 and Iris weights, identical fresh fusion weights.
  Frame caption vs tokenized empty string; both padded/truncated to 77 tokens.
  Native causal CLIP; no additional attention mask passed to CLIP or U-Net.
  This differs from historical dynamic padding and is not claimed to reproduce
  the old Iris standalone metrics.
- Same percentile uint16 conversion for both inputs. DA2 retains ImageNet
  normalization/right-bottom reflect padding to multiples of 14. Iris gets the
  same uint8 pixels normalized to [-1,1] at native 256x640. Sparse GT is not resized.
- Iris inference features only: VAE posterior mode, pure Gaussian depth input,
  t=999, depth task sin/cos embedding. Noise seed = SHA256(`iris-fusion:43:` +
  frame ID), independent of caption/batch/order, fixed in training and validation.
  No GT, pseudo-GT or GT-derived scale/bounds enter either feature branch.
- Tap `up_blocks[2].attentions[2]`: Bx640x16x40 -> DPT `scratch.layer3_rn`.
  Tap `up_blocks[3].attentions[2]`: Bx320x32x80 -> DPT `scratch.layer2_rn`.
  Runtime shape/finite assertions. Native DA2 token extraction/readout is retained.
- Two fusion modules: 1x1 source projection -> GroupNorm(8) -> GELU, 128 channels;
  coordinate-aware resampling into the padded DPT grid; concatenate with 128-channel
  DA2 features; 3x3 conv -> GELU -> 1x1 conv; add to original feature. Last conv
  weight/bias are zero-initialized, no scalar alpha. 746,752 trainable parameters.
- All DA2 (including DPT), Iris U-Net, VAE and CLIP weights frozen and eval.
  Only the fusion modules train. Frozen downstream DPT runs WITH autograd so loss
  reaches the fusion modules; frozen feature extraction naturally needs no graph.
- Per-frame valid-LiDAR log-L1, then mean over nonempty frames, .001<GT<80.
  Same loss as B0. No new SiLog/pseudo/latent/reconstruction/teacher losses.
- seed43, 20,000 updates, batch4 x accumulation3, no augmentation, AdamW
  lr5e-5, betas(.9,.999), eps1e-8, weight_decay.01, grad clip1;
  lr multiplier `(1-step/20000)**.9`. FP32 weights, BF16 compute for
  U-Net/DA2/fusion, FP32 VAE/text/loss. No automatic unfreezing/extension/sweep.
- Same official eight train sequences (75,688 frames), three validation sequences
  (14,249). Enforce original B0 manifest/caption hashes and asset/source contracts.
- Only fusion parameters are saved. Milestones every2k; resumable `last.pt` every1k.
  Resume restores optimizer/scheduler/RNG/data cursor and repairs interrupted val/log.

## Minimal checks and jobs

CPU tests use actual tiny Diffusers U-Net/VAE and Transformers CLIP, plus a small
spatial decoder fixture. They verify initial exact B0, projection gradients through
frozen decoder, unchanged frozen weights, live text path, mask policy, padding
coordinates, deterministic per-ID noise, matching preprocessing, checkpoint resume,
and pair gate rejection of altered noise. Run the file directly to avoid Aire's
unrelated installed `tests` package.

GPU smoke loads REAL checkpoints and runs both arms for32 updates. It verifies:

1. Exact initial B0 reproduction on four frames in both FP32 and BF16.
2. Same initial backbones/fusions/prediction hashes; same per-update IDs, actual
   image/GT bytes and actual noise hashes between arms.
3. Both source projections receive nonzero gradients, fusion parameters update,
   all frozen weights remain byte-identical and have no gradients.
4. Finite predictions/loss/gradients and eight validation frames per arm.

Training jobs require `afterok` on the paired smoke AND recheck its contracts.
They reload original B0/Iris and fresh fusion initialization, not smoke weights.
Two formal jobs each include validation; no test-set jobs or mail jobs are submitted.

Validation: stride4 every2k; select minimum SSI AbsRel, tie -> earlier step.
Then full-frame validation for fixed20k and selected checkpoint (reuse if identical).
SSI and none are both reported with the existing DA2 Bridge protocol (.001,80).
This is NOT the old 5810-frame route-selection protocol nor a final-paper unified
protocol result. Do not mix old SSI-log/disparity tables with these depth-space scores.
Frame must improve both Empty and B0 to justify further evaluation. Existing test
sets have already informed research; do not describe them as never-seen holdouts.

Submit on Aire after syncing the six new files:

```bash
( set -euo pipefail
  cd ~/Iris
  unset IRIS_MS2_ROOT IRIS_MANIFEST_DIR IRIS_DATA
  IRIS_RUNS=/mnt/scratch/dwdh0363/runs bash slurm/submit_da2_iris_fusion.sh
)
```

If the exact old Iris checkpoint is elsewhere, export `IRIS_FEATURE_CKPT` before
submission. Missing readable input weights fail on the login node before queuing.
The first smoke resolves existing HF cache only; no model downloads on GPU nodes.
Outputs never copy the two frozen backbones into checkpoints. Preserve their source
files and asset caches while the experiment is needed.

## Comparison with prior AnyThermal adapter experiments

This is a documentary comparison, not an additional run. See
`ADAPTER_V2_HANDOFF_20260703.md`, `AIRE_RESULTS_20260802.md` section2 and
`RESULTS_20260926_WHERE_THE_BLUR_IS.md` section6.

The old d1/d2 routes mapped AnyThermal features through an adapter into the Lotus
condition latent; Lotus still produced depth. This required the adapter to satisfy
the expected latent interface. The old V1 target-orientation/zero-filled-target bug
is separate and must not be generalized to all later adapter runs. Later d1/d2
results did not establish a general improvement over the thermal-VAE route.

Here the entire trained DA2 depth pathway is preserved. Iris features supplement
DPT rather than replacing its encoder or its input latent. No VAE depth target,
depth decoder or latent-matching loss participates. Cross-model feature mismatch
is still a risk: retain zero-init B0 gate and frozen weights; do not claim fusion
must work merely because it is different from replacement.

Both arms inherit caption-trained Iris weights: this tests the marginal benefit
of supplying frame text to the same pretrained pipeline, not never-text-trained
versus text-trained histories. Improvement over B0 alone cannot establish a caption
contribution because the empty arm also has an extra visual feature branch.
