# Adapter V2 Phase C design gate

## Decision

The first V2 route uses **condition latent distillation**. The dense teacher is
the same MS2 thermal image, decoded from its original high-bit array, converted
with `AnyThermalEncoder._array_to_uint8`, replicated to three equal channels,
mapped to `[-1,1]`, and encoded by the frozen Lotus VAE with a fixed per-sample
seed.

The Adapter receives frozen AnyThermal features from that thermal image and is
trained to match the teacher latent. AnyThermal, Lotus VAE and Lotus U-Net stay
frozen. The U-Net is not involved in the distillation loss.

Sparse thermal-view LiDAR GT is not passed to the VAE. Its zero values mean
missing measurements, not zero metres. They are excluded before reciprocal,
quantile and normalization and remain available only for target-convention
auditing and later upstream Lotus evaluation.

## Alternatives considered

1. **Selected: thermal condition latent distillation.** Dense input, no GT
   completion assumption, no invalid-pixel contamination, and no Val/Test
   teacher leakage.
2. **Deferred: U-Net training with a dense disparity teacher or audited depth
   completion.** This could supervise geometry more directly, but requires a
   separately validated Train-only teacher and must demonstrate that completion
   errors do not become false ground truth.
3. **Rejected: fill sparse invalid GT with zero and VAE-encode the full map.**
   VAE convolutions mix the fill value into neighbouring latent cells before a
   latent mask can be applied.

## Gates before any training

- Converted thermal min/max/std must be recorded and must not be constant or
  all white.
- Teacher latent sampling must be exactly reproducible for a fixed seed.
- Adapter output and teacher latent shapes must match exactly.
- Teacher and prediction must be finite.
- Only Adapter parameters may receive gradients.
- Report latent MSE, channel mean/std, cosine similarity and Pearson
  correlation together.
- Run `tools/audit_adapter_v2_phase_c.py` on exactly eight fixed Train samples;
  do not read Val or Test.
