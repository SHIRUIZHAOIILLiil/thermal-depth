# Adapter V2 Phase B: Lotus `trunc_disparity` target audit

Audited on 2026-07-03 against:

- `lotus/utils/hypersim_dataset.py`
- `lotus/utils/vkitti_dataset.py`
- `lotus/train_iris_g.py`

The training entry point defaults to `norm_type="trunc_disparity"` and
`truncnorm_min=0.02`. Both upstream datasets therefore use the per-image valid
disparities' 2% and 98% quantiles. For valid positive depth `z`:

```text
d = 1 / z
d_lo = quantile(d_valid, 0.02)
d_hi = quantile(d_valid, 0.98)
target = clip(2 * ((d - d_lo) / (d_hi - d_lo + 1e-5) - 0.5), -1, 1)
```

This orientation assigns larger target values to nearer pixels. The V2
implementation is `models/lotus_target_v2.py::trunc_disparity_target` and does
not call the legacy `tools/overfit_32_anythermal_lotus.py::load_depth`.

VKITTI establishes the relevant invalid-pixel rule: quantiles are computed
only over its valid positive mask. V2 tightens the implementation so masking
happens before reciprocal and normalization. Invalid output positions remain
`NaN`, preventing the sparse target from being silently zero-filled and sent
to the VAE before the separate Phase C dense-target design is approved.

`seeded_target_latent_and_noise` accepts only finite dense `[B,3,H,W]` targets
and uses explicit generators for both the stochastic VAE latent and diffusion
noise. Its reproducibility contract is covered by the Phase B unit tests.
