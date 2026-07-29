# MS2 Unified Depth Evaluation and Visualization Protocol v1

Status: implementation frozen; real-checkpoint re-evaluation not yet run.

## Non-negotiable data contract

- Evaluation is JSONL-manifest driven. Record manifest absolute path and SHA256.
- Input is the left thermal image.
- GT is thermal-view filtered LiDAR depth. RGB-view GT is not a fallback and may
  not appear in the same comparison table.
- Preserve sample ID, sequence, condition, split, and manifest line.
- The existing `thermal-depth/src/data/ms2_dataset.py` loader confirms MS2 depth
  PNGs are `uint16` with `depth_m = raw / 256.0`.
- Protocol-v1 defaults are `min_depth=0.1 m`, `max_depth=80.0 m`, with the strict
  validity rule `finite(gt) & gt > min_depth & gt < max_depth`. This deliberately
  differs from the old loader's inclusive upper bound and must be recorded.
- At least eight uniformly spaced GT samples must be audited before inference.

## Geometry contract

Dense predictions are resized to untouched GT resolution with bilinear
interpolation. Masks use nearest interpolation. Sparse GT is never interpolated.
Original and final resolutions are logged.

Each inference route exports a raw `.npy` named by sample ID. Its registered
adapter declares representation, orientation, metric-scale availability,
positive-depth conversion, and clipping. The four route names are:

- `iris-lotus`: relative disparity; no native metric scale.
- `adapter-only`: Lotus relative disparity; no native metric scale.
- `adapter+u-net`: Lotus relative disparity; no native metric scale.
- `sp-dit`: restored metric depth in metres.

Non-metric routes do not receive raw `RMSE (m)` values. They receive aligned
metric results only. Any future calibration that changes this must create a new
adapter declaration and protocol revision.

## Metrics and aggregation

Native metric metrics, where legal: AbsRel, SqRel, RMSE (m), RMSElog, delta1,
delta2, delta3. Per-image affine alignment fits
`aligned = scale * native_depth + shift` using valid GT pixels only. Aligned
metrics are AbsRel, RMSE (m), RMSElog, delta1, Pearson, and optional Spearman.

Save image-wise mean, population standard deviation, median, global-pixel
aggregation, valid-pixel counts, condition summaries, and paired 95% bootstrap
confidence intervals. Image-wise (macro) results are primary; global pixel
results are secondary and labelled. Normalized-space losses are not metres.

## Controlled ablations

Caption modes `correct`, `empty`, and `hard-wrong` must share image, checkpoint,
seed, scheduler, denoising steps, and initial latent/noise. Report paired win
rates, mean/median improvements, and bootstrap CI. Attention entropy and
prediction-change ratio live in mechanism diagnostics, never metric tables.

Adapter-only versus Adapter+U-Net must use the same validation manifest and
checkpoint selection rule. Diffusion loss is reported separately and cannot be
used as a proxy for geometry quality.

## Main visualization

Every route uses `ms2_eval.visualize.save_shared_visualization`: left thermal,
sparse thermal-view GT, raw metric prediction when legal, raw prediction sampled
on the GT mask, affine-aligned prediction, and valid-pixel absolute error. Use
one fixed `magma_r` depth map and one fixed metre range per comparison, with a
shared color bar. Invalid GT is black. Main figures never use per-image min-max.
Optional structure views must say `relative depth (not metres)`.

## Required output tree

```text
config_resolved.yaml
run_metadata.json
checkpoint_info.json
predictions/*.npy
metrics/per_image.csv
metrics/summary.json
metrics/summary_by_condition.json
metrics/bootstrap_comparison.json
visualizations/*
logs/*
```

## Gates

Unit tests must pass before real inference. A final report has seven separate
sections: implementation audit, native protocol, unified raw metric, unified
affine-aligned, caption ablation, training ablation, and qualitative failures.
Tables must never mix GT view, sample count, alignment, or representation.

## Commands

Audit only:

```powershell
python tools/run_unified_ms2_evaluation.py `
  --config configs/ms2_unified_evaluation_v1.example.yaml `
  --audit-only
```

Evaluation uses the same command after setting `audit_only: false`, selecting a
route/checkpoint, and exporting route-native predictions to `prediction_dir`.
