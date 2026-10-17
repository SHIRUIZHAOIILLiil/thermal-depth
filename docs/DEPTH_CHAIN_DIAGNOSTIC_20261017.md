# Three-bound target/VAE/model diagnostic

No training, no optimizer, no checkpoint modification, no test-set input. This is
a mechanism diagnostic, not a main-table evaluation or a new checkpoint selector.

## Fixed design

- Same seed-fixed random sample of 256 official training frames for q098/q0995/q100;
  a separate fixed 96-frame official validation sample for model inference only.
  Selection does not depend on prediction errors. Coverage in each GT depth bin is
  recorded. Zero far-GT coverage aborts; increase the count instead of picking good
  examples. These are small samples, not the full validation benchmark.
- Lower quantile .02, upper .98/.995/1.0. Actual `MS2ThermalDataset` target path,
  no augmentation, no sky override, real LiDAR overwrite on calibrated pseudo.
- Three matching arms at **step16000**, not independently selected checkpoints.
  Existing converted checkpoint files are required; no conversion/training starts
  automatically if one is absent.
- Correct frame caption; percentile thermal conversion; legacy text padding;
  single inference, no flip/ensemble, condition VAE mode. All arms have identical
  subset ordering and inference seed. Noise uses subset position, so these sampled
  evaluations need not reproduce the same frame from a full-manifest evaluation.
- VAE target roundtrip uses fp32 and reports both posterior mode and one sampled
  latent. The sample uses identical standard-normal epsilon across three bounds
  for each ID. This isolates representation, not exact fp16 stochastic training.

## Four phases, one GPU job

1. `prepare`: verify official train/val sequence membership, freeze manifests,
   checkpoint/source-code hashes, selection seed and far-GT coverage.
2. `targets`: calibrated pseudo before overwrite -> LiDAR-fused dense target ->
   clipped normalized target -> VAE roundtrip. Uses actual training dataset code.
   Saves dense arrays, bounds, per-frame pseudo/GT hashes, VAE parameter hash and
   thermal range/std. No invalid sparse depth is encoded directly.
3. `models`: invokes the established `train_route_suite.py --eval-checkpoint` path
   for each arm on the frozen train and val lists. That option returns before any
   training loop; output contains raw unit predictions.
4. `score`: paired stage-wise/GT-depth-bin metrics and model-versus-target unit MAE.

## Interpretation safeguards

`pseudo` is the **calibrated** AnyThermal output before LiDAR overwrite. Its
calibration already used same-frame LiDAR, so it is not an independent teacher
accuracy/held-out test. True pre-calibration output is not available from this
directory and is not silently substituted.

`dense` is expected to match GT on measured pixels by construction. That says
nothing about its accuracy at missing pixels. `no_lidar` model-to-target unit MAE
measures agreement, not depth accuracy.

`metric_no_fit`: pseudo/dense in metres, no per-frame fit. `known_target_scale`:
unit target, VAE or model output inverted with that frame's known training-target
bounds. For the model this is **oracle-scale diagnostic**, NOT native metric
depth inference. Decoder output is not clipped to [0,1]; log exponent clamp
[-9,9] matches training. Metrics then clamp depths to [.001,80], with out-of-range
shares reported. `ssi_log`: official per-frame fit on all valid LiDAR, then reuse
the fit for every bin (no per-bin fitting). Val has only this aligned result and
never gets target-derived bounds.

All GT is thermal-view filtered LiDAR at original resolution. Bins are 0–5,
5–20, 20–40, 40–60, 60–80 m, plus whole-frame. Summary is a macro mean over frames
having valid pixels in a bin; bin RMSE is mean per-frame bin RMSE, NOT the weighted
pooled-bin RMSE in the earlier `diagnose_error_sources` table. Compare arms within
this output; do not splice its bin numbers into the older table.

`stage_deltas.csv` gives paired metric changes dense -> clipped target -> VAE ->
model. These are error differences, NOT an additive causal attribution. VAE
roundtrip is not a strict model ceiling. Unit MAE is not metres. Train-vs-val gaps
also reflect scene composition; they alone cannot prove overfitting.

## Aire commands (after syncing new files to ~/Iris)

Smoke is separate and never auto-submits a larger job:

```bash
cd ~/Iris
CHAIN_SMOKE=1 sbatch --job-name=chain_smoke --time=01:00:00 --export=ALL,CHAIN_SMOKE slurm/diagnose_depth_chain.sbatch
```

This runs 8 train + 8 val frames through **all three arms**, including scoring.
Check exit status COMPLETED, `COMPLETED: .../analysis/summary.csv`, three arms in
summary, near-zero dense/no-fit GT error, and finite stage metrics. Zero far-GT
coverage or missing checkpoints fails. Other empty bins are omitted and their
zero coverage is recorded in config, never reported as zero error.

After reviewing smoke, submit the larger diagnostic explicitly:

```bash
cd ~/Iris
CHAIN_SMOKE=0 sbatch --job-name=depth_chain --export=ALL,CHAIN_SMOKE slurm/diagnose_depth_chain.sbatch
```

Fresh output: `$IRIS_RUNS/analysis/depth_chain/attempt_<job>_<random>/`.
Read `config.json`, `analysis/summary.csv`, `analysis/stage_deltas.csv` and
`analysis/model_vs_target_unit.json`. Raw NPZ/NPY predictions are retained.
No shell `SMOKE` or `RESUME_FROM` variables are used. Do not set unrelated data
root/pseudo directory overrides unless intended. Each phase refuses stale outputs.

## Local checks

```powershell
& E:\conda\envs\pytorch_\python.exe -m unittest tests.test_depth_chain -v
& E:\conda\envs\pytorch_\python.exe -m unittest tests.test_ms2_unified_evaluator -v
```

CPU tests cover split/ID safety, deterministic selection, actual dataset's three
upper bounds, inverse clipping, official alignment consistency, inference-only
commands and full scoring with synthetic arrays. Real cached diffusion weights,
GPU inference and cluster environment still require the smoke run.
