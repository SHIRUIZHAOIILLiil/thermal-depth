# SD2.1 direct-depth: existing pseudo-GT auxiliary pair

## Frozen design

Independent experiment: `outputs/lotus_line_v2/sd21_tadp_pseudo_pair_v1`.
No old experiments are edited. No teacher is added or run. Read existing
calibrated pseudo depths in metres; never use normalized Lotus targets.

Both arms start from the same original cached SD2.1-base U-Net and identically
initialized depth head, seed 43. Frame caption versus tokenized empty string;
identical architecture and native text-attention convention. Train U-Net and
head; freeze VAE and text encoder. Preserve the previous direct-depth recipe:
20,000 updates, batch 4 x accumulation 3, U-Net LR 5e-6, head LR 5e-4,
500-update warmup then polynomial decay, BF16 model / FP32 loss, no augmentation.
Do not resume the previous sparse-only 20k checkpoints into this experiment.

Loss = masked real-LiDAR SiLog (lambda 0.5) + 0.1 * pseudo-depth log-L1.
Each loss is independently averaged over its valid pixels per microbatch.
Pseudo supervision is used only where LiDAR is missing (nonfinite or <=0.001 m).
Finite LiDAR >=80 m is out of scope, not treated as missing. Pseudo depth must
be finite and strictly between 0.001 and 80 m. Invalid values are excluded, not
clipped. An empty pseudo mask contributes zero; a microbatch without any valid
real LiDAR fails as in the original recipe. Beta is fixed; no sweep option.

Validation uses real LiDAR only, exactly the previous ssi/none evaluator and
mask. Curve every 2k on stride 4; select minimum ssi AbsRel, ties earlier.
Evaluate selected checkpoint and fixed 20k over all 14,249 validation frames.
No test jobs are submitted. This tests whether the existing dense supervision
helps the caption/empty pair; improvement is not guaranteed.

## Completeness and smoke gates

Default source:
`$IRIS_RUNS/pseudo_gt/official_train/calibrated_pseudo_depth/<sample_id>.npy`.
Override only the location using `PSEUDO_DIR`. Its parent must contain the
existing builder's `build_config.json`, certifying `depth_scale: 256` and
`stored: calibrated depth in metres, no range handling applied`.
Missing provenance fails rather than silently assuming units or generating data.

Smoke checks all 75,688 official training IDs, required files, floating dtype,
256x640 shape, GT shape and usable missing-pixel coverage. It records invalid
pixel counts and SHA256 for every file in `pseudo_inventory.json`. No dataset
copy is made. Subsequent reads verify each file's content hash; source metadata
and manifest identity must also match. Validation/test IDs cannot enter this
pseudo loader. Isolated empty pseudo masks are allowed; globally zero coverage
fails. Full integrity audit occurs on Aire, not in the local synthetic tests.

The paired smoke runs 32 updates per arm and checks same initialization,
configuration and image order, frozen encoders, updated U-Net/head and nonzero
cross-attention gradients. It checks that the pseudo term is exercised and
logged total = real + 0.1 * pseudo. Successful smoke is required by both
Slurm afterok dependencies and the trainer's contract validation.

Logs include `real_loss`, `pseudo_log_l1`, `weighted_pseudo`, `pseudo_pixels`,
`lidar_pixels`, total `loss` and image-order hashes.

## Submit on Aire after syncing the new files

```bash
cd ~/Iris && IRIS_RUNS=/mnt/scratch/dwdh0363/runs bash slurm/submit_sd21_pseudo_pair.sh
```

Three jobs: `sd21_pg_smoke`, then `sd21_pg_frame` and `sd21_pg_empty` with
`afterok:<smoke ID>` and `--kill-on-invalid-dep=yes`. Smoke requests 4h for the
full file audit and both GPU passes; formal jobs request 48h, not an estimate
of actual duration. Existing formal output directories cause submission to
stop; no overwrite or implicit continuation. No job is submitted locally.

```bash
grep -hE '^\[(pseudo-audit|paired-smoke|train|completed)\]' ~/Iris/sd21_pg_smoke-*.out
```

If smoke fails, inspect its stderr before retrying; do not bypass the gate.

## Necessary local checks

```bash
python tests/test_sd21_pseudo_pair.py
python tests/test_sd21_tadp_pair.py
bash -n slurm/sd21_pseudo_pair.sbatch
bash -n slurm/submit_sd21_pseudo_pair.sh
```

Seven new CPU tests plus seven existing direct-depth regression tests passed
in the existing WSL environment. Includes real tiny Diffusers backward,
exact beta-zero value/gradient equivalence, masks/normalization, missing/corrupt
files, content changes, paired-smoke accounting and mocked Slurm dependencies.
These do not certify Aire's dataset or full SD2.1 GPU run; the smoke does that.
