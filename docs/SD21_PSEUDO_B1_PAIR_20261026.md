# SD2.1: beta=1.0 frame/empty pair

Only experimental change from `sd21_tadp_pseudo_pair_v1`: pseudo log-L1
coefficient 0.1 -> 1.0. Independent source files and output directory:
`$IRIS_RUNS/outputs/lotus_line_v2/sd21_tadp_pseudo_b1_pair_v1`.
Old experiments and calibrated pseudo files are not modified.

Loss: real-LiDAR masked SiLog (lambda=.5) + 1.0 * missing-pixel pseudo log-L1.
Both terms retain separate microbatch pixel means. No clipping or recalibration;
same valid masks and existing calibrated metric pseudo loader as beta=.1.
Default pseudo location remains
`$IRIS_RUNS/pseudo_gt/official_train/calibrated_pseudo_depth` (PSEUDO_DIR override).
Full training-file completeness, shape, provenance and SHA256 audit remains in
smoke; subsequent reads verify hashes. No new teacher or pseudo generation.

Both arms initialize from original cached SD2.1-base plus the same random depth
head, seed43; no warm-start from beta=.1. U-Net/head trainable, VAE/text frozen.
Frame caption versus tokenized empty; identical architecture/mask convention.
20k steps, batch4 x accumulation3, U-Net/head LR5e-6/5e-4, 500 warmup,
polynomial decay, BF16, same image order and no augmentation. Validation uses
real LiDAR only, stride4 curve every2k, minimum ssi AbsRel with earlier tie-break;
selected and fixed20k full validation reported separately. No test jobs.

## Checks and dependency

8 beta=1 tests plus 7 beta=.1 regression tests passed in WSL. Includes exact
trainer-recipe parity except beta/identity, loss and gradient scaling, masks,
beta-zero equivalence, pseudo-file integrity, tiny Diffusers backward, paired
smoke accounting and mocked submission. Both shell scripts pass bash -n.
This does not replace the actual Aire GPU and dataset smoke.

One smoke job runs both arms for32 updates and requires same initialization,
image order and contracts, nonzero pseudo contribution, frozen encoders,
updated U-Net/head and text cross-attention gradients. Both formal jobs depend
on afterok and recheck the matching smoke contract. Invalid dependencies cancel
the pending training jobs. Existing output directories cannot be overwritten.

## One-shot Aire submission (after syncing the six new files)

```bash
cd ~/Iris && IRIS_RUNS=/mnt/scratch/dwdh0363/runs bash slurm/submit_sd21_pseudo_b1_pair.sh
```

Creates sd21_pg1_smoke (4h requested), then sd21_pg1_frame and sd21_pg1_empty
(48h requested each, training plus validation). Requested time is not runtime
prediction. No training was submitted from the development machine.

```bash
grep -hE '^\[(pseudo-audit|paired-smoke|train|completed)\]' ~/Iris/sd21_pg1_smoke-*.out
```

New files: models/sd21_pseudo_b1_loss.py, tools/train_sd21_pseudo_b1_pair.py,
slurm/sd21_pseudo_b1_pair.sbatch, slurm/submit_sd21_pseudo_b1_pair.sh,
tests/test_sd21_pseudo_b1_pair.py, this document.

Interpretation: beta1 tests stronger dense supervision, not a proven optimal
weight or an equivalent of Iris latent supervision. Judge frame vs empty on
validation before deciding on any test evaluation; do not select on day test.
