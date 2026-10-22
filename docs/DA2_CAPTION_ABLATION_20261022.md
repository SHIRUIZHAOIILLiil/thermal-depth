# DA V2 caption interventions at step 8000

Validation-only mechanism diagnostic. No training, no test split, no new model
weights. This uses the existing pilot `ms2_eval.official_protocol` evaluator
(.001 < GT < 80, frame-mean metrics; ssi and none separately), NOT the separate
final-paper unified evaluation protocol.

## Frozen experiment

- One checkpoint: `da2_caption_v1/caption_s43/step008000.pt`.
- Require internal step 8000 and consumed image cursor 96000, caption contract,
  original source hashes, asset hashes, software versions, manifest and captions.
- Same full official validation manifest: 14249 frames in original order.
- Same BF16 inference, native thermal preprocessing, native thermal-view LiDAR.
- Correct: original frame caption. First reproduce saved `val_step008000.json`
  overall AND sequence metrics in both alignments to absolute tolerance 1e-6.
  Failure stops the job; never silently loosen the tolerance. Different GPUs or
  kernels can cause numeric differences: investigate before accepting results.
- Shuffled: seed 43, global validation-frame permutation conditioned on no fixed
  points. Token IDs and masks move together; image/GT stay fixed. Full mapping
  and counts of identical strings/token inputs are recorded. No guarantee that
  another frame's caption is semantically different; this is not a hard-negative
  test and is not a within-sequence shuffle.
- Adapter off: original checkpoint visual core, same padding/cropping; bypass
  text encoder and adapters without modifying their weights. This is NOT the
  separately trained no-caption checkpoint.

## Checks performed locally

7 new CPU tests + 13 existing DA V2 CPU tests passed. 13 official metric tests
passed separately. Bash syntax passed. Tiny CPU models exercise interventions,
mapping, evaluator outputs, pairing, reproduction failure, checkpoint guards,
and weight immutability. Real GPU/weights have NOT been tested locally.
Training sources are unchanged, so existing training contracts are preserved.

## Submit on Aire

First sync the new script, test and sbatch file to Aire (local edits are not
automatically committed/pushed). If already committed and pushed, pull first.

```bash
(
set -euo pipefail
cd ~/Iris
bash -n slurm/da2_caption_ablation.sbatch
test -f /mnt/scratch/dwdh0363/runs/da2_caption_v1/caption_s43/step008000.pt
sbatch slurm/da2_caption_ablation.sbatch
)
```

The job runs CPU unit tests in the iris environment, then all three inference
passes on one GPU. No dependency on a training job and no training submission.
New output directory per job prevents overwriting previous results.

```bash
cd ~/Iris
grep -E '^\[(condition|shuffle|reproduction|result|completed)\]' da2cap_ablate-JOBID.out
tail -n 30 da2cap_ablate-JOBID.err
```

Replace JOBID with the returned job ID. Outputs:
`/mnt/scratch/dwdh0363/runs/da2_caption_v1/ablation_s8000_JOBID/`.
Includes metadata/checkpoint SHA256, shuffle mapping, reproduction report,
three metric JSONs and per-frame CSVs, final summary with overall, per-sequence,
depth-band and paired differences. No dense prediction arrays or model copies.

## Time and interpretation

Total inference = 3 x 14249 = 42747 frames. Rough planning estimate 30–90 minutes
on comparable hardware/storage, excluding queue time; this is not a benchmark.
The supplied pilot job durations include training and two validation passes, so
cannot yield an exact inference rate. Four-hour walltime is safety margin.
After the correct pass, logs report its duration to estimate the two remaining
passes. Storage should be on the order of tens of MB, not extra checkpoints.

Paired delta = intervention minus correct. Positive AbsRel/RMSE delta or
negative a1 delta favors correct caption. Means, medians and paired win fractions
are descriptive, not significance claims (adjacent video frames correlated).
Wrong-caption degradation alone may reflect harmful misinformation; consider
adapter-off results jointly. This diagnoses the 8000-step model only and does
not decide convergence, future 20000-step utility, or performance vs NeWCRF.
