# Final targeted diagnostic before choosing a training hypothesis

Scope: read existing validation predictions for l100, tmax1 and NeWCRF. No
training, inference, test access, checkpoint selection or changes to evaluation.
Runs inside `slurm/far_bias.sbatch` with `--structure-reference newcrf`.

## Added outputs

`far_structure_per_frame.csv` contains per-model/sequence/frame statistics for
GT >=40 m and >=60 m (at least 50 valid pixels). All models use identical GT masks
and IDs, with the original whole-frame fit. Unlike the near/far fit comparison,
these diagnostics do not require a successful far-only fit.

- MSE gap = model minus NeWCRF, in metres squared. Positive is worse. Do not call
  this an RMSE gap. Also reports far SSE divided by the full-frame valid count,
  to distinguish per-far-pixel errors from contribution to whole-frame MSE.
- Spearman: average ranks for ties, at most 4096 sampled pixels per frame.
  Reports both raw-output and whole-frame-aligned ordering. Constant predictions
  have undefined correlation (null), not a silently imputed zero.
- Ordering accuracy: randomly paired sampled pixels with GT separation >=5 m;
  strict correct ordering and tied-prediction share are separate. Each model uses
  the same ID-derived seed, pixels and pairs. No eligible pairs gives null.
- Range ratio: (prediction P90-P10)/(GT P90-P10) on all valid pixels in the band,
  using unclipped whole-frame-aligned metres. Near-zero GT span gives null.
  Also exports GT and prediction spans and the evaluation clamp fraction.

`far_structure_summary.json` includes by-sequence signed and positive-excess gap
shares and the worst ceil(10%) paired frames. Signed contributions can be negative
or above 100% because another sequence can offset a loss. Positive-excess shares
measure only where ours is worse. The full-frame-contribution calculation uses
the same eligible frames, NOT the entire validation-set denominator.

Structural summaries are medians of defined per-frame values, with counts.
Check undefined counts and GT spans before comparing them. Bands overlap and
must not be added together. Sequence duration affects total contributions;
inspect both mean gap and contribution share.

## Interpretation and stopping rule

Use the same-frame NeWCRF reference, not invented absolute thresholds:

1. If ordering is comparable but aligned range is smaller, prioritize a single
   hypothesis about depth-range representation/supervision while protecting near
   accuracy. Range ratio alone cannot distinguish smooth compression from noise.
2. If ordering is materially worse, a scalar calibration is insufficient; select
   a structural-supervision hypothesis supported by the existing target-chain
   evidence. This does not on its own prove that pseudo GT or features caused it.
3. If a sequence dominates, examine mean paired gap as well as its frame share;
   do not train on validation frames or conclude distribution mismatch solely from
   the sequence name.

This diagnostic informs one proposed training arm; it does not auto-launch one.
Retain l100, the frozen validation gates and untouched test sets. If evidence is
mixed, report the uncertainty and choose the smallest justified hypothesis rather
than starting another broad parameter sweep. No promise of exceeding NeWCRF.

## Submission after syncing local changes to Aire

```bash
(
set -euo pipefail
cd ~/Iris
python -m unittest tests.test_far_structure tests.test_far_bias_breakdown tests.test_diagnose_error_sources -q
bash -n slurm/far_bias.sbatch
SMOKE=$(sbatch --parsable --export=ALL,LIMIT=32 slurm/far_bias.sbatch)
SMOKE=${SMOKE%%;*}
echo "smoke=$SMOKE"
sbatch --dependency=afterok:$SMOKE --export=ALL,LIMIT=0 slurm/far_bias.sbatch
)
```

Read sections 4, 4b and 5 from the full job's log. Sections 1–3 remain unchanged.
Local checks: 16 CPU tests passed, including a synthetic end-to-end output test,
correct-order compression, reversed order, constant predictions, tied ranks,
model-independent sampling, signed-gap cancellation and unmatched-ID rejection.
Actual Aire inputs are checked by the smoke; no real-data run was performed locally.
