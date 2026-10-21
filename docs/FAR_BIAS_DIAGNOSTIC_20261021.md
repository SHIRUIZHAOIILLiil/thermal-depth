# Far-bias CPU diagnostic

Validation only, using existing full-val raw predictions for l100, tmax1 and
NeWCRF at step 16000. No training, inference or test-set evaluation. This is not
the three-upper-bound comparison: the 0.995 arm is not included by default.

## Outputs and interpretation

- `far_bias_by_bin.csv`: `available` retains each fit's successful frames (the
  `all` fit must succeed on every selected frame). `paired` restricts every model
  and fit to the same intersection of successful frames. Use **paired** for
  all/near/far comparisons; report its coverage. Empty depth bins do not count
  toward that bin's frame mean.
- `pixel_abs_rel`: pooled pixels; `frame_mean_abs_rel`: equal weight per frame
  with a nonempty bin. Neither a sub-bin nor a limited smoke result is a full-val
  headline score. The full-frame all-fit AbsRel is checked per frame against the
  existing evaluator before aggregation.
- `median_ratio`: approximate median using up to 200 sampled pixels per frame
  and bin, not an exact pooled median or an equal-frame median. The per-frame
  far medians in `far_bias_per_frame.csv` use all eligible far pixels.
- `excluded_fits.csv`: model, fit, sequence, ID, fit-pixel count and failure
  reason. All-fit failures and invalid raw predictions abort.
- `far_bias_summary.json`: selected and paired IDs, settings, exclusion counts,
  per-frame bias distribution, and squared-error tails. Tails rank by far MSE
  descending (ID breaks ties), select ceil(10% of eligible frames), and report
  both their share of pooled SSE and their share of summed frame MSE. Zero total
  error gives null, not a fabricated percentage. This tail uses the full all-fit
  population with at least `frame_min_far` far pixels.

Default near/far boundary is 40 m; the 60–80 m bin is also reported. Far-only
alignment uses far GT and scores on it: it is an oracle sensitivity diagnostic,
not a deployable improvement or proof of causality. All fits retain each model's
existing alignment space (ours: ssi_log, NeWCRF: ssi).

## Aire submission (after syncing these changes)

```bash
cd ~/Iris
python -m unittest tests.test_far_bias_breakdown tests.test_diagnose_error_sources -q
bash -n slurm/far_bias.sbatch
SMOKE=$(sbatch --parsable --export=ALL,LIMIT=32 slurm/far_bias.sbatch)
SMOKE=${SMOKE%%;*}
sbatch --dependency=afterok:$SMOKE --export=ALL,LIMIT=0 slurm/far_bias.sbatch
```

The smoke reads the first 32 validation frames, not a representative scientific
sample. Full evaluation is a separate CPU job dependent on its success. Defaults
reuse attempts 8263904 and 8285836; override `A_L100`/`A_TMAX1` if their locations
changed. Outputs are unique per job under `$IRIS_RUNS/analysis/far_bias/`.

Local validation: 11 tests in the two modules above passed. Real Aire data and
paths remain to be checked by the smoke. An additional old depth-chain dataset
test could not run in the local Python environment because PyTorch is absent;
the CPU far-bias diagnostic does not need it.
