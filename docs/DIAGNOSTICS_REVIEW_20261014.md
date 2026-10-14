# Validation diagnostics: review fixes (2026-10-14)

Diagnostic only: no test-set analysis and no optimizer steps. This does not change
the frozen main-table checkpoint selection protocol.

## Submission and retained outputs

The existing submission interface is unchanged:

```bash
bash ~/Iris/slurm/submit_diagnostics.sh sd2_cap_pct_logtn_img_l0100_s43 0.1
```

Each validation submission writes a new `attempt_<job>_<random>` directory under
`$IRIS_RUNS/analysis/diag_val/<run>_step16000/`. It intentionally reruns inference
instead of trusting an old nonempty directory. Interrupted attempts remain intact.
Both raw prediction directories are retained; there is no automatic cleanup.
To change CPU analysis settings, rerun `tools/diagnose_error_sources.py` against
the retained directories, using a new `--out` directory. All manifest IDs must
have both predictions; missing predictions cause failure, never a partial table.

`summary.json` includes checkpoint SHA256, manifest metadata, analysis-code SHA256,
settings and command. `per_frame.csv` includes alignment scale/shift and clamp
counts. Raw predictions plus these parameters reproduce the scored predictions;
the analysis does not store another dense aligned copy.

## Interpretation

- Boundaries: horizontal/vertical adjacent **valid** GT pixels with depth ratio
  above 1.25. The 7x7 window defines an edge neighbourhood. It does not compare
  arbitrary pixels across a hole. Sparse centres without an adjacent valid pair
  or sufficient valid neighbours are excluded from supported regions.
- `interior` means supported non-boundary (adjacent ratios below 1.05), not a
  semantic object interior. Thresholds are diagnostic choices, not ground truth
  segmentation. Between-threshold pixels remain a separate bucket.
- `tail_by_sequence.csv`: own-error tails for each model and paired excess-error
  tails, using ceil(N*pct/100), descending error and ID tie-breaks. Reports frame
  IDs, counts, sequence-normalized rates and share of all tail frames. For a1,
  lower accuracy is worse. A paired top tail can include nonpositive differences
  if fewer than k frames are worse; it is a ranking, not a count of failures.
- Signed gap contribution can exceed 100% or be negative due to cancellation.
  Bin RMSE is sqrt(frame-weighted pixel MSE), not mean per-frame RMSE. Do not
  substitute it for the headline benchmark metric.

## Gradient probe

`PROBE_SEED` defaults to 20261014. A seed-fixed permutation selects the same
training subset for both checkpoints; no shuffle, one process, zero loader
workers. After checkpoint restoration, probe RNG is reset and reseeded for each
batch; fp16 backward scale is fixed at 1024 rather than restored from training.
The trainer records sample IDs, paths, input tensor hashes, batch seeds, manifest
SHA256 and checkpoint U-Net hashes. Compare IDs/hashes before interpreting
early/late differences. GPU numerical determinism is not claimed across hardware.

Per-term gradients share one forward graph. Reports include the actual sum
`lambda_dense*g_depth + lambda_recon*g_recon` versus weighted image L1, global
and per-module norms/cosines, and the unweighted image-L1 norm. Nonfinite gradients
abort. Existing probe output files are not appended to; submissions use a unique
attempt suffix. No optimizer step or checkpoint save occurs in probe mode.

## Local verification and cluster gate

Run the suites in separate processes on Windows to avoid the local mixed OpenMP
runtime conflict:

```powershell
& E:\conda\envs\pytorch_\python.exe -m unittest tests.test_grad_probe tests.test_diagnose_error_sources -v
& E:\conda\envs\pytorch_\python.exe -m unittest tests.test_ms2_unified_evaluator -v
```

Probe-only smoke submission (no validation inference):

```bash
PROBE_ONLY=1 PROBE_BATCHES=2 PROBE_STEPS="4000 16000" bash slurm/submit_diagnostics.sh sd2_cap_pct_logtn_img_l0100_s43 0.1
```

CPU tests do not validate the real diffusion/Accelerate/fp16 stack. Before treating
the probe as production-ready, run each checkpoint with `PROBE_BATCHES=2` and
verify finite gradients, matching sample IDs/input hashes, and absence of new
training checkpoints. This variable reduces only the probes, not validation
inference. Full defaults remain 50 batches and steps 4000/16000.
