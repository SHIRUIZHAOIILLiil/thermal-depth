# tmax1 continuation: NeWCRF-oriented validation selection

This is a performance exploration, not a single-variable normalization ablation.
Start from `sd2_cap_pct_logtn_img_l0100_tmax1_s43/converted/step16000_weights.pt`
(original image loss weight 0.1). Continue with image weight 0.05, unchanged
latent/reconstruction losses, local-pair weight 0.01 and far frame-mean-bias
square weight 0.0005. Use q02/q100 and train only the last two U-Net up-block
convolutions for 2000 additional updates at 3e-6. Existing training functions
are reused without edits. Legacy labels 2000/3000/4000 mean added 0/1000/2000;
they are not the original checkpoint's training steps. Fresh AdamW and the
existing continuation image cursor are retained.

Selection is frozen before this run:

1. Relative to the reproduced tmax1 start, allow AbsRel increase <=0.002
   (0.2 percentage points), delta1 decrease <=0.003 (0.3 percentage points).
2. Require RMSE improvement >=0.02 m from that start.
3. Compare with verified NeWCRF on the SAME 14249 validation frames. Iris uses
   its existing ssi_log; NeWCRF uses its existing ssi. No main-table protocol change.
4. Internal deficit score = mean of max(0, ratio-1), for ratios
   AbsRel/CRF_AbsRel, RMSE/CRF_RMSE, (1-delta1)/(1-CRF_delta1).
   Better-than-CRF metrics do not cancel worse metrics. Require this score
   to improve over the start, or all three metrics to meet NeWCRF.
5. Among eligible candidates, minimize deficit score, then RMSE, AbsRel,
   and step. Otherwise retain the start. No near-band hard gates this round.

The deficit score is an experimental selection utility, not an official or
paper metric, and the equal weighting is a design choice. Full candidate
metrics and gates are saved; a composite-score improvement is not a claim
of overall superiority. Status is one of:

- retained_start: no continuation candidate qualifies;
- narrowed_gap_on_val: qualifying improvement, at least one primary metric
  remains worse than NeWCRF;
- matched_newcrf_primary3_on_val: AbsRel/RMSE/delta1 each no worse than NeWCRF
  on validation. Does not establish test performance or all-seven superiority.

Reference comes from the exact NeWCRF diagnostic cache used by job 9167854.
Read its analysis summary/CSV and verify validation identity, per-frame IDs,
alignment, depth range, checkpoint hash and reproduction of the saved RMSE
moments. No guessed AbsRel/delta1, test data or additional NeWCRF inference.
Smoke and train require identical reference hashes and selection policy.

New directory: `outputs/lotus_line_v2/iris_tmax1_bias_newcrf_v2/run_*`.
Submit: `bash slurm/submit_iris_tmax1_bias.sh`.
Dependency: smoke -> train+val. No test jobs. Can run independently of
`slurm/diagnose_iris_far_restore.sbatch`; neither writes the other's inputs.
