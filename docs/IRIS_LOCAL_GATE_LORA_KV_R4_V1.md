# Native Iris local-gate LoRA continuation

Independent run: `outputs/lotus_line_v2/iris_local_gate_lora_kv_r4_v1`.
No old trainer, source checkpoint, result or manifest is modified. This is the
existing pilot `official_protocol` SSI/none experiment, not a final-paper unified
evaluation. The training entry point and submission script do not load or submit
the test split; the separately authorized attachment below adds test jobs.

## Frozen recipe

- Shared frozen base: `da2_iris_fusion_matched_pair_v2/empty_s43/step006000.pt`.
- Matched native Iris sources: old `sd2_cap_pct_logtn_s43` / `sd2_nocap_pct_logtn_s43`
  converted step18000 weights, resolved and hashed from the existing contracts.
- Gate parents: `iris_visual_gate_matched_v1/frame/step002000.pt` and
  `iris_visual_gate_matched_v1/empty/step006000.pt`. Their saved selections and
  full-validation outputs must agree. Historical optimization budgets differ;
  this is a matched-pipeline continuation, not a same-weight text-content test.
- Caption arm uses unchanged frame captions; empty arm uses empty text. Native
  legacy padding and existing mask behavior are retained. Single-frame forwards,
  VAE condition mode, timestep999, noise seed20260703+manifest index, unclipped
  FP32 depth decoding, percentile thermal preprocessing stay unchanged.
- Only the correction Iris receives LoRA, NOT the frozen Iris inside the base.
- Exact targets: `up_blocks.{2,3}.attentions.{0,1,2}.transformer_blocks.0.attn2.to_{k,v}`.
  Rank4, alpha4, dropout0, same seed43 A initialization and zero B. Expected widths
  are 1024->640 / 1024->320, giving 72192 added parameters. Original weights,
  Q/output projections, self-attention, convolutions, CLIP and VAE remain frozen.
- The inherited local gate also trains. Anchor gate is a frozen copy.

## Prediction and loss

Let `b` be the shared frozen base, `r0=w0*delta0` the original selected arm's
correction, and `r=w*delta` the current candidate. Native decoded raw Iris maps
use the same whole-image standardization as the previous gate experiment.

`depth = b * exp(r0 + log(1.05) * tanh((r-r0)/log(1.05)))`

At initialization `r=r0`, reproducing the original selected prediction. The
ratio to that original prediction is bounded to `[1/1.05, 1.05]` BEFORE SSI
alignment. This is not a guarantee of improved aligned metrics. Gate remains
in `[0,0.1]`. The final constraint is centered on the inherited arm prediction,
not the common bare base. Frozen anchor evaluation disables LoRA and runs under
no_grad; trainable candidate forward and frozen FP32 VAE decode retain autograd.
No auxiliary teacher/label is introduced. Anchors are recomputed, not cached to
scratch, to avoid adding large persistent storage requirements.

Loss is per-frame official SSI AbsRel at real thermal-view LiDAR pixels,
`.001<GT<80`, averaged over valid frames. Alignment remains differentiable.
No latent, reconstruction, pseudo-depth, language-label or inter-arm ranking loss.

## Budget and selection

- New AdamW: LoRA lr1e-5/wd0; gate lr3e-5/wd0.01. Default Adam betas/epsilon.
- Linear warmup100 updates; cosine decay to 10% peak at update4000.
- Microbatch1, accumulate12, average over valid frames, global gradient clip1.
- Both arms use identical ImageStream seed43 from cursor0 (new-stage order),
  4000 additional updates / 48000 frame draws. No added geometric augmentation.
- Trainable tensors/optimizer saved every250 updates for resume; checkpoints at
  500/1000/2000/4000. Resume validates the exact common recipe and cursor.
- Validation curve at 0/500/1000/2000/4000, stride4, same frames for both arms.
- Eligible if RMSE<=own step0 and a1>=own step0, then lowest AbsRel; ties earlier.
  Step0 is always eligible. Full validation scores 0, selected and fixed4000;
  selected point is not replaced after inspecting full validation or test data.
- Full step0 must reproduce each arm's old selected full-val metrics. Report both
  fixed-budget and validation-selected comparisons; a gap caused only by empty
  degradation is not evidence of improved caption-arm accuracy.

## Required checks and submission

Local test: `python tests/test_iris_local_gate_lora.py` (direct file execution,
avoiding Aire's conflicting `tests` package). Exercises actual small diffusers
U-Net/native RouteModel/frozen VAE backward, zero initialization, K/V gradients,
frozen parameters, checkpoint restore, correction bounds, selection guards,
paired recipe rejection and mocked Slurm dependencies/duplicate prevention.

Real paired GPU smoke separately loads both full models, verifies selected
parent predictions against saved CSV, performs two accumulated updates per arm,
checks K/V and gate gradients/updates, hashes frozen parameters before/after,
and compares LoRA initial tensors plus common images/base features/GT traces.
Its success is required by BOTH training jobs via `afterok` and matching receipt.
Real-model GPU memory/runtime remain subject to this smoke; local tests do not
claim to have run Aire weights.

```bash
(
  set -euo pipefail
  cd ~/Iris
  git pull --ff-only
  unset IRIS_MS2_ROOT IRIS_MANIFEST_DIR IRIS_DATA
  IRIS_RUNS=/mnt/scratch/dwdh0363/runs bash slurm/submit_iris_local_gate_lora.sh
)
```

Jobs: `iris_ll_smoke`, `iris_ll_frame`, `iris_ll_empty`. Train jobs include the
validation curve and selected/fixed full validation. No test jobs are submitted.
Read `submission.txt` in the new run directory if submission partially fails;
the one-shot script refuses a second submission rather than duplicating jobs.

After both arms finish:

```bash
python tools/train_iris_local_gate_lora.py report --out \
  /mnt/scratch/dwdh0363/runs/outputs/lotus_line_v2/iris_local_gate_lora_kv_r4_v1
```

## Attach selected-checkpoint tests (subsequently authorized)

`tools/test_iris_local_gate_lora.py` evaluates each arm's validation-selected
checkpoint on full day/night/rain manifests, logging SSI, none and official
stride10 SSI. It recomputes the frozen guarded selection from the saved curve;
neither the fixed4000 score nor test scores can change checkpoint selection.

Nonzero selection restores the COMPLETE selected LoRA + gate state, with the
same frozen parent anchor and 5% trust bound used in training. Step0 instead
verifies and re-folds the SAME ARM's previous local-gate test CSVs: frame2k or
empty6k. It does not reuse the bare shared fusion base. Source checkpoint,
selection, manifests, official IDs, counts, provenance and CSV hashes are checked.
Outputs are stored separately under this new run's `test_selected_v1/{arm}`.
Existing complete per-condition CSVs can be reused only under matching provenance.

Four necessary runner tests cover both step0 arms, wrong-source rejection,
nonzero checkpoint selection/restore wiring, resume without duplicate inference,
selection guards and mocked Slurm attachment. Real model inference is exercised
on Aire; these tests do not claim to evaluate real checkpoints locally.

If training is already submitted, attach ONLY the two test jobs:

```bash
(
  set -euo pipefail
  cd ~/Iris
  git pull --ff-only
  unset IRIS_MS2_ROOT IRIS_MANIFEST_DIR IRIS_DATA
  export IRIS_RUNS=/mnt/scratch/dwdh0363/runs
  bash slurm/attach_iris_local_gate_lora_tests.sh
)
```

This reads `submission.txt`, verifies owner/job names, and attaches
`iris_ll_test_frame` / `iris_ll_test_empty` with `afterok` to their respective
`iris_ll_frame` / `iris_ll_empty` jobs. It never submits training. Optional
explicit arguments are `FRAME_TRAIN_ID EMPTY_TRAIN_ID`. Attachment receipts
prevent duplicate test submissions. If a job ID is no longer eligible for a
Slurm dependency, the submission fails instead of silently dropping the dependency.

For a fresh run, execute the existing training submitter and then the attachment
script in the same `set -euo pipefail` shell. No change to the already-submitted
training code, contracts or source hashes is required.
