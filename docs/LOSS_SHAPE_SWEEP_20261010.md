# Loss sweep and aligned-tail plan (2026-10-10)

## What is isolated

Run the three decisions in order. Do not submit the next stage until the
previous stage has been selected on validation.

1. **Image-loss weight:** change only `lambda_image`.
2. **Tail shape:** keep the selected image weight and add a metre-space Huber
   term without alignment.
3. **Train alignment:** keep both weights and the Huber shape, and change only
   `tail_align=none` to `tail_align=ssi_log`.

The fixed control is `sd2_cap_pct_logtn_img_s43`, step 16000: SD2.1-base
start, seed 43, captions, percentile thermal conversion, `log_truncnorm`,
legacy text padding, random flip on, and image-L1 weight 0.05. Existing test
results are not used for any choice below.

## Frozen protocol

- Train: `ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl`
  (75,688 frames).
- Validation: `ms2_val_official3_thermalcap_20260821.jsonl`.
- Route `b_thermal_unet`, prompt `correct`, condition latent `mode`.
- Official upstream Iris/Lotus evaluator with `ssi_log`.
- Full validation: `VAL_STRIDE=1`.
- Primary comparison: step 16000 for every arm. `FIXED_STEP=16000` overrides
  the printed validation minimum.
- `VAL_ONLY=1` exits before any test-manifest check, fingerprint, or inference.
- Huber transition is 5 m. Tail weight is 0.01 in both tail stages.
- Every new code path first runs a fresh 20-step smoke. Full training is a
  separate command and is submitted only after the smoke log is inspected.

A constant decoded frame has no identifiable affine scale. It is excluded only
from the aligned auxiliary term and counted as `tail_frames_degenerate`;
latent and image-L1 supervision remain active.

## Frozen selection rule

The previous-stage control is always eligible, so a stage can select "no
change". A candidate is eligible only if, relative to that control:

- AbsRel regression is no more than 0.0005;
- delta1 regression is no more than 0.001.

Rank eligible arms by lowest RMSE, then SqRel, then AbsRel. This preserves the
current AbsRel/delta1 advantage while targeting the NeWCRF gap in RMSE and
SqRel. `tools/select_loss_sweep.py` prints all seven metrics and the decision.

## Submit stage 1 on AIRE

After synchronising this repository to `~/Iris`:

```bash
cd ~/Iris
bash slurm/submit_loss_sweep.sh lambda-smoke
```

Inspect the smoke gates below. Only after they pass:

```bash
bash slurm/submit_loss_sweep.sh lambda
```

This re-evaluates the existing 0.05 control and submits independent 0.025, 0.1,
and 0.2 arms. Each evaluation is full-val, fixed-step, and validation-only.

New run names:

```text
sd2_cap_pct_logtn_img_l0025_s43
sd2_cap_pct_logtn_img_l0100_s43
sd2_cap_pct_logtn_img_l0200_s43
```

Select the lambda:

```bash
source ~/Iris/slurm/common.sh
VM="$IRIS_MANIFEST_DIR/ms2_val_official3_thermalcap_20260821.jsonl"
VF=$(sha256sum "$VM" | cut -c1-8)
SFX=eval_eval_affine_invariant_log_space.json
ROOT="$IRIS_RUNS/eval"

python tools/select_loss_sweep.py \
  --control "l050=$ROOT/sd2_cap_pct_logtn_img_s43_val_${VF}_stride1_correct_s16000/$SFX" \
  --candidate \
    "l025=$ROOT/sd2_cap_pct_logtn_img_l0025_s43_val_${VF}_stride1_correct_s16000/$SFX" \
    "l100=$ROOT/sd2_cap_pct_logtn_img_l0100_s43_val_${VF}_stride1_correct_s16000/$SFX" \
    "l200=$ROOT/sd2_cap_pct_logtn_img_l0200_s43_val_${VF}_stride1_correct_s16000/$SFX"
```

## Submit stages 2 and 3 separately

Pass the validation-selected numeric value to the shape stage:

```bash
bash slurm/submit_loss_sweep.sh shape-smoke 0.05
# inspect the completed smoke log, then:
bash slurm/submit_loss_sweep.sh shape 0.05
```

Replace 0.05 with the winner. This adds
`lambda_tail=0.01, tail_loss=huber, tail_align=none, delta=5m` and changes
nothing else. Compare it against that lambda's no-tail control using the same
selector.

Only if Huber survives the gate, submit alignment:

```bash
bash slurm/submit_loss_sweep.sh aligned-smoke 0.05
# inspect the completed smoke log, then:
bash slurm/submit_loss_sweep.sh aligned 0.05
```

The aligned run is identical to the Huber run except
`tail_align=ssi_log`. Compare it against the unaligned Huber run, not against
the original model; this isolates alignment.

## Smoke gates

Inspect each smoke log before submitting the corresponding full job:

- the configuration line shows the requested image/tail weights and alignment;
- `SL_A`, `SL_R`, `SL_I`, and, for tail stages, `SL_T` are finite;
- `tFrames` is nonzero and aligned frames are not all degenerate;
- thermal conversion statistics are non-saturated;
- target coverage remains 100%.

Do not submit the full job after a failed smoke, and do not change another
experimental factor to make the smoke pass.

## Test boundary

Do not evaluate any sweep arm on test. After the aligned-stage winner is frozen,
evaluate exactly that final model once on day/night/rain using the same official
route-selection evaluator. Final-paper checkpoint re-evaluation remains
separate under `docs/MS2_UNIFIED_EVALUATION_PROTOCOL_V1.md`.

