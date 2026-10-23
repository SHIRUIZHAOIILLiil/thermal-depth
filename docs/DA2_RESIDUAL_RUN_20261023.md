# Residual experiment: one-shot submission

Approved: run alongside existing DA2 test jobs; no dependence on their results.
Original DA2 training/model/data sources and all old checkpoints remain unchanged.
Implementation is new, not a reproduction of missing TFDepth source.

## Fixed recipe

- Frozen no-caption_s43/step018000.pt + frozen SD2.1 CLIP.
- frame: per-frame caption; fixed: `A thermal image of a scene.`.
- Identical two-scale residual head initialization; no scalar alpha gate.
- log residual bounded to +/-log(2); only head trained; sparse real LiDAR only.
- 20000 NEW updates each, seed43, batch4 x accumulation3, AdamW 5e-5,
  weight decay .01, poly20000, gradient clipping1, BF16.
- Full validation at4000/8000/12000/16000/20000; fixed20000 comparison,
  no test evaluation, no automatic unfreezing or extension.
- Baseline full validation performed once per arm; independent outputs.
- Loss uses log of positive final prediction directly. Base is floored at .001,
  but corrected prediction can reach .0005: unlike the old loss, it is NOT floored
  a second time before log. This exactly implements the documented residual target.

## Submit (iris environment; first sync new files to Aire)

```bash
(
set -euo pipefail
cd ~/Iris
python tests/test_da2_residual.py
bash slurm/submit_da2_residual.sh
)
```

Three jobs: paired smoke (frame32 then fixed32, one GPU) -> two formal jobs,
both afterok dependent on smoke. Smoke verifies zero-head baseline identity,
finite loss/gradient, head and attention gradient updates, frozen weight hashes,
eight-frame validation, common initialization/config and exact paired frame order.
Formal jobs start FRESH residual heads, never use smoke-trained weights.
Fresh output collision fails; do not repeatedly submit the whole chain.
If sbatch fails mid-submission, printed earlier IDs still exist; inspect before retry.

Outputs: `/mnt/scratch/dwdh0363/runs/da2_caption_residual_v1/`:
`smoke_JOBID/{frame,fixed}`, `frame_s43`, `fixed_s43`, `tokens`.
The existing DA2 tests read different checkpoints/directories and can run concurrently.

```bash
grep -E '^\[(paired-smoke|completed)\]' ~/Iris/da2_res_smoke-JOBID.out
tail -n 30 ~/Iris/da2_res_smoke-JOBID.err
```

Replace JOBID. CPU unit tests also emit a synthetic pair-gate message: only a
COMPLETED real smoke job with both completed.json markers authorizes formal runs.
If smoke fails, afterok prevents either formal job starting. Do not bypass the gate.

After both formal jobs finish:

```bash
cd ~/Iris
python tools/check_da2_residual.py \
  --base /mnt/scratch/dwdh0363/runs/da2_caption_residual_v1
```

Reports all20000 paired frame hashes, full validation, and the document's
vs-fixed / vs-baseline criteria. It never submits extra work.
The final-checkpoint text perturbation diagnostic from the plan is not auto-submitted
by this chain; if needed, implement/authorize it separately after the main comparison.

## Interrupted formal run only

After checking the error, resume only that arm, keeping the same source/config:

```bash
export SMOKE_OUTPUT=/mnt/scratch/dwdh0363/runs/da2_caption_residual_v1/smoke_ORIGINAL_JOBID
sbatch -J da2_res_frame --export=ALL,ACTION=train,ARM=frame,RESUME=1 slurm/da2_residual.sbatch
```

Use ARM=fixed for the other arm. Restore head, optimizer, scheduler, RNG and
consumed image cursor. Missing milestone validations are repaired before continuing.
Head-only checkpoints retain the immutable baseline path/hash and asset hashes:
do not delete/move the baseline checkpoint or text assets while these runs are needed.

## Checks and resources

Six necessary CPU tests passed, including exact next-update resume, frozen weights,
zero identity, mask/bounds, fixed text and paired-smoke/dependency guards.
Both shell scripts passed bash -n. Real GPU weights not exercised locally.
Smoke reserves2h; formal jobs24h each (limits, not runtime predictions).
Smoke completed.json records measured seconds/update and peak GPU allocation;
formal validation adds five full passes plus the baseline pass.
No repeated frozen model copies: new checkpoint files contain residual parameters
and optimizer/RNG state only. Keep sufficient scratch space for token caches/logs.
