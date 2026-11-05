# Restore censored far LiDAR: target-chain diagnostic

Purpose: test whether the existing frozen Iris VAE retains recovered real
60–80 m values without degrading near/mid reconstruction. No training, test
evaluation, U-Net inference, new teacher, or pseudo-label generation.

Inputs are read-only from
`iris_val_chain_bias4k_v1/job_9174706`: its frozen `val96.jsonl`,
`diagnostic_pseudo/*.npy`, and `summary.json`. Require the same official
validation manifest (14249 frames, SHA256 prefix c26a5859), deterministic
96-frame selection (seed 20261017), per-frame far-point coverage, and
reproduction of the old target/VAE far bias and RMSE within 0.0005 m.

The original dataset builds targets exactly as before, with real LiDAR written
over calibrated pseudo depth and log quantiles q02/q98. Copy this target; only
where valid real GT satisfies `60 <= GT < 80` and its original unbounded
normalized value exceeds 1, restore
`2 * ((log(GT)-lo)/(hi-lo+1e-5) - 0.5)` to all three channels. Do not recompute
bounds or alter any other pixel. Assert exact equality elsewhere and inverse
recovery of the restored GT values. Lower clipping, sky and pseudo-only pixels
are unchanged.

Run the original and restored targets through the same frozen float32 VAE,
using posterior mode, no sampling, and no decoder unit clamp. Four stages:
original_unit, restored_unit, original_vae, restored_vae. Score whole frame,
0–20, 20–40, 40–60, 60–80 m and restored points, with paired masks. Report
per-frame CSV and per-sequence/ALL frame-mean summaries for both:

- target_scale: invert the original known bounds; isolates representation
  effects, but is not a deployable metric-depth prediction.
- ssi_log: fit on all valid points of the frame before binning. Near-depth
  changes here can include changed alignment; compare target_scale too.

Report bias, RMSE, AbsRel, delta1 and prediction fraction above 80 m before
evaluation clamping. At restored points, report signed decoded depth lift
divided by input depth lift (target_scale). This ratio may be negative or
above one; it is not accuracy and does not prove generalization or caption gain.

Interpretation: positive input lift with negligible decoded lift means this
target repair does not survive the current codec. Preserved lift with reduced
far error and stable near/mid reconstruction supports a subsequent controlled
training trial, not automatic submission. A large near/mid degradation argues
against proceeding unchanged. No model selection or automatic training gate.

New outputs: `outputs/lotus_line_v2/iris_far_restore_chain_v1/job_JOBID`.
Submit: `sbatch slurm/diagnose_iris_far_restore.sbatch` after pulling.
The source diagnostic must remain on scratch; missing files fail explicitly.
