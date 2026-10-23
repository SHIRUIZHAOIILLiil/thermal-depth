# Frozen DA2 test evaluation

User-approved checkpoints: no-caption step18000, caption step20000, seed43.
No new selection, training, calibration, or test-driven configuration changes.
Original training/model/data sources are unchanged and verified against contracts.

Three frozen manifest SHA256 prefixes/counts:
- day3_common: f50d4506 / 23311
- night3: 273310de / 22916
- rainy3: 5a63743f / 25023

All three manifests are audited before scoring: full hashes recorded, strict
sequence allowlists, no train/validation sequence overlap, no cross-condition
frame overlap, ID/path agreement, thermal-view GT, nonempty captions.
Tokens are built from test captions without using GT and cached in test outputs.

Reuse pilot official_protocol evaluator, original native thermal preprocessing,
BF16, .001<GT<80, frame-mean metrics. Report ssi and none separately.
This is NOT the separate final-paper unified-protocol re-evaluation. No dense
predictions or extra checkpoint copies; only metrics, per-frame CSV and tokens.
The evaluator retains its generic val_step*.json filename inside TEST directories.

## Submit after syncing files to Aire

In the iris environment:

```bash
(
set -euo pipefail
cd ~/Iris
python tests/test_da2_fixed_test.py
bash -n slurm/da2_fixed_test.sbatch
sbatch -J da2_test_nocap --export=ALL,ARM=no-caption slurm/da2_fixed_test.sbatch
sbatch -J da2_test_cap --export=ALL,ARM=caption slurm/da2_fixed_test.sbatch
)
```

Two single-GPU jobs, each scores day/night/rain in sequence. No training jobs.
Same-arm flock prevents concurrent writes. Retry only a failed arm; completed
conditions are reused only when code/data/checkpoint provenance and counts match.
Successful outputs are never used to change checkpoint or select another run.

```bash
grep -hE '^\[(test-result|completed-test)\]' ~/Iris/da2_test_nocap-*.out ~/Iris/da2_test_cap-*.out
```

Results: /mnt/scratch/dwdh0363/runs/da2_caption_v1/test_fixed_20261023/
with no-caption/summary.json and caption/summary.json, plus per-condition files.

Prior measured caption validation: 14249 frames in 14.5 minutes. Scaling 71250
test frames gives about 73 minutes per arm before setup/I/O variability. Budget
roughly 1-2 hours per arm, excluding queue, not a guarantee; sbatch reserves 8h.

Local CPU tests use mocks/tiny data, not real GPU checkpoints. Run initial unit
tests on login node before submitting to avoid import errors wasting a queue.
Do not claim real test success until Aire completes all six arm/condition pairs.
