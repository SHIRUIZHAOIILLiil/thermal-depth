# Repository evaluation rule

## Current handoff (2026-07-03)

Before changing or running AnyThermal → Lotus training, read
`docs/ADAPTER_V2_HANDOFF_20260703.md` in full.  It contains the fixed Windows,
WSL, Conda, dataset and manifest context, plus the Adapter V2 task checklist.

The V1 MS2 training target is a confirmed legacy bug: it directly min-max
normalizes positive depth while loading the
`lotus-depth-g-v2-1-disparity` checkpoint, and it VAE-encodes zero-filled
semi-dense GT.  Do not resume, extend, or present V1 checkpoints as final
results.  Preserve them as diagnostic evidence.  V2 must first reproduce the
upstream `trunc_disparity` convention and pass near/far orientation and invalid
pixel tests.  Do not start a large run before the V2 smoke gates in the handoff
document pass.

All V1 outputs, earlier archives, slides, reports and qualitative diagnostics
are consolidated under `archive/lotus_line_legacy_20260703`.  Do not move them
back into active output paths and do not delete them.  New experiments must use
`outputs/lotus_line_v2/...`.

The current Direct vs Adapter-only vs Adapter+U-Net route-selection phase must
use the upstream Iris/Lotus evaluator end to end:

- `lotus/evaluation/evaluation.py::evaluation_depth` for alignment and metrics;
- `lotus/utils/image_utils.py::colorize_depth_map` for saved prediction images;
- the same MS2 left-thermal input and thermal-view filtered LiDAR GT adapter;
- the same evaluator options for every route.

MS2 thermal PNGs are high-bit-depth (`uint16` / PIL `I;16`). Never call
`PIL.Image.convert("RGB")` directly on the raw thermal image: it clips the
sample to an all-white input. Decode the original array first and use the
shared `AnyThermalEncoder._array_to_uint8` high-bit-depth conversion before
replicating it to three channels for Lotus VAE input. Record or assert the
converted thermal min/max/std so saturation cannot pass silently.

Do not use `ms2_eval.visualize`, experiment-specific panels, alternative colour
maps, GT/error mosaics, or recoloured predictions in route-selection results.
Presentation comparisons may only place the untouched official `vis/*.png`
outputs side by side.

The separate final-paper checkpoint re-evaluation phase must use
`docs/MS2_UNIFIED_EVALUATION_PROTOCOL_V1.md` and the `ms2_eval` package. Never
mix its metrics or visualizations with the route-selection tables.

Before accepting a final-paper unified result, run:

```powershell
python -m unittest tests.test_ms2_unified_evaluator -v
```

Do not compare RGB-view GT with thermal-view GT. Do not call relative or
normalized-space errors RMSE in metres. Diagnostic attention/response measures
must remain separate from depth-quality metrics. Real checkpoint re-evaluation
is a separate execution phase and requires a frozen manifest/config.
