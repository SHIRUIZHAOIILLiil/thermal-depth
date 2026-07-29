# Repository evaluation rule

All MS2 depth experiments in this repository must follow the canonical protocol
at `E:/project/Iris/docs/MS2_UNIFIED_EVALUATION_PROTOCOL_V1.md` and use the
tested implementation in `E:/project/Iris/ms2_eval` until that package is moved
to a shared installable location.

Experiment-specific code may perform inference, but must not redefine GT
decoding, validity masks, resizing, alignment, metrics, aggregation, or main
visualization normalization. Export route-native `.npy` predictions keyed by
the frozen manifest sample IDs, then invoke the canonical evaluator.

Do not compare RGB-view GT with thermal-view GT. Do not label relative or
normalized-space errors as RMSE in metres. Attention entropy and prediction
change are mechanism diagnostics, not depth-quality metrics. Real checkpoint
re-evaluation begins only after evaluator tests pass and the manifest/config are
frozen.
