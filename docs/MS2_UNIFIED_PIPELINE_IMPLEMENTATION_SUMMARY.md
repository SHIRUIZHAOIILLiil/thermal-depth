# Unified MS2 pipeline implementation summary

Implemented in this phase:

- canonical metric and affine-alignment implementation;
- strict validity-mask contract and official uint16/256 GT audit;
- manifest hash and thermal-view enforcement;
- dense-prediction and nearest-mask resize utilities;
- explicit adapters for Iris/Lotus, Adapter-only, Adapter+U-Net, and SP-DiT;
- image summaries, condition summaries, global-pixel aggregation, paired
  bootstrap utilities, caption and training ablation contracts;
- shared fixed-scale visualization;
- required output tree, CLI, config template, final-report skeleton;
- synthetic evaluator unit tests.

Deferred by instruction: exporting/running every real checkpoint and producing
the populated seven-section experimental report. Existing experiment-specific
metrics are not imported or trusted automatically.
