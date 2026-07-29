"""Entry point that replaces only the resize dependency for lean environments."""

from tests.test_ms2_unified_evaluator import UnifiedEvaluatorTests

import tests.test_ms2_unified_evaluator as original
from ms2_eval.resize_pil import resize_dense_prediction, resize_mask_nearest

original.resize_dense_prediction = resize_dense_prediction
original.resize_mask_nearest = resize_mask_nearest
