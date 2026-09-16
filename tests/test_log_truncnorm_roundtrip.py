"""log_truncnorm closes, and only ssi_log can score it.

Three normalisations already live in this pipeline and each needs a different
inverse. The pair of bounds a frame carries cannot say which space it quantised,
so the inverse is selected by `norm_type` and getting that wrong yields a finite,
plausible-looking depth that is not the one the network predicted. These tests
pin the arithmetic instead of trusting that.

The third test is the expensive one to skip. Marigold was once reported at
AbsRel 0.265 because it was scored in the wrong space; here the same mistake is
reproduced deliberately, and it costs a factor of thirty.
"""

from __future__ import annotations

import numpy as np
import pytest

from ms2_eval.official_protocol import evaluate_sample

TRUNCNORM_MIN, TRUNCNORM_MAX = 0.02, 0.98


def normalise(depth: np.ndarray) -> tuple[np.ndarray, float, float]:
    """The lotus/utils/ms2_thermal_dataset.py `log_truncnorm` branch."""
    log_depth = np.log(np.maximum(depth, 1e-6))
    lo = float(np.quantile(log_depth, TRUNCNORM_MIN))
    hi = float(np.quantile(log_depth, TRUNCNORM_MAX))
    return ((log_depth - lo) / (hi - lo + 1e-5) - 0.5) * 2.0, lo, hi


def invert(unit: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """The train_iris_ms2_g.py `image_depth_l1_loss` inverse."""
    y = np.clip(unit, -1, 1) / 2.0 + 0.5
    return np.exp(np.clip(lo + y * (hi - lo + 1e-5), -9.0, 9.0))


@pytest.fixture
def depth() -> np.ndarray:
    return np.random.default_rng(0).uniform(3.0, 70.0, (32, 48)).astype(np.float32)


def test_inverse_recovers_metres(depth):
    """Everything inside [-1, 1] survives normalise -> invert unchanged."""
    unit, lo, hi = normalise(depth)
    inside = np.abs(unit) <= 1
    assert inside.mean() > 0.9, "the 2/98 quantiles should clip only the tails"
    assert np.abs(invert(unit, lo, hi)[inside] - depth[inside]).max() < 1e-3


def test_exponent_is_clamped_before_exp():
    """A decoder output far outside its range must stay finite.

    inf in a masked mean is nan, which kills the step rather than costing it,
    so the clamp sits on the exponent and not on the depth it produces.
    """
    assert np.isfinite(invert(np.array([-40.0, 40.0]), -3.0, 5.0)).all()


def test_ssi_log_absorbs_the_per_frame_normalisation(depth):
    """The unit-range output the network emits scores near zero under ssi_log.

    The per-frame bounds come from that frame's own GT, so at test time they
    cannot be read back -- doing so would be using test GT to preprocess. The
    two-parameter fit has to absorb them, and it can only do that in the space
    the normalisation was affine in.
    """
    unit, _lo, _hi = normalise(depth)
    prediction = (np.clip(unit, -1, 1) / 2.0 + 0.5).astype(np.float32)

    correct = evaluate_sample(prediction, depth, align="ssi_log")["abs_rel"]
    assert correct < 0.02

    for wrong_space in ("ssi", "ssi_disparity"):
        wrong = evaluate_sample(prediction, depth, align=wrong_space)["abs_rel"]
        assert wrong > 10 * correct, (
            f"{wrong_space} should be far off, not quietly plausible: {wrong:.4f}")
