"""The one place the metric inverse-depth convention is written down.

Every earlier target in this project was normalised *per frame* -- the
`trunc_disparity` block in `lotus/utils/ms2_thermal_dataset.py` takes the 2nd
and 98th percentile of `1/depth` **of that frame** and maps them to [-1, 1].
That divides out each frame's own scale and shift before the VAE ever sees it,
which is why the checkpoint's output has no units and why evaluation has had to
fit an affine to test GT.

This module replaces those per-frame quantiles with two constants estimated once
on the **training split** and then frozen.  Same functional form, same [-1, 1]
target range, same [0, 1] decoded range -- only the two numbers stop moving.
That is the whole of the metric adaptation, representationally.

The four spaces, and the names used for them everywhere downstream
--------------------------------------------------------------------------
    D   metric depth, metres                    (GT: PNG / 256.0)
    q   metric inverse depth, 1/m               q = 1 / D
    u   normalised target, [-1, 1]              what the VAE *encoder* eats
    y   decoded prediction, [0, 1]              what the VAE *decoder* gives back

Verified against the code rather than assumed (see `tests/test_metric_depth_norm.py`):

  * the encoder side eats `u`.  `ms2_thermal_dataset.__getitem__` builds
    `depth_norm` in [-1, 1], `.repeat(3, 1, 1)`, and `train_iris_ms2_g.py:1054`
    hands that straight to `vae.encode(...)`.  There is no rescaling in between.
  * the decoder side gives back `y`.  `decode_to_disparity`
    (`tools/train_ms2_joint_gt_v3.py:420`) ends in `decoded.mean(dim=1) / 2 + 0.5`,
    i.e. `y = (u + 1) / 2` applied to the decoder's [-1, 1] output.

so `u` and `y` are two encodings of the same number and

    y = (u + 1) / 2        u = 2y - 1
    u = 2 (q - q_lo) / (q_hi - q_lo) - 1
    q = q_lo + y (q_hi - q_lo)                  <- exact inverse of the line above
    D = 1 / q

`unit_to_inverse` and `decoded_to_inverse` are the same function reached through
the two encodings; `test_metric_depth_norm.py` asserts they agree, so a future
edit cannot drift one without failing the other.

Clipping is never silent.  Both directions return a `ClipReport` saying which
rule fired, on how many pixels, and what the bound was.  A pixel outside
[q_lo, q_hi] is *expected* -- the quantiles are 2/98, so about 4% of training
pixels land outside by construction -- but the fraction has to be visible, since
a large one means the constants no longer describe the data being fed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:  # torch is optional here so the analysis tools can import this on any env
    import torch
except ImportError:  # pragma: no cover - exercised only on a torch-less box
    torch = None


SCHEMA = "iris.metric_depth_norm.v1"


# --------------------------------------------------------------------------- #
# the constants
# --------------------------------------------------------------------------- #


@dataclass
class MetricNorm:
    """Two frozen numbers plus everything needed to say where they came from.

    `q_lo` / `q_hi` are inverse depths in 1/m.  Everything else is provenance and
    is written to the JSON artifact so a result can be traced back to the split
    the constants were estimated on.
    """

    q_lo: float
    q_hi: float
    quantile_lo: float
    quantile_hi: float
    min_depth: float
    max_depth: float
    source_split: str
    source_manifest: str
    valid_pixels: int
    frames: int
    frame_stride: int = 1
    depth_scale: float = 256.0
    sequences: list[str] = field(default_factory=list)
    q_min_observed: float | None = None
    q_max_observed: float | None = None
    histogram_bins: int | None = None
    created_utc: str = ""
    schema: str = SCHEMA
    notes: str = ""

    def __post_init__(self) -> None:
        if not np.isfinite(self.q_lo) or not np.isfinite(self.q_hi):
            raise ValueError(f"Non-finite constants: q_lo={self.q_lo}, q_hi={self.q_hi}")
        if self.q_lo <= 0:
            # q = 1/D with D <= max_depth, so q_lo must clear zero or the
            # reciprocal at the far end of the range is unbounded.
            raise ValueError(f"q_lo must be positive, got {self.q_lo}")
        if self.q_hi <= self.q_lo:
            raise ValueError(f"q_hi ({self.q_hi}) must exceed q_lo ({self.q_lo})")
        if self.source_split != "train":
            # Loud rather than advisory: estimating these on val or test would
            # leak the split the paper reports on into the model's units, and
            # the leak would be invisible in every downstream number.
            raise ValueError(
                f"source_split is {self.source_split!r}; the normalisation constants "
                "may only be estimated on 'train'."
            )
        if not self.created_utc:
            self.created_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # -- derived ---------------------------------------------------------- #

    @property
    def span(self) -> float:
        return float(self.q_hi - self.q_lo)

    @property
    def depth_at_q_lo(self) -> float:
        """The far end of the represented range, in metres."""
        return 1.0 / self.q_lo

    @property
    def depth_at_q_hi(self) -> float:
        """The near end of the represented range, in metres."""
        return 1.0 / self.q_hi

    @property
    def q_floor(self) -> float:
        """Smallest inverse depth allowed through the reciprocal.

        Set by the evaluation range, not by the constants: a prediction is
        clamped to at most `max_depth` metres anyway, inside
        `official_protocol.official_depth_errors`.  Applying the same bound
        before `1/q` keeps the reciprocal finite without changing any metric.
        """
        return 1.0 / self.max_depth

    @property
    def q_ceil(self) -> float:
        return 1.0 / self.min_depth

    def as_affine(self) -> tuple[float, float]:
        """`q = a * y + b` for a decoded prediction `y`.

        The globally-normalised model and the train-fitted affine baseline are
        the same operation with different provenance for `(a, b)`, so both go
        through one code path in the evaluator.
        """
        return self.span, float(self.q_lo)

    # -- io --------------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["depth_at_q_lo_m"] = self.depth_at_q_lo
        payload["depth_at_q_hi_m"] = self.depth_at_q_hi
        payload["span"] = self.span
        return payload

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MetricNorm":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"{path}: schema {payload.get('schema')!r}, expected {SCHEMA!r}")
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in fields})

    def summary(self) -> str:
        return (
            f"q_lo={self.q_lo:.6f} q_hi={self.q_hi:.6f} 1/m "
            f"(depth {self.depth_at_q_hi:.2f}-{self.depth_at_q_lo:.2f} m), "
            f"Q{self.quantile_lo:g}/Q{self.quantile_hi:g} of {self.valid_pixels:,} "
            f"valid pixels over {self.frames:,} {self.source_split} frames"
        )


# --------------------------------------------------------------------------- #
# clip accounting
# --------------------------------------------------------------------------- #


@dataclass
class ClipReport:
    """What a clamp actually did.  Never discard one silently."""

    rule: str
    low_bound: float
    high_bound: float
    clipped_low: int
    clipped_high: int
    total: int

    @property
    def fraction(self) -> float:
        return (self.clipped_low + self.clipped_high) / max(1, self.total)

    def __str__(self) -> str:
        return (
            f"{self.rule}: [{self.low_bound:.6g}, {self.high_bound:.6g}] "
            f"clipped {self.clipped_low}+{self.clipped_high}/{self.total} "
            f"({self.fraction * 100:.3f}%)"
        )


def _is_torch(x) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def _clip(x, low, high):
    return x.clamp(low, high) if _is_torch(x) else np.clip(x, low, high)


def _count(mask) -> int:
    return int(mask.sum().item()) if _is_torch(mask) else int(np.count_nonzero(mask))


# --------------------------------------------------------------------------- #
# the conversions
# --------------------------------------------------------------------------- #


def depth_to_inverse(depth, norm: MetricNorm, valid=None):
    """metres -> 1/m, with invalid pixels left at zero rather than 1/0.

    `valid` defaults to the official mask, `min_depth < D < max_depth`, which is
    the same test `official_protocol.official_valid_mask` applies at evaluation.
    """
    if valid is None:
        valid = valid_depth_mask(depth, norm)
    if _is_torch(depth):
        q = torch.zeros_like(depth)
        q[valid] = 1.0 / depth[valid]
    else:
        q = np.zeros_like(depth, dtype=np.float64)
        q[valid] = 1.0 / depth[valid].astype(np.float64)
        q = q.astype(np.float32)
    return q, valid


def valid_depth_mask(depth, norm: MetricNorm):
    """The official validity test, `min_depth < D < max_depth` and finite."""
    if _is_torch(depth):
        return torch.isfinite(depth) & (depth > norm.min_depth) & (depth < norm.max_depth)
    depth = np.asarray(depth)
    return np.isfinite(depth) & (depth > norm.min_depth) & (depth < norm.max_depth)


def inverse_to_unit(q, norm: MetricNorm, clip: bool = True):
    """1/m -> the [-1, 1] map the VAE encoder eats.  Returns (u, ClipReport)."""
    u = 2.0 * (q - norm.q_lo) / norm.span - 1.0
    if not clip:
        return u, None
    low = _count(u < -1.0)
    high = _count(u > 1.0)
    total = int(u.numel()) if _is_torch(u) else int(u.size)
    report = ClipReport("unit_target_clip", -1.0, 1.0, low, high, total)
    return _clip(u, -1.0, 1.0), report


def unit_to_inverse(u, norm: MetricNorm):
    """The [-1, 1] target back to 1/m.  Exact inverse of `inverse_to_unit`."""
    return norm.q_lo + (u + 1.0) * 0.5 * norm.span


def unit_to_decoded(u):
    """[-1, 1] -> [0, 1].  `decode_to_disparity`'s `/ 2 + 0.5`, in one place."""
    return (u + 1.0) * 0.5


def decoded_to_unit(y):
    """[0, 1] -> [-1, 1]."""
    return y * 2.0 - 1.0


def decoded_to_inverse(y, norm: MetricNorm):
    """The decoded prediction straight to metric inverse depth.

    `q = q_lo + y (q_hi - q_lo)`, which is `unit_to_inverse(decoded_to_unit(y))`
    with the algebra done once.  Nothing here touches GT.
    """
    return norm.q_lo + y * norm.span


def affine_to_inverse(y, a: float, b: float):
    """`q = a*y + b` for a frozen affine fitted elsewhere (the train-only baseline).

    Same shape as `decoded_to_inverse`, which is that function with
    `(a, b) = norm.as_affine()`.  Kept separate so the evaluator can say which
    pair of numbers it used and where they came from.
    """
    return a * y + b


def inverse_to_depth(q, norm: MetricNorm):
    """1/m -> metres, with the positive clamp made explicit.  Returns (D, ClipReport).

    The clamp is `[1/max_depth, 1/min_depth]`, i.e. depth clamped to the
    official evaluation range.  `official_depth_errors` clamps aligned
    predictions to exactly that range before computing metrics, so this changes
    no reported number -- it only keeps the reciprocal finite and positive so a
    stray negative cannot become a negative "depth" and poison an average.
    """
    low = _count(q < norm.q_floor)
    high = _count(q > norm.q_ceil)
    total = int(q.numel()) if _is_torch(q) else int(q.size)
    report = ClipReport("inverse_depth_positive_clamp", norm.q_floor, norm.q_ceil, low, high, total)
    return 1.0 / _clip(q, norm.q_floor, norm.q_ceil), report


def depth_to_unit(depth, norm: MetricNorm, valid=None):
    """metres -> [-1, 1] target in one call.  Invalid pixels come back at -1.

    Invalid pixels get `q = 0`, which is below `q_lo`, so the clip pins them to
    -1: the far end of the range.  For the dense completed target that is the
    right answer anyway -- completion clips everything beyond `max_depth` to
    exactly `max_depth`, which is how sky is encoded -- and for the sparse real
    lidar these pixels are excluded by the mask before any loss sees them.
    """
    q, valid = depth_to_inverse(depth, norm, valid)
    u, report = inverse_to_unit(q, norm)
    return u, valid, report


def non_positive_fraction(q) -> float:
    """Share of inverse-depth values that cannot be reciprocated as-is."""
    total = int(q.numel()) if _is_torch(q) else int(q.size)
    bad = _count(q <= 0)
    return bad / max(1, total)


def nonfinite_count(x) -> int:
    if _is_torch(x):
        return int((~torch.isfinite(x)).sum().item())
    return int(np.count_nonzero(~np.isfinite(np.asarray(x))))


def range_line(name: str, x) -> str:
    """A one-line min/max/mean, for the range logging Step 3 asks for."""
    if _is_torch(x):
        finite = x[torch.isfinite(x)]
        if finite.numel() == 0:
            return f"{name}: all non-finite ({x.numel()} values)"
        return (
            f"{name}: min={finite.min().item():.6g} max={finite.max().item():.6g} "
            f"mean={finite.mean().item():.6g} n={finite.numel()} "
            f"nonfinite={x.numel() - finite.numel()}"
        )
    x = np.asarray(x)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return f"{name}: all non-finite ({x.size} values)"
    return (
        f"{name}: min={finite.min():.6g} max={finite.max():.6g} "
        f"mean={finite.mean():.6g} n={finite.size} nonfinite={x.size - finite.size}"
    )
