"""Build a diagnostic explaining the legacy GT-mask visualization effect.

This is not a route-selection result and must not be mixed into the official
Iris/Lotus comparison figures.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = (
    ROOT
    / "outputs/_legacy_archive_20260701/adapter_v0_caption_sensitivity_v2"
    / "samples/00_2021-08-06-11-23-45_000000"
)
OUT = ROOT / "docs/diagnostics"


def normalize(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    lo, hi = float(array.min()), float(array.max())
    if hi <= lo:
        return np.zeros(array.shape, dtype=np.uint8)
    return (np.clip((array - lo) / (hi - lo), 0, 1) * 255).round().astype(np.uint8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prediction = np.load(SAMPLE / "pred_real.npy")
    valid = np.load(SAMPLE / "valid_mask.npy").astype(bool)

    full = normalize(prediction)
    mask = valid.astype(np.uint8) * 255
    sampled = np.where(valid, full, 0).astype(np.uint8)

    panels = [
        ("A  SAME RAW PREDICTION - FULL FIELD", full),
        (f"B  THERMAL GT VALID MASK - {valid.mean() * 100:.1f}%", mask),
        ("C  OLD DISPLAY - A SAMPLED ON B", sampled),
    ]
    width, height = full.shape[1], full.shape[0]
    header = 42
    canvas = Image.new("RGB", (width * 3, height + header), (16, 23, 36))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (label, array) in enumerate(panels):
        x = index * width
        canvas.paste(Image.fromarray(array, mode="L").convert("RGB"), (x, header))
        draw.text((x + 12, 15), label, fill=(245, 248, 255), font=font)
    target = OUT / "same_prediction_full_vs_gt_masked.png"
    canvas.save(target)

    readme = OUT / "ANYTHERMAL_VISUALIZATION_GAP.md"
    readme.write_text(
        """# AnyThermal + Lotus visualization gap diagnostic

This diagnostic uses one legacy AnyThermal + Lotus prediction from sample
`2021-08-06-11-23-45_000000`.

- Panel A is the complete dense `pred_real.npy` array. Every pixel has a value.
- Panel B is the thermal-view LiDAR GT validity mask. It covers 28.1% of pixels.
- Panel C applies Panel B to Panel A and sets all other pixels to black. This is
  how the legacy `pred_real.png` was generated.

The object/road scan pattern in Panel C is therefore partly imposed by the GT
validity mask. It is not evidence that the dense prediction in Panel A contains
the same sharp geometry across the full image.

The legacy implementation is in `tools/train_ms2_adapter_v0.py`:

- `save_image` accepts `valid` at line 811;
- line 824 applies `np.where(valid, norm, 0.0)`;
- lines 969–970 save raw and aligned predictions with that GT mask.

The current official Iris/Lotus evaluator saves the complete aligned prediction
without applying the GT mask. Metrics in both cases are still evaluated only on
valid GT pixels; this diagnostic concerns qualitative display, not metric-mask
correctness.

## Prompt to give another model

Please inspect `same_prediction_full_vs_gt_masked.png` and explain why Panel C
looks more geometrically detailed than Panel A even though both originate from
the same dense prediction array. Panel B is a sparse LiDAR ground-truth validity
mask, and Panel C is computed as `where(mask, minmax(prediction), 0)`. Distinguish
between: (1) a dense prediction array, (2) a prediction visualized only at GT-valid
pixels, and (3) whether the model recovered dense scene geometry. Also explain
whether metrics restricted to valid GT pixels can remain legitimate while the
qualitative visualization is misleading if it is labelled simply as “prediction”.
""",
        encoding="utf-8",
    )
    print(target)
    print(readme)


if __name__ == "__main__":
    main()
