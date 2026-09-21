# -*- coding: utf-8 -*-
"""The official lidar GT of one frame, as an image that cannot overstate itself.

Asked whether a 25.4% map really looks that sparse, the honest answer is one
picture at a scale that cannot lie, not a side-by-side of the same picture at
three scales -- that was tried and the three panels were indistinguishable,
because the differences live at the pixel level and the figure gets resampled
again on the way to a screen.

So: integer nearest upscale, nothing else. Tripling means one array pixel
becomes a 3x3 block and no two pixels ever compete for one, which is why the
measured non-black fraction of the output equals the coverage of the array
exactly. Shrinking is what inflates it -- at 600 pixels wide this frame reads
as 37% and at 300 as 44%, against a true 25.4%, and interpolation='nearest'
does not prevent that: it stops the renderer blurring between pixels and does
nothing about several of them landing on one.

The same caveat applies to viewing the result. Dropped into a slide and dragged
small, or opened at 83% zoom, the inflation comes back.
"""
import numpy as np
from PIL import Image
import matplotlib

FRAME = "figs_report/f1_2021-08-13-16-31-10_004184"
SCALE = 3

lidar = np.asarray(Image.open(f"{FRAME}/lidar.png"), np.float32) / 256.0
real = np.isfinite(lidar) & (lidar > 1e-3) & (lidar < 80.0)
lo, hi = np.percentile(lidar[real], [2, 98])

rgba = matplotlib.colormaps["turbo_r"]((np.clip(lidar, lo, hi) - lo) / (hi - lo))
rgb = (rgba[..., :3] * 255).astype(np.uint8)
rgb[~real] = 0                                  # unmeasured stays black

image = Image.fromarray(rgb)
big = image.resize((image.width * SCALE, image.height * SCALE), Image.NEAREST)
big.save("lidar_raw_3x.png")

shown = float((np.asarray(big.convert("RGB")).sum(axis=2) > 20).mean())
print(f"数组 {lidar.shape}  覆盖 {real.mean():.2%}")
print(f"输出 {big.width}x{big.height}  非黑 {shown:.2%}")
assert abs(shown - real.mean()) < 1e-4, "整数倍放大不该改变比例"
print("一致 —— 放大不改变稀疏程度")
