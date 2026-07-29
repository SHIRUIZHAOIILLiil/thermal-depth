"""One shared MS2 visualization implementation for every model route."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _thermal_for_display(path: str | Path) -> np.ndarray:
    with Image.open(path) as image: raw = np.asarray(image).astype(np.float32)
    if raw.ndim == 3: return np.asarray(Image.open(path).convert("RGB"))
    finite = np.isfinite(raw)
    if not finite.any(): return np.zeros(raw.shape, np.float32)
    lo, hi = np.percentile(raw[finite], [1, 99])
    return np.clip((raw - lo) / max(float(hi - lo), 1e-6), 0, 1)


def save_shared_visualization(
    output_dir: str | Path, *, sample_id: str, thermal_path: str | Path,
    gt_m: np.ndarray, valid: np.ndarray, native_depth: np.ndarray,
    aligned_depth_m: np.ndarray, raw_is_metric: bool, depth_range_m: tuple[float, float],
    colormap: str = "magma_r", error_max_m: float = 10.0,
) -> None:
    import matplotlib.pyplot as plt

    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    gt, mask = np.asarray(gt_m, np.float32), np.asarray(valid, bool)
    native, aligned = np.asarray(native_depth, np.float32), np.asarray(aligned_depth_m, np.float32)
    if not (gt.shape == mask.shape == native.shape == aligned.shape):
        raise ValueError("Visualization arrays must share the GT resolution")
    vmin, vmax = map(float, depth_range_m)
    sparse_gt = np.ma.masked_where(~mask, gt)
    native_masked = np.ma.masked_where(~mask, native)
    aligned_masked = np.ma.masked_where(~mask, aligned)
    error = np.ma.masked_where(~mask, np.abs(aligned - gt))
    cmap = plt.get_cmap(colormap).copy(); cmap.set_bad("black", alpha=1.0)
    error_cmap = plt.get_cmap("inferno").copy(); error_cmap.set_bad("black", alpha=1.0)

    def one(array, name, *, cm=cmap, lo=vmin, hi=vmax, colorbar=True, label="Depth (m)"):
        fig, ax = plt.subplots(figsize=(7, 3)); im = ax.imshow(array, cmap=cm, vmin=lo, vmax=hi)
        ax.axis("off")
        if colorbar: fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=label)
        fig.tight_layout(); fig.savefig(output / name, dpi=160, bbox_inches="tight"); plt.close(fig)

    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(output / "gt_validity_mask.png")
    thermal = _thermal_for_display(thermal_path)
    plt.imsave(output / "thermal_input.png", thermal, cmap="gray" if thermal.ndim == 2 else None)
    one(sparse_gt, "sparse_gt_depth_m.png")
    if raw_is_metric:
        one(native, "raw_metric_prediction_m.png")
        one(native_masked, "raw_metric_prediction_on_gt_mask_m.png")
    else:
        # A relative native view is diagnostic only and never receives a metre label.
        one(native, "native_relative_depth_not_metric.png", lo=float(np.nanpercentile(native, 1)),
            hi=float(np.nanpercentile(native, 99)), label="Relative depth (not metres)")
    one(aligned, "affine_aligned_prediction_m.png")
    one(error, "absolute_error_on_valid_gt_m.png", cm=error_cmap, lo=0, hi=float(error_max_m), label="Absolute error (m)")

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    axes[0, 0].imshow(thermal, cmap="gray" if thermal.ndim == 2 else None); axes[0, 0].set_title("Left thermal input")
    depth_images = []
    for ax, array, title in [
        (axes[0, 1], sparse_gt, "Thermal-view filtered LiDAR GT"),
        (axes[0, 2], native if raw_is_metric else None, "Raw metric prediction" if raw_is_metric else "Raw metric unavailable"),
        (axes[1, 0], native_masked if raw_is_metric else None, "Raw prediction sampled on GT mask" if raw_is_metric else "Raw metric unavailable"),
        (axes[1, 1], aligned, "Per-image affine-aligned prediction"),
    ]:
        if array is None:
            ax.set_facecolor("black")
        else:
            depth_images.append(ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax))
        ax.set_title(title)
    err_im = axes[1, 2].imshow(error, cmap=error_cmap, vmin=0, vmax=error_max_m); axes[1, 2].set_title("|Aligned prediction - GT| on valid GT")
    for ax in axes.flat: ax.axis("off")
    fig.colorbar(depth_images[-1], ax=axes[:, :2], fraction=0.025, pad=0.02, label="Depth (m)")
    fig.colorbar(err_im, ax=axes[:, 2], fraction=0.05, pad=0.02, label="Absolute error (m)")
    fig.suptitle(f"MS2 unified protocol v1 — {sample_id}")
    fig.savefig(output / "comparison_panel.png", dpi=170, bbox_inches="tight"); plt.close(fig)
