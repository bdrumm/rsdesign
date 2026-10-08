"""Pixel-level comparison between two RGB images (target screenshot vs rendered IR).

All functions take numpy uint8 RGB arrays of shape (H, W, 3). When the two images differ
in size they are cropped (top-left aligned) to the common size; callers that care about
size mismatch should check shapes themselves first.

Perceptual distance is CIE76 ΔE in L*a*b* (D65), computed vectorised with
``skimage.color.rgb2lab``. Rough scale: <1 imperceptible, ~2.3 just-noticeable,
>10 clearly different colours, ~100 black-vs-white.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity

from dt.common.image import same_size, to_gray
from dt.ir import Box
from dt.params import P, register

register("compare.pixel.bad_de", 6.0,
         "ΔE above which a pixel counts as 'bad' (frac_bad, residual regions).", (2.0, 20.0))
register("compare.pixel.residual_min_area", 16,
         "Minimum connected-component area (px) for a residual region.", (4, 400))
register("compare.pixel.residual_dilate", 2,
         "Dilation radius (px) applied to the bad-pixel mask before connected components, "
         "so fragmented edges of one error merge into a single region.", (0, 8))
register("compare.pixel.residual_max_regions", 64,
         "Keep at most this many residual regions (largest by summed ΔE).", (4, 512))
register("compare.pixel.psnr_max", 100.0,
         "PSNR reported for identical images (true value is +inf).", (60.0, 200.0))


# --------------------------------------------------------------------------- colour space
def to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB (H, W, 3) -> float32 L*a*b* (H, W, 3)."""
    arr = np.ascontiguousarray(rgb[..., :3])
    return rgb2lab(arr.astype(np.float32) / 255.0).astype(np.float32)


def diff_map(a: np.ndarray, b: np.ndarray, fast: bool | None = None) -> np.ndarray:
    """Per-pixel CIE76 ΔE between two RGB images, as float32 (H, W).

    Images of different size are cropped to the common size (top-left aligned).
    ``fast`` (default: inside ``dt.accel.search()``) computes it on the GPU (MLX) for search loops such
    as refine trials: ~13x faster on a 1280x800 frame, within 0.05 ΔE. Outside search the float32 CPU
    path below is used so benchmark baselines do not move.
    """
    a, b = same_size(a, b)
    from dt import accel
    if fast if fast is not None else accel.search_active():
        return accel.delta_e76(a, b)
    la, lb = to_lab(a), to_lab(b)
    d = la - lb
    return np.sqrt(np.einsum("ijk,ijk->ij", d, d)).astype(np.float32)


# --------------------------------------------------------------------------- scalar metrics
def pixel_metrics(a: np.ndarray, b: np.ndarray, bad_thr: float | None = None,
                  diff: np.ndarray | None = None) -> dict[str, float]:
    """Scalar similarity metrics between two RGB images.

    Returns ``{"mse", "psnr", "ssim", "mean_de", "frac_bad", "max_de"}``:
      * mse / psnr on RGB 0..255 (psnr capped at ``compare.pixel.psnr_max`` for identical images)
      * ssim on grayscale (skimage structural_similarity, data_range 255)
      * mean_de: mean ΔE over all pixels; max_de: the largest ΔE
      * frac_bad: fraction of pixels with ΔE > ``bad_thr`` (default ``compare.pixel.bad_de``)

    ``diff`` may be passed to reuse an already computed ``diff_map(a, b)``.
    """
    a, b = same_size(a, b)
    thr = float(P["compare.pixel.bad_de"] if bad_thr is None else bad_thr)
    d = diff_map(a, b) if diff is None else diff[: a.shape[0], : a.shape[1]]
    af = a[..., :3].astype(np.float32)
    bf = b[..., :3].astype(np.float32)
    mse = float(np.mean((af - bf) ** 2))
    psnr = float(P["compare.pixel.psnr_max"]) if mse <= 1e-12 else float(10.0 * math.log10(255.0 ** 2 / mse))
    psnr = min(psnr, float(P["compare.pixel.psnr_max"]))
    ga, gb = to_gray(a), to_gray(b)
    win = min(7, (min(ga.shape) // 2) * 2 - 1)  # odd, <= image size
    ssim = 1.0 if win < 3 else float(structural_similarity(ga, gb, data_range=255, win_size=win))
    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "mean_de": float(d.mean()) if d.size else 0.0,
        "max_de": float(d.max()) if d.size else 0.0,
        "frac_bad": float((d > thr).mean()) if d.size else 0.0,
    }


# --------------------------------------------------------------------------- residual regions
def residual_regions(diff: np.ndarray, thr: float | None = None, min_area: int | None = None,
                     dilate: int | None = None, max_regions: int | None = None) -> list[Box]:
    """Connected components of ``diff > thr`` as boxes, largest error mass first.

    The bad-pixel mask is dilated by ``dilate`` px (default ``compare.pixel.residual_dilate``)
    so that the fragmented outline of a single misplaced element becomes one region; the
    returned boxes are the bounds of the *un-dilated* bad pixels inside each component.
    Components with fewer than ``min_area`` bad pixels are dropped.
    """
    thr = float(P["compare.pixel.bad_de"] if thr is None else thr)
    min_area = int(P["compare.pixel.residual_min_area"] if min_area is None else min_area)
    dilate = int(P["compare.pixel.residual_dilate"] if dilate is None else dilate)
    max_regions = int(P["compare.pixel.residual_max_regions"] if max_regions is None else max_regions)
    bad = diff > thr
    if not bad.any():
        return []
    grown = ndimage.binary_dilation(bad, iterations=dilate) if dilate > 0 else bad
    labels, n = ndimage.label(grown)
    if n == 0:
        return []
    # bad pixels only, labelled by the dilated component they belong to
    lab_bad = np.where(bad, labels, 0)
    idx = np.arange(1, n + 1)
    counts = ndimage.sum(bad, lab_bad, idx)
    mass = ndimage.sum(diff, lab_bad, idx)
    slices = ndimage.find_objects(lab_bad)
    regions: list[tuple[float, Box]] = []
    for i, sl in enumerate(slices):
        if sl is None or counts[i] < min_area:
            continue
        ys, xs = sl
        regions.append((float(mass[i]), Box(xs.start, ys.start, xs.stop - xs.start, ys.stop - ys.start)))
    regions.sort(key=lambda t: -t[0])
    return [b for _, b in regions[:max_regions]]


# --------------------------------------------------------------------------- alignment
def alignment_offset(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Translation of ``b`` relative to ``a`` via phase correlation.

    Returns ``(dx, dy, response)`` such that ``b`` ≈ ``a`` shifted right by ``dx`` and down by
    ``dy`` (``b[y, x] ≈ a[y - dy, x - dx]``). ``response`` in 0..1 is the correlation peak
    strength; values well under ~0.1 mean the shift is unreliable (e.g. flat crops).

    Accepts RGB or grayscale crops; sizes are reconciled by cropping to the common size. A
    Hanning window suppresses the border discontinuity so partial-content crops still lock on.
    """
    a, b = same_size(a, b)
    if a.size == 0 or min(a.shape[:2]) < 2:
        return 0.0, 0.0, 0.0
    ga = (to_gray(a) if a.ndim == 3 else a).astype(np.float32)
    gb = (to_gray(b) if b.ndim == 3 else b).astype(np.float32)
    if float(ga.std()) < 1e-6 or float(gb.std()) < 1e-6:
        return 0.0, 0.0, 0.0
    h, w = ga.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ga, gb, win)
    if not (math.isfinite(dx) and math.isfinite(dy)):
        return 0.0, 0.0, 0.0
    return float(dx), float(dy), float(max(0.0, min(1.0, resp)))
