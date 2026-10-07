"""Shared image I/O helpers. All images are numpy uint8 arrays in RGB (H, W, 3) or RGBA (H, W, 4)."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PIL import Image

from dt.ir import Box, Color


def load_rgb(path: str) -> np.ndarray:
    """Load any image as RGB uint8 (alpha composited over white)."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg.convert("RGB")
    elif im.mode != "RGB":
        im = im.convert("RGB")
    return np.asarray(im, dtype=np.uint8).copy()


def load_rgba(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8).copy()


def save_rgb(arr: np.ndarray, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(np.ascontiguousarray(arr[..., :3]).astype(np.uint8)).save(path)
    return path


def crop(arr: np.ndarray, box: Box, pad: int = 0) -> np.ndarray:
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = box.expand(pad).as_int()
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return arr[0:0, 0:0]
    return arr[y0:y1, x0:x1]


def downscale_dpr(arr: np.ndarray, dpr: float) -> np.ndarray:
    """Resize a @2x/@3x screenshot down to DPR 1 (area resampling)."""
    if dpr == 1:
        return arr
    im = Image.fromarray(arr)
    w, h = im.size
    im = im.resize((int(round(w / dpr)), int(round(h / dpr))), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8).copy()


def dominant_color(arr: np.ndarray, ignore: Optional[Color] = None, quant: int = 8) -> Optional[Color]:
    """Most frequent quantized color in region (optionally ignoring one color ± quant)."""
    if arr.size == 0:
        return None
    px = arr[..., :3].reshape(-1, 3)
    q = (px // quant).astype(np.int32)
    keys = q[:, 0] * 1_000_000 + q[:, 1] * 1000 + q[:, 2]
    uniq, counts = np.unique(keys, return_counts=True)
    order = np.argsort(-counts)
    for idx in order:
        mask = keys == uniq[idx]
        mean = px[mask].mean(axis=0)
        c = Color.from_rgb(*mean)
        if ignore is not None and c.delta_e(ignore) < 3:
            continue
        return c
    return None


def mean_color(arr: np.ndarray) -> Optional[Color]:
    if arr.size == 0:
        return None
    return Color.from_rgb(*arr[..., :3].reshape(-1, 3).mean(axis=0))


def to_gray(arr: np.ndarray) -> np.ndarray:
    a = arr[..., :3].astype(np.float32)
    return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.uint8)


def same_size(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop both to common size (top-left aligned)."""
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]
