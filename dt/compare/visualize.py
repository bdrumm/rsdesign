"""Diff visualisations for reports: target | rendered | ΔE heatmap, with residual boxes."""
from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from dt.common.image import same_size, save_rgb
from dt.ir import Box
from dt.params import P, register

register("compare.visualize.heat_max_de", 30.0,
         "ΔE mapped to the hottest heatmap colour (higher values saturate).", (5.0, 100.0))

_RED = (230, 30, 30)
_HEADER = 18


def heatmap(diff: np.ndarray, max_de: float | None = None) -> np.ndarray:
    """ΔE map -> RGB uint8 heatmap (black = 0, saturating at ``max_de``)."""
    max_de = float(P["compare.visualize.heat_max_de"] if max_de is None else max_de)
    norm = np.clip(diff / max(max_de, 1e-6), 0.0, 1.0)
    u8 = (norm * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_boxes(rgb: np.ndarray, boxes: Iterable[Box], color: tuple[int, int, int] = _RED, thickness: int = 1) -> np.ndarray:
    """Return a copy of ``rgb`` with ``boxes`` outlined."""
    out = np.ascontiguousarray(rgb[..., :3]).copy()
    h, w = out.shape[:2]
    for b in boxes:
        x0, y0, x1, y1 = b.as_int()
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w - 1, max(x0, x1 - 1)), min(h - 1, max(y0, y1 - 1))
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
    return out


def _panel(img: np.ndarray, label: str) -> np.ndarray:
    """Add a white header strip with a label above ``img``."""
    h, w = img.shape[:2]
    out = np.full((h + _HEADER, w, 3), 255, dtype=np.uint8)
    out[_HEADER:, :] = img[..., :3]
    cv2.putText(out, label, (4, _HEADER - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    return out


def compose_diff_image(target: np.ndarray, rendered: np.ndarray, diff: np.ndarray,
                       regions: Optional[Iterable[Box]] = None) -> np.ndarray:
    """Side-by-side RGB image: target | rendered | heatmap, residual boxes drawn on all three."""
    target, rendered = same_size(target, rendered)
    h, w = target.shape[:2]
    diff = diff[:h, :w]
    boxes = list(regions or [])
    panels = [
        _panel(draw_boxes(target, boxes), "target"),
        _panel(draw_boxes(rendered, boxes), "rendered"),
        _panel(draw_boxes(heatmap(diff), boxes, color=(0, 255, 0)), f"dE heat (mean {float(diff.mean()):.2f})"),
    ]
    return np.concatenate(panels, axis=1)


def save_diff_image(target: np.ndarray, rendered: np.ndarray, diff: np.ndarray, path: str,
                    regions: Optional[Iterable[Box]] = None) -> str:
    """Write :func:`compose_diff_image` to ``path`` (PNG). Returns the path."""
    return save_rgb(compose_diff_image(target, rendered, diff, regions), path)
