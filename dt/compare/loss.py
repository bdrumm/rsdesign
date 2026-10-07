"""Scalar loss + per-node error map for the adversarial refine loop.

``evaluate(doc, target)`` renders ``doc`` (unless a render is supplied), compares it with the
target screenshot and returns a :class:`LossReport` whose ``total`` is the number the
optimizer minimises. Everything in the report is derived from pixels and the IR; nothing
here looks at the image with a model.

    total = w_pixel * mean_de / de_norm
          + w_bad   * frac_bad
          + w_text  * text_term        (OCR of target vs doc texts; 0 when OCR unavailable)
          + w_size  * size_term        (1 - IoU of the two image extents; 0 when equal)

Per-node error = mean ΔE inside the node's box excluding its visible children's boxes, from
an integral image of the diff map (O(1) per box), so the critic can rank nodes cheaply.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from dt.compare.pixel import alignment_offset, diff_map, pixel_metrics, residual_regions
from dt.compare.structural import normalize_text, structural_metrics, text_cer
from dt.ir import Box, Document, Node
from dt.params import P, register

register("compare.loss.w_pixel", 1.0, "Weight of normalised mean ΔE in the total loss.", (0.0, 5.0))
register("compare.loss.w_bad", 1.0, "Weight of the bad-pixel fraction in the total loss.", (0.0, 5.0))
register("compare.loss.w_text", 0.5, "Weight of the OCR text term in the total loss.", (0.0, 5.0))
register("compare.loss.w_size", 1.0, "Weight of the image-extent mismatch term.", (0.0, 5.0))
register("compare.loss.de_norm", 50.0, "ΔE that maps to a pixel term of 1.0.", (10.0, 100.0))
register("compare.loss.ocr_min_conf", 0.5, "Ignore OCR lines of the target below this confidence.", (0.0, 1.0))


# --------------------------------------------------------------------------- report
@dataclass
class LossReport:
    """Result of :func:`evaluate`. ``to_dict`` is JSON-safe (diff map omitted unless asked)."""

    total: float
    pixel: dict = field(default_factory=dict)
    structure: dict = field(default_factory=dict)
    per_node: dict[str, float] = field(default_factory=dict)
    residual_regions: list[Box] = field(default_factory=list)
    diff_map: Optional[np.ndarray] = None
    alignment: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (dx, dy, response) of rendered vs target
    size: tuple[int, int] = (0, 0)  # (W, H) of the compared area

    def worst_nodes(self, k: int = 5) -> list[tuple[str, float]]:
        """Node ids with the highest per-node error, descending."""
        return sorted(self.per_node.items(), key=lambda kv: -kv[1])[:k]

    def to_dict(self, include_diff: bool = False) -> dict:
        d = {
            "total": float(self.total),
            "pixel": dict(self.pixel),
            "structure": {k: v for k, v in self.structure.items() if k != "matches"},
            "per_node": {k: float(v) for k, v in self.per_node.items()},
            "residual_regions": [b.to_dict() for b in self.residual_regions],
            "alignment": {"dx": self.alignment[0], "dy": self.alignment[1], "response": self.alignment[2]},
            "size": {"w": self.size[0], "h": self.size[1]},
        }
        if "matches" in self.structure:
            d["structure"]["matches"] = [list(m) for m in self.structure["matches"]]
        if include_diff and self.diff_map is not None:
            d["diff_map"] = self.diff_map.tolist()
        return d


# --------------------------------------------------------------------------- per-node errors
def _clip_box(b: Box, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = b.as_int()
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def _box_sum(integral: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])


def per_node_errors(root: Node, diff: np.ndarray) -> dict[str, float]:
    """Mean ΔE inside each visible node's box excluding its visible children's boxes.

    Uses an integral image so each node costs O(1 + #children). Children are assumed not to
    overlap each other (the common case); overlapping children over-subtract and the result is
    clamped at 0. A node fully covered by its children falls back to the mean over its whole box
    so it still carries a signal. Nodes entirely outside the image get 0.
    """
    h, w = diff.shape[:2]
    integral = cv2.integral(np.ascontiguousarray(diff, dtype=np.float32), sdepth=cv2.CV_64F)
    out: dict[str, float] = {}
    for n in root.walk():
        if not n.visible:
            continue
        x0, y0, x1, y1 = _clip_box(n.box, w, h)
        area = (x1 - x0) * (y1 - y0)
        if area <= 0:
            out[n.id] = 0.0
            continue
        total = _box_sum(integral, x0, y0, x1, y1)
        child_sum, child_area = 0.0, 0
        for c in n.children:
            if not c.visible:
                continue
            cx0, cy0, cx1, cy1 = _clip_box(c.box, w, h)
            cx0, cy0, cx1, cy1 = max(cx0, x0), max(cy0, y0), min(cx1, x1), min(cy1, y1)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            child_sum += _box_sum(integral, cx0, cy0, cx1, cy1)
            child_area += (cx1 - cx0) * (cy1 - cy0)
        rem = area - child_area
        if rem <= 0:
            out[n.id] = total / area
        else:
            out[n.id] = max(0.0, total - child_sum) / rem
    return out


# --------------------------------------------------------------------------- OCR text term
def _ocr_texts(rgb: np.ndarray) -> Optional[list[str]]:
    """Target text lines via ``dt.perceive.ocr`` if that module is importable and works; else None."""
    try:
        from dt.perceive.ocr import ocr_lines  # type: ignore
    except Exception:
        return None
    try:
        min_conf = float(P["compare.loss.ocr_min_conf"])
        lines = ocr_lines(rgb)
    except Exception:
        return None
    out = []
    for ln in lines:
        text = normalize_text(getattr(ln, "text", ""))
        conf = float(getattr(ln, "conf", 1.0))
        if text and conf >= min_conf:
            out.append(text)
    return out


def text_term_from_lines(target_lines: list[str], doc: Document) -> Optional[float]:
    """Mean over target text lines of the best CER against any doc text (or any single line of one).

    0 when every target line is reproduced exactly; 1 when none is. ``None`` if the target has
    no text lines (nothing to score).
    """
    if not target_lines:
        return None
    cands: list[str] = []
    for n in doc.texts():
        t = n.text or ""
        cands.append(normalize_text(t))
        cands.extend(normalize_text(part) for part in t.split("\n") if normalize_text(part))
    cands = [c for c in cands if c]
    if not cands:
        return 1.0
    return float(np.mean([min(text_cer(c, line) for c in cands) for line in target_lines]))


# --------------------------------------------------------------------------- evaluate
def evaluate(doc: Document, target_rgb: np.ndarray, rendered_rgb: Optional[np.ndarray] = None,
             gt: Optional[Document] = None, use_ocr: bool = True, keep_diff: bool = True) -> LossReport:
    """Compare ``doc`` (rendered) against the ``target_rgb`` screenshot.

    * ``rendered_rgb``: pass an existing render of ``doc`` to skip rendering.
    * ``gt``: optional ground-truth IR; when given, ``structure`` also holds
      :func:`structural_metrics(doc, gt)`.
    * ``use_ocr``: OCR the target (if ``dt.perceive.ocr`` is available) for the text term.
    * ``keep_diff``: keep the ΔE map on the report (set False to save memory in long loops).
    """
    if rendered_rgb is None:
        from dt.render.screenshot import render_doc
        rendered_rgb = render_doc(doc)
    th, tw = target_rgb.shape[:2]
    rh, rw = rendered_rgb.shape[:2]
    d = diff_map(target_rgb, rendered_rgb)
    h, w = d.shape
    pix = pixel_metrics(target_rgb, rendered_rgb, diff=d)
    pix["size_mismatch"] = (th, tw) != (rh, rw)
    inter = w * h
    size_term = 1.0 - inter / float(tw * th + rw * rh - inter) if inter else 1.0
    pix["size_term"] = size_term

    regions = residual_regions(d)
    per_node = per_node_errors(doc.root, d)
    align = alignment_offset(target_rgb, rendered_rgb)

    structure: dict = {}
    text_term: Optional[float] = None
    if use_ocr:
        lines = _ocr_texts(target_rgb)
        if lines is not None:
            structure["ocr_lines"] = len(lines)
            text_term = text_term_from_lines(lines, doc)
    structure["text_term"] = text_term
    if gt is not None:
        structure.update(structural_metrics(doc, gt))

    de_norm = float(P["compare.loss.de_norm"])
    total = (
        float(P["compare.loss.w_pixel"]) * min(1.0, pix["mean_de"] / de_norm)
        + float(P["compare.loss.w_bad"]) * pix["frac_bad"]
        + float(P["compare.loss.w_text"]) * (text_term or 0.0)
        + float(P["compare.loss.w_size"]) * size_term
    )
    return LossReport(
        total=float(total), pixel=pix, structure=structure, per_node=per_node,
        residual_regions=regions, diff_map=d if keep_diff else None, alignment=align, size=(w, h),
    )
