"""Device-pixel-ratio detection for real screenshots (docs/SELF_IMPROVEMENT.md, mechanism 6).

Phone and laptop screenshots are captured at 2x-3.5x. The pipeline works in CSS pixels (DPR 1), so
a 1179px-wide iPhone screenshot must be downscaled by 3 before perception, or every size, radius
and font estimate is off by that factor.

Evidence, combined into one score per candidate DPR:
  * width prior: the image width divided by the DPR should be a common CSS viewport width
    (360-430 phones, 600-1024 tablets, 1280-1920 desktops), within a tolerance;
  * text prior: after downscaling, the median OCR'd line height should look like UI type (the
    body/label sizes 11-16px dominate every Material/iOS/web UI), not 2-3x that.

    dpr, evidence = detect_dpr(rgb)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from dt.params import P, register

register("perceive.dpr.candidates", [1.0, 1.5, 2.0, 2.625, 3.0, 3.5], "DPRs considered by detect_dpr")
register("perceive.dpr.viewports", [320, 360, 375, 390, 393, 400, 412, 414, 428, 430, 600, 768, 800, 834, 1024, 1280, 1366, 1440, 1512, 1536, 1600, 1728, 1920],
         "common CSS viewport widths (px)")
register("perceive.dpr.width_tol", 0.02, "relative tolerance when matching width/dpr to a common viewport", (0.0, 0.1))
register("perceive.dpr.text_target", 12.0, "expected median OCR ink line height at DPR 1 (measured 11.5 on the corpora)", (8.0, 24.0))
register("perceive.dpr.text_weight", 1.0, "weight of the text prior vs the width prior", (0.0, 4.0))
register("perceive.dpr.probe_width", 1600, "OCR the image downscaled to at most this width when probing", (600, 4000))


def _width_score(width: int, dpr: float) -> float:
    css = width / dpr
    tol = float(P["perceive.dpr.width_tol"])
    best = min(abs(css - v) / v for v in P["perceive.dpr.viewports"])
    return 1.0 if best <= tol else max(0.0, 1.0 - (best - tol) / 0.1)


def _median_line_height(rgb: np.ndarray) -> Optional[float]:
    """Median OCR line ink height in the image's own pixels (None if no text)."""
    from PIL import Image
    from dt.perceive.ocr import ocr_lines
    h, w = rgb.shape[:2]
    scale = min(1.0, float(P["perceive.dpr.probe_width"]) / w)
    probe = rgb if scale >= 1.0 else np.asarray(Image.fromarray(rgb[..., :3]).resize((int(w * scale), int(h * scale)), Image.LANCZOS))
    lines = [l for l in ocr_lines(probe) if l.text and len(l.text.strip()) >= 3]
    if not lines:
        return None
    return float(np.median([l.box.h for l in lines])) / scale


def detect_dpr(rgb: np.ndarray) -> tuple[float, dict]:
    """Best DPR for a screenshot plus the evidence behind it."""
    h, w = rgb.shape[:2]
    mlh = _median_line_height(rgb)
    target = float(P["perceive.dpr.text_target"])
    tw = float(P["perceive.dpr.text_weight"])
    scores = {}
    for d in P["perceive.dpr.candidates"]:
        d = float(d)
        ws = _width_score(w, d)
        ts = 0.0
        if mlh is not None:
            # log-distance between the line height at this DPR and the expected UI line height
            ts = max(0.0, 1.0 - abs(np.log((mlh / d) / target)) / np.log(2.0))
        scores[d] = ws + tw * ts
    best = max(scores, key=lambda d: (round(scores[d], 6), -d))
    return best, {"width": w, "median_line_px": mlh, "scores": {str(k): round(v, 3) for k, v in scores.items()}}
