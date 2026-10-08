"""Case sources: IR-generated (``render_doc``), HTML-generated (real browser layout, DOM ground
truth via the Material Web corpus extractor) and real crops (target pixels only).

All randomness belongs to the family generator (seeded); everything here is deterministic.
"""
from __future__ import annotations

import hashlib
import math
import os
from typing import Any, Optional

import numpy as np

from dt.ir import Box, Color, Document, assign_ids
from dt.scenarios.spec import ScenarioCase

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK_DIR = os.path.join(ROOT, "out", "scenarios", "_work")


def work_dir() -> str:
    """Per-process scratch dir (parallel workers must not share measurement pages)."""
    d = os.path.join(WORK_DIR, str(os.getpid()))
    os.makedirs(d, exist_ok=True)
    return d


# ----------------------------------------------------------------------------- colour helpers
def _lab(c: Color) -> np.ndarray:
    from skimage import color as skcolor
    return skcolor.rgb2lab(np.array([[[c.r, c.g, c.b]]], dtype=np.float64) / 255.0)[0, 0]


def de2000(a: Color, b: Color) -> float:
    """CIEDE2000 between two colours (the validator's measure)."""
    from skimage import color as skcolor
    return float(skcolor.deltaE_ciede2000(_lab(a)[None, None], _lab(b)[None, None])[0, 0])


def color_at_de(base: Color, de: float, hue_deg: float = 0.0, chroma_frac: float = 0.3,
                lighter: Optional[bool] = None) -> Color:
    """An sRGB colour about ``de`` (ΔE2000) from ``base``: moves along a Lab direction mixing a
    lightness step (towards the middle grey unless ``lighter`` is given) with a chroma step at
    ``hue_deg`` (``chroma_frac`` of the direction), then picks the 8-bit colour whose actual ΔE2000
    is closest to ``de``."""
    from skimage import color as skcolor
    L, a, b = _lab(base)
    if lighter is None:
        lighter = L < 50
    sl = 1.0 if lighter else -1.0
    h = math.radians(hue_deg)
    cf = max(0.0, min(1.0, chroma_frac))
    d = np.array([sl * (1.0 - cf), cf * math.cos(h), cf * math.sin(h)])
    d /= max(1e-9, float(np.linalg.norm(d)))
    best, best_err = base, 1e9
    for t in np.linspace(0.0, max(4.0, de * 6.0), 241):
        lab = np.array([L, a, b]) + t * d
        lab[0] = min(100.0, max(0.0, lab[0]))
        rgb = np.clip(skcolor.lab2rgb(lab[None, None])[0, 0] * 255.0, 0, 255).round()
        c = Color(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        err = abs(de2000(base, c) - de)
        if err < best_err:
            best, best_err = c, err
    return best


# ----------------------------------------------------------------------------- IR source
def ir_case(family: str, seed: int, params: dict, doc: Document, roi: Optional[Box] = None,
            meta: Optional[dict] = None, fit_text: bool = True) -> ScenarioCase:
    """Render ``doc`` with the real renderer; the IR is the ground truth. Text boxes are tightened
    to their measured glyph widths first (as in the synthetic corpus)."""
    from dt.render.screenshot import render_doc
    if fit_text and any(n.type == "text" for n in doc.walk()):
        from dt.selftest.synth import fit_text_boxes
        fit_text_boxes(doc, work_dir())
    assign_ids(doc.root)
    doc.meta = {**(doc.meta or {}), "scenario": family, "seed": seed}
    rgb = render_doc(doc)
    return ScenarioCase(id=f"{family}_{seed}", family=family, params=params, target_rgb=rgb, gt=doc, roi=roi,
                        seed=seed, meta=dict(meta or {}))


# ----------------------------------------------------------------------------- HTML source
def html_case(family: str, seed: int, params: dict, body_html: str, width: int, height: int,
              extra_css: str = "", roi: Optional[Box] = None, meta: Optional[dict] = None) -> ScenarioCase:
    """Lay out ``body_html`` in the real browser (Roboto, Material Symbols and the M3 tokens of
    the Material Web corpus pages) and extract the ground-truth IR from the DOM with
    :data:`dt.selftest.mwc_corpus.EXTRACT_JS` (svg -> ``vector`` nodes, text -> one node per
    line, painted boxes -> frames)."""
    import random

    from dt.render.screenshot import render_url
    from dt.selftest import mwc_corpus
    wd = work_dir()
    body = (f"<style>{extra_css}</style>" if extra_css else "") + body_html
    html = mwc_corpus.page_html(random.Random(0), width, height, body_html=body, html_dir=wd)
    digest = hashlib.sha1(html.encode()).hexdigest()[:12]
    path = os.path.join(wd, f"{family}_{seed}_{digest}.html")
    with open(path, "w") as f:
        f.write(html)
    try:
        import dt.params as _p
        rgb, res = render_url("file://" + path, width, height, wait_ms=int(_p.P["selftest.mwc.wait_ms"]),
                              script=mwc_corpus.extract_script())
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    if not isinstance(res, dict) or "nodes" not in res:
        raise RuntimeError(f"{family}: DOM extraction returned no result")
    gt = mwc_corpus.build_document(res, width, height)
    assign_ids(gt.root)
    gt.meta = {"scenario": family, "seed": seed}
    return ScenarioCase(id=f"{family}_{seed}", family=family, params=params, target_rgb=rgb, gt=gt, roi=roi,
                        html=html, seed=seed, meta=dict(meta or {}))


# ----------------------------------------------------------------------------- real crops
def crop(rgb: np.ndarray, box: Box) -> np.ndarray:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box.as_int()
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    return np.ascontiguousarray(rgb[y0:y1, x0:x1, :3])


def real_case(family: str, seed: int, params: dict, rgb: np.ndarray, meta: dict[str, Any]) -> ScenarioCase:
    return ScenarioCase(id=str(meta.get("id") or f"{family}_{seed}"), family=family, params=params,
                        target_rgb=np.ascontiguousarray(rgb[..., :3]), gt=None, roi=None, seed=seed, meta=dict(meta))
