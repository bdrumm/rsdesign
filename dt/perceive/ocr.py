"""OCR + text-style estimation.

Pipeline (all programmatic, no model "looks" at pixels):

    rgb --backend.recognize()--> raw lines (text, coarse box, words, conf)
        --_refine_line()--> TextLine with a tight *ink* box, baseline and glyph classes
        --estimate_text_style()--> TextStyle(size, weight, color)
        --group_paragraphs()--> Paragraph(lines, line_height)
        --paragraph_node()--> IR text Node whose box reproduces the renderer's line box

Backends (``get_backend``): ``vision`` (macOS, pyobjc, lazily imported), ``rapidocr`` (ONNX,
pip), ``tesseract`` (pytesseract). The first available one is used unless ``DT_OCR`` names one.

Font size is NOT read from the OCR box (Vision's boxes are quantised); it is derived from the
measured ink height, through linear fits ``size = a * ink_h + b`` calibrated against
``dt.render`` for three glyph classes (tallest glyph: ascender / cap-height / x-height, decided
from the recognised characters); the baseline itself comes from the ink row profile. Weight comes from the mean
stroke width (ink mass / skeleton length, normalised by size). Colour is the dominant colour of
fully covered glyph pixels. The fitted constants live in ``dt.params`` so the tuner can refine
them (see ``scripts`` in the test file for how they were produced).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

from dt.common.image import dominant_color
from dt.ir import Box, Color, Node, TextStyle, new_id
from dt.params import P, register

# --------------------------------------------------------------------------- params
register("perceive.ocr.backend", "auto", "OCR backend: auto|vision|rapidocr|tesseract")
register("perceive.ocr.min_conf", 0.3, "drop OCR lines below this confidence", (0.0, 0.9))
register("perceive.ocr.upscale_min_h", 48, "upscale images shorter than this (px) before OCR", (16, 128))
register("perceive.ocr.crop_pad_ratio", 0.45, "vertical padding (x box height) around a raw OCR box when measuring ink", (0.2, 0.8))
register("perceive.ocr.crop_pad_px", 3, "horizontal padding (px) around a raw OCR box when measuring ink", (1, 8))
register("perceive.ocr.cov_thr", 0.5, "glyph coverage threshold (0..1) for the binary ink mask", (0.3, 0.7))
register("perceive.ocr.min_contrast", 24.0, "min RGB distance text<->background to accept ink", (8.0, 60.0))
register("perceive.ocr.baseline_band", 0.45, "baseline is searched in the lower fraction of the ink height", (0.25, 0.7))
register("perceive.ocr.icon_word_gap", 0.6, "single-char word separated from its neighbour by an ink gap >= ratio x size -> icon, not text", (0.4, 1.0))
# size = a*ink_h + b where ink_h = baseline - glyph top, per top class (calibrated on Roboto via dt.render)
for _cls, _a, _b in (("asc", 1.3269, -0.058), ("cap", 1.3869, 0.19), ("x", 1.8659, -0.082)):
    register(f"perceive.ocr.size_fit.{_cls}.a", _a, f"font-size fit slope, lines whose tallest glyphs are {_cls} (ascender|cap|x-height)", (0.9, 2.4))
    register(f"perceive.ocr.size_fit.{_cls}.b", _b, f"font-size fit intercept for top class {_cls}", (-3.0, 3.0))
register("perceive.ocr.ascent_ratio", 0.9287, "renderer text box: baseline - box.y = ratio * size (Roboto hhea ascent)", (0.8, 1.0))
register("perceive.ocr.line_height_ratio", 1.1719, "renderer 'normal' line-height / size (Roboto)", (1.0, 1.4))
register("perceive.ocr.lsb_ratio", 0.05, "left/right side bearing / size added around ink for the text box", (0.0, 0.15))
register("perceive.ocr.text_w_slack", 0.35, "extra width / size added to text boxes so a slight size/weight over-estimate never wraps", (0.0, 1.0))
register("perceive.ocr.stroke_thr_500", 0.094, "stroke-width/size above which weight is >= 500", (0.06, 0.14))
register("perceive.ocr.stroke_thr_700", 0.118, "stroke-width/size above which weight is 700", (0.09, 0.2))
register("perceive.ocr.color_min_rel", 0.9, "pixels with >= rel * max distance from bg are used for the colour estimate", (0.6, 1.0))
register("perceive.ocr.para_gap_ratio", 1.9, "max baseline distance / size between lines of one paragraph", (1.2, 3.0))
register("perceive.ocr.para_x_tol", 3.0, "px tolerance on x-start or centre for paragraph grouping", (1.0, 8.0))
register("perceive.ocr.para_size_tol", 1.4, "px tolerance on font size between lines of one paragraph; just above the 1px ink-height quantum of the asc/cap fits (1.33/1.39 px), so same-size lines rasterised 1px apart still group (16px vs 14px measure 15.9 vs 14.5: colour must separate those)", (0.5, 4.0))
register("perceive.ocr.para_color_de", 5.0, "max dE between line colours of one paragraph (M3 on-surface vs on-surface-variant differ by ~20)", (2.0, 15.0))
register("perceive.ocr.band_merge_gap", 0.25, "ink row bands separated by an empty gap < ratio x tallest band are one line (i-dots, accents); farther bands are neighbouring lines and are dropped from the line's ink", (0.1, 0.6))
register("perceive.ocr.fg_refit_de", 3.0, "glyph colour is re-estimated on the line's own ink band; coverage is recomputed when it moved by more than this dE", (1.0, 10.0))
# second-chance OCR: text-like blobs that the full-image pass returned nothing for (badges, dark surfaces, ...)
register("perceive.ocr.sc.enabled", 1, "run the second-chance OCR pass on uncovered text-like blobs (0 = off)", (0, 1))
register("perceive.ocr.sc.scale", 3, "integer upscale factor of second-chance crops before OCR", (1, 4))
register("perceive.ocr.sc.max_crops", 32, "max OCR calls of the second-chance pass per image (bounds OCR time)", (0, 96))
register("perceive.ocr.sc.min_conf", 0.5, "min OCR confidence of a second-chance result", (0.3, 1.0))
register("perceive.ocr.sc.min_alnum", 1, "min alphanumeric characters of a second-chance result", (1, 3))
register("perceive.ocr.sc.edge_thr", 48, "min 3x3 morphological colour gradient (max over RGB) for a pixel to count as glyph edge", (16, 128))
register("perceive.ocr.sc.join_px", 5, "horizontal closing width (px) that joins glyph edges into word blobs", (2, 12))
register("perceive.ocr.sc.min_h", 7, "min blob height (px) of a second-chance candidate", (4, 16))
register("perceive.ocr.sc.max_h", 40, "max blob height (px) of a second-chance candidate", (20, 96))
register("perceive.ocr.sc.max_w", 480, "max blob width (px) of a second-chance candidate", (100, 1200))
register("perceive.ocr.sc.pad_ratio", 0.4, "context padding around a candidate blob (x blob height) in its OCR crop", (0.1, 1.0))
register("perceive.ocr.sc.border_px", 16, "replicated border (px, after upscaling) added around a second-chance crop", (0, 48))
register("perceive.ocr.short_alnum", 2, "full-pass lines with at most this many alphanumerics must pass the ink-width plausibility check, else they are icon glyphs", (0, 4))
register("perceive.ocr.sc.width_ratio_min", 0.7, "second-chance line rejected when ink width / expected Roboto width of its text is below this (icon read as letters)", (0.4, 0.95))
register("perceive.ocr.sc.width_ratio_max", 1.5, "second-chance line rejected when ink width / expected Roboto width of its text is above this", (1.1, 2.5))
register("perceive.ocr.sc.container_density", 0.6, "second-chance 'ink' this solid (fraction of its bbox) with glyph holes is a badge/pill container; glyphs are re-measured inside it", (0.4, 0.9))
register("perceive.ocr.sc.badge_max_h", 20, "max height (px) of a filled shape with glyph holes searched inside an unreadable blob (badge on an icon)", (12, 32))
register("perceive.ocr.sc.max_sub", 2, "max filled sub-shapes OCR'd per unreadable second-chance blob", (0, 4))
register("perceive.ocr.sc.dark_lum", 0.2, "crops whose local background luminance is below this are OCR'd inverted first (light-on-dark text)", (0.05, 0.5))
# RapidOCR (PP-OCR rec model) often drops the spaces of Latin text ('GoogleWorkspace'): spaces are
# restored from the measured ink column gaps of the line (see restore_spaces)
register("perceive.ocr.rapid.space_gap_h", 0.24, "an ink column gap >= this x line ink height may be a word space", (0.12, 0.45))
register("perceive.ocr.rapid.space_gap_ratio", 1.8, "... and it must be >= this x the line's median letter gap (tracked caps stay one word)", (1.2, 3.0))
register("perceive.ocr.rapid.space_gap_excess_px", 3.0, "... and exceed the median letter gap by this many px (pixel-rounded small text: 0 false spaces on 19.6k rendered letter gaps incl. tracked caps)", (1.0, 6.0))
register("perceive.ocr.rapid.snap_h", 0.6, "a word gap farther than this x ink height from every character boundary stays unexplained (no space)", (0.2, 1.5))
register("perceive.ocr.rapid.cov_thr", 0.35, "glyph coverage above which a column counts as ink when measuring word gaps", (0.15, 0.6))
register("perceive.ocr.rapid.use_cls", 1, "run RapidOCR's text orientation classifier (1 = on; off is faster but turned an icon glyph into text on synth_1_006)", (0, 1))

_ASC_CHARS = set("bdfhkl()[]{}|/\\?!@#$%&ÀÁÂÃÄÅÈÉÊËÌÍÎÏÒÓÔÕÖÙÚÛÜ")
_CAP_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ij\"'")
_DESC_CHARS = set("gjpqy()[]{}|/\\@,;")


# --------------------------------------------------------------------------- data
@dataclass
class RawLine:
    """What a backend returns: text + coarse boxes (pixel coords, y down)."""
    text: str
    box: Box
    words: list[tuple[str, Box]]
    conf: float


@dataclass
class TextLine:
    """One recognised line with measured ink geometry.

    ``box`` is the tight ink bounding box (descenders included); ``baseline`` the y of the
    baseline (px, float, found from the ink row profile); ``top_class`` in {asc, cap, x} says
    what the tallest glyphs are and ``has_desc`` whether descender characters are present;
    ``cov`` is the glyph coverage map (crop-local, see ``cov_origin``) kept for style estimation.
    """
    text: str
    box: Box
    words: list[tuple[str, Box]]
    conf: float
    baseline: float = 0.0
    top_class: str = "asc"
    has_desc: bool = False
    bg: Color = field(default_factory=lambda: Color(255, 255, 255))
    fg: Color = field(default_factory=Color)
    cov: Optional[np.ndarray] = None
    cov_origin: tuple[int, int] = (0, 0)
    meta: dict = field(default_factory=dict)


@dataclass
class Paragraph:
    lines: list[TextLine]
    style: TextStyle
    box: Box  # tight ink box of all lines
    text: str
    line_height: Optional[float]  # px, None if single line


# --------------------------------------------------------------------------- backends
class OcrBackend(Protocol):
    name: str

    def recognize(self, rgb: np.ndarray) -> list[RawLine]: ...


def _box_from_vision(bb, W: int, H: int) -> Box:
    """Vision normalised rect (origin bottom-left) -> pixel Box (origin top-left)."""
    x, y, w, h = bb.origin.x, bb.origin.y, bb.size.width, bb.size.height
    return Box(x * W, (1.0 - y - h) * H, w * W, h * H)


class VisionBackend:
    """macOS Vision (VNRecognizeTextRequest, accurate, no language correction)."""
    name = "vision"

    def __init__(self) -> None:
        import Vision  # noqa: F401  (lazy; never at module top level)
        import Quartz  # noqa: F401

    def recognize(self, rgb: np.ndarray) -> list[RawLine]:
        import io
        import Quartz
        import Vision
        from Foundation import NSData
        from PIL import Image

        H, W = rgb.shape[:2]
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgb[..., :3])).save(buf, format="PNG")
        data = NSData.dataWithBytes_length_(buf.getvalue(), len(buf.getvalue()))
        src = Quartz.CGImageSourceCreateWithData(data, None)
        cg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setUsesLanguageCorrection_(False)
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
        ok, err = handler.performRequests_error_([req], None)
        if not ok:
            raise RuntimeError(f"Vision OCR failed: {err}")
        out: list[RawLine] = []
        for obs in req.results() or []:
            cand = obs.topCandidates_(1)
            if not cand:
                continue
            cand = cand[0]
            text = str(cand.string())
            words: list[tuple[str, Box]] = []
            pos = 0
            for w in text.split(" "):
                if w:
                    rb, _ = cand.boundingBoxForRange_error_((pos, len(w)), None)
                    wb = _box_from_vision(rb.boundingBox(), W, H) if rb is not None else _box_from_vision(obs.boundingBox(), W, H)
                    words.append((w, wb))
                pos += len(w) + 1
            out.append(RawLine(text, _box_from_vision(obs.boundingBox(), W, H), words, float(cand.confidence())))
        return out


class RapidOcrBackend:
    """RapidOCR (ONNX). ``pip install rapidocr-onnxruntime`` (or the ``ocr`` extra).

    The PP-OCR recogniser frequently omits the spaces of Latin text and has no word geometry;
    per-character boxes (``return_word_box``) plus the line's own ink gaps restore both
    (``restore_spaces``)."""
    name = "rapidocr"

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        self._engine = RapidOCR()

    def recognize(self, rgb: np.ndarray) -> list[RawLine]:
        kw = {"use_cls": bool(int(P["perceive.ocr.rapid.use_cls"]))}
        try:
            result, _ = self._engine(np.ascontiguousarray(rgb[..., ::-1]), return_word_box=True, **kw)
        except TypeError:  # older rapidocr without return_word_box / use_cls keywords
            result, _ = self._engine(np.ascontiguousarray(rgb[..., ::-1]))
        out: list[RawLine] = []
        for item in result or []:
            quad, text, conf = item[0], str(item[1]), float(item[2])
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            b = Box(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            char_boxes = None
            if len(item) > 3 and item[3] is not None and len(item[3]) == len(text):
                char_boxes = [(float(min(p[0] for p in q)), float(max(p[0] for p in q))) for q in item[3]]
            text, words = restore_spaces(rgb, text, b, char_boxes)
            if text:
                out.append(RawLine(text, b, words, conf))
        return out


class TesseractBackend:
    """Tesseract via pytesseract (word boxes from image_to_data)."""
    name = "tesseract"

    def __init__(self) -> None:
        import pytesseract  # type: ignore

        self._t = pytesseract
        self._t.get_tesseract_version()

    def recognize(self, rgb: np.ndarray) -> list[RawLine]:
        from PIL import Image

        d = self._t.image_to_data(Image.fromarray(rgb), output_type=self._t.Output.DICT)
        groups: dict[tuple, list] = {}
        for i, txt in enumerate(d["text"]):
            if not txt.strip():
                continue
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            conf = float(d["conf"][i]) / 100.0
            groups.setdefault(key, []).append((txt, Box(d["left"][i], d["top"][i], d["width"][i], d["height"][i]), conf))
        out: list[RawLine] = []
        for ws in groups.values():
            b = ws[0][1]
            for _, wb, _ in ws[1:]:
                b = b.union(wb)
            out.append(RawLine(" ".join(w for w, _, _ in ws), b, [(w, wb) for w, wb, _ in ws], float(np.mean([c for _, _, c in ws]))))
        return out


def _split_words_evenly(text: str, box: Box) -> list[tuple[str, Box]]:
    """Fallback word boxes for backends without per-word geometry: proportional split."""
    words = [w for w in text.split(" ") if w]
    if not words:
        return []
    total = len(text)
    out: list[tuple[str, Box]] = []
    pos = 0
    for w in words:
        out.append((w, Box(box.x + box.w * pos / total, box.y, box.w * len(w) / total, box.h)))
        pos += len(w) + 1
    return out


def _empty_runs(cols: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) of the runs of False inside the first..last True column (interior gaps only)."""
    idx = np.flatnonzero(cols)
    if idx.size < 2:
        return []
    runs: list[tuple[int, int]] = []
    start = None
    for i in range(int(idx[0]), int(idx[-1]) + 1):
        if not cols[i] and start is None:
            start = i
        elif cols[i] and start is not None:
            runs.append((start, i))
            start = None
    return runs


def restore_spaces(rgb: np.ndarray, text: str, box: Box,
                   char_x: Optional[list[tuple[float, float]]] = None) -> tuple[str, list[tuple[str, Box]]]:
    """Re-insert word spaces an OCR backend dropped, using the line's ink column gaps.

    ``char_x`` are the backend's coarse per-character (x0, x1) extents in image coordinates, one per
    character of ``text`` (spaces included); without them characters are spread evenly across the
    ink. A gap becomes a space when it is wide relative to the line's ink height
    (``perceive.ocr.rapid.space_gap_h``) *and* an outlier against the line's own letter gaps
    (``perceive.ocr.rapid.space_gap_ratio``), so letter-spaced caps stay one word. Gaps are aligned
    in reading order to the character boundaries (existing spaces absorb their own gap; a gap far
    from every boundary is left alone); spaces are only ever added, never removed. Returns
    (text, word boxes) with word boxes spanning their characters and the line height.
    """
    text = text.strip() if char_x is None else text
    H, W = rgb.shape[:2]
    x0, y0 = max(0, int(np.floor(box.x))), max(0, int(np.floor(box.y)))
    x1, y1 = min(W, int(np.ceil(box.x2))), min(H, int(np.ceil(box.y2)))
    if not text.strip() or x1 - x0 < 2 or y1 - y0 < 2:
        return text.strip(), _split_words_evenly(text.strip(), box)
    crop = rgb[y0:y1, x0:x1]
    cov, _ = glyph_coverage(crop, local_background(crop))
    ink = cov > float(P["perceive.ocr.rapid.cov_thr"])
    rows = np.flatnonzero(ink.any(axis=1))
    cols = ink.any(axis=0)
    chars = list(text)
    measured = char_x is not None and len(char_x) == len(chars)
    ci = np.flatnonzero(cols)
    lo, hi = (x0 + float(ci[0]), x0 + float(ci[-1]) + 1) if ci.size else (box.x, box.x2)
    space_at: set[int] = set()  # a space is inserted after chars[k]
    gaps: list[float] = []
    ink_h = 0.0
    if rows.size:
        ink_h = float(rows[-1] - rows[0] + 1)
        runs = _empty_runs(cols)
        widths = [e - s for s, e in runs]
        if widths:
            med = float(np.median(widths))
            thr = max(float(P["perceive.ocr.rapid.space_gap_h"]) * ink_h, float(P["perceive.ocr.rapid.space_gap_ratio"]) * med,
                      med + float(P["perceive.ocr.rapid.space_gap_excess_px"]))
            gaps = [x0 + (s + e) / 2.0 for (s, e), w in zip(runs, widths) if w >= thr]
    n_sp = sum(1 for k in range(1, len(chars) - 1) if chars[k] == " ")
    scale = 0.0
    if not measured:  # model positions from Roboto advances scaled onto the ink span
        adv = [0.25 if c == " " else _expected_ink_width(c, 1.0) + 0.1 for c in chars]
        n_new = max(0, len(gaps) - n_sp)
        scale = (hi - lo) / max(1e-6, sum(adv) + 0.25 * n_new)
        cum = np.cumsum(adv)
        char_x = [(lo + scale * (cum[k] - adv[k]), lo + scale * cum[k]) for k in range(len(chars))]
    # candidate boundaries in reading order: existing spaces (absorb their own gap) and adjacent letters
    bnd: list[tuple[float, int, bool]] = []
    for k in range(len(chars) - 1):
        if chars[k] == " ":
            if 0 < k:
                bnd.append(((char_x[k][0] + char_x[k][1]) / 2.0, k, True))
        elif chars[k + 1] != " ":
            bnd.append(((char_x[k][1] + char_x[k + 1][0]) / 2.0, k, False))
    if gaps and bnd:
        # order-preserving alignment of big gaps to boundaries (edit-distance DP): a gap either matches
        # one boundary (cost = distance) or stays unexplained (cost = snap penalty); boundaries may
        # stay unmatched for free. Modelled positions shift right by the spaces inserted before them.
        m, nb = len(gaps), len(bnd)
        pen = float(P["perceive.ocr.rapid.snap_h"]) * max(ink_h, 1.0)
        frac_new = 0.0 if measured or not m else max(0, m - n_sp) / m
        INF = float("inf")
        dp = np.full((m + 1, nb + 1), INF)
        mv = np.zeros((m + 1, nb + 1), np.int8)  # 1 skip boundary, 2 skip gap, 3 match
        dp[0, :] = 0.0
        mv[0, 1:] = 1
        for i in range(m + 1):
            for j in range(nb + 1):
                if i == 0 and j == 0:
                    continue
                best, how = INF, 0
                if j > 0 and dp[i, j - 1] < best:
                    best, how = dp[i, j - 1], 1
                if i > 0 and dp[i - 1, j] + pen < best:
                    best, how = dp[i - 1, j] + pen, 2
                if i > 0 and j > 0:
                    x = bnd[j - 1][0] + scale * 0.25 * frac_new * (i - 0.5)
                    c = dp[i - 1, j - 1] + abs(x - gaps[i - 1])
                    if c < best:
                        best, how = c, 3
                dp[i, j], mv[i, j] = best, how
        i, j = m, nb
        while i > 0 or j > 0:
            how = mv[i, j]
            if how == 3:
                _, k, is_sp = bnd[j - 1]
                if not is_sp:
                    space_at.add(k)
                i, j = i - 1, j - 1
            elif how == 2:
                i -= 1
            else:
                j -= 1
    # rebuild text and word boxes
    words: list[tuple[str, Box]] = []
    cur, cx0, cx1 = "", None, None

    def flush():
        if cur:
            words.append((cur, Box(cx0, box.y, max(0.0, cx1 - cx0), box.h)))

    for k, c in enumerate(chars):
        if c == " ":
            flush(); cur, cx0, cx1 = "", None, None
            continue
        cur += c
        cx0 = char_x[k][0] if cx0 is None else min(cx0, char_x[k][0])
        cx1 = char_x[k][1] if cx1 is None else max(cx1, char_x[k][1])
        if k in space_at:
            flush(); cur, cx0, cx1 = "", None, None
    flush()
    return " ".join(w for w, _ in words), words


_BACKENDS = {"vision": VisionBackend, "rapidocr": RapidOcrBackend, "tesseract": TesseractBackend}
_backend_cache: dict[str, OcrBackend] = {}


def available_backends() -> list[str]:
    """Names of backends that can be constructed on this machine (probes imports)."""
    names: list[str] = []
    for n, cls in _BACKENDS.items():
        if n == "vision" and sys.platform != "darwin":
            continue
        try:
            cls()
            names.append(n)
        except Exception:
            continue
    return names


def get_backend(name: Optional[str] = None) -> OcrBackend:
    """Return an OCR backend. Order: explicit name > $DT_OCR > params > vision > rapidocr > tesseract."""
    want = name or os.environ.get("DT_OCR") or str(P["perceive.ocr.backend"])
    order = list(_BACKENDS) if want == "auto" else [want]
    errors: list[str] = []
    for n in order:
        if n in _backend_cache:
            return _backend_cache[n]
        if n not in _BACKENDS:
            raise ValueError(f"unknown OCR backend {n!r}; choose from {list(_BACKENDS)}")
        if n == "vision" and sys.platform != "darwin":
            errors.append("vision: macOS only")
            continue
        try:
            be = _BACKENDS[n]()
        except Exception as e:  # import or init failure
            errors.append(f"{n}: {type(e).__name__}: {e}")
            continue
        _backend_cache[n] = be
        return be
    raise RuntimeError(
        "no OCR backend available. Install one of: macOS Vision (pyobjc-framework-Vision), "
        "rapidocr-onnxruntime, or tesseract + pytesseract. Details: " + "; ".join(errors)
    )


# --------------------------------------------------------------------------- ink analysis
def _rgb_dist(a: np.ndarray, c: Color) -> np.ndarray:
    """Euclidean RGB distance of each pixel to colour c (float32, HxW)."""
    d = a[..., :3].astype(np.float32) - np.array([c.r, c.g, c.b], np.float32)
    return np.sqrt((d * d).sum(axis=-1))


def _border_ring(a: np.ndarray, t: int = 1) -> np.ndarray:
    h, w = a.shape[:2]
    t = max(1, min(t, h // 2, w // 2))
    return np.concatenate([a[:t].reshape(-1, 3), a[-t:].reshape(-1, 3), a[:, :t].reshape(-1, 3), a[:, -t:].reshape(-1, 3)])


def local_background(a: np.ndarray) -> Color:
    """Dominant colour of the 1px border ring of a crop (the glyphs sit inside)."""
    ring = _border_ring(a, 1)
    c = dominant_color(ring.reshape(-1, 1, 3), quant=4)
    return c if c is not None else Color(255, 255, 255)


def glyph_coverage(a: np.ndarray, bg: Color) -> tuple[np.ndarray, Color]:
    """Per-pixel glyph coverage (0..1) of an antialiased text crop and the estimated text colour.

    Model: pixel = bg + cov * (fg - bg). fg = median of the pixels farthest from bg.
    """
    dist = _rgb_dist(a, bg)
    if dist.max() < float(P["perceive.ocr.min_contrast"]):
        return np.zeros(a.shape[:2], np.float32), bg
    top = dist >= float(P["perceive.ocr.color_min_rel"]) * dist.max()
    fg_rgb = np.median(a[top][:, :3], axis=0)
    fg = Color.from_rgb(*fg_rgb)
    return _coverage(a, bg, fg), fg


def _fg_in(a: np.ndarray, bg: Color, sel: np.ndarray) -> Optional[Color]:
    """Text colour estimated only from the pixels selected by `sel` (farthest from bg)."""
    if not sel.any():
        return None
    dist = _rgb_dist(a, bg)
    d = dist[sel]
    top = sel & (dist >= float(P["perceive.ocr.color_min_rel"]) * float(d.max()))
    return Color.from_rgb(*np.median(a[top][:, :3], axis=0))


def _coverage(a: np.ndarray, bg: Color, fg: Color) -> np.ndarray:
    """Projection of each pixel on the bg->fg axis, clipped to 0..1."""
    v = np.array([fg.r - bg.r, fg.g - bg.g, fg.b - bg.b], np.float32)
    n2 = float((v * v).sum()) or 1.0
    cov = ((a[..., :3].astype(np.float32) - np.array([bg.r, bg.g, bg.b], np.float32)) * v).sum(axis=-1) / n2
    return np.clip(cov, 0.0, 1.0).astype(np.float32)


def _top_class(text: str) -> str:
    chars = set(text)
    if chars & _ASC_CHARS:
        return "asc"
    if chars & _CAP_CHARS:
        return "cap"
    return "x"


def _ink_rows(mask: np.ndarray) -> tuple[int, int]:
    """(top, baseline_row) of the glyph ink.

    The baseline is the row in the lower ``baseline_band`` of the ink with the largest drop in
    ink count to the row below it: bowl bottoms end there, only descender stems continue.
    """
    rows = mask.sum(axis=1).astype(np.int64)
    nz = np.where(rows > 0)[0]
    if nz.size == 0:
        return 0, mask.shape[0] - 1
    top, bottom = int(nz[0]), int(nz[-1])
    start = top + int((bottom - top + 1) * (1.0 - float(P["perceive.ocr.baseline_band"])))
    padded = np.concatenate([rows, [0]])
    drops = padded[start:bottom + 1] - padded[start + 1:bottom + 2]
    return top, start + int(np.argmax(drops))


def _components_touching(mask: np.ndarray, core: tuple[int, int, int, int]) -> np.ndarray:
    """Keep only connected components of mask whose bbox intersects the core rect (x0,y0,x1,y1)."""
    import cv2

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    cx0, cy0, cx1, cy1 = core
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        if x < cx1 and x + w > cx0 and y < cy1 and y + h > cy0:
            keep[i] = True
    return keep[lab]


def _line_band(mask: np.ndarray, core: tuple[int, int, int, int]) -> np.ndarray:
    """Rows (bool, crop-local) of the text line that the OCR box is about.

    Ink rows form bands separated by empty rows; bands closer than ``band_merge_gap`` x the
    tallest band are one line (i-dots, accents). When several lines remain (the crop padding,
    or an over-tall OCR box, reached a neighbouring line) the one with the most ink inside the
    OCR box's columns wins.
    """
    rows = mask.any(axis=1)
    idx = np.where(rows)[0]
    if idx.size == 0:
        return rows
    runs: list[list[int]] = [[int(idx[0]), int(idx[0])]]
    for y in idx[1:]:
        if y == runs[-1][1] + 1:
            runs[-1][1] = int(y)
        else:
            runs.append([int(y), int(y)])
    if len(runs) == 1:
        return rows
    tallest = max(e - s + 1 for s, e in runs)
    gap_max = float(P["perceive.ocr.band_merge_gap"]) * tallest
    groups: list[list[int]] = [list(runs[0])]
    for s, e in runs[1:]:
        if s - groups[-1][1] - 1 < gap_max:
            groups[-1][1] = e
        else:
            groups.append([s, e])
    if len(groups) == 1:
        return rows
    cx0, _, cx1, _ = core
    cols = mask[:, max(0, cx0):max(cx0 + 1, cx1)]
    best = max(groups, key=lambda g: int(cols[g[0]:g[1] + 1].sum()))
    out = np.zeros_like(rows)
    out[best[0]:best[1] + 1] = True
    return out


def _refine_line(rgb: np.ndarray, raw: RawLine) -> Optional[TextLine]:
    """Measure the real ink box / baseline for a raw OCR line. Returns None if no ink found."""
    H, W = rgb.shape[:2]
    pad_y = max(2, int(round(raw.box.h * float(P["perceive.ocr.crop_pad_ratio"]))))
    pad_x = int(P["perceive.ocr.crop_pad_px"])
    x0 = max(0, int(np.floor(raw.box.x)) - pad_x)
    y0 = max(0, int(np.floor(raw.box.y)) - pad_y)
    x1 = min(W, int(np.ceil(raw.box.x2)) + pad_x)
    y1 = min(H, int(np.ceil(raw.box.y2)) + pad_y)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    a = rgb[y0:y1, x0:x1]
    bg = local_background(a)
    cov, fg = glyph_coverage(a, bg)
    mask = cov >= float(P["perceive.ocr.cov_thr"])
    if not mask.any():
        return None
    core = (int(raw.box.x) - x0, int(raw.box.y) - y0, int(np.ceil(raw.box.x2)) - x0, int(np.ceil(raw.box.y2)) - y0)
    mask &= _components_touching(mask, core)
    if not mask.any():
        return None
    rows = _line_band(mask, core)
    mask[~rows] = False
    if not mask.any():
        return None
    # the glyph colour of the crop may have come from a neighbouring (darker) line: re-fit it
    fg2 = _fg_in(a, bg, mask)
    if fg2 is not None and fg2.delta_e(fg) > float(P["perceive.ocr.fg_refit_de"]):
        fg = fg2
        cov = _coverage(a, bg, fg)
        m2 = cov >= float(P["perceive.ocr.cov_thr"])
        m2[~rows] = False
        m2 &= _components_touching(m2, core)
        if m2.any():
            mask = m2
    ys, xs = np.where(mask)
    top, base_row = _ink_rows(mask)
    if raw.text.strip().isdigit():
        # digits sit on the baseline with no descenders, and the largest row drop of '4'/'7' is
        # the crossbar, not the baseline: use the ink bottom
        base_row = int(ys.max())
    ink = Box(x0 + xs.min(), y0 + top, xs.max() - xs.min() + 1, ys.max() - top + 1)
    return TextLine(
        text=raw.text, box=ink, words=list(raw.words), conf=raw.conf, baseline=float(y0 + base_row + 1),
        top_class=_top_class(raw.text), has_desc=bool(set(raw.text) & _DESC_CHARS), bg=bg, fg=fg,
        cov=cov * mask, cov_origin=(x0, y0), meta={"raw_box": raw.box.to_dict()},
    )


def _refine_line_inner(rgb: np.ndarray, raw: RawLine) -> Optional[TextLine]:
    """``_refine_line`` for text inside a small filled container (badge, avatar, pill).

    When the measured 'ink' is really the container (solid mask, density >= ``sc.container_density``,
    e.g. a red badge around white digits), the container colour becomes the background: pixels
    outside the container's (hole-filled, 1px eroded) shape are painted with it and the line is
    measured again, so box / baseline / colour describe the glyphs, not the badge.
    """
    from scipy.ndimage import binary_erosion, binary_fill_holes

    line = _refine_line(rgb, raw)
    if line is None or line.cov is None:
        return line
    m = line.cov >= float(P["perceive.ocr.cov_thr"])
    ys, xs = np.where(m)
    if ys.size == 0:
        return line
    sub = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if sub.mean() < float(P["perceive.ocr.sc.container_density"]):
        return line
    shape = binary_erosion(binary_fill_holes(m), iterations=1)
    if not (shape & ~m).any():  # no holes: a solid blob, not glyphs inside a container
        return line
    ox, oy = line.cov_origin
    h, w = m.shape
    patch = rgb[oy:oy + h, ox:ox + w].copy()
    cont = line.fg
    patch[~shape] = (cont.r, cont.g, cont.b)
    raw2 = RawLine(raw.text, raw.box.translate(-ox, -oy), [(t, b.translate(-ox, -oy)) for t, b in raw.words], raw.conf)
    inner = _refine_line(patch, raw2)
    if inner is None:
        return line
    inner.box = inner.box.translate(ox, oy)
    inner.baseline += oy
    inner.words = list(raw.words)
    inner.cov_origin = (inner.cov_origin[0] + ox, inner.cov_origin[1] + oy)
    inner.meta["raw_box"] = raw.box.to_dict()
    inner.meta["container"] = cont.hex()
    return inner


def _ink_runs(line: TextLine) -> list[tuple[int, int]]:
    """Horizontal ink runs (x_start, x_end exclusive, absolute px) of a line's coverage mask."""
    if line.cov is None:
        return []
    cols = (line.cov >= float(P["perceive.ocr.cov_thr"])).sum(axis=0) > 0
    xs = np.where(cols)[0]
    if xs.size == 0:
        return []
    x0 = line.cov_origin[0]
    runs: list[tuple[int, int]] = []
    start = prev = int(xs[0])
    for x in xs[1:]:
        if x - prev > 1:
            runs.append((x0 + start, x0 + prev + 1))
            start = int(x)
        prev = int(x)
    runs.append((x0 + start, x0 + prev + 1))
    return runs


def _ink_gap_after(line: TextLine, wb: Box) -> float:
    """Width of the empty column run right after the ink of word box `wb` (px)."""
    runs = _ink_runs(line)
    inside = [r for r in runs if r[1] > wb.x and r[0] < wb.x2]
    if not inside:
        return 0.0
    end = max(r[1] for r in inside)
    after = [r[0] for r in runs if r[0] >= end]
    return float(min(after) - end) if after else 0.0


def _ink_gap_before(line: TextLine, wb: Box) -> float:
    runs = _ink_runs(line)
    inside = [r for r in runs if r[1] > wb.x and r[0] < wb.x2]
    if not inside:
        return 0.0
    start = min(r[0] for r in inside)
    before = [r[1] for r in runs if r[1] <= start]
    return float(start - max(before)) if before else 0.0


def split_icon_words(rgb: np.ndarray, line: TextLine) -> tuple[Optional[TextLine], list[Box]]:
    """Detach single-character 'words' that are really icon glyphs OCR'd as text.

    A leading or trailing single character is an icon when it is not alphanumeric (+ ✓ • ...)
    or when the ink gap between it and the neighbouring word is >= icon_word_gap x size (M3
    icon-label gaps are ~0.6-0.9em, word spaces ~0.35-0.45em). A non-alphanumeric character
    *inside* a line at word spacing is punctuation ("draft 0 - please", "1 – 50 of 2,345",
    "Drive · Shared"): it is only an icon when icon-sized gaps separate it from both neighbours.
    Returns (line or None if nothing remains, icon boxes).
    """
    if len(line.words) < 2:
        return line, []
    size = estimate_size(line)
    gap_px = float(P["perceive.ocr.icon_word_gap"]) * size
    icons: list[Box] = []
    keep: list[tuple[str, Box]] = []
    last = len(line.words) - 1
    for i, (w, wb) in enumerate(line.words):
        if len(w) == 1:
            gap = _ink_gap_after(line, wb) if i == 0 else _ink_gap_before(line, wb)
            if 0 < i < last and not w.isalnum():
                is_icon = min(gap, _ink_gap_after(line, wb)) >= gap_px
            else:
                is_icon = not w.isalnum() or gap >= gap_px
            if is_icon:
                icons.append(wb)
                continue
        keep.append((w, wb))
    if not icons:
        return line, []
    if not keep:
        return None, icons
    line.words = keep
    line.text = " ".join(w for w, _ in keep)
    line.meta["icon_words"] = [b.to_dict() for b in icons]
    return line, icons


def ocr_lines(rgb: np.ndarray, backend: Optional[OcrBackend] = None) -> list[TextLine]:
    """Recognise text lines in an RGB image and measure their ink geometry.

    Lines are re-measured from the pixels (tight ink box, baseline) and sorted top-to-bottom,
    left-to-right. Icon glyphs that the OCR merged into a line as single characters are split
    off (their boxes are in ``line.meta['icon_words']``). Text-like blobs the full-image pass
    returned nothing for (tiny badge digits, light-on-dark labels, isolated titles) get a
    second, bounded OCR pass on upscaled (and, on dark backgrounds, inverted) crops
    (``line.meta['second_chance']``).
    """
    be = backend or get_backend()
    H, W = rgb.shape[:2]
    scale = 1
    if H < int(P["perceive.ocr.upscale_min_h"]) or W < int(P["perceive.ocr.upscale_min_h"]):
        scale = int(np.ceil(int(P["perceive.ocr.upscale_min_h"]) / max(1, min(H, W))))
    src = rgb if scale == 1 else np.kron(rgb, np.ones((scale, scale, 1), np.uint8))
    raws = be.recognize(src)
    out: list[TextLine] = []
    seen: list[Box] = []
    for r in raws:
        if r.conf < float(P["perceive.ocr.min_conf"]) or not r.text.strip():
            continue
        if scale != 1:
            r = RawLine(r.text, _scale_box(r.box, 1 / scale), [(w, _scale_box(b, 1 / scale)) for w, b in r.words], r.conf)
        n0 = len(out)
        _add_line(rgb, _normalize_raw(r), out)
        for k in range(n0, len(out)):  # what the pass really read as text (not icon-only / inkless lines)
            l = out[k]
            if not l.text:
                continue
            if sum(ch.isalnum() for ch in l.text) <= int(P["perceive.ocr.short_alnum"]) and not plausible_text_geometry(l):
                # an icon glyph read as 1-2 letters ('Do' for a person icon): keep it as an icon word
                out[k] = TextLine("", Box(), [], 0.0, meta={"icon_words": [l.box.to_dict()], "rejected_text": l.text})
                seen.append(l.box)
                continue
            seen.extend(wb for _, wb in l.words)
    if int(P["perceive.ocr.sc.enabled"]) and scale == 1:
        for r in second_chance_raws(rgb, seen, be):
            got: list[TextLine] = []
            _add_line(rgb, _normalize_raw(r), got, {"second_chance": True}, refine=_refine_line_inner)
            out.extend(l for l in got if l.text and plausible_text_geometry(l))
    out.sort(key=lambda l: (round(l.box.y / 4), l.box.x))
    return out


def _add_line(rgb: np.ndarray, r: RawLine, out: list[TextLine], meta: Optional[dict] = None, refine=None) -> None:
    """Refine one raw OCR line, split off icon glyphs and append the result(s) to `out`."""
    refine = refine or _refine_line
    line = refine(rgb, r)
    if line is None:
        return
    line, icons = split_icon_words(rgb, line)
    if line is None:
        out.append(TextLine("", Box(), [], 0.0, meta={"icon_words": [b.to_dict() for b in icons]}))
        return
    if icons:  # re-measure ink without the icon glyphs
        raw2 = RawLine(line.text, line.words[0][1].union(line.words[-1][1]), line.words, line.conf)
        line2 = refine(rgb, raw2)
        if line2 is not None:
            line2.meta["icon_words"] = line.meta["icon_words"]
            line = line2
    if meta:
        line.meta.update(meta)
    out.append(line)


# Cyrillic / Greek letters that Vision (language correction off) returns for Latin glyphs
_HOMOGLYPHS = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "Х": "X", "У": "Y", "І": "I", "Ј": "J", "Ѕ": "S", "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s", "Α": "A", "Β": "B", "Ε": "E", "Η": "H", "Ι": "I",
    "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X", "Υ": "Y", "Ζ": "Z", "ο": "o",
})


def normalize_homoglyphs(text: str) -> str:
    """Map Cyrillic/Greek look-alikes to Latin when the text is otherwise Latin ('ОK' -> 'OK')."""
    mapped = text.translate(_HOMOGLYPHS)
    if mapped == text:
        return text
    letters = [c for c in text if c.isalpha()]
    latin = sum(1 for c in letters if c.isascii())
    foreign = sum(1 for c in letters if not c.isascii() and c.translate(_HOMOGLYPHS) == c)
    return mapped if latin > 0 and foreign == 0 else text


def _normalize_raw(r: RawLine) -> RawLine:
    t = normalize_homoglyphs(r.text)
    if t == r.text:
        return r
    return RawLine(t, r.box, [(normalize_homoglyphs(w), b) for w, b in r.words], r.conf)


# --------------------------------------------------------------------------- second chance
def second_chance_candidates(rgb: np.ndarray, covered: list[Box]) -> list[Box]:
    """Text-like blobs (glyph-edge clusters of word size) that no OCR line covers.

    Edges = 3x3 morphological colour gradient >= ``sc.edge_thr``; joined horizontally by
    ``sc.join_px``; components of ``sc.min_h..sc.max_h`` height (<= ``sc.max_w`` wide) whose box
    neither contains the centre of, nor mostly overlaps, an already recognised line. Sorted
    small-first (badges and short labels are the usual misses), then top-to-bottom.
    """
    import cv2

    H, W = rgb.shape[:2]
    k = np.ones((3, 3), np.uint8)
    img = np.ascontiguousarray(rgb[..., :3])
    grad = (cv2.dilate(img, k).astype(np.int16) - cv2.erode(img, k).astype(np.int16)).max(axis=2)
    mask = (grad >= int(P["perceive.ocr.sc.edge_thr"])).astype(np.uint8)
    for b in covered:
        x0, y0 = max(0, int(b.x) - 1), max(0, int(b.y) - 1)
        x1, y1 = min(W, int(np.ceil(b.x2)) + 1), min(H, int(np.ceil(b.y2)) + 1)
        mask[y0:y1, x0:x1] = 0
    jw = max(1, int(P["perceive.ocr.sc.join_px"]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (jw, 3)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_h, max_h, max_w = int(P["perceive.ocr.sc.min_h"]), int(P["perceive.ocr.sc.max_h"]), int(P["perceive.ocr.sc.max_w"])
    out: list[Box] = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i, :4])
        if h < min_h or h > max_h or w > max_w or w < 0.5 * min_h:
            continue
        b = Box(x, y, w, h)
        if any(b.contains_point(c.cx, c.cy) or b.intersect(c).area > 0.5 * min(b.area, c.area) for c in covered):
            continue
        out.append(b)
    out.sort(key=lambda b: (b.area > 4 * max_h * max_h, b.y, b.x))
    return out


def _ocr_crop(be: OcrBackend, rgb: np.ndarray, b: Box, invert: bool) -> list[RawLine]:
    """OCR an upscaled (optionally inverted) crop around `b`; boxes mapped back to image px."""
    import cv2

    H, W = rgb.shape[:2]
    pad = max(2, int(round(float(P["perceive.ocr.sc.pad_ratio"]) * b.h)))
    x0, y0 = max(0, int(b.x) - pad), max(0, int(b.y) - pad)
    x1, y1 = min(W, int(np.ceil(b.x2)) + pad), min(H, int(np.ceil(b.y2)) + pad)
    c = np.ascontiguousarray(rgb[y0:y1, x0:x1, :3])
    s = max(1, int(P["perceive.ocr.sc.scale"]))
    if s > 1:
        c = cv2.resize(c, ((x1 - x0) * s, (y1 - y0) * s), interpolation=cv2.INTER_CUBIC)
    if invert:
        c = 255 - c
    bd = int(P["perceive.ocr.sc.border_px"])
    if bd > 0:
        c = cv2.copyMakeBorder(c, bd, bd, bd, bd, cv2.BORDER_REPLICATE)

    def back(q: Box) -> Box:
        return Box((q.x - bd) / s + x0, (q.y - bd) / s + y0, q.w / s, q.h / s)

    return [RawLine(r.text, back(r.box), [(w, back(wb)) for w, wb in r.words], r.conf)
            for r in be.recognize(np.ascontiguousarray(c))]


# Roboto advance widths (em) by glyph class; used only as a plausibility check of OCR results
_ADV_NARROW = {**dict.fromkeys("iljI|!.,:;'", 0.25), **dict.fromkeys("frt()[]-1", 0.36), "J": 0.55}
_ADV_WIDE = dict.fromkeys("mwMW@", 0.86)


def _expected_ink_width(text: str, size: float) -> float:
    """Approximate Roboto ink width (px) of `text` at `size` (advances minus side bearings)."""
    em = 0.0
    for ch in text:
        if ch in _ADV_NARROW:
            em += _ADV_NARROW[ch]
        elif ch in _ADV_WIDE:
            em += _ADV_WIDE[ch]
        elif ch == " ":
            em += 0.25
        elif ch.isdigit():
            em += 0.56
        elif ch.isupper():
            em += 0.64
        else:
            em += 0.55
    return max(0.2, em - 0.1) * size


def plausible_text_geometry(line: TextLine) -> bool:
    """Is the measured ink width consistent with the recognised characters at the measured size?

    Rejects icon glyphs that a crop-level OCR reads as letters ('Do' for a person icon, 'J' for
    a videocam): their ink is far wider or narrower than those letters would be at a size
    derived from the ink height.
    """
    exp = _expected_ink_width(line.text, estimate_size(line))
    r = line.box.w / max(1.0, exp)
    line.meta["width_ratio"] = round(r, 3)
    return float(P["perceive.ocr.sc.width_ratio_min"]) <= r <= float(P["perceive.ocr.sc.width_ratio_max"])


def _accept_second(r: RawLine) -> bool:
    if r.conf < float(P["perceive.ocr.sc.min_conf"]):
        return False
    return sum(1 for ch in r.text if ch.isalnum()) >= int(P["perceive.ocr.sc.min_alnum"])


def _filled_subshapes(rgb: np.ndarray, b: Box) -> list[Box]:
    """Small filled shapes with glyph holes inside blob `b` (a badge anchored on an icon).

    Pixels of the blob crop that contrast with its background are grouped by quantised colour;
    connected components of one colour that enclose holes and are ``sc.min_h``..``sc.badge_max_h``
    tall are returned (image px), largest-first.
    """
    import cv2
    from scipy.ndimage import binary_fill_holes

    x0, y0, x1, y1 = int(b.x), int(b.y), int(np.ceil(b.x2)), int(np.ceil(b.y2))
    a = rgb[y0:y1, x0:x1, :3]
    if a.size == 0:
        return []
    bg = local_background(a)
    fgm = _rgb_dist(a, bg) >= float(P["perceive.ocr.min_contrast"])
    q = (a.astype(np.int32) // 32)
    key = (q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2])
    key[~fgm] = -1
    vals, counts = np.unique(key[key >= 0], return_counts=True)
    min_h, max_h = int(P["perceive.ocr.sc.min_h"]), int(P["perceive.ocr.sc.badge_max_h"])
    out: list[tuple[int, Box]] = []
    for v, c in zip(vals, counts):
        if c < min_h * min_h // 2:
            continue
        m = (key == v).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = (int(t) for t in stats[i])
            if not (min_h <= h <= max_h and w >= 0.5 * h and area >= min_h * min_h // 2):
                continue
            comp = lab[y:y + h, x:x + w] == i
            holes = binary_fill_holes(comp) & ~comp
            if holes.sum() < 4:
                continue
            out.append((area, Box(x0 + x, y0 + y, w, h)))
    out.sort(key=lambda t: -t[0])
    return [bx for _, bx in out]


def _read_blob(be: OcrBackend, rgb: np.ndarray, b: Box, budget: int) -> tuple[list[RawLine], int]:
    """OCR one candidate (inverted first on a dark background); returns (accepted lines, calls used)."""
    H, W = rgb.shape[:2]
    pad = max(1, int(round(0.25 * b.h)))
    ctx = rgb[max(0, int(b.y) - pad):min(H, int(np.ceil(b.y2)) + pad), max(0, int(b.x) - pad):min(W, int(np.ceil(b.x2)) + pad)]
    dark = local_background(ctx).luminance() < float(P["perceive.ocr.sc.dark_lum"])
    used = 0
    for inv in ((True, False) if dark else (False,)):
        if used >= budget:
            break
        used += 1
        found = [r for r in _ocr_crop(be, rgb, b, inv) if r.text.strip() and _accept_second(r)]
        if found:
            return found, used
    return [], used


def second_chance_raws(rgb: np.ndarray, covered: list[Box], backend: Optional[OcrBackend] = None) -> list[RawLine]:
    """Second OCR pass over uncovered text-like blobs (bounded by ``sc.max_crops`` OCR calls).

    Each crop is upscaled x ``sc.scale``; crops on a dark local background are read inverted
    first and, if that finds nothing, once more as-is. A blob that yields nothing is searched
    for small filled shapes with holes (a badge on an icon) which are read on their own. Only
    results with conf >= ``sc.min_conf`` and >= ``sc.min_alnum`` alphanumerics, whose box lies
    on the candidate and overlaps no recognised line, are returned (image coordinates).
    """
    be = backend or get_backend()
    budget = int(P["perceive.ocr.sc.max_crops"])
    taken = list(covered)
    out: list[RawLine] = []
    for b in second_chance_candidates(rgb, covered):
        if budget <= 0:
            break
        if any(b.contains_point(c.cx, c.cy) for c in taken):
            continue
        found, used = _read_blob(be, rgb, b, budget)
        budget -= used
        areas = [b]
        if not found:
            for sb in _filled_subshapes(rgb, b)[:int(P["perceive.ocr.sc.max_sub"])]:
                if budget <= 0:
                    break
                got, used = _read_blob(be, rgb, sb, budget)
                budget -= used
                if got:
                    found, areas = got, [sb]
                    break
        for r in found:
            if not any(a.contains_point(r.box.cx, r.box.cy) or r.box.contains_point(a.cx, a.cy) for a in areas):
                continue
            if any(r.box.contains_point(c.cx, c.cy) or c.contains_point(r.box.cx, r.box.cy) for c in taken):
                continue
            taken.append(r.box)
            out.append(r)
    return out


def _scale_box(b: Box, s: float) -> Box:
    return Box(b.x * s, b.y * s, b.w * s, b.h * s)


# --------------------------------------------------------------------------- style
def estimate_size(line: TextLine) -> float:
    """Font size (px) from the ink height above the baseline using the calibrated per-class fit."""
    ink_h = max(1.0, line.baseline - line.box.y)
    a = float(P[f"perceive.ocr.size_fit.{line.top_class}.a"])
    b = float(P[f"perceive.ocr.size_fit.{line.top_class}.b"])
    return max(4.0, a * ink_h + b)


def stroke_width(cov: np.ndarray) -> float:
    """Mean stroke width of an antialiased glyph coverage map: ink mass / skeleton length."""
    from skimage.morphology import skeletonize

    mask = cov >= float(P["perceive.ocr.cov_thr"])
    if not mask.any():
        return 0.0
    skel = skeletonize(mask)
    length = float(skel.sum())
    if length <= 0:
        return 0.0
    return float(cov.sum()) / length


def estimate_weight(line: TextLine, size: float) -> int:
    """Weight class (400/500/700) from stroke width normalised by font size."""
    if line.cov is None:
        return 400
    ratio = stroke_width(line.cov) / max(1.0, size)
    line.meta["stroke_ratio"] = round(ratio, 4)
    if ratio >= float(P["perceive.ocr.stroke_thr_700"]):
        return 700
    if ratio >= float(P["perceive.ocr.stroke_thr_500"]):
        return 500
    return 400


def estimate_text_style(rgb: np.ndarray, line: TextLine) -> TextStyle:
    """Size / weight / colour of one line. ``line_height`` is left None (set by paragraphs)."""
    size = estimate_size(line)
    weight = estimate_weight(line, size)
    return TextStyle(family="Roboto", size=round(size, 1), weight=weight, color=line.fg, align="left", valign="top")


# --------------------------------------------------------------------------- paragraphs
def _same_paragraph(a: TextLine, sa: TextStyle, b: TextLine, sb: TextStyle) -> bool:
    if abs(sa.size - sb.size) > float(P["perceive.ocr.para_size_tol"]):
        return False
    if sa.color.delta_e(sb.color) > float(P["perceive.ocr.para_color_de"]):
        return False
    gap = b.baseline - a.baseline
    if gap <= 0 or gap > float(P["perceive.ocr.para_gap_ratio"]) * sa.size:
        return False
    tol = float(P["perceive.ocr.para_x_tol"])
    aligned = abs(a.box.x - b.box.x) <= tol or abs(a.box.cx - b.box.cx) <= tol or abs(a.box.x2 - b.box.x2) <= tol
    return aligned


def group_paragraphs(rgb: np.ndarray, lines: list[TextLine]) -> list[Paragraph]:
    """Merge vertically stacked, aligned, same-style lines into paragraphs; estimate line_height."""
    lines = [l for l in lines if l.text]
    styles = [estimate_text_style(rgb, l) for l in lines]
    order = sorted(range(len(lines)), key=lambda i: (lines[i].baseline, lines[i].box.x))
    used = [False] * len(lines)
    paras: list[Paragraph] = []
    for i in order:
        if used[i]:
            continue
        group = [i]
        used[i] = True
        last = i
        for j in order:
            if used[j] or j == i:
                continue
            if _same_paragraph(lines[last], styles[last], lines[j], styles[j]):
                group.append(j)
                used[j] = True
                last = j
        ls = [lines[k] for k in group]
        box = ls[0].box
        for l in ls[1:]:
            box = box.union(l.box)
        st = styles[group[0]]
        lh = None
        if len(ls) > 1:
            lh = float(np.median([ls[k + 1].baseline - ls[k].baseline for k in range(len(ls) - 1)]))
            st.line_height = round(lh, 1)
        st.align = _para_align(ls)
        paras.append(Paragraph(ls, st, box, "\n".join(l.text for l in ls), lh))
    return paras


def _para_align(ls: list[TextLine]) -> str:
    if len(ls) < 2:
        return "left"
    tol = float(P["perceive.ocr.para_x_tol"])
    if all(abs(l.box.x - ls[0].box.x) <= tol for l in ls):
        return "left"
    if all(abs(l.box.cx - ls[0].box.cx) <= tol for l in ls):
        return "center"
    if all(abs(l.box.x2 - ls[0].box.x2) <= tol for l in ls):
        return "right"
    return "left"


def paragraph_node(p: Paragraph) -> Node:
    """IR text node whose box reproduces the renderer's line box for this paragraph.

    The renderer lays text at the top of the box with ``line-height: normal`` (or the given
    line_height, which adds half-leading above the ascent), so:
        y = first_baseline - ascent*size - (L - normal)/2 ; h = n_lines * L
    """
    st = p.style
    size = st.size
    normal = float(P["perceive.ocr.line_height_ratio"]) * size
    L = p.line_height if p.line_height else normal
    ascent = float(P["perceive.ocr.ascent_ratio"]) * size
    lsb = float(P["perceive.ocr.lsb_ratio"]) * size
    first = p.lines[0].baseline
    y = first - ascent - (L - normal) / 2.0
    h = L * len(p.lines)
    x = p.box.x - lsb
    w = p.box.w + 2 * lsb + float(P["perceive.ocr.text_w_slack"]) * size
    n = Node(id=new_id("t"), type="text", name=p.text.split("\n")[0][:24], box=Box(x, y, w, h), text=p.text, text_style=st)
    n.meta = {
        "src": "ocr", "conf": round(float(np.mean([l.conf for l in p.lines])), 3),
        "ink": p.box.to_dict(), "baseline": first, "top_class": p.lines[0].top_class,
        "stroke_ratio": p.lines[0].meta.get("stroke_ratio"),
    }
    return n


def text_nodes(rgb: np.ndarray, lines: Optional[list[TextLine]] = None) -> list[Node]:
    """Convenience: OCR (unless lines given) -> paragraphs -> IR text nodes."""
    if lines is None:
        lines = ocr_lines(rgb)
    return [paragraph_node(p) for p in group_paragraphs(rgb, lines)]
