"""Adversary 'ir-space': compare target and candidate in IR space (the perceptual fixed point).

Idea. Instead of comparing pixels, run OUR OWN perceiver (``dt.perceive.perceive``, with its icon and
font annotations) on BOTH images, match the two perceived trees element by element, and type every
difference from the matched pairs (moved, resized, recoloured, re-rounded, restyled text, different
text, different glyph, shadow, faded) and from the unmatched elements (missing / extra, split / merge).
Because the same perceiver reads both sides, its *systematic* errors (a mis-OCR'd word, a merged line,
a wrong font family) appear on both sides and cancel; only what differs between the two readings is
reported. This is what makes the findings *typed in the vocabulary of the IR*: every finding names the
IR property that is wrong, which is exactly what the refine loop needs.

Fixed-point test. ``fixed_point(rgb)`` runs perceive -> render -> perceive and diffs the two readings
with the same differ. Where perceive(render(perceive(x))) != perceive(x) the perceiver is not at a fixed
point: its reading there is unstable (a threshold sits on a knife edge), so a target-vs-candidate
difference on such an element is weak evidence. ``find`` uses those boxes to lower confidence.

Bias risk and mitigation. The same perceiver judges both images, so (a) it is blind wherever it is
blind on both (an element it never sees cannot be reported missing) and (b) it can *hallucinate*
differences: a one-pixel change may flip a grouping / OCR-line / icon-name decision and produce a
cascade of differences in places whose pixels did not change at all. Mitigations, all on by default:

1. **Pixel support.** A finding is kept only if the (lightly blurred) ΔE map between the two images has
   at least ``adversary.irspace.pix_min_px`` pixels above a noise-adaptive threshold inside its box.
   The threshold is ``max(pix_de, pix_k x median edge ΔE)``: on a noisy target (blur, JPEG, another
   browser) the floor rises with the page-wide edge noise, so nuisance alone does not count.
2. **Fixed-point down-weighting** (above).
3. **Optional residual fallback** (``adversary.irspace.residual_fallback``): pixel residual regions that
   no IR finding explains are reported with a low confidence and a pixel-typed guess, which measures
   (and covers) the perceiver's blind spots instead of hiding them.

Public API
----------
* ``find(target_rgb, candidate_rgb, candidate_ir=None) -> list[Finding]`` (the adversary)
* ``compare_docs(a, b) -> list[Diff]`` (the IR differ, no pixel gating; used by the fixed point)
* ``fixed_point(rgb, doc=None) -> dict`` (instability findings + stability rate)
* ``perceive_cached(rgb)`` / ``warm(items, workers)`` (perception cache; perception is ~5 s per image)
* ``main()``: tune on CALIBRATION, score on EVALUATION, run on out/real_r3m, write out/adversary/ir-space.md

All thresholds are registered in ``dt.params`` under ``adversary.irspace.*``; the registered defaults are the values
tuned on the benchmark CALIBRATION split (coordinate descent, ``tune``; see out/adversary/irspace_tuned.json).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

import cv2
import numpy as np

from dt.adversary.taxonomy import Finding
from dt.ir import Box, Color, Document, Node
from dt.params import P, register

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE_DIR = os.environ.get("DT_IRSPACE_CACHE", os.path.join(ROOT, "out", "adversary", "irspace_cache"))

# --------------------------------------------------------------------------- params
_R = "adversary.irspace."
register(_R + "match_max_disp", 40.0, "largest edge displacement (px) for two elements to be matched (shift/spacing reach)", (8.0, 96.0))
register(_R + "match_min_ratio", 0.5, "matched elements keep at least this ratio of width and of height", (0.2, 0.9))
register(_R + "match_w_disp", 1.0, "matching cost weight: edge displacement / match_max_disp", (0.0, 4.0))
register(_R + "match_w_color", 0.6, "matching cost weight: colour ΔE / 50", (0.0, 4.0))
register(_R + "match_w_text", 0.8, "matching cost weight: text CER", (0.0, 4.0))
register(_R + "match_w_icon", 0.3, "matching cost weight: different icon name", (0.0, 4.0))
register(_R + "match_max_cost", 1.8, "assignments costing more than this are left unmatched", (0.5, 5.0))
register(_R + "move_min_px", 1.0, "edge displacement (px, rounded) that counts as moved", (0.5, 4.0))
register(_R + "size_tol_px", 0.5, "|Δw|,|Δh| up to this (px) is 'same size' (a move); above = resized", (0.5, 4.0))
register(_R + "color_min_de", 2.5, "ΔE76 between matched colours that counts as recoloured", (1.0, 15.0))
register(_R + "radius_min_px", 3.0, "|Δradius| (px) that counts as re-rounded", (1.0, 8.0))
register(_R + "text_size_min", 1.0, "|Δsize| (px) that counts as restyled text", (0.5, 4.0))
register(_R + "text_weight_min", 100, "|Δweight| that counts as restyled text (the perceiver's weight steps are 100)", (100, 400))
register(_R + "text_width_rel", 0.0, "same text and style: width change must also exceed this fraction of the width", (0.0, 0.1))
register(_R + "text_width_min", 2.0, "same text, same style, width change (px) above this = restyled/split text", (1.0, 12.0))
register(_R + "shadow_min_px", 3.0, "|Δ(blur+|dy|)| (px) of the perceived shadow that counts as a shadow change", (0.5, 8.0))
register(_R + "opacity_res", 10.0, "RGB residual (0..255) of the 'blended toward the backdrop' fit for an opacity finding", (2.0, 30.0))
register(_R + "opacity_min_contrast", 24.0, "element-vs-backdrop RGB distance needed to judge opacity", (5.0, 80.0))
register(_R + "opacity_alpha_tol", 0.12, "elements faded by alphas this close (and adjacent) form one opacity finding", (0.02, 0.3))
register(_R + "group_pad", 8.0, "px: moved/missing/extra elements whose boxes are this close are grouped into one finding", (0.0, 24.0))
register(_R + "gap_min_px", 1.5, "a restyled text line whose widest internal ink gap changed by this (px, beyond the size change) is a split/merge", (1.0, 1000.0))
register(_R + "ink_thr", 60, "|RGB - line background| (sum over channels) that marks an ink column", (10, 200))
register(_R + "reread_iou", 0.5, "an unmatched target and candidate element overlapping this much are one element read differently", (0.2, 0.95))
register(_R + "split_iou", 0.75, "IoU of the union of two parts with the whole for a split/merge", (0.4, 0.95))
register(_R + "tint_min_n", 3, "colour pairs needed to judge a global tint", (2, 12))
register(_R + "tint_min_de", 1.5, "per-pair ΔE76 above which a pair is 'tinted'", (0.5, 6.0))
register(_R + "tint_frac", 0.6, "fraction of colour pairs that must be tinted (consistently) for a global tint", (0.3, 1.0))
register(_R + "tint_spread", 0.6, "max std/|mean| of the per-pair Lab offsets for a consistent global tint", (0.1, 1.5))
register(_R + "tint_pix_de", 1.0, "median page ΔE needed as pixel support for a global tint", (0.2, 5.0))
register(_R + "pix_sigma", 1.0, "Gaussian sigma (px) applied to both images before the pixel-support ΔE map", (0.0, 2.0))
register(_R + "pix_de", 4.0, "pixel support: ΔE floor for a supporting pixel", (2.0, 30.0))
register(_R + "pix_k", 3.0, "pixel support: threshold = max(pix_de, pix_k x the pix_q quantile of ΔE over page edge pixels)", (0.0, 10.0))
register(_R + "pix_q", 0.9, "pixel support: quantile of the page-wide edge ΔE used as the noise floor", (0.5, 0.995))
register(_R + "pix_color_frac", 0.5, "colour support: a raw pixel supports a recolour claiming ΔE m when its ΔE > this x m (capped at pix_de)", (0.1, 1.0))
register(_R + "pix_ratio", 0.0, "pixel support: the box's mean edge ΔE must exceed this x the page-wide median edge ΔE (+0.5)", (0.0, 20.0))
register(_R + "pix_min_px", 12, "pixel support: supporting pixels needed inside a finding's box", (1, 100))
register(_R + "pix_conf_scale", 40.0, "confidence = 1 - exp(-supporting px / this)", (2.0, 400.0))
register(_R + "edge_grad", 40.0, "gradient magnitude (Sobel, gray) that marks a target edge pixel (noise floor estimate)", (5.0, 200.0))
register(_R + "nuis_enabled", True, "match the target's nuisance (sub-pixel offset, blur, JPEG) on the candidate before perceiving / comparing")
register(_R + "nuis_gain", 0.15, "a nuisance degradation is kept when it lowers the median edge ΔE by this fraction", (0.0, 0.8))
register(_R + "nuis_min_de", 0.3, "skip nuisance matching when the median edge ΔE is already below this", (0.0, 3.0))
register(_R + "jnd_de", 2.3, "ΔE of a just-noticeable difference (global disturbance level)", (1.0, 6.0))
register(_R + "noisy_frac", 0.2, "a target whose fraction of changed edge pixels (after nuisance matching) reaches this is 'globally disturbed'", (0.05, 1.01))
register(_R + "noisy_geom_px", 1.5, "on a disturbed target, geometric findings up to this many px are dropped", (0.0, 4.0))
register(_R + "noisy_text", 2.5, "on a disturbed target, text.style findings up to this many style steps are dropped", (0.0, 4.0))
register(_R + "noisy_de", 8.0, "on a disturbed target, recolours up to this ΔE are dropped", (0.0, 20.0))
register(_R + "noisy_alpha", 0.15, "on a disturbed target, opacity findings up to this |Δopacity| are dropped", (0.0, 0.5))
register(_R + "noisy_cer", 0.2, "on a disturbed target, text.content findings up to this CER are dropped", (0.0, 0.6))
register(_R + "fp_enabled", True, "run the fixed-point test on the target and down-weight findings on unstable elements")
register(_R + "fp_conf_scale", 0.5, "confidence multiplier for a finding overlapping a perception-unstable box", (0.0, 1.0))
register(_R + "fp_drop_px", 0, "drop a finding on an unstable box unless it has at least this many supporting px (0 = never drop)", (0, 400))
register(_R + "residual_fallback", False, "also report pixel residual regions that no IR finding explains (low confidence)")
register(_R + "residual_conf", 0.15, "confidence of a residual-fallback finding", (0.0, 1.0))
register(_R + "residual_min_px", 24, "supporting px for a residual-fallback region", (4, 400))
register(_R + "min_conf", 0.0, "findings below this confidence are not reported", (0.0, 0.9))

TYPE_PRIOR = {  # how much a typed reading is trusted before pixel support (orders AP)
    "color.tint_global": 1.0, "text.content": 0.95, "icon.glyph": 0.9, "geometry.shift": 0.9, "text.position": 0.9,
    "layout.spacing": 0.9, "geometry.size": 0.85, "color.fill": 0.85, "color.stroke": 0.8, "structure.missing": 0.8,
    "icon.missing": 0.85, "structure.extra": 0.75, "structure.split": 0.7, "structure.merge": 0.7, "text.style": 0.75,
    "geometry.radius": 0.7, "effect.shadow": 0.6, "effect.opacity": 0.75,
}


# --------------------------------------------------------------------------- perception cache
def _src_hash(*dirs: str) -> str:
    h = hashlib.sha1()
    for d in dirs:
        p = os.path.join(ROOT, "dt", d)
        for f in sorted(os.listdir(p)):
            if f.endswith(".py"):
                with open(os.path.join(p, f), "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()[:10]


_PERC_VERSION: Optional[str] = None


def perceiver_version() -> str:
    """Hash of the perceiver source + its params (cache invalidation)."""
    global _PERC_VERSION
    if _PERC_VERSION is None:
        params = {k: v for k, v in sorted(P.all().items()) if k.startswith("perceive.")}
        _PERC_VERSION = hashlib.sha1((_src_hash("perceive") + json.dumps(params, sort_keys=True, default=str)).encode()).hexdigest()[:12]
    return _PERC_VERSION


def image_key(rgb: np.ndarray) -> str:
    a = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    return hashlib.sha1(str(a.shape).encode() + a.tobytes()).hexdigest()[:20]


_MEM: dict[str, Document] = {}


def _cache_path(kind: str, key: str) -> str:
    return os.path.join(CACHE_DIR, perceiver_version(), f"{kind}_{key}.json")


def _cached(kind: str, key: str, make) -> Document:
    mk = f"{kind}:{key}"
    if mk in _MEM:
        return copy.deepcopy(_MEM[mk])
    path = _cache_path(kind, key)
    if os.path.exists(path):
        try:
            doc = Document.load(path)
            _MEM[mk] = doc
            return copy.deepcopy(doc)
        except Exception:
            pass
    doc = make()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp"
    doc.save(tmp)
    os.replace(tmp, path)
    _MEM[mk] = doc
    return copy.deepcopy(doc)


def perceive_cached(rgb: np.ndarray) -> Document:
    """``dt.perceive.perceive(rgb)`` memoised on the image bytes and the perceiver version."""
    from dt.perceive import perceive
    return _cached("p", image_key(rgb), lambda: perceive(np.ascontiguousarray(rgb[..., :3])))


def reperceive_cached(rgb: np.ndarray) -> Document:
    """perceive(render(perceive(rgb))), memoised (the fixed-point second reading)."""
    from dt.perceive import perceive
    from dt.render.screenshot import render_doc

    def make() -> Document:
        doc = perceive_cached(rgb)
        doc.source_image = None
        return perceive(render_doc(doc))
    return _cached("fp", image_key(rgb), make)


def _warm_one(args: tuple) -> tuple[str, float, float]:
    path, fp = args[0], args[1]
    from dt.common.image import load_rgb
    if len(args) > 2:  # (candidate path, False, target path): the nuisance-matched candidate
        t0 = time.perf_counter()
        cand, recipe = matched_candidate(load_rgb(args[2]), load_rgb(path))
        if recipe:
            perceive_cached(cand)
        return path, time.perf_counter() - t0, 0.0
    rgb = load_rgb(path)
    t0 = time.perf_counter()
    perceive_cached(rgb)
    t1 = time.perf_counter()
    if fp:
        reperceive_cached(rgb)
    return path, t1 - t0, time.perf_counter() - t1


def warm(items: Iterable[tuple[str, bool]], workers: int = 3, progress=None) -> list[tuple[str, float, float]]:
    """Fill the perception cache for image files ``[(path, with_fixed_point)]`` with a process pool.
    Returns ``[(path, perceive_s, fixed_point_s)]`` (cold timings; 0-ish when already cached)."""
    items = list(dict.fromkeys(items))
    if workers <= 1:
        return [_warm_one(it) for it in items]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out = []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_warm_one, it): it for it in items}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                out.append(f.result())
            except Exception as e:  # keep going: find() will perceive it in-process
                out.append((futs[f][0], float("nan"), float("nan")))
                if progress:
                    progress(f"warm failed {futs[f][0]}: {type(e).__name__}: {e}")
            if progress and k % 10 == 0:
                progress(f"warm {k}/{len(items)}")
    return out


# --------------------------------------------------------------------------- elements
CLASS_OF = {"text": "text", "icon": "icon", "vector": "icon", "line": "line", "image": "image"}


@dataclass
class El:
    id: str
    cls: str                  # text | icon | line | image | shape
    box: Box
    node: Node
    fill: Optional[Color]
    stroke: Optional[Color]
    radius: float
    text: str = ""
    size: float = 0.0
    weight: int = 0
    icon: Optional[str] = None
    shadow: float = 0.0       # blur + |dy| of the perceived shadow (0 = none)
    backdrops: list[Color] = field(default_factory=list)  # ancestor fills, nearest first
    parent: Optional[str] = None


def _solid(n: Node) -> Optional[Color]:
    for f in n.fills:
        if f.kind == "solid" and f.color is not None and f.color.a > 0.02 and f.opacity > 0.02:
            return f.color
    return None


def _paints(n: Node) -> bool:
    if not n.visible or n.opacity <= 0.02:
        return False
    if n.type == "text":
        return bool((n.text or "").strip())
    if n.type in ("icon", "vector", "image"):
        return True
    return _solid(n) is not None or bool(n.strokes) or bool(n.effects)


def elements(doc: Document) -> list[El]:
    """Every painted, non-root node of a (perceived) document as a comparable element."""
    out: list[El] = []

    def rec(n: Node, backs: list[Color], parent: Optional[str]) -> None:
        own = _solid(n)
        if n is not doc.root and _paints(n):
            cls = CLASS_OF.get(n.type, "shape")
            ts = n.text_style
            fill = (ts.color if ts is not None else None) if cls == "text" else own
            sh = max([e.blur + abs(e.dy) for e in n.effects if not e.inner] or [0.0])
            out.append(El(n.id, cls, n.box, n, fill, n.strokes[0].color if n.strokes else None, float(sum(n.radius) / 4.0),
                          " ".join((n.text or "").split()), float(ts.size) if ts else 0.0, int(ts.weight) if ts else 0,
                          n.icon_name, float(sh), list(backs), parent))
        nb = ([own] + backs) if own is not None else backs
        for c in n.children:
            rec(c, nb, n.id if n is not doc.root else None)
    rec(doc.root, [], None)
    return out


def _lev(a: str, b: str) -> int:
    from dt.compare.structural import levenshtein
    return levenshtein(a, b)


def _cer(t: str, c: str) -> float:
    if t == c:
        return 0.0
    return _lev(t, c) / max(1, len(t))


def _edge_disp(a: Box, b: Box) -> float:
    return max(abs(a.x - b.x), abs(a.y - b.y), abs(a.x2 - b.x2), abs(a.y2 - b.y2))


def _compatible(a: El, b: El) -> bool:
    if a.cls == b.cls:
        return True
    return {a.cls, b.cls} <= {"shape", "image", "line"}


def _de(a: Optional[Color], b: Optional[Color]) -> float:
    if a is None or b is None:
        return 0.0 if a is None and b is None else 50.0
    return a.delta_e(b)


def match(ta: list[El], ca: list[El]) -> list[tuple[int, int, float]]:
    """One-to-one assignment of target elements to candidate elements (Hungarian) with a cost mixing
    IoU, edge displacement, colour, text and icon agreement; type classes gate the pairs."""
    if not ta or not ca:
        return []
    from scipy.optimize import linear_sum_assignment
    md = float(P[_R + "match_max_disp"])
    mr = float(P[_R + "match_min_ratio"])
    wd, wc, wt, wi = (float(P[_R + k]) for k in ("match_w_disp", "match_w_color", "match_w_text", "match_w_icon"))
    big = 1e6
    C = np.full((len(ta), len(ca)), big)
    tb = np.array([[e.box.x, e.box.y, e.box.x2, e.box.y2] for e in ta])
    cb = np.array([[e.box.x, e.box.y, e.box.x2, e.box.y2] for e in ca])
    disp = np.abs(tb[:, None, :] - cb[None, :, :]).max(axis=2)
    for i, a in enumerate(ta):
        for j in np.nonzero(disp[i] <= md)[0]:
            b = ca[j]
            if not _compatible(a, b):
                continue
            if min(a.box.w, b.box.w) < mr * max(a.box.w, b.box.w) and min(a.box.h, b.box.h) < mr * max(a.box.h, b.box.h):
                continue
            iou = a.box.iou(b.box)
            cost = (1.0 - iou) + wd * disp[i, j] / md + wc * min(1.0, _de(a.fill, b.fill) / 50.0)
            if a.cls == "text":
                cost += wt * min(1.0, _cer(a.text, b.text))
            if a.cls == "icon" and (a.icon or "") != (b.icon or ""):
                cost += wi
            C[i, j] = cost
    r, c = linear_sum_assignment(C)
    mx = float(P[_R + "match_max_cost"])
    return [(int(i), int(j), float(C[i, j])) for i, j in zip(r, c) if C[i, j] <= mx]


# --------------------------------------------------------------------------- the IR differ
@dataclass
class Diff:
    """One raw difference between two perceived documents (before pixel gating / grouping)."""
    type: str
    box: Box
    magnitude: Optional[float]
    t: Optional[El] = None
    c: Optional[El] = None
    ev: dict = field(default_factory=dict)


def _ubox(boxes: Iterable[Box]) -> Box:
    out: Optional[Box] = None
    for b in boxes:
        out = b if out is None else out.union(b)
    return out or Box(0, 0, 0, 0)


def _blend_alpha(t: Color, c: Color, backs: list[Color]) -> Optional[tuple[float, float]]:
    """(alpha, residual) if ``c`` ≈ alpha·t + (1-alpha)·backdrop for one of the target's backdrops."""
    best = None
    T = np.array([t.r, t.g, t.b], float)
    Cc = np.array([c.r, c.g, c.b], float)
    for bk in backs[:3]:
        B = np.array([bk.r, bk.g, bk.b], float)
        v = T - B
        nv = float(v @ v)
        if math.sqrt(nv) < float(P[_R + "opacity_min_contrast"]):
            continue
        a = float((Cc - B) @ v / nv)
        res = float(np.linalg.norm(Cc - (a * T + (1 - a) * B)))
        if best is None or res < best[1]:
            best = (a, res)
    return best


def _union_find(n: int, linked) -> list[list[int]]:
    par = list(range(n))

    def f(i: int) -> int:
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if linked(i, j):
                par[f(i)] = f(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(f(i), []).append(i)
    return list(groups.values())


def compare_docs(a: Document, b: Document) -> list[Diff]:
    """Typed differences of ``b`` (candidate reading) w.r.t. ``a`` (target reading), no pixel gating.

    Pairwise readings: moved (``geometry.shift`` / ``text.position``, grouped by common offset into one
    subtree move, or ``layout.spacing`` when offsets are multiples of one step along one axis), resized
    (``geometry.size``), recoloured (``color.fill`` / ``color.stroke``, or ``effect.opacity`` when the new
    colour is the old one blended toward its backdrop, or ``color.tint_global`` when all colours move by
    one Lab offset), re-rounded (``geometry.radius``), shadow (``effect.shadow``), text (``text.content``,
    ``text.style``), glyph (``icon.glyph``). Unmatched: ``structure.missing`` / ``icon.missing`` /
    ``structure.extra``, and ``structure.split`` / ``structure.merge`` when two parts tile one whole.
    """
    ta, ca = elements(a), elements(b)
    pairs = match(ta, ca)
    mt = {i for i, _, _ in pairs}
    mc = {j for _, j, _ in pairs}
    un_t = [i for i in range(len(ta)) if i not in mt]
    un_c = [j for j in range(len(ca)) if j not in mc]
    diffs: list[Diff] = []
    used_t: set[int] = set()
    used_c: set[int] = set()
    siou = float(P[_R + "split_iou"])

    # ---- split / merge: two parts on one side tile one element on the other
    def _parts_tile(whole: El, p1: El, p2: El) -> bool:
        return _compatible(whole, p1) and _compatible(whole, p2) and p1.box.union(p2.box).iou(whole.box) >= siou \
            and p1.box.intersect(p2.box).area <= 0.1 * min(p1.box.area, p2.box.area) + 1e-6

    def _gap(p1: El, p2: El) -> float:
        gx = max(p1.box.x, p2.box.x) - min(p1.box.x2, p2.box.x2)
        gy = max(p1.box.y, p2.box.y) - min(p1.box.y2, p2.box.y2)
        return float(max(0.0, max(gx, gy)))

    pair_of_t = {i: j for i, j, _ in pairs}
    pair_of_c = {j: i for i, j, _ in pairs}
    for i in range(len(ta)):  # split: target whole i ~ candidate (matched j or unmatched) + another unmatched part
        cands = ([pair_of_t[i]] if i in pair_of_t else []) + un_c
        found = None
        for x in cands:
            if ca[x].box.intersect(ta[i].box).area <= 0:
                continue
            for y in un_c:
                if x == y or x in used_c or y in used_c or ca[y].box.intersect(ta[i].box).area <= 0:
                    continue
                if _parts_tile(ta[i], ca[x], ca[y]):
                    if ta[i].cls == "text" and _cer(ta[i].text.replace(" ", ""), (ca[x].text + ca[y].text).replace(" ", "")) > 0.34 \
                            and _cer(ta[i].text.replace(" ", ""), (ca[y].text + ca[x].text).replace(" ", "")) > 0.34:
                        continue
                    found = (x, y)
                    break
            if found:
                break
        if found:
            x, y = found
            used_t.add(i)
            used_c.update(found)
            diffs.append(Diff("structure.split", ta[i].box.union(ca[x].box).union(ca[y].box), _gap(ca[x], ca[y]), ta[i], ca[x],
                              {"parts": [ca[x].id, ca[y].id]}))
    for j in range(len(ca)):  # merge: candidate whole j ~ two target parts
        if j in used_c:
            continue
        cands = ([pair_of_c[j]] if j in pair_of_c else []) + un_t
        found = None
        for x in cands:
            if ta[x].box.intersect(ca[j].box).area <= 0:
                continue
            for y in un_t:
                if x == y or x in used_t or y in used_t or ta[y].box.intersect(ca[j].box).area <= 0:
                    continue
                if _parts_tile(ca[j], ta[x], ta[y]):
                    found = (x, y)
                    break
            if found:
                break
        if found:
            x, y = found
            used_c.add(j)
            used_t.update(found)
            diffs.append(Diff("structure.merge", ca[j].box.union(ta[x].box).union(ta[y].box), _gap(ta[x], ta[y]), ta[x], ca[j],
                              {"parts": [ta[x].id, ta[y].id]}))

    # ---- matched pairs
    mv = float(P[_R + "move_min_px"])
    stol = float(P[_R + "size_tol_px"])
    cmin = float(P[_R + "color_min_de"])
    moves: list[tuple[El, El, float, float]] = []
    recolors: list[Diff] = []
    color_pairs: list[tuple[Color, Color]] = []
    ra, rb = _solid(a.root), _solid(b.root)
    if ra is not None and rb is not None:
        color_pairs.append((ra, rb))
    for i, j, _ in pairs:
        if i in used_t or j in used_c:
            continue
        t, c = ta[i], ca[j]
        tb, cb = t.box, c.box
        dl, dtp, dr, db = cb.x - tb.x, cb.y - tb.y, cb.x2 - tb.x2, cb.y2 - tb.y2
        dw, dh = cb.w - tb.w, cb.h - tb.h
        if t.fill is not None and c.fill is not None:
            color_pairs.append((t.fill, c.fill))
        if t.cls == "text":
            content = t.text != c.text
            dsize = abs(c.size - t.size)
            dweight = abs(c.weight - t.weight)
            if content:
                diffs.append(Diff("text.content", tb.union(cb).expand(1.0), _cer(t.text, c.text), t, c, {"from": t.text, "to": c.text}))
            elif dsize >= float(P[_R + "text_size_min"]) or dweight >= float(P[_R + "text_weight_min"]):
                diffs.append(Diff("text.style", tb.union(cb).expand(1.0), dsize + dweight / 100.0, t, c,
                                  {"size": [t.size, c.size], "weight": [t.weight, c.weight]}))
            elif abs(dw) > max(float(P[_R + "text_width_min"]), float(P[_R + "text_width_rel"]) * tb.w) and abs(dh) <= stol + 1:
                diffs.append(Diff("text.style", tb.union(cb).expand(1.0), abs(dw) / max(1.0, t.size) * 2, t, c, {"width": [tb.w, cb.w]}))
            elif max(abs(dl), abs(dtp)) >= mv and abs(dw) <= stol + 1 and abs(dh) <= stol + 1:
                moves.append((t, c, round(dl), round(dtp)))
        else:
            if abs(dw) <= stol and abs(dh) <= stol:
                if max(abs(dl), abs(dtp)) >= mv:
                    moves.append((t, c, round(dl), round(dtp)))
            else:
                diffs.append(Diff("geometry.size", tb.union(cb), float(round(max(abs(dl), abs(dtp), abs(dr), abs(db)))), t, c,
                                  {"from": tb.to_dict(), "to": cb.to_dict()}))
            if t.cls == "icon" and (t.icon or "") != (c.icon or "") and t.icon and c.icon:
                diffs.append(Diff("icon.glyph", tb.union(cb), None, t, c, {"from": t.icon, "to": c.icon}))
            if t.cls in ("shape", "image") and abs(c.radius - t.radius) >= float(P[_R + "radius_min_px"]):
                diffs.append(Diff("geometry.radius", tb.union(cb), abs(c.radius - t.radius), t, c, {"radius": [t.radius, c.radius]}))
            if abs(c.shadow - t.shadow) >= float(P[_R + "shadow_min_px"]):
                reach = max(t.shadow, c.shadow) + 2
                diffs.append(Diff("effect.shadow", tb.union(cb).expand(reach), abs(c.shadow - t.shadow), t, c, {"shadow": [t.shadow, c.shadow]}))
            if t.stroke is not None and c.stroke is not None and t.stroke.delta_e(c.stroke) >= cmin:
                recolors.append(Diff("color.stroke", tb.union(cb), t.stroke.delta_e(c.stroke), t, c, {"from": t.stroke.hex(), "to": c.stroke.hex()}))
                color_pairs.append((t.stroke, c.stroke))
        if t.fill is not None and c.fill is not None and t.fill.delta_e(c.fill) >= cmin:
            de = t.fill.delta_e(c.fill)
            box = tb.union(cb).expand(1.0) if t.cls == "text" else tb.union(cb)
            ab = _blend_alpha(t.fill, c.fill, t.backdrops)
            if ab is not None and 0.05 <= ab[0] <= 0.95 and ab[1] <= float(P[_R + "opacity_res"]):
                recolors.append(Diff("effect.opacity", box, 1.0 - ab[0], t, c, {"alpha": ab[0], "res": ab[1], "de": de}))
            else:
                recolors.append(Diff("color.fill", box, de, t, c, {"from": t.fill.hex(), "to": c.fill.hex()}))

    # ---- global tint: one Lab offset explains most colour pairs
    tint = _global_tint(color_pairs)
    if tint is not None:
        off, mag = tint
        diffs.append(Diff("color.tint_global", Box(0, 0, a.width, a.height), mag, None, None, {"offset_lab": [round(float(v), 2) for v in off]}))
        keep = []
        for d in recolors:
            tc = d.t.fill if d.type != "color.stroke" else d.t.stroke
            cc = d.c.fill if d.type != "color.stroke" else d.c.stroke
            o = np.array(cc.lab()) - np.array(tc.lab())
            if np.linalg.norm(o - off) > max(3.0, 0.5 * np.linalg.norm(off)):
                keep.append(d)
        recolors = keep
    # ---- opacity: elements faded by one alpha, adjacent -> one finding
    opa = [d for d in recolors if d.type == "effect.opacity"]
    recolors = [d for d in recolors if d.type != "effect.opacity"]
    pad = float(P[_R + "group_pad"])
    atol = float(P[_R + "opacity_alpha_tol"])
    for g in _union_find(len(opa), lambda i, j: abs(opa[i].ev["alpha"] - opa[j].ev["alpha"]) <= atol
                         and opa[i].box.expand(pad).intersect(opa[j].box).area > 0):
        ds = [opa[k] for k in g]
        diffs.append(Diff("effect.opacity", _ubox(d.box for d in ds), float(np.median([d.magnitude for d in ds])), ds[0].t, ds[0].c,
                          {"n": len(ds), "alpha": float(np.median([d.ev["alpha"] for d in ds]))}))
    diffs.extend(recolors)

    # ---- moves: group by common offset (a subtree moves as one); spacing = offsets k·d along one axis
    groups = _union_find(len(moves), lambda i, j: abs(moves[i][2] - moves[j][2]) <= 1 and abs(moves[i][3] - moves[j][3]) <= 1
                         and moves[i][0].box.expand(pad).intersect(moves[j][0].box).area > 0)
    clusters = []
    for g in groups:
        ms = [moves[k] for k in g]
        dx = float(np.median([m[2] for m in ms]))
        dy = float(np.median([m[3] for m in ms]))
        box = _ubox([m[0].box for m in ms] + [m[1].box for m in ms])
        clusters.append({"ms": ms, "dx": dx, "dy": dy, "box": box, "text_only": all(m[0].cls == "text" for m in ms)})
    used_cl: set[int] = set()
    for axis in ("x", "y"):
        ax = [k for k, cl in enumerate(clusters) if (cl["dy"] == 0 if axis == "x" else cl["dx"] == 0) and (cl["dx"] if axis == "x" else cl["dy"]) != 0]
        for sign in (1, -1):
            mem = [k for k in ax if np.sign(clusters[k]["dx" if axis == "x" else "dy"]) == sign and k not in used_cl]
            if len(mem) < 2:
                continue
            vals = {k: abs(clusters[k]["dx" if axis == "x" else "dy"]) for k in mem}
            step = min(vals.values())
            mult = {k: v / step for k, v in vals.items() if abs(v / step - round(v / step)) <= 0.2}
            if len({round(v) for v in mult.values()}) < 2:
                continue
            ks = list(mult)
            box = _ubox(clusters[k]["box"] for k in ks)
            used_cl.update(ks)
            diffs.append(Diff("layout.spacing", box, float(step), clusters[ks[0]]["ms"][0][0], clusters[ks[0]]["ms"][0][1],
                              {"axis": axis, "step": step * sign, "n": len(ks)}))
    for k, cl in enumerate(clusters):
        if k in used_cl:
            continue
        t, c = cl["ms"][0][0], cl["ms"][0][1]
        diffs.append(Diff("text.position" if cl["text_only"] else "geometry.shift", cl["box"].expand(1.0 if cl["text_only"] else 0.0),
                          math.hypot(cl["dx"], cl["dy"]), t, c, {"dx": cl["dx"], "dy": cl["dy"], "n": len(cl["ms"])}))

    # ---- re-read: an unmatched target and an unmatched candidate element in the same place (the class or
    # the reading changed, e.g. a glyph read as text on one side and as an icon on the other)
    rr = float(P[_R + "reread_iou"])
    for i in [i for i in un_t if i not in used_t]:
        best, bj = rr, None
        for j in [j for j in un_c if j not in used_c]:
            iou = ta[i].box.iou(ca[j].box)
            if iou >= best:
                best, bj = iou, j
        if bj is None:
            continue
        t, c = ta[i], ca[bj]
        used_t.add(i)
        used_c.add(bj)
        if "icon" in (t.cls, c.cls):
            typ, mag = "icon.glyph", None
        elif t.cls == c.cls == "text":
            typ, mag = "text.content", _cer(t.text, c.text)
        else:
            typ, mag = "geometry.size", float(round(_edge_disp(t.box, c.box)))
        diffs.append(Diff(typ, t.box.union(c.box), mag, t, c, {"reread": [t.cls, c.cls], "iou": round(best, 3)}))

    # ---- unmatched: missing (target only) / extra (candidate only), grouped by overlap
    for side, idx, els in (("t", [i for i in un_t if i not in used_t], ta), ("c", [j for j in un_c if j not in used_c], ca)):
        es = [els[k] for k in idx]
        for g in _union_find(len(es), lambda i, j: es[i].box.expand(2.0).intersect(es[j].box).area > 0):
            ms = [es[k] for k in g]
            box = _ubox(m.box.expand(m.shadow) for m in ms)
            if side == "t":
                typ = "icon.missing" if all(m.cls == "icon" for m in ms) else "structure.missing"
                diffs.append(Diff(typ, box, box.area, ms[0], None, {"n": len(ms), "classes": sorted({m.cls for m in ms})}))
            else:
                diffs.append(Diff("structure.extra", box, box.area, None, ms[0], {"n": len(ms), "classes": sorted({m.cls for m in ms})}))
    return diffs


def _global_tint(pairs: list[tuple[Color, Color]]) -> Optional[tuple[np.ndarray, float]]:
    if len(pairs) < int(P[_R + "tint_min_n"]):
        return None
    offs = np.array([np.array(c.lab()) - np.array(t.lab()) for t, c in pairs])
    mags = np.linalg.norm(offs, axis=1)
    tinted = mags >= float(P[_R + "tint_min_de"])
    if tinted.mean() < float(P[_R + "tint_frac"]):
        return None
    sel = offs[tinted]
    mean = np.median(sel, axis=0)
    nm = float(np.linalg.norm(mean))
    if nm < float(P[_R + "tint_min_de"]):
        return None
    close = np.linalg.norm(sel - mean, axis=1) <= float(P[_R + "tint_spread"]) * nm + 1.0
    if close.sum() < float(P[_R + "tint_frac"]) * len(offs):
        return None
    return np.mean(sel[close], axis=0), float(np.mean(mags[tinted][close]))


# --------------------------------------------------------------------------- pixel support
@dataclass
class PixelEvidence:
    d: np.ndarray            # blurred ΔE map (geometry / structure support)
    d0: np.ndarray           # raw ΔE map (colour support: blur dilutes thin strokes and glyphs)
    floor: float             # noise floor on d  = pix_k x median ΔE over target edge pixels
    floor0: float            # noise floor on d0
    page_median: float       # median raw ΔE over the page (tint support)
    edge: Optional[np.ndarray] = None   # edge pixels of either image
    ref: float = 0.0         # median ΔE over edge pixels page-wide (d)
    ref0: float = 0.0        # same on d0
    changed_edges: float = 0.0  # fraction of edge pixels with raw ΔE > JND (global disturbance level)

    def contrast(self, box: Box, raw: bool = False, pad: float = 1.0) -> float:
        """Local-vs-page contrast: mean ΔE over the edge pixels in ``box`` / (page-wide median edge ΔE + 0.5).
        Nuisance (blur, JPEG, sub-pixel, another browser) raises ΔE on every edge of the page alike; a real
        local change raises it here more than elsewhere."""
        m = self.d0 if raw else self.d
        H, W = m.shape
        x0, y0, x1, y1 = box.expand(pad).as_int()
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        sub = m[y0:y1, x0:x1]
        e = self.edge[y0:y1, x0:x1] if self.edge is not None else None
        local = float(sub[e].mean()) if e is not None and e.any() else float(sub.mean())
        return local / ((self.ref0 if raw else self.ref) + 0.5)

    def count(self, box: Box, thr: float, raw: bool = False, pad: float = 1.0) -> int:
        m = self.d0 if raw else self.d
        H, W = m.shape
        x0, y0, x1, y1 = box.expand(pad).as_int()
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            return 0
        return int((m[y0:y1, x0:x1] > thr).sum())

    def support(self, typ: str, box: Box, magnitude: Optional[float], ev: dict) -> tuple[int, float]:
        """(supporting px, threshold) for a finding: colour readings are checked on the raw map against a
        threshold proportional to the ΔE they claim; everything else on the blurred map at ``pix_de``."""
        if typ in ("color.fill", "color.stroke", "effect.opacity"):
            de = float(ev.get("de", magnitude or 0.0)) if typ == "effect.opacity" else float(magnitude or 0.0)
            thr = max(self.floor0, min(float(P[_R + "pix_de"]), float(P[_R + "pix_color_frac"]) * de))
            return self.count(box, thr, raw=True), thr
        thr = max(self.floor, float(P[_R + "pix_de"]))
        return self.count(box, thr), thr


def _noise_floor(d: np.ndarray, edge: np.ndarray) -> float:
    """``pix_k`` x the ``pix_q`` quantile of ΔE over the page's edge pixels: nuisance raises ΔE on most edges,
    a real change only on the few edges it touches, so a high quantile tracks the noise and not the change."""
    return float(P[_R + "pix_k"]) * (float(np.quantile(d[edge], float(P[_R + "pix_q"]))) if edge.any() else 0.0)


_PIX: dict[tuple, PixelEvidence] = {}


def pixel_evidence(target: np.ndarray, cand: np.ndarray) -> PixelEvidence:
    """Memoised :func:`_pixel_evidence` (tuning re-scores the same pairs many times)."""
    k = (image_key(target), image_key(cand), float(P[_R + "pix_sigma"]), float(P[_R + "edge_grad"]), float(P[_R + "pix_k"]), float(P[_R + "pix_q"]),
         float(P[_R + "jnd_de"]))
    if k not in _PIX:
        if len(_PIX) > 256:
            _PIX.clear()
        _PIX[k] = _pixel_evidence(target, cand)
    return _PIX[k]


def _pixel_evidence(target: np.ndarray, cand: np.ndarray) -> PixelEvidence:
    from dt.common.image import same_size, to_gray
    from dt.compare.pixel import diff_map
    t, c = same_size(target, cand)
    s = float(P[_R + "pix_sigma"])
    d0 = diff_map(t, c)
    d = diff_map(cv2.GaussianBlur(t, (0, 0), s), cv2.GaussianBlur(c, (0, 0), s)) if s > 0 else d0
    edge = np.zeros(d.shape, bool)
    for im in (t, c):
        g = to_gray(im).astype(np.float32)
        edge |= np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) > float(P[_R + "edge_grad"])
    ref = float(np.median(d[edge])) if edge.any() else 0.0
    ref0 = float(np.median(d0[edge])) if edge.any() else 0.0
    ch = float((d0[edge] > float(P[_R + "jnd_de"])).mean()) if edge.any() else 0.0
    return PixelEvidence(d, d0, _noise_floor(d, edge), _noise_floor(d0, edge), float(np.median(d0)), edge, ref, ref0, ch)


# --------------------------------------------------------------------------- nuisance matching
def _edge_level(t: np.ndarray, c: np.ndarray, edge: np.ndarray) -> float:
    """Median ΔE over edge pixels: robust to the few edges a real perturbation touches."""
    from dt.compare.pixel import diff_map
    d = diff_map(t, c)
    return float(np.median(d[edge])) if edge.any() else float(np.median(d))


def _subpixel(c: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(c, M, (c.shape[1], c.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def match_nuisance(target: np.ndarray, cand: np.ndarray) -> tuple[np.ndarray, dict]:
    """Degrade the candidate the way the target appears degraded (sub-pixel offset, resampling blur, JPEG),
    so both sides reach the perceiver with the same nuisance and only real differences remain.

    Each degradation family is searched in the order the nuisance is produced (offset, blur, compression)
    and kept only when it lowers the page-wide median edge ΔE by ``nuis_gain`` (relative). The median is
    robust to the minority of edges a real perturbation touches. Returns (candidate', applied recipe)."""
    from dt.common.image import same_size, to_gray
    from dt.adversary.perturb import jpeg
    t, c = same_size(target, cand)
    if not bool(P[_R + "nuis_enabled"]) or t.shape != c.shape:
        return cand, {}
    g = to_gray(t).astype(np.float32)
    edge = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) > float(P[_R + "edge_grad"])
    base = _edge_level(t, c, edge)
    if base < float(P[_R + "nuis_min_de"]):
        return cand, {}
    gain = float(P[_R + "nuis_gain"])
    recipe: dict = {}
    cur, cur_l = c, base
    gc = to_gray(c).astype(np.float32)
    win = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
    (dx, dy), _resp = cv2.phaseCorrelate(gc, g, win)
    if 0.1 <= max(abs(dx), abs(dy)) <= 0.9:
        trial = _subpixel(cur, dx, dy)
        lv = _edge_level(t, trial, edge)
        if lv < cur_l * (1 - gain):
            cur, cur_l, recipe["subpixel"] = trial, lv, [round(dx, 3), round(dy, 3)]
    best = None
    for sg in (0.3, 0.4, 0.5, 0.6, 0.7, 0.85):
        trial = cv2.GaussianBlur(cur, (0, 0), sg)
        lv = _edge_level(t, trial, edge)
        if best is None or lv < best[0]:
            best = (lv, sg, trial)
    if best is not None and best[0] < cur_l * (1 - gain):
        cur_l, recipe["blur"], cur = best[0], best[1], best[2]
    best = None
    for q in (65, 70, 75, 80, 85, 90, 95):
        trial = jpeg(cur, q)
        lv = _edge_level(t, trial, edge)
        if best is None or lv < best[0]:
            best = (lv, q, trial)
    if best is not None and best[0] < cur_l * (1 - gain):
        cur_l, recipe["jpeg"], cur = best[0], best[1], best[2]
    if recipe:
        recipe.update(level_before=round(base, 3), level_after=round(cur_l, 3))
        return np.ascontiguousarray(cur, dtype=np.uint8), recipe
    return cand, {}


_NUIS: dict[tuple, tuple[np.ndarray, dict]] = {}


def matched_candidate(target: np.ndarray, cand: np.ndarray) -> tuple[np.ndarray, dict]:
    k = (image_key(target), image_key(cand), bool(P[_R + "nuis_enabled"]), float(P[_R + "nuis_gain"]), float(P[_R + "nuis_min_de"]))
    if k not in _NUIS:
        if len(_NUIS) > 256:
            _NUIS.clear()
        _NUIS[k] = match_nuisance(target, cand)
    return _NUIS[k]


# --------------------------------------------------------------------------- fixed point
def fixed_point(rgb: np.ndarray, doc: Optional[Document] = None) -> dict:
    """The perceptual fixed-point test: perceive(render(perceive(x))) vs perceive(x).

    Returns ``{"findings": [Finding], "elements": n, "unstable": n_unstable, "stable_frac": f}``.
    Each finding is a difference the perceiver introduces by itself (its reading of x does not
    reproduce from its own render): it localises perception instability, not a translation error.
    """
    a = doc if doc is not None else perceive_cached(rgb)
    b = reperceive_cached(rgb) if doc is None else _reperceive_doc(a)
    diffs = compare_docs(a, b)
    els = elements(a)
    unstable = {id(d.t.node) for d in diffs if d.t is not None}
    fs = [Finding(d.type, d.box, d.magnitude, 1.0, {"fixed_point": True, **_jsonable(d.ev)}, d.t.id if d.t else None) for d in diffs]
    return {"findings": fs, "elements": len(els), "unstable": len(unstable),
            "stable_frac": 1.0 - len(unstable) / len(els) if els else 1.0}


def _reperceive_doc(doc: Document) -> Document:
    from dt.perceive import perceive
    from dt.render.screenshot import render_doc
    d = copy.deepcopy(doc)
    d.source_image = None
    return perceive(render_doc(d))


def _jsonable(ev: dict) -> dict:
    return json.loads(json.dumps(ev, default=lambda o: o.to_dict() if hasattr(o, "to_dict") else str(o)))


# --------------------------------------------------------------------------- the adversary
def _attribute(box: Box, ir: Optional[Document], prefer_text: bool = False) -> Optional[str]:
    """Candidate IR node best covering ``box`` (largest IoU among painted nodes)."""
    if ir is None:
        return None
    best, bid = 0.0, None
    for n in ir.walk():
        if n is ir.root or not n.visible:
            continue
        iou = n.box.iou(box)
        if prefer_text and n.type == "text":
            iou += 0.05
        if iou > best:
            best, bid = iou, n.id
    return bid if best >= 0.3 else None


def _residual_type(target: np.ndarray, cand: np.ndarray, box: Box) -> str:
    """Pixel-typed guess for a residual region the IR differ did not explain: where is the ink?"""
    x0, y0, x1, y1 = box.as_int()
    t = target[y0:y1, x0:x1].astype(np.float32)
    c = cand[y0:y1, x0:x1].astype(np.float32)
    if t.size == 0:
        return "structure.missing"
    ref = np.median(np.concatenate([t.reshape(-1, 3), c.reshape(-1, 3)]), axis=0)
    it = float((np.abs(t - ref).sum(axis=2) > 40).mean())
    ic = float((np.abs(c - ref).sum(axis=2) > 40).mean())
    if it > 2 * ic + 0.01:
        return "structure.missing"
    if ic > 2 * it + 0.01:
        return "structure.extra"
    return "color.fill"


def find(target_rgb: np.ndarray, candidate_rgb: np.ndarray, candidate_ir: Optional[Document] = None) -> list[Finding]:
    """The ir-space adversary: perceive both images, diff the readings, keep pixel-supported findings."""
    pt = perceive_cached(target_rgb)
    cand, recipe = matched_candidate(target_rgb, candidate_rgb)
    pc = perceive_cached(cand)
    diffs = _retype_width_changes(compare_docs(pt, pc), target_rgb, cand)
    ev = pixel_evidence(target_rgb, cand)
    unstable: list[Box] = []
    if bool(P[_R + "fp_enabled"]):
        try:
            fp = compare_docs(pt, reperceive_cached(target_rgb))
            unstable = [d.box for d in fp]
        except Exception:  # renderer unavailable: no down-weighting
            unstable = []
    out = _finalise(diffs, ev, unstable, target_rgb, cand, candidate_ir)
    for f in out:
        if recipe:
            f.evidence["nuisance_matched"] = recipe
    return out


def _finalise(diffs: list[Diff], ev: PixelEvidence, unstable: list[Box], target_rgb: np.ndarray, candidate_rgb: np.ndarray,
              candidate_ir: Optional[Document]) -> list[Finding]:
    min_px = int(P[_R + "pix_min_px"])
    scale = float(P[_R + "pix_conf_scale"])
    fps = float(P[_R + "fp_conf_scale"])
    fpd = int(P[_R + "fp_drop_px"])
    out: list[Finding] = []
    tinted = any(d.type == "color.tint_global" and ev.page_median >= float(P[_R + "tint_pix_de"]) for d in diffs)
    noisy = (not tinted) and ev.changed_edges >= float(P[_R + "noisy_frac"])
    for d in diffs:
        if noisy and _below_noise(d):
            continue
        if d.type == "color.tint_global":
            if ev.page_median < float(P[_R + "tint_pix_de"]):
                continue
            n = int(ev.d.size)
            conf = TYPE_PRIOR[d.type]
        else:
            n, _thr = ev.support(d.type, d.box, d.magnitude, d.ev)
            if n < min_px:
                continue
            ratio = ev.contrast(d.box, raw=d.type in ("color.fill", "color.stroke", "effect.opacity"))
            if ratio < float(P[_R + "pix_ratio"]):
                continue
            conf = TYPE_PRIOR.get(d.type, 0.7) * (1.0 - math.exp(-n / scale))
            hit = any(u.intersect(d.box).area > 0.25 * min(u.area, d.box.area) for u in unstable)
            if hit:
                if fpd and n < fpd:
                    continue
                conf *= fps
        if conf < float(P[_R + "min_conf"]):
            continue
        node = (d.c.id if d.c is not None else None)
        nid = _attribute(d.c.box if d.c is not None else d.box, candidate_ir, d.type.startswith("text.")) if candidate_ir is not None and node else None
        evd = {"support_px": n, **({"contrast": round(ratio, 2)} if d.type != "color.tint_global" else {}), **_jsonable(d.ev)}
        if d.t is not None:
            evd["target_el"] = {"cls": d.t.cls, "box": d.t.box.to_dict()}
        if d.c is not None:
            evd["candidate_el"] = {"cls": d.c.cls, "box": d.c.box.to_dict()}
        out.append(Finding(d.type, d.box, None if d.magnitude is None else float(d.magnitude), float(conf), evd, nid))
    if bool(P[_R + "residual_fallback"]):
        from dt.compare.pixel import residual_regions
        explained = [f.box.expand(2) for f in out if f.type != "color.tint_global"]
        if not any(f.type == "color.tint_global" for f in out):
            thr = max(ev.floor, float(P[_R + "pix_de"]))
            for b in residual_regions(ev.d, thr=thr):
                if any(b.intersect(e).area > 0.3 * b.area for e in explained):
                    continue
                n = ev.count(b, thr)
                if n < int(P[_R + "residual_min_px"]):
                    continue
                out.append(Finding(_residual_type(target_rgb, candidate_rgb, b), b, None, float(P[_R + "residual_conf"]),
                                   {"support_px": n, "residual": True}, _attribute(b, candidate_ir)))
    out.sort(key=lambda f: -f.confidence)
    return out


def _with(overrides: dict, fn):
    """Run ``fn()`` with temporary param overrides (restored exactly afterwards)."""
    from dt.params import _OVERRIDES
    saved = {k: (_OVERRIDES[k] if k in _OVERRIDES else _MISSING) for k in overrides}
    for k, v in overrides.items():
        P.set(k, v)
    try:
        return fn()
    finally:
        for k, v in saved.items():
            if v is _MISSING:
                _OVERRIDES.pop(k, None)
            else:
                _OVERRIDES[k] = v


_MISSING = object()


def make_finder(name: str, **overrides):
    """An adversary function = ``find`` under param overrides (keys without the ``adversary.irspace.`` prefix)."""
    ov = {(k if k.startswith("adversary.") else _R + k): v for k, v in overrides.items()}

    def fn(target_rgb, candidate_rgb, candidate_ir=None):
        return _with(ov, lambda: find(target_rgb, candidate_rgb, candidate_ir))
    fn.__name__ = name
    return fn


_GEOM_PX = ("geometry.shift", "geometry.size", "geometry.radius", "text.position", "layout.spacing", "structure.split",
            "structure.merge", "effect.shadow")


def _below_noise(d: Diff) -> bool:
    """On a globally disturbed target (unmatched nuisance: font smoothing, another rasteriser, a sub-pixel
    layout offset) differences no larger than the nuisance itself produces are not trusted."""
    m = d.magnitude
    if m is None:
        return False
    if d.type in _GEOM_PX:
        return m <= float(P[_R + "noisy_geom_px"])
    if d.type == "text.style":
        return m <= float(P[_R + "noisy_text"])
    if d.type in ("color.fill", "color.stroke"):
        return m <= float(P[_R + "noisy_de"])
    if d.type == "effect.opacity":
        return m <= float(P[_R + "noisy_alpha"])
    if d.type == "text.content":
        return m <= float(P[_R + "noisy_cer"])
    return False


def _max_gap(rgb: np.ndarray, box: Box) -> float:
    """Widest run of ink-free columns strictly inside the ink extent of a text line (px)."""
    x0, y0, x1, y1 = box.as_int()
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return 0.0
    sub = rgb[y0:y1, x0:x1].astype(np.int16)
    border = np.concatenate([sub[0], sub[-1], sub[:, 0], sub[:, -1]])
    bg = np.median(border, axis=0)
    ink = (np.abs(sub - bg).sum(axis=2) > int(P[_R + "ink_thr"])).any(axis=0)
    xs = np.nonzero(ink)[0]
    if len(xs) < 2:
        return 0.0
    best = run = 0
    for v in ink[xs[0]:xs[-1] + 1]:
        run = 0 if v else run + 1
        best = max(best, run)
    return float(best)


def _retype_width_changes(diffs: list[Diff], target: np.ndarray, cand: np.ndarray) -> list[Diff]:
    """Same text, restyled reading: if the widest internal ink gap of the line grew (shrank) by more than
    ``gap_min_px`` beyond what the size change explains, a seam opened (closed) inside the line: a split
    (merge), not a style change. The perceiver's font fit tends to explain an opened seam with a bolder
    weight, so the gap test runs before trusting the style reading."""
    out = []
    for d in diffs:
        if d.type == "text.style" and d.t is not None and d.c is not None:
            gt_, gc = _max_gap(target, d.t.box.expand(1)), _max_gap(cand, d.c.box.expand(1))
            exp = gt_ * (d.c.size / d.t.size if d.t.size > 0 and d.c.size > 0 else 1.0)
            m = float(P[_R + "gap_min_px"])
            if gc - exp >= m:
                d = Diff("structure.split", d.box, gc - gt_, d.t, d.c, {**d.ev, "gap": [gt_, gc]})
            elif exp - gc >= m:
                d = Diff("structure.merge", d.box, gt_ - gc, d.t, d.c, {**d.ev, "gap": [gt_, gc]})
        out.append(d)
    return out


def find_hybrid(target_rgb: np.ndarray, candidate_rgb: np.ndarray, candidate_ir: Optional[Document] = None) -> list[Finding]:
    """``find`` with the residual fallback on (IR findings + low-confidence pixel residuals)."""
    return _with({_R + "residual_fallback": True}, lambda: find(target_rgb, candidate_rgb, candidate_ir))


# --------------------------------------------------------------------------- harness: tune / score / real pages / report
OUT = os.path.join(ROOT, "out", "adversary")
REAL_DIR = os.path.join(ROOT, "out", "real_r3m")

# coordinate-descent search space (CALIBRATION only)
TUNE_GRID: dict[str, list] = {
    "pix_ratio": [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0],
    "pix_min_px": [3, 6, 12, 24, 48],
    "pix_de": [4.0, 6.0, 8.0, 12.0, 16.0],
    "pix_k": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    "pix_q": [0.5, 0.75, 0.9, 0.95, 0.98],
    "pix_sigma": [0.0, 0.5, 0.7, 1.0],
    "color_min_de": [2.5, 3.0, 4.0, 6.0, 8.0],
    "pix_color_frac": [0.3, 0.5, 0.7],
    "match_max_cost": [1.4, 1.8, 2.2, 2.8],
    "match_max_disp": [24.0, 40.0, 56.0],
    "size_tol_px": [0.5, 1.0, 2.0],
    "move_min_px": [0.5, 1.0, 2.0],
    "radius_min_px": [1.0, 2.0, 3.0, 5.0],
    "text_width_min": [2.0, 3.0, 6.0, 10.0, 1000.0],
    "text_weight_min": [100, 200, 300, 1000],
    "text_width_rel": [0.0, 0.02, 0.04, 0.06],
    "gap_min_px": [1.5, 2.0, 3.0, 1000.0],
    "text_size_min": [0.5, 1.0, 1.5, 2.0],
    "shadow_min_px": [1.0, 1.5, 3.0, 6.0],
    "reread_iou": [0.3, 0.5, 0.7, 0.95],
    "noisy_frac": [0.1, 0.15, 0.2, 1.01],
    "noisy_geom_px": [1.0, 1.5, 2.5, 4.0],
    "noisy_text": [1.0, 1.5, 2.5, 4.0],
    "noisy_de": [5.0, 8.0, 12.0, 20.0],
    "noisy_alpha": [0.1, 0.15, 0.3],
    "noisy_cer": [0.1, 0.2, 0.4],
    "fp_conf_scale": [0.25, 0.5, 1.0],
    "fp_drop_px": [0, 12, 48],
    "tint_pix_de": [0.5, 1.0, 2.0],
    "min_conf": [0.0, 0.05, 0.1, 0.2],
}


def bench_images(bench) -> list[tuple]:
    items: list[tuple] = []
    for c in bench.cases:
        items.append((c.paths["target"], True))
        items.append((c.paths["candidate"], False))
        if c.noise:
            items.append((c.paths["candidate"], False, c.paths["target"]))
    return items


def real_pages() -> list[tuple[str, str, str, str]]:
    """``[(name, target.png, render.png, ir.mapped.json)]`` for the translated real pages."""
    out = []
    if not os.path.isdir(REAL_DIR):
        return out
    for name in sorted(os.listdir(REAL_DIR)):
        d = os.path.join(REAL_DIR, name)
        t, r, ir = os.path.join(d, "validation", "target.png"), os.path.join(d, "render.png"), os.path.join(d, "ir.mapped.json")
        if all(os.path.exists(p) for p in (t, r, ir)):
            out.append((name, t, r, ir))
    return out


def tune(bench, rounds: int = 2, progress=None) -> tuple[dict, float, list]:
    """Coordinate descent over ``TUNE_GRID`` maximising the CALIBRATION headline. Returns (best overrides,
    best headline, history)."""
    from dt.adversary import benchmark as B
    cases = bench.split("calibration")
    best: dict = {}
    hist = []

    def obj(ov: dict) -> float:
        return B.score(make_finder("t", **ov), cases, split="calibration")["headline"]
    cur = obj(best)
    hist.append(({}, cur))
    if progress:
        progress(f"tune start headline {cur:.4f}")
    for r in range(rounds):
        improved = False
        for k, vals in TUNE_GRID.items():
            for v in vals:
                if best.get(k, P[_R + k]) == v:
                    continue
                ov = {**best, k: v}
                h = obj(ov)
                hist.append((ov, h))
                if h > cur + 1e-4:
                    cur, best, improved = h, ov, True
                    if progress:
                        progress(f"round {r} {k}={v} -> {cur:.4f}")
        if not improved:
            break
    return best, cur, hist


def run_real(fn, out_dir: str) -> list[dict]:
    """Run an adversary on the real translated pages; returns per-page summaries and writes evidence panels."""
    from dt.common.image import load_rgb, save_rgb
    from dt.compare.pixel import diff_map
    from dt.compare.visualize import draw_boxes, heatmap
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name, tp, rp, irp in real_pages():
        t, r = load_rgb(tp), load_rgb(rp)
        if t.shape != r.shape:  # the validator pads; here we compare the common frame
            h, w = min(t.shape[0], r.shape[0]), min(t.shape[1], r.shape[1])
            t, r = t[:h, :w], r[:h, :w]
        ir = Document.load(irp)
        t0 = time.perf_counter()
        fs = fn(t, r, ir)
        secs = time.perf_counter() - t0
        fpi = fixed_point(t)
        loud = _with({_R + "noisy_frac": 1.01}, lambda: find(t, r, ir))  # the default reading without the disturbed-page guard
        evp = pixel_evidence(t, matched_candidate(t, r)[0])
        counts: dict[str, int] = {}
        for f in fs:
            counts[f.type] = counts.get(f.type, 0) + 1
        hm = heatmap(diff_map(t, r))
        panels = [draw_boxes(img, [f.box for f in fs[:40]], (220, 0, 0), 2) for img in (t, r, hm)]
        sep = np.full((t.shape[0], 6, 3), 255, np.uint8)
        im = np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)
        sc = min(1.0, 2400 / im.shape[1])
        if sc < 1.0:
            im = cv2.resize(im, (int(im.shape[1] * sc), int(im.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        img_path = os.path.join(out_dir, f"real_{name}.png")
        save_rgb(im, img_path)
        rows.append({"page": name, "size": [int(t.shape[1]), int(t.shape[0])], "n": len(fs), "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
                     "seconds": secs, "changed_edges": evp.changed_edges, "n_unguarded": len(loud), "counts_unguarded": _type_counts(loud),
                     "fixed_point": {"elements": fpi["elements"], "unstable": fpi["unstable"], "stable_frac": fpi["stable_frac"],
                                                      "types": _type_counts(fpi["findings"])},
                     "top": [f.to_dict() for f in fs[:12]], "image": os.path.relpath(img_path, ROOT)})
    return rows


def fixed_point_study(bench, split: str, overrides: dict) -> dict:
    """Does perception instability predict false findings? Runs ``find`` without the fixed-point weighting and
    splits its predictions by whether they overlap a box where perceive(render(perceive(target))) disagrees
    with perceive(target). Also the per-target stability rate."""
    from dt.adversary import benchmark as B
    fn = make_finder("irspace_no_fixed_point", **{**overrides, "fp_enabled": False})
    rows = B.run(fn, bench.split(split))
    stab: dict[str, float] = {}
    on = {"n": 0, "tp": 0}
    off = {"n": 0, "tp": 0}
    for r in rows:
        c = r["case"]
        t = c.target_rgb
        k = image_key(t)
        pt = perceive_cached(t)
        fpd = compare_docs(pt, reperceive_cached(t))
        if k not in stab:
            els = elements(pt)
            uns = {id(d.t.node) for d in fpd if d.t is not None}
            stab[k] = 1.0 - len(uns) / len(els) if els else 1.0
        unstable = [d.box for d in fpd]
        hit = {i for i, _ in r["pairs"]}
        for i, p in enumerate(r["preds"]):
            u = any(b.intersect(p.box).area > 0.25 * min(b.area, p.box.area) for b in unstable)
            dd = on if u else off
            dd["n"] += 1
            dd["tp"] += int(i in hit)
    return {"targets": len(stab), "stable_frac_mean": float(np.mean(list(stab.values()))) if stab else 1.0,
            "stable_frac_min": float(np.min(list(stab.values()))) if stab else 1.0,
            "on_unstable": {**on, "precision": on["tp"] / on["n"] if on["n"] else None},
            "on_stable": {**off, "precision": off["tp"] / off["n"] if off["n"] else None}}


def _type_counts(fs: list[Finding]) -> dict:
    out: dict[str, int] = {}
    for f in fs:
        out[f.type] = out.get(f.type, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    from dt.adversary import benchmark as B
    ap = argparse.ArgumentParser(description="ir-space adversary: warm the perception cache, tune on CALIBRATION, score on EVALUATION, run on real pages")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--no-tune", action="store_true")
    a = ap.parse_args(argv)
    bench = B.build(out_dir=os.path.join(OUT, "bench"))
    log = print
    timings = warm(bench_images(bench) + [(t, True) for _, t, _, _ in real_pages()] + [(r, False) for _, _, r, _ in real_pages()],
                   a.workers, log)
    cold = os.path.join(OUT, "irspace_warm_timings.json")  # cold timings of the first warm-up (later runs hit the cache)
    if os.path.exists(cold):
        with open(cold) as f:
            timings = [tuple(t) for t in json.load(f)] + list(timings)
    tuned_path = os.path.join(OUT, "irspace_tuned.json")
    if a.no_tune and os.path.exists(tuned_path):
        with open(tuned_path) as f:
            tj = json.load(f)
        best, cal_h = tj["overrides"], tj["calibration_headline"]
    elif a.no_tune:
        best, cal_h = {}, None
    else:
        best, cal_h, hist = tune(bench, a.rounds, log)
        with open(tuned_path, "w") as f:
            json.dump({"overrides": best, "calibration_headline": cal_h, "evaluations": len(hist)}, f, indent=1)
    variants = [  # the registered defaults ARE the calibration-tuned values; `best` is applied again for --no-tune runs
        ("irspace", make_finder("irspace", **best)),
        ("irspace_no_pixel_support", make_finder("irspace_no_pixel_support", **{**best, "pix_min_px": 0, "pix_de": 0.0, "pix_k": 0.0, "tint_pix_de": 0.0})),
        ("irspace_no_fixed_point", make_finder("irspace_no_fixed_point", **{**best, "fp_enabled": False})),
        ("irspace_no_page_guard", make_finder("irspace_no_page_guard", **{**best, "noisy_frac": 1.01})),
        ("irspace_no_nuisance_matching", make_finder("irspace_no_nuisance_matching", **{**best, "nuis_enabled": False})),
        ("irspace_hybrid", make_finder("irspace_hybrid", **{**best, "residual_fallback": True})),
    ]
    reports = []
    for name, fn in variants:
        for split in ("calibration", "evaluation"):
            reports.append(B.score(fn, bench, split, name=name))
            log(f"{name} {split} {reports[-1]['headline']:.4f}")
    base = [B.score(B.baseline_residual, bench, s) for s in ("calibration", "evaluation")]
    real = run_real(variants[0][1], os.path.join(OUT, "irspace"))
    ex = B.save_examples(bench, variants[0][1], os.path.join(OUT, "irspace", "examples"))
    with open(os.path.join(OUT, "irspace_results.json"), "w") as f:
        json.dump({"tuned": best, "reports": reports, "baseline": base, "real": real, "warm_timings": timings}, f, indent=1, default=str)
    fps = {sp: fixed_point_study(bench, sp, best) for sp in ("calibration", "evaluation")}
    notes_path = os.path.join(OUT, "irspace_notes.md")
    notes = open(notes_path).read() if os.path.exists(notes_path) else ""
    md = write_report(bench, best, cal_h, reports, base, real, timings, ex, fps, notes)
    with open(os.path.join(OUT, "ir-space.md"), "w") as f:
        f.write(md)
    log(md[:3000])
    return 0


def write_report(bench, best, cal_h, reports, base, real, timings, examples, fp_study=None, notes: str = "") -> str:
    """Markdown report: headline table over variants, per-type table + confusion (B.format_report), runtime,
    tuning, fixed-point study, real pages, evidence. ``notes`` (failure analysis etc.) is appended verbatim."""
    from dt.adversary import benchmark as B
    pct = lambda v: f"{100 * v:.1f}"  # noqa: E731
    L = ["# Adversary 'ir-space': comparison in IR space (the perceptual fixed point)", "",
         "Our own perceiver (`dt.perceive.perceive`, icon + font annotations) reads BOTH images; the two readings are "
         "matched element by element (Hungarian over IoU, edge displacement, colour, text and glyph costs) and every "
         "difference is typed from the matched pairs (moved / resized / recoloured / re-rounded / shadow / faded / text / "
         "glyph) and from the unmatched elements (missing / extra / split / merge); a common Lab offset over all pairs is a "
         "global tint, offsets k·d along one axis are a spacing drift. Bias mitigations: nuisance matching on the candidate, "
         "pixel support for every finding, a globally-disturbed-page guard, and fixed-point down-weighting. "
         "Implementation: `dt/adversary/irspace.py`; tests: `tests/test_adversary_irspace.py`.", "",
         "## Scores", "",
         "| adversary | split | headline | det P | det R | det F1 | AP | macro leaf F1 | macro parent F1 | type acc leaf / parent | noise false-case rate | spurious / perturbed case (clean / noisy target) | s/case (warm cache) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in reports + base:
        d, n = r["detection"], r["noise"]
        L.append(f"| {r['adversary']} | {r['split']} | **{r['headline']:.3f}** | {pct(d['precision'])} | {pct(d['recall'])} | {pct(d['f1'])} | {pct(r['ap'])} | "
                 f"{pct(r['macro_leaf']['f1'])} | {pct(r['macro_parent']['f1'])} | {pct(r['type_accuracy']['leaf'])} / {pct(r['type_accuracy']['parent'])} | "
                 f"{pct(n['false_case_rate'])} | {n['spurious_per_perturbed_case_clean_target']:.2f} / {n['spurious_per_perturbed_case_noisy_target']:.2f} | {r['runtime']['mean_s']:.3f} |")
    L += ["", "Variants (evaluation is never used for tuning): `irspace` = the tuned adversary; `irspace_no_pixel_support` = the "
          "same IR differ with pixel support switched off (what the perceiver alone claims); `irspace_no_fixed_point` = no "
          "fixed-point down-weighting; `irspace_hybrid` = plus low-confidence pixel residuals the IR differ did not explain; "
          "`irspace_no_page_guard` = no globally-disturbed-page guard; `irspace_no_nuisance_matching` = the candidate is perceived as "
          "rendered (no sub-pixel/blur/JPEG matching); `baseline_residual` = the benchmark's floor.", ""]
    ev = [r for r in reports if r["adversary"] == "irspace"]
    full = B.format_report(ev, bench, "x")
    full = full[full.index("## irspace on"):] if "## irspace on" in full else full
    L += ["## Per-type results and confusion matrices", "", full.replace("## irspace on", "### irspace on"), ""]
    # runtime
    ps = [t[1] for t in timings if t[1] == t[1] and t[1] > 0.5]
    fs = [t[2] for t in timings if t[2] == t[2] and t[2] > 0.5]
    L += ["## Runtime", "",
          f"* Cold perception (one image, process pool of 4 under a shared machine load of ~15-70): median {np.median(ps):.1f} s, "
          f"p95 {np.percentile(ps, 95):.1f} s over {len(ps)} images." if ps else "* Cold perception: (all cached)",
          f"* Fixed point (render + re-perceive the target reading): median {np.median(fs):.1f} s over {len(fs)} targets." if fs else "",
          "* A cold `find()` on a new pair therefore costs ~2 perceptions + 1 render + 1 perception for the fixed point "
          "(~15-20 s; the target side is cached across candidates of the same target, so a refine loop pays ~5 s per candidate).",
          "* With the perception cache warm (as the scores above were computed) the differ, nuisance matching and pixel support "
          "cost the s/case column.", ""]
    L += ["## Tuning (CALIBRATION only)", "", f"Coordinate descent over {len(TUNE_GRID)} registered params "
          f"(`adversary.irspace.*`); calibration headline {cal_h if cal_h is None else f'{cal_h:.3f}'}. Values the search moved (now the registered defaults):", "",
          "| param | tuned |", "|---|---|"] + [f"| `{_R}{k}` | {v} |" for k, v in sorted(best.items())] + [""]
    if fp_study:
        L += ["## Fixed-point test: perceive(render(perceive(x))) vs perceive(x)", "",
              "| split | targets | mean stable elements | min | predictions on unstable boxes (precision) | on stable boxes (precision) |", "|---|---|---|---|---|---|"]
        for sp, st in fp_study.items():
            u, o = st["on_unstable"], st["on_stable"]
            f = lambda v: "–" if v is None else pct(v)  # noqa: E731
            L.append(f"| {sp} | {st['targets']} | {pct(st['stable_frac_mean'])} | {pct(st['stable_frac_min'])} | {u['n']} ({f(u['precision'])}) | {o['n']} ({f(o['precision'])}) |")
        L.append("")
    L += ["## Real translated pages (out/real_r3m: target = validation/target.png, candidate = render.png, IR = ir.mapped.json)", "",
          "| page | size | changed edges | findings (guarded) | by type | unguarded findings | fixed point: stable elements | evidence |", "|---|---|---|---|---|---|---|---|"]
    for r in real:
        bt = ", ".join(f"{k} {v}" for k, v in r["counts"].items()) or "–"
        L.append(f"| {r['page']} | {r['size'][0]}x{r['size'][1]} | {pct(r['changed_edges'])} | {r['n']} | {bt} | {r['n_unguarded']} | "
                 f"{pct(r['fixed_point']['stable_frac'])} ({r['fixed_point']['unstable']}/{r['fixed_point']['elements']} unstable) | `{r['image']}` |")
    L.append("")
    if examples:
        L += ["Benchmark evidence panels (target | candidate | ΔE heatmap; green = ground truth, red = ir-space findings): "
              + ", ".join(f"`{os.path.relpath(p, ROOT)}`" for p in examples), ""]
    if notes:
        L += [notes, ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
