"""Novel finding types: discrepancies the taxonomy does not explain become candidate types and scenario stubs.

The taxonomy (``dt.adversary.taxonomy``) is the loop's vocabulary; a vocabulary learned from synthetic
perturbations cannot contain the failure modes that only real screenshots exhibit. This module finds
them, without labels, and turns each into a draft scenario family so the loop can learn it next round.

1. **Items.** Every discrepancy on a page is an item: the residual regions of ``dt.compare.pixel``
   (ΔE > ``compare.pixel.bad_de``) and the fused findings of the ensemble. Each item gets 16
   interpretable features measured on the two images and the candidate IR (size, aspect, ΔE, signed
   lightness/chroma change, texture and edge density on both sides, which side owns the ink, text /
   icon coverage, how much an integer shift or a per-channel affine colour transfer explains, ...).
2. **Known distribution.** The same items on the fusion-calibration benchmark (never evaluation, never
   real pages), labelled with the ground-truth type they localise, or ``noise`` (pure-noise cases and
   unmatched residue). Features are robust-standardised on it (median / IQR).
3. **Novelty test.** Distance to the k-th nearest known item, converted to a conformal p-value against
   the known items' own leave-one-out k-NN distances; plus the k-NN label purity (how well a single
   type explains the neighbourhood). An item is a novelty candidate when ``p < alpha`` (unlike
   anything seen) or when it is a fused finding whose type posterior is below ``type_conf_min``.
4. **Clusters.** Candidates are clustered with DBSCAN (eps from the candidates' own k-distance
   distribution). A cluster is a candidate *type* only when it is supported by >= ``min_pages``
   distinct pages: a one-page cluster is a one-off and is reported but never promoted (no overfitting
   to one screenshot).
5. **Report + stubs.** Per cluster: size, pages, centroid in original units, the features that
   distinguish it from the known distribution, nearest known type, exemplar crops, a proposed name and
   a draft ``ScenarioFamily`` stub compatible with the scenario harness spec (``dt.scenarios.spec``:
   name, description, stage, failure_refs, param_space, generate(params, seed), criteria) with a
   generator sketch, criteria on the validator's measures, failure refs mined from
   ``knowledge/failures.jsonl``, and a Python skeleton.

CLI: ``python -m dt.adversary.novelty [--real out/real_r3m]`` -> ``out/adversary/synthesis/novelty/``.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

import cv2
import numpy as np

from dt.adversary.taxonomy import ERROR_TYPES, TYPES, Finding, is_noise
from dt.ir import Box, Document
from dt.params import P, register

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "out", "adversary", "synthesis", "novelty")
REAL_DIR = os.path.join(ROOT, "out", "real_r3m")
FAILURES = os.path.join(ROOT, "knowledge", "failures.jsonl")

_K = "adversary.novelty."
register(_K + "k", 5, "k of the k-NN novelty distance and label purity", (1, 25))
register(_K + "alpha", 0.02, "conformal p-value below which an item is unlike every known item", (0.001, 0.2))
register(_K + "type_conf_min", 0.5, "a fused finding whose type posterior is below this is a novelty candidate (no type explains it well)", (0.1, 0.95))
register(_K + "purity_min", 0.5, "or: k-NN label purity below this AND p < 2*alpha", (0.1, 1.0))
register(_K + "min_area", 24, "items smaller than this (px²) are ignored", (1, 400))
register(_K + "min_pts", 3, "DBSCAN: points within eps for a core point", (2, 20))
register(_K + "eps_quantile", 0.5, "DBSCAN eps = this quantile of the candidates' min_pts-NN distances", (0.1, 0.95))
register(_K + "min_pages", 2, "a cluster becomes a candidate type only when its items come from at least this many pages", (1, 10))
register(_K + "max_items_per_page", 400, "cap on residual items per page (largest first)", (20, 5000))
register(_K + "z_clip", 6.0, "clustering clips standardised features at +-this: 'far outside the known range' is one state, so one extreme feature cannot split a mode", (2.0, 50.0))
register(_K + "ink_de", 10.0, "ΔE76 from the box's border median for a pixel to count as ink", (2.0, 40.0))

FEATURES = ("log_area", "log_aspect", "page_frac", "de_mean", "de_p90", "dL_share", "d_chroma", "tex_t", "tex_c",
            "edge_t", "edge_c", "ink_balance", "text_frac", "icon_frac", "shift_gain", "colour_gain")
FEATURE_DOC = {
    "log_area": "log10 box area (px²)", "log_aspect": "log(w/h)", "page_frac": "box area / page area",
    "de_mean": "mean ΔE76 target vs candidate in the box", "de_p90": "90th percentile ΔE76",
    "dL_share": "signed mean lightness change (target - candidate) / mean ΔE", "d_chroma": "mean chroma change (target - candidate)",
    "tex_t": "lightness std in the target box", "tex_c": "lightness std in the candidate box",
    "edge_t": "Canny edge density of the target box", "edge_c": "Canny edge density of the candidate box",
    "ink_balance": "(target ink - candidate ink) / (sum): +1 only the target paints, -1 only the candidate",
    "text_frac": "share of the box covered by candidate text nodes", "icon_frac": "share covered by icon / vector / image nodes",
    "shift_gain": "share of the box error removed by the best integer shift (|d| <= 3 px)",
    "colour_gain": "share of the box error removed by a per-channel affine colour transfer",
}


# --------------------------------------------------------------------------- features
def _lab(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3], dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


@dataclass
class PageCtx:
    """Per-page precomputation shared by all items of the page."""
    target: np.ndarray
    cand: np.ndarray
    ir: Optional[Document]
    LT: np.ndarray = field(init=False)
    LC: np.ndarray = field(init=False)
    D: np.ndarray = field(init=False)
    ET: np.ndarray = field(init=False)
    EC: np.ndarray = field(init=False)
    text_mask: np.ndarray = field(init=False)
    icon_mask: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        H, W = self.target.shape[:2]
        c = self.cand
        if c.shape[:2] != (H, W):  # judge on the target frame (pad / crop the candidate)
            pad = np.zeros_like(self.target)
            h, w = min(H, c.shape[0]), min(W, c.shape[1])
            pad[:h, :w] = c[:h, :w, :3]
            self.cand = c = pad
        self.LT, self.LC = _lab(self.target), _lab(c)
        self.D = np.sqrt(((self.LT - self.LC) ** 2).sum(-1))
        gt = cv2.cvtColor(np.ascontiguousarray(self.target[..., :3]), cv2.COLOR_RGB2GRAY)
        gc = cv2.cvtColor(np.ascontiguousarray(c[..., :3]), cv2.COLOR_RGB2GRAY)
        self.ET = cv2.Canny(gt, 40, 120) > 0
        self.EC = cv2.Canny(gc, 40, 120) > 0
        self.text_mask = np.zeros((H, W), bool)
        self.icon_mask = np.zeros((H, W), bool)
        if self.ir is not None:
            for n in self.ir.walk():
                if not n.visible or n.box.area <= 0:
                    continue
                x0, y0, x1, y1 = _clip(n.box, W, H)
                if n.type == "text":
                    self.text_mask[y0:y1, x0:x1] = True
                elif n.type in ("icon", "vector", "image"):
                    self.icon_mask[y0:y1, x0:x1] = True


def _clip(b: Box, W: int, H: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = b.as_int()
    return max(0, x0), max(0, y0), min(W, max(x0 + 1, x1)), min(H, max(y0 + 1, y1))


def _ink(L: np.ndarray, thr: float) -> float:
    border = np.concatenate([L[0], L[-1], L[:, 0], L[:, -1]])
    med = np.median(border, axis=0)
    return float((np.sqrt(((L - med) ** 2).sum(-1)) > thr).mean())


def features(ctx: PageCtx, box: Box) -> Optional[dict]:
    H, W = ctx.D.shape
    x0, y0, x1, y1 = _clip(box, W, H)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    d = ctx.D[y0:y1, x0:x1]
    lt, lc = ctx.LT[y0:y1, x0:x1], ctx.LC[y0:y1, x0:x1]
    de_mean = float(d.mean())
    e0 = float(d.sum()) + 1e-6
    # best integer shift of the candidate crop (search |d| <= 3 on a padded window)
    best = e0
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            if dx == 0 and dy == 0:
                continue
            sx0, sy0, sx1, sy1 = x0 + dx, y0 + dy, x1 + dx, y1 + dy
            if sx0 < 0 or sy0 < 0 or sx1 > W or sy1 > H:
                continue
            e = float(np.sqrt(((lt - ctx.LC[sy0:sy1, sx0:sx1]) ** 2).sum(-1)).sum())
            best = min(best, e)
    shift_gain = max(0.0, 1.0 - best / e0)
    # per-channel affine colour transfer candidate -> target
    fit = np.empty_like(lc)
    for ch in range(3):
        a = lc[..., ch].ravel()
        b = lt[..., ch].ravel()
        A = np.stack([a, np.ones_like(a)], 1)
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        fit[..., ch] = (A @ coef).reshape(lc.shape[:2])
    colour_gain = max(0.0, 1.0 - float(np.sqrt(((lt - fit) ** 2).sum(-1)).sum()) / e0)
    ch_t = np.sqrt(lt[..., 1] ** 2 + lt[..., 2] ** 2)
    ch_c = np.sqrt(lc[..., 1] ** 2 + lc[..., 2] ** 2)
    thr = float(P[_K + "ink_de"])
    it, ic = _ink(lt, thr), _ink(lc, thr)
    area = float((x1 - x0) * (y1 - y0))
    return {"log_area": math.log10(area), "log_aspect": math.log((x1 - x0) / (y1 - y0)), "page_frac": area / float(W * H),
            "de_mean": de_mean, "de_p90": float(np.percentile(d, 90)),
            "dL_share": float((lt[..., 0] - lc[..., 0]).mean()) / max(de_mean, 1e-3), "d_chroma": float((ch_t - ch_c).mean()),
            "tex_t": float(lt[..., 0].std()), "tex_c": float(lc[..., 0].std()),
            "edge_t": float(ctx.ET[y0:y1, x0:x1].mean()), "edge_c": float(ctx.EC[y0:y1, x0:x1].mean()),
            "ink_balance": (it - ic) / (it + ic + 1e-6), "text_frac": float(ctx.text_mask[y0:y1, x0:x1].mean()),
            "icon_frac": float(ctx.icon_mask[y0:y1, x0:x1].mean()), "shift_gain": shift_gain, "colour_gain": colour_gain}


def vec(f: dict) -> np.ndarray:
    return np.array([f[k] for k in FEATURES], np.float64)


# --------------------------------------------------------------------------- items
@dataclass
class Item:
    page: str
    box: Box
    source: str                     # "residual" | "finding"
    feats: dict
    finding: Optional[dict] = None  # the fused finding (type, confidence, p_type) when source == finding
    label: Optional[str] = None     # known items only
    novelty: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"page": self.page, "box": self.box.to_dict(), "source": self.source, "features": self.feats, "finding": self.finding,
                "label": self.label, "novelty": self.novelty}


def page_items(name: str, target: np.ndarray, cand: np.ndarray, ir: Optional[Document], findings: Iterable[Finding] = ()) -> list[Item]:
    """Residual regions + (non-noise) findings of one page, with features."""
    from dt.compare.pixel import diff_map, residual_regions
    ctx = PageCtx(target, cand, ir)
    regs = residual_regions(diff_map(ctx.target, ctx.cand))
    regs = sorted(regs, key=lambda b: -b.area)[: int(P[_K + "max_items_per_page"])]
    out = []
    for b in regs:
        if b.area < float(P[_K + "min_area"]):
            continue
        f = features(ctx, b)
        if f:
            out.append(Item(name, b, "residual", f))
    for fd in findings:
        if is_noise(fd.type) or fd.box.area < float(P[_K + "min_area"]):
            continue
        f = features(ctx, fd.box)
        if f:
            ev = fd.evidence or {}
            out.append(Item(name, fd.box, "finding", f, {"type": fd.type, "confidence": float(fd.confidence),
                                                         "p_type": float(ev.get("p_type", 1.0)), "members": ev.get("members")}))
    return out


def known_items(cases: list, findings_by_uid: Optional[dict] = None) -> list[Item]:
    """Labelled items on benchmark cases: residual regions localising a gt finding take its type,
    the others (and everything on pure-noise cases) are ``noise``."""
    from dt.adversary.benchmark import localises
    from dt.adversary.ensemble import uid
    out = []
    for c in cases:
        fs = (findings_by_uid or {}).get(uid(c), [])
        items = page_items(uid(c), c.target_rgb, c.candidate_rgb, c.candidate_ir, fs)
        for it in items:
            lab = "noise"
            best = 0.0
            for g in c.gt:
                if localises(it.box, g, c.page):
                    s = it.box.iou(g.box) + 0.01
                    if s > best:
                        best, lab = s, g.type
            it.label = lab
            out.append(it)
    return out


# --------------------------------------------------------------------------- the novelty model
@dataclass
class NoveltyModel:
    centre: np.ndarray
    scale: np.ndarray
    X: np.ndarray                   # standardised known items
    labels: list[str]
    loo: np.ndarray                 # leave-one-out k-NN distances of the known items (sorted)
    k: int

    @staticmethod
    def fit(items: list[Item]) -> "NoveltyModel":
        A = np.stack([vec(i.feats) for i in items])
        centre = np.median(A, 0)
        q75, q25 = np.percentile(A, 75, 0), np.percentile(A, 25, 0)
        scale = np.where(q75 - q25 > 1e-6, q75 - q25, np.maximum(A.std(0), 1e-3))
        X = (A - centre) / scale
        k = int(P[_K + "k"])
        D = _pairwise(X, X)
        np.fill_diagonal(D, np.inf)
        loo = np.sort(np.sort(D, 1)[:, k - 1])
        return NoveltyModel(centre, scale, X, [i.label or "noise" for i in items], loo, k)

    def z(self, f: dict) -> np.ndarray:
        return (vec(f) - self.centre) / self.scale

    def score(self, items: list[Item]) -> None:
        if not items:
            return
        Z = np.stack([self.z(i.feats) for i in items])
        D = _pairwise(Z, self.X)
        idx = np.argsort(D, 1)[:, : self.k]
        for r, it in enumerate(items):
            dk = float(D[r, idx[r, -1]])
            p = (1.0 + float((self.loo >= dk).sum())) / (1.0 + len(self.loo))
            labs = Counter(self.labels[j] for j in idx[r])
            top, cnt = labs.most_common(1)[0]
            it.novelty = {"knn_dist": dk, "p_value": p, "nearest_type": top, "purity": cnt / self.k,
                          "neighbours": dict(labs)}


def _pairwise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    aa = (A * A).sum(1)[:, None]
    bb = (B * B).sum(1)[None, :]
    return np.sqrt(np.maximum(aa + bb - 2 * A @ B.T, 0.0))


def is_candidate(it: Item) -> bool:
    nv = it.novelty
    alpha = float(P[_K + "alpha"])
    if nv.get("p_value", 1.0) < alpha:
        return True
    if it.finding is not None and it.finding.get("p_type", 1.0) < float(P[_K + "type_conf_min"]):
        return True
    return nv.get("purity", 1.0) < float(P[_K + "purity_min"]) and nv.get("p_value", 1.0) < 2 * alpha


def dbscan(X: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """Plain DBSCAN (labels -1 = noise). O(n²) memory; fine for a few thousand points."""
    n = len(X)
    D = _pairwise(X, X)
    nb = [np.flatnonzero(D[i] <= eps) for i in range(n)]
    core = np.array([len(v) >= min_pts for v in nb])
    lab = np.full(n, -1)
    c = 0
    for i in range(n):
        if lab[i] != -1 or not core[i]:
            continue
        lab[i] = c
        stack = [i]
        while stack:
            j = stack.pop()
            if not core[j]:
                continue
            for q in nb[j]:
                if lab[q] == -1:
                    lab[q] = c
                    stack.append(q)
        c += 1
    return lab


# --------------------------------------------------------------------------- naming + scenario stubs
def _distinguishing(z: np.ndarray, n: int = 4) -> list[tuple[str, float]]:
    order = np.argsort(-np.abs(z))[:n]
    return [(FEATURES[i], float(z[i])) for i in order]


def propose_name(cent: dict, z: np.ndarray) -> tuple[str, str]:
    """A taxonomy-style name and a one-line hypothesis from the centroid (original units) and its z."""
    if cent["text_frac"] > 0.5 and cent["shift_gain"] < 0.5 and cent["colour_gain"] < 0.6 and cent["edge_t"] > 0.05 and cent["edge_c"] > 0.05:
        return "text.glyph_shape", ("text whose glyph outlines differ although position and colour match: a font family / "
                                    "rasterisation mismatch (substituted font, hinting, smoothing)")
    if cent["tex_t"] > 1.5 * max(cent["tex_c"], 1.0) and cent["log_area"] > 3.0 and cent["ink_balance"] > 0.2 and cent["icon_frac"] < 0.5:
        return "structure.flattened_art", "textured target art (illustration, photo, mock-up) rendered as a flat or near-flat block"
    if cent["tex_c"] > 2.0 * max(cent["tex_t"], 1.0) and cent["log_area"] > 3.3:
        return "structure.spurious_texture", "the candidate paints texture or a raster where the target is flat"
    if cent["ink_balance"] > 0.5 and cent["log_area"] < 3.3:
        return "icon.unrendered_glyph", "small target-only ink (custom glyph, logo, bullet) that the candidate does not paint"
    if cent["ink_balance"] < -0.5 and cent["log_area"] < 3.3:
        return "structure.extra_small", "small candidate-only ink the target does not have"
    if cent["shift_gain"] > 0.6:
        return "geometry.subpixel_drift", "an offset of 1-3 px that an integer shift explains, below the geometry types' magnitudes"
    if cent["colour_gain"] > 0.7 and abs(cent["d_chroma"]) > 3:
        return "color.local_transfer", "a region-wide colour transform (chroma change) that no single-fill type covers"
    top = _distinguishing(z, 2)
    return "residual." + "_".join(f"{'hi' if v > 0 else 'lo'}_{k}" for k, v in top), "an unexplained residual pattern; see its distinguishing features"


_REF_WORDS = {
    "text.glyph_shape": ("font", "glyph", "smoothing", "typeface", "family"),
    "structure.flattened_art": ("illustration", "raster", "art", "placeholder", "crop"),
    "structure.spurious_texture": ("raster", "crop", "texture"),
    "icon.unrendered_glyph": ("icon", "glyph", "logo", "bullet", "sparkle", "unnamed"),
    "structure.extra_small": ("extra", "spurious"),
    "geometry.subpixel_drift": ("offset", "sub-pixel", "subpixel", "2px", "shift"),
    "color.local_transfer": ("tint", "colour", "color", "theme", "palette"),
}


def failure_refs(name: str, limit: int = 4) -> list[str]:
    """``knowledge/failures.jsonl`` entries whose symptom/root cause mention the cluster's keywords."""
    words = _REF_WORDS.get(name, ())
    if not words or not os.path.exists(FAILURES):
        return []
    hits = []
    with open(FAILURES) as f:
        for i, line in enumerate(f):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = " ".join(str(d.get(k, "")) for k in ("case", "symptom", "root_cause", "root_cause_hypothesis")).lower()
            score = sum(text.count(w) for w in words)
            if score:
                hits.append((score, i, str(d.get("case", ""))[:120]))
    hits.sort(key=lambda t: (-t[0], t[1]))
    return [f"failures.jsonl#{i}: {c}" for _, i, c in hits[:limit]]


_SKETCH = {
    "text.glyph_shape": ("perceive", "ir", ["perceive.fonts"],
                         "IR page of 6-20 text lines (12-22 px, weights 400/500); target rendered with family A from a pool "
                         "{Roboto, Google Sans Text, Inter, Arial, Helvetica}, candidate pipeline must recover A; params: family, size, "
                         "weight, tracking, smoothing on/off, background contrast.",
                         [{"metric": "mean_de", "op": "<=", "threshold": 1.0, "roi": "roi", "scale": 2.0, "doc": "text pixels reproduce"},
                          {"metric": "text_cer", "op": "<=", "threshold": "param:validate.gate.struct.text_cer", "roi": None, "scale": 0.05,
                           "doc": "characters stay right"}]),
    "structure.flattened_art": ("translate", "ir", ["refine.raster", "perceive.segment"],
                                "IR page with one image node (procedural texture: gradient + shapes + noise, 120-420 px per side) "
                                "among cards and text; success = the art is reproduced (raster within budget) without baking text.",
                                [{"metric": "region_de_worst", "op": "<", "threshold": "param:validate.gate.ident.worst_region_de", "roi": "roi",
                                  "scale": 10.0, "doc": "art region reproduced"},
                                 {"metric": "raster_frac", "op": "<=", "threshold": "param:validate.gate.raster_frac", "roi": None, "scale": 0.35,
                                  "doc": "editability budget"}]),
    "structure.spurious_texture": ("translate", "ir", ["refine.raster"],
                                   "IR page with flat surfaces next to textured art; success = no raster or texture over the flat surfaces.",
                                   [{"metric": "jnd_frac_nontext", "op": "<=", "threshold": "param:validate.gate.ident.jnd_frac", "roi": "roi",
                                     "scale": 0.05, "doc": "flat stays flat"}]),
    "icon.unrendered_glyph": ("perceive", "ir", ["perceive.icons", "refine.raster"],
                              "IR page with 4-12 small non-Material glyphs (bullets, sparkles, logos as vector paths, 8-24 px) next to "
                              "Material Symbols; success = every glyph painted (named icon or raster), none misnamed.",
                              [{"metric": "jnd_frac_nontext", "op": "<=", "threshold": "param:validate.gate.ident.jnd_frac", "roi": "roi",
                                "scale": 0.05, "doc": "glyph pixels reproduced"}]),
    "structure.extra_small": ("perceive", "ir", ["perceive.segment"],
                              "IR page with clean flat surfaces and JPEG/blur nuisance; success = no spurious small nodes.",
                              [{"metric": "jnd_frac_nontext", "op": "<=", "threshold": "param:validate.gate.ident.jnd_frac", "roi": None,
                                "scale": 0.05, "doc": "nothing extra painted"}]),
    "geometry.subpixel_drift": ("translate", "ir", ["perceive.ocr", "refine.critic"],
                                "IR page captured at a fractional page offset (0.1-0.9 px) and DPR 1-3 downscale; success = boxes within 1 px.",
                                [{"metric": "chamfer", "op": "<=", "threshold": "param:validate.gate.ident.chamfer", "roi": None, "scale": 1.0,
                                  "doc": "edges in place"}]),
    "color.local_transfer": ("translate", "ir", ["perceive.segment", "map.color"],
                             "IR page whose one region (card, sheet, hero) is colour-transformed (Lab affine, 3-15 ΔE); success = fills recovered.",
                             [{"metric": "region_de_worst", "op": "<", "threshold": "param:validate.gate.ident.worst_region_de", "roi": "roi",
                               "scale": 10.0, "doc": "region colours"}]),
}


def scenario_stub(name: str, hypothesis: str, cluster: dict) -> dict:
    """A draft ``ScenarioFamily`` (``dt.scenarios.spec``) for the candidate type: the fields of
    ``ScenarioFamily.summary()`` plus a generator sketch and a Python skeleton. ``param_space`` is
    widened from the cluster's observed ranges (a family must cover the mode, not the pages it was seen on)."""
    stage, source, prefixes, sketch, criteria = _SKETCH.get(name, (
        "translate", "real", [], "crops of the cluster's exemplar regions (finite real family) until a generator is written",
        [{"metric": "region_de_worst", "op": "<", "threshold": "param:validate.gate.ident.worst_region_de", "roi": "roi", "scale": 10.0,
          "doc": "region reproduced"}]))
    rng = cluster["ranges"]
    side = [round(10 ** (rng["log_area"][0] / 2) * 0.5), round(10 ** (rng["log_area"][1] / 2) * 1.5)]
    param_space = {"region_side_px": [max(4, int(side[0])), max(8, int(side[1]))],
                   "contrast_de": [round(max(1.0, rng["de_mean"][0] * 0.5), 2), round(rng["de_mean"][1] * 1.5, 2)],
                   "page": ["mobile_412", "desktop_1280"], "theme": ["light", "dark"],
                   "nuisance": ["none", "jpeg_q85", "subpixel_0.33", "smoothing_off"]}
    fam = "adv_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    stub = {"name": fam, "description": f"{hypothesis}. Discovered by dt.adversary.novelty as cluster {cluster['id']} "
                                        f"({cluster['n']} items on {len(cluster['pages'])} pages).",
            "stage": stage, "source": source, "refine_iters": 4 if stage == "translate" else 0,
            "failure_refs": failure_refs(name), "param_space": param_space, "criteria": criteria,
            "tune_prefixes": prefixes, "version": 1, "generator_sketch": sketch, "candidate_type": name,
            "evidence": {"pages": cluster["pages"], "exemplars": cluster["exemplars"], "distinguishing": cluster["distinguishing"]}}
    stub["python"] = _python_stub(stub)
    return stub


def _python_stub(s: dict) -> str:
    crit = ",\n              ".join(
        f"Criterion({c['metric']!r}, {c['op']!r}, {c['threshold']!r}, roi={c['roi']!r}, scale={c['scale']}, doc={c['doc']!r})" for c in s["criteria"])
    ps = ", ".join(f"{k!r}: {tuple(v) if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v) else v!r}"
                   for k, v in s["param_space"].items())
    return (f'"""{s["name"]}: {s["description"]}\n\nGenerator sketch: {s["generator_sketch"]}\n"""\n'
            "from dt.scenarios.spec import Criterion, ScenarioCase, ScenarioFamily\n\n\n"
            "def generate(params: dict, seed: int) -> ScenarioCase:\n"
            "    raise NotImplementedError('draft from dt.adversary.novelty: implement the generator sketch above')\n\n\n"
            f"FAMILY = ScenarioFamily(\n    name={s['name']!r},\n    description={s['description']!r},\n    stage={s['stage']!r},\n"
            f"    failure_refs={s['failure_refs']!r},\n    param_space={{{ps}}},\n    generate=generate,\n"
            f"    criteria=[{crit}],\n    source={s['source']!r},\n    refine_iters={s['refine_iters']},\n"
            f"    tune_prefixes={tuple(s['tune_prefixes'])!r},\n)\n")


def to_family(stub: dict):
    """Instantiate the stub as a ``dt.scenarios.spec.ScenarioFamily`` when the scenario harness is
    installed (its generator raises until implemented). Returns None otherwise."""
    try:
        from dt.scenarios.spec import Criterion, ScenarioFamily
    except Exception:
        return None

    def generate(params, seed):
        raise NotImplementedError(stub["generator_sketch"])
    return ScenarioFamily(name=stub["name"], description=stub["description"], stage=stub["stage"], failure_refs=list(stub["failure_refs"]),
                          param_space={k: (tuple(v) if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v) else v)
                                       for k, v in stub["param_space"].items()},
                          generate=generate, criteria=[Criterion.from_dict(c) for c in stub["criteria"]], source=stub["source"],
                          refine_iters=stub["refine_iters"], tune_prefixes=tuple(stub["tune_prefixes"]))


# --------------------------------------------------------------------------- clustering + report
def cluster_candidates(cands: list[Item], model: NoveltyModel) -> list[dict]:
    if len(cands) < int(P[_K + "min_pts"]):
        return []
    zc_ = float(P[_K + "z_clip"])
    Z = np.clip(np.stack([model.z(i.feats) for i in cands]), -zc_, zc_)
    mp_ = int(P[_K + "min_pts"])
    D = _pairwise(Z, Z)
    kd = np.sort(D, 1)[:, min(mp_, len(cands) - 1)]
    eps = float(np.quantile(kd, float(P[_K + "eps_quantile"])))
    lab = dbscan(Z, eps, mp_)
    out = []
    for c in sorted(set(lab) - {-1}):
        idx = np.flatnonzero(lab == c)
        members = [cands[i] for i in idx]
        A = np.stack([vec(m.feats) for m in members])
        cent = {k: float(v) for k, v in zip(FEATURES, A.mean(0))}
        zc = Z[idx].mean(0)
        name, hyp = propose_name(cent, zc)
        # exemplars: closest to the centroid, one per page first
        dist = np.linalg.norm(Z[idx] - zc, axis=1)
        order = np.argsort(dist)
        ex, seen = [], set()
        for j in order:
            m = members[j]
            if m.page not in seen:
                ex.append({"page": m.page, "box": m.box.to_dict(), "source": m.source, "finding": m.finding})
                seen.add(m.page)
            if len(ex) >= 4:
                break
        pages = sorted({m.page for m in members})
        near = Counter(m.novelty.get("nearest_type") for m in members).most_common(1)[0][0]
        out.append({"id": f"C{c}", "n": len(members), "pages": pages, "name": name, "hypothesis": hyp,
                    "promotable": len(pages) >= int(P[_K + "min_pages"]),
                    "centroid": cent, "distinguishing": [{"feature": k, "z": round(v, 2), "doc": FEATURE_DOC[k]} for k, v in _distinguishing(zc)],
                    "ranges": {k: [float(A[:, i].min()), float(A[:, i].max())] for i, k in enumerate(FEATURES)},
                    "nearest_known_type": near, "median_p": float(np.median([m.novelty["p_value"] for m in members])),
                    "sources": dict(Counter(m.source for m in members)), "exemplars": ex, "eps": eps,
                    "member_boxes": [(m.page, m.box.to_dict()) for m in members][:200]})
    out.sort(key=lambda d: (-len(d["pages"]), -d["n"]))
    for d in out:
        d["stub"] = scenario_stub(d["name"], d["hypothesis"], d) if d["promotable"] else None
    return out


def save_exemplars(cl: dict, images: dict[str, tuple[np.ndarray, np.ndarray]], out_dir: str) -> list[str]:
    """target | candidate crops (with 8 px context) of a cluster's exemplars."""
    from dt.common.image import save_rgb
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for k, e in enumerate(cl["exemplars"]):
        if e["page"] not in images:
            continue
        t, c = images[e["page"]]
        b = Box.from_dict(e["box"]).expand(8)
        H, W = t.shape[:2]
        x0, y0, x1, y1 = _clip(b, W, H)
        ct, cc = t[y0:y1, x0:x1], c[y0:min(y1, c.shape[0]), x0:min(x1, c.shape[1])]
        if cc.shape != ct.shape:
            pad = np.full_like(ct, 255)
            pad[: cc.shape[0], : cc.shape[1]] = cc
            cc = pad
        scale = max(1, int(96 / max(1, min(ct.shape[:2]))))
        scale = min(scale, 4)
        im = np.concatenate([ct, np.full((ct.shape[0], 4, 3), 255, np.uint8), cc], 1)
        if scale > 1:
            im = cv2.resize(im, (im.shape[1] * scale, im.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        if im.shape[1] > 1600:
            f = 1600 / im.shape[1]
            im = cv2.resize(im, (1600, max(1, int(im.shape[0] * f))), interpolation=cv2.INTER_AREA)
        p = os.path.join(out_dir, f"{cl['id']}_{k}_{e['page']}.png")
        save_rgb(np.ascontiguousarray(im), p)
        paths.append(p)
    return paths


@dataclass
class RealCase:
    """A real translated page (``out/real_r3m/<page>``) shaped like a benchmark case for the ensemble."""
    id: str
    paths: dict
    bench_key: str = "real_r3m"
    kind: str = "real"
    gt: list = field(default_factory=list)
    family: str = "real"
    meta: dict = field(default_factory=dict)

    @property
    def target_rgb(self) -> np.ndarray:
        from dt.common.image import load_rgb
        return load_rgb(self.paths["target"])

    @property
    def candidate_rgb(self) -> np.ndarray:
        from dt.common.image import load_rgb
        return load_rgb(self.paths["candidate"])

    @property
    def candidate_ir(self) -> Optional[Document]:
        return Document.load(self.paths["ir"]) if os.path.exists(self.paths["ir"]) else None

    @property
    def page(self) -> Box:
        t = self.target_rgb
        return Box(0, 0, t.shape[1], t.shape[0])


def real_cases(root: str = REAL_DIR) -> list[RealCase]:
    out = []
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        v = os.path.join(root, name, "validation")
        if os.path.exists(os.path.join(v, "target.png")) and os.path.exists(os.path.join(v, "render.png")):
            out.append(RealCase(name, {"target": os.path.join(v, "target.png"), "candidate": os.path.join(v, "render.png"),
                                       "ir": os.path.join(root, name, "ir.mapped.json")}))
    return out


def run(real_root: str = REAL_DIR, workers: int = 2, out_dir: str = OUT, progress=print) -> dict:
    """Fit the known distribution on fusion-calibration, score + cluster the real pages, write the report."""
    from dt.adversary import ensemble as E
    os.makedirs(out_dir, exist_ok=True)
    model_f = E.load_model()
    fcal = E.fusion_calibration()
    preds_fcal, _ = E._load_all(fcal, model_f.approaches, workers, None)
    fused_fcal = E.fuse_preds(model_f, preds_fcal, fcal)
    progress(f"known items on {len(fcal)} fusion-calibration cases")
    known = known_items(fcal, fused_fcal)
    nm = NoveltyModel.fit(known)
    # sanity: novelty rate on EVALUATION benchmark items (should be ~alpha: they are known types)
    ev = E.main_bench().split("evaluation")
    preds_ev, _ = E._load_all(ev, model_f.approaches, workers, None)
    fused_ev = E.fuse_preds(model_f, preds_ev, ev)
    ev_items = known_items(ev, fused_ev)
    nm.score(ev_items)
    ev_rate = float(np.mean([i.novelty["p_value"] < float(P[_K + "alpha"]) for i in ev_items])) if ev_items else 0.0
    ev_cand = [i for i in ev_items if is_candidate(i)]
    ev_clusters = cluster_candidates(ev_cand, nm)
    # real pages
    reals = real_cases(real_root)
    preds_r, _ = E._load_all(reals, model_f.approaches, workers, progress)
    fused_r = E.fuse_preds(model_f, preds_r, reals)
    items, images, per_page = [], {}, {}
    for rc in reals:
        t, c = rc.target_rgb, rc.candidate_rgb
        images[rc.id] = (t, c)
        its = page_items(rc.id, t, c, rc.candidate_ir, fused_r.get(E.uid(rc), []))
        nm.score(its)
        items += its
        per_page[rc.id] = {"items": len(its), "candidates": sum(is_candidate(i) for i in its),
                           "fused_findings": [f.to_dict() for f in fused_r.get(E.uid(rc), [])]}
    cands = [i for i in items if is_candidate(i)]
    clusters = cluster_candidates(cands, nm)
    for cl in clusters:
        cl["exemplar_images"] = [os.path.relpath(p, ROOT) for p in save_exemplars(cl, images, os.path.join(out_dir, "exemplars"))]
    res = {"known": {"items": len(known), "labels": dict(Counter(i.label for i in known)), "k": nm.k},
           "evaluation_sanity": {"items": len(ev_items), "p_below_alpha": ev_rate, "candidates": len(ev_cand),
                                 "clusters": [{k: d[k] for k in ("id", "n", "pages", "name", "promotable")} for d in ev_clusters]},
           "real": {"pages": len(reals), "items": len(items), "candidates": len(cands), "per_page": per_page},
           "clusters": clusters, "params": {k: v for k, v in P.all().items() if k.startswith(_K)}}
    with open(os.path.join(out_dir, "novelty.json"), "w") as f:
        json.dump(E._jsonable(res), f, indent=1, default=str)
    for cl in clusters:
        if cl.get("stub"):
            with open(os.path.join(out_dir, f"{cl['stub']['name']}.family.json"), "w") as f:
                json.dump(E._jsonable(cl["stub"]), f, indent=1)
            with open(os.path.join(out_dir, f"{cl['stub']['name']}.py.txt"), "w") as f:
                f.write(cl["stub"]["python"])
    return res


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Find, cluster and draft scenario families for novel finding types on real pages.")
    ap.add_argument("--real", default=REAL_DIR)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    res = run(a.real, a.workers)
    print(json.dumps({"known": res["known"], "evaluation_sanity": res["evaluation_sanity"],
                      "real": {k: v for k, v in res["real"].items() if k != "per_page"},
                      "clusters": [{k: d[k] for k in ("id", "n", "pages", "name", "promotable", "nearest_known_type")} for d in res["clusters"]]},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
