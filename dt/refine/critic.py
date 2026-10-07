"""Critic: turns a :class:`dt.compare.LossReport` into concrete, testable edit hypotheses.

Every hypothesis is derived programmatically from the target screenshot, the current render
and the IR (phase correlation, edge profiles, colour sampling, OCR, shape-mask fitting). The
critic never decides anything by itself: it proposes, the optimizer renders and keeps only
what lowers the loss.

Kinds (``Hypothesis.kind``):

* ``shift``   per-node phase correlation of target vs render crops -> integer (dx, dy)
* ``resize``  edge search: for each side, where the fill/stroke colour ends in the target
* ``recolor`` dominant target colour under the pixels the render paints with the node colour
* ``text``    re-OCR of the target region -> characters, or size/weight (+ re-fitted box)
* ``radius``  shape-scale candidates scored by predicted-corner ΔE against the target
* ``missing`` residual region covered by no leaf -> ``perceive_region`` -> inserted nodes
* ``extra``   node region flat in the target (and painted like what is behind it) -> delete
* ``opacity`` fill looks like a blend of the fill and what is behind it -> set opacity
* ``shadow``  halo under the node present/absent in the target -> add / remove effects

Each hypothesis carries ``region`` (the pixels it may change) so the optimizer can batch
independent ones, and ``expected_gain``: the share of the total loss inside that region times
a kind-specific confidence (an upper bound on what accepting it could save).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from dt.common.image import crop, dominant_color
from dt.compare.loss import LossReport
from dt.compare.pixel import alignment_offset
from dt.compare.structural import normalize_text, text_cer
from dt.ir import Box, Color, Document, Fill, Node, Shadow, TextStyle
from dt.params import P, register

# --------------------------------------------------------------------------- params
register("refine.critic.max_nodes", 24, "Rank at most this many worst nodes per critique.", (4, 128))
register("refine.critic.min_bad_px", 24, "Nodes with at least this many bad pixels inside their box are always candidates.", (4, 400))
register("refine.critic.min_node_de", 0.25, "Ignore nodes whose mean ΔE (own or inside the box) is below this.", (0.05, 10.0))
register("refine.critic.shift_pad", 6, "Padding (px) around a node box for the shift crops.", (2, 24))
register("refine.critic.shift_min_resp", 0.05, "Minimum phase-correlation response to propose a shift.", (0.0, 0.5))
register("refine.critic.shift_max", 24, "Largest shift (px) the critic proposes.", (4, 100))
register("refine.critic.resize_max", 4, "Edge search range (px) on each side for resize.", (1, 12))
register("refine.critic.fill_tol", 3.5, "ΔE below which a pixel counts as 'painted with' a colour.", (1.0, 12.0))
register("refine.critic.min_contrast", 1.0, "Minimum ΔE between a node colour and its backdrop for edge/radius search to be meaningful.", (0.2, 6.0))
register("refine.critic.edge_frac", 0.5, "Fraction of an edge strip that must match for the edge to be there.", (0.2, 0.9))
register("refine.critic.recolor_min_de", 2.0, "Minimum ΔE between the current and sampled colour to propose a recolor.", (0.5, 10.0))
register("refine.critic.min_mask_px", 12, "Minimum number of sample pixels for colour estimates.", (4, 200))
register("refine.critic.text_pad", 4, "Padding (px) around a text box before re-OCR.", (0, 16))
register("refine.critic.text_size_tol", 0.75, "Propose a size change only if OCR disagrees by more than this (px).", (0.2, 3.0))
register("refine.critic.text_size_snap", 0.5, "Snap an OCR size to the nearest integer when within this (px).", (0.0, 0.5))
register("refine.critic.text_size_snap_doc", 1.0, "Snap an OCR size to a size already used in the document when within this (px).", (0.0, 2.0))
register("refine.critic.tracking_min", 0.1, "Propose letter-spacing only if the ink-width estimate differs from the current value by more than this (px).", (0.02, 1.0))
register("refine.critic.text_max_cer", 0.6, "Re-OCR text with a higher CER is a different element, not a correction.", (0.2, 1.0))
register("refine.critic.radius_candidates", [0, 4, 8, 12, 16, 20, 28], "Shape-scale radii tried in addition to min(w,h)/2.")
register("refine.critic.radius_min_gain", 0.1, "Relative corner-cost improvement needed to propose a radius.", (0.0, 0.8))
register("refine.critic.missing_cover", 0.5, "A residual region overlapped by leaves above this fraction is not 'missing'.", (0.1, 0.9))
register("refine.critic.missing_pad", 3, "Padding (px) around a residual region before perceive_region.", (0, 12))
register("refine.critic.missing_merge", 12, "Residual regions closer than this (px) are perceived together (word gaps, icon+label).", (0, 40))
register("refine.critic.max_missing", 3, "Perceive at most this many residual regions per critique.", (0, 12))
register("refine.critic.flat_std", 2.5, "RGB std below which a target region counts as flat (for 'extra').", (0.5, 12.0))
register("refine.critic.behind_ring", 8, "Width (px) of the ring outside a node sampled for the backdrop colour.", (2, 24))
register("refine.critic.min_err_mass", 100.0, "Nodes whose box holds at least this summed ΔE are candidates (catches low-contrast errors).", (10.0, 2000.0))
register("refine.critic.shadow_ring", 6, "Height (px) of the strip under a node probed for a shadow halo.", (2, 16))
register("refine.critic.shadow_min_de", 3.0, "Mean ΔE in the under-strip (target vs behind colour) that signals a shadow.", (1.0, 15.0))
register("refine.critic.opacity_range", [0.1, 0.9], "Blend factors inside this range become opacity hypotheses.")

register("refine.critic.text_verified_de", 1.5, "text whose style perception already render-verified (meta.font_de <= this) gets content fixes only, no style/box moves", (0.0, 10.0))
register("refine.raster.enabled", True, "propose replacing badly explained, text-free subtrees by an image of the target crop (as-is fallback)")
register("refine.raster.min_side", 20, "smallest raster (px per side)", (8, 64))
register("refine.raster.max_frac", 0.25, "largest single raster as a fraction of the screen", (0.02, 0.6))
register("refine.raster.budget_frac", 0.35, "total rasterised area allowed as a fraction of the screen", (0.0, 0.8))
register("refine.raster.max_text_px", 12.0, "subtrees containing text at or above this size are never rasterised (text stays editable)", (6.0, 24.0))
register("refine.raster.min_de", 4.0, "mean ΔE inside the subtree before a raster is proposed", (1.0, 20.0))
register("refine.raster.region_gap", 16, "residual regions closer than this (px) merge into one raster candidate", (0, 64))
register("refine.raster.logo_hue_spread", 90.0, "a text run whose glyph hues span at least this many degrees is a logo (rasterisable)", (30.0, 180.0))
register("refine.raster.min_texture", 0.12, "a raster needs at least this fraction of crop pixels away from its dominant colour (ΔE>5); flat areas get delete/recolor instead", (0.0, 0.6))
register("refine.raster.min_parts", 6, "or: unexplained parts (unnamed icons / small rects) inside the subtree", (2, 50))
register("refine.raster.overlap_frac", 0.3, "a raster may cover at most this fraction of an editable text or component outside the subtree it replaces (the validator's text_under_raster rule)", (0.0, 1.0))
register("refine.raster.logo_bg_dist", 30.0, "RGB distance from the text crop's border median that counts as glyph ink for the logo test", (10.0, 80.0))
register("refine.raster.logo_flat_border", 0.8, "a logo sits on a flat backdrop: this fraction of the text crop's border must be within logo_bg_dist of its median (text over art is not a logo)", (0.3, 1.0))
register("refine.raster.logo_gap", 0.5, "a logo's ink extends past its OCR box over gaps up to this many text heights (OCR boxes clip wordmark letters)", (0.0, 2.0))
register("refine.raster.logo_pad", 4, "px of flat backdrop kept around a logo's ink in its crop", (0, 12))

# M3 elevation level 1 (not a threshold: the default effect added by a 'shadow' hypothesis)
_DEFAULT_SHADOW = [Shadow(Color(0, 0, 0, 0.3), 0, 1, 2, 0), Shadow(Color(0, 0, 0, 0.15), 0, 1, 3, 1)]
_SHAPE_TYPES = ("frame", "rect", "instance", "image", "line")


# --------------------------------------------------------------------------- hypothesis
@dataclass
class Hypothesis:
    """One proposed edit. ``apply`` returns a modified deep copy; the input is never mutated."""

    kind: str
    node_id: Optional[str]
    params: dict
    expected_gain: float
    region: Box
    _apply: Callable[[Document], None] = field(repr=False, compare=False)
    # mutually exclusive variants of this edit (e.g. "refit the box" vs "keep the box"): the optimizer
    # scores this hypothesis and each alternative from the SAME base state and accepts only the best
    alternatives: list["Hypothesis"] = field(default_factory=list, repr=False, compare=False)

    def apply(self, doc: Document) -> Document:
        new = copy.deepcopy(doc)
        self._apply(new)
        return new

    def describe(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.params.items() if not k.startswith("_"))
        return f"{self.kind}({self.node_id}; {p}) gain~{self.expected_gain:.4f}"


# --------------------------------------------------------------------------- helpers
def _de(rgb: np.ndarray, c: Color) -> np.ndarray:
    """Per-pixel CIE76 ΔE of an RGB array against one colour."""
    if rgb.size == 0:
        return np.zeros(rgb.shape[:2], np.float32)
    lab = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3], dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    ref = np.array(c.lab(), np.float32)
    d = lab - ref
    return np.sqrt((d * d).sum(axis=-1))


def _clip(b: Box, W: int, H: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = b.as_int()
    return max(0, x0), max(0, y0), min(W, x1), min(H, y1)


def _box_sum(integral: np.ndarray, b: Box) -> float:
    H, W = integral.shape[0] - 1, integral.shape[1] - 1
    x0, y0, x1, y1 = _clip(b, W, H)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])


def _mask_pixels(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pixels of ``rgb`` where ``mask`` is True, as an (N, 1, 3) array for ``dominant_color``."""
    px = rgb[mask]
    return px.reshape(-1, 1, 3)


def _behind_color(rendered: np.ndarray, box: Box, ring: Optional[int] = None) -> Optional[Color]:
    """Dominant colour of the render in a ring outside ``box``: what the node is painted over.

    The ring is ``refine.critic.behind_ring`` px wide and skips the first pixel (anti-aliased
    edge); a shadow gradient spreads over many quantised colours, so the dominant one is the
    true backdrop."""
    H, W = rendered.shape[:2]
    ring = int(P["refine.critic.behind_ring"] if ring is None else ring)
    outer = _clip(box.expand(ring + 1), W, H)
    inner = _clip(box.expand(1), W, H)
    m = np.zeros((H, W), bool)
    m[outer[1]:outer[3], outer[0]:outer[2]] = True
    m[inner[1]:inner[3], inner[0]:inner[2]] = False
    if m.sum() < int(P["refine.critic.min_mask_px"]):
        return None
    return dominant_color(_mask_pixels(rendered, m))


def _node_color(n: Node) -> Optional[Color]:
    if n.type == "text":
        return n.text_style.color if n.text_style else None
    c = n.fill_color
    if c is None and n.strokes:
        c = n.strokes[0].color
    return c


def _children_mask(n: Node, W: int, H: int) -> np.ndarray:
    m = np.zeros((H, W), bool)
    for c in n.children:
        if c.visible:
            x0, y0, x1, y1 = _clip(c.box, W, H)
            if x1 > x0 and y1 > y0:
                m[y0:y1, x0:x1] = True
    return m


class _Ctx:
    """Per-critique precomputation: diff/bad-pixel integral images and image size."""

    def __init__(self, doc: Document, target: np.ndarray, rendered: np.ndarray, report: LossReport) -> None:
        self.doc, self.target, self.rendered, self.report = doc, target, rendered, report
        self.H, self.W = target.shape[:2]
        diff = report.diff_map
        if diff is None:
            from dt.compare.pixel import diff_map
            diff = diff_map(target, rendered)
        self.diff = diff
        bad = (diff > float(P["compare.pixel.bad_de"])).astype(np.float32)
        self.diff_int = cv2.integral(np.ascontiguousarray(diff, dtype=np.float32), sdepth=cv2.CV_64F)
        self.bad_int = cv2.integral(bad, sdepth=cv2.CV_64F)

    def share(self, box: Box) -> float:
        """Share of the total loss attributable to ``box`` (pixel + bad-fraction terms)."""
        area = float(self.W * self.H)
        if area <= 0:
            return 0.0
        de_norm = float(P["compare.loss.de_norm"])
        pix = float(P["compare.loss.w_pixel"]) * _box_sum(self.diff_int, box) / (area * de_norm)
        bad = float(P["compare.loss.w_bad"]) * _box_sum(self.bad_int, box) / area
        return pix + bad

    def tcrop(self, box: Box, pad: int = 0) -> np.ndarray:
        return crop(self.target, box, pad)

    def rcrop(self, box: Box, pad: int = 0) -> np.ndarray:
        return crop(self.rendered, box, pad)


# --------------------------------------------------------------------------- edit closures
def _set_box(node_id: str, box: Box, move_children: tuple[float, float] | None) -> Callable[[Document], None]:
    def f(doc: Document) -> None:
        n = doc.find(node_id)
        if n is None:
            return
        if move_children is not None:
            dx, dy = move_children
            for m in n.walk():
                if m is not n:
                    m.box = m.box.translate(dx, dy)
        n.box = box

    return f


def _set_fill(node_id: str, color: Color) -> Callable[[Document], None]:
    def f(doc: Document) -> None:
        n = doc.find(node_id)
        if n is None:
            return
        if n.type == "text":
            if n.text_style is None:
                n.text_style = TextStyle()
            n.text_style.color = color
        elif n.fills:
            n.fills[0] = Fill.solid(color)
        elif n.strokes and n.type != "icon":
            n.strokes[0].color = color
        else:
            n.fills = [Fill.solid(color)]

    return f


def _set_attr(node_id: str, **attrs: object) -> Callable[[Document], None]:
    def f(doc: Document) -> None:
        n = doc.find(node_id)
        if n is None:
            return
        for k, v in attrs.items():
            setattr(n, k, copy.deepcopy(v))

    return f


def _set_text(node_id: str, text: Optional[str], style: Optional[dict], box: Optional[Box]) -> Callable[[Document], None]:
    def f(doc: Document) -> None:
        n = doc.find(node_id)
        if n is None:
            return
        if text is not None:
            n.text = text
        if style:
            if n.text_style is None:
                n.text_style = TextStyle()
            for k, v in style.items():
                setattr(n.text_style, k, v)
        if box is not None:
            n.box = box

    return f


def _delete(node_id: str) -> Callable[[Document], None]:
    def f(doc: Document) -> None:
        p = doc.parent_of(node_id)
        if p is not None:
            p.children = [c for c in p.children if c.id != node_id]

    return f


def _stable_ids(doc: Document, nodes: list[Node], salt: str) -> None:
    """Give inserted nodes ids derived from what they are (not uuid4), so a refine run is repeatable:
    the same input yields the same history and document. Unique within ``doc``."""
    import hashlib
    taken = {m.id for m in doc.walk()}
    i = 0
    for top in nodes:
        for m in top.walk():
            prefix = m.id.split("_", 1)[0] if "_" in m.id else "n"
            key = f"{salt}|{i}|{m.type}|{m.box.x:.2f},{m.box.y:.2f},{m.box.w:.2f},{m.box.h:.2f}|{m.text or ''}"
            nid = f"{prefix}_{hashlib.sha1(key.encode()).hexdigest()[:8]}"
            while nid in taken:
                nid += "x"
            m.id = nid
            taken.add(nid)
            i += 1


def _insert(parent_id: str, nodes: list[Node]) -> Callable[[Document], None]:
    def f(doc: Document) -> None:
        p = doc.find(parent_id) or doc.root
        new = copy.deepcopy(nodes)
        _stable_ids(doc, new, f"missing|{parent_id}")
        p.children.extend(new)

    return f


# --------------------------------------------------------------------------- shift
def _shift_cost(t: np.ndarray, r: np.ndarray, dx: int, dy: int) -> float:
    """Mean |target - rendered| after undoing a rendered displacement of (dx, dy)."""
    h, w = t.shape[:2]
    if abs(dx) >= w or abs(dy) >= h:
        return float("inf")
    tx0, tx1 = max(0, -dx), min(w, w - dx)
    ty0, ty1 = max(0, -dy), min(h, h - dy)
    a = t[ty0:ty1, tx0:tx1].astype(np.float32)
    b = r[ty0 + dy:ty1 + dy, tx0 + dx:tx1 + dx].astype(np.float32)
    return float(np.abs(a - b).mean()) if a.size else float("inf")


def _refine_offset(t: np.ndarray, r: np.ndarray, dx: int, dy: int) -> tuple[int, int]:
    """Snap a phase-correlation estimate to the integer offset that best explains the crops
    (3x3 neighbourhood search), so sub-pixel rounding errors do not become 1px misplacements."""
    t, r = t[: min(t.shape[0], r.shape[0]), : min(t.shape[1], r.shape[1])], r[: min(t.shape[0], r.shape[0]), : min(t.shape[1], r.shape[1])]
    best, best_cost = (dx, dy), float("inf")
    for j in (-1, 0, 1):
        for i in (-1, 0, 1):
            c = _shift_cost(t, r, dx + i, dy + j)
            if c < best_cost - 1e-6:
                best, best_cost = (dx + i, dy + j), c
    return best


def _shift_hyps(ctx: _Ctx, n: Node) -> list[Hypothesis]:
    pad = int(P["refine.critic.shift_pad"])
    t, r = ctx.tcrop(n.box, pad), ctx.rcrop(n.box, pad)
    if t.size == 0 or r.size == 0:
        return []
    dx, dy, resp = alignment_offset(t, r)
    dx, dy = _refine_offset(t, r, int(round(dx)), int(round(dy)))
    mx = int(P["refine.critic.shift_max"])
    if resp < float(P["refine.critic.shift_min_resp"]) or (dx == 0 and dy == 0) or max(abs(dx), abs(dy)) > mx:
        return []
    new = n.box.translate(-dx, -dy)
    region = n.box.union(new)
    gain = ctx.share(region) * resp
    out = [Hypothesis("shift", n.id, {"dx": -dx, "dy": -dy, "subtree": True}, gain, region, _set_box(n.id, new, (-dx, -dy)))]
    if n.children:  # maybe only the container moved and its children are right
        out.append(Hypothesis("shift", n.id, {"dx": -dx, "dy": -dy, "subtree": False}, gain * 0.5, region, _set_box(n.id, new, None)))
    return out


# --------------------------------------------------------------------------- resize
def _edge_profile(ctx: _Ctx, n: Node, color: Color, side: str, rng: int, tol: float) -> Optional[int]:
    """Offset (px, outward positive) of the target's edge for ``side``, or None if undetectable.

    Looks at strips parallel to the side, from ``rng`` px inside to ``rng`` px outside, over the
    middle of the side (corners/radii excluded), and finds the last strip still painted with
    ``color`` in the target.
    """
    X0, Y0, X1, Y1 = n.box.as_int()
    rad = max(n.radius) if n.type != "ellipse" else min(n.box.w, n.box.h) / 2
    inset = int(max(rad, 0.25 * (n.box.h if side in ("left", "right") else n.box.w)))
    if side in ("left", "right"):
        a0, a1 = Y0 + inset, Y1 - inset
    else:
        a0, a1 = X0 + inset, X1 - inset
    if a1 <= a0:
        a0, a1 = (Y0 + Y1) // 2, (Y0 + Y1) // 2 + 1
        if side in ("top", "bottom"):
            a0, a1 = (X0 + X1) // 2, (X0 + X1) // 2 + 1
    a0, a1 = max(0, a0), min(ctx.H if side in ("left", "right") else ctx.W, a1)
    if a1 <= a0:
        return None
    frac = float(P["refine.critic.edge_frac"])
    is_fill = n.fill_color is not None and n.type != "line"
    behind = _behind_color(ctx.rendered, n.box)
    if behind is not None and behind.delta_e(color) < float(P["refine.critic.min_contrast"]):
        return None  # node colour equals what is behind it: no edge to find

    def painted(strip: np.ndarray) -> np.ndarray:
        d = _de(strip, color)
        return d < _de(strip, behind) if behind is not None else d < tol

    ks = list(range(-rng, rng + 1))
    m: dict[int, float] = {}
    for k in ks:
        if side == "right":
            c = X1 - 1 + k
            strip = ctx.target[a0:a1, c:c + 1] if 0 <= c < ctx.W else None
        elif side == "left":
            c = X0 - k
            strip = ctx.target[a0:a1, c:c + 1] if 0 <= c < ctx.W else None
        elif side == "bottom":
            c = Y1 - 1 + k
            strip = ctx.target[c:c + 1, a0:a1] if 0 <= c < ctx.H else None
        else:
            c = Y0 - k
            strip = ctx.target[c:c + 1, a0:a1] if 0 <= c < ctx.H else None
        if strip is None or strip.size == 0:
            m[k] = 0.0
        else:
            m[k] = float(painted(strip).mean())
    matched = [k for k in ks if m[k] > frac]
    if is_fill:
        # The fill must be continuous from inside the node to the edge. When the whole window
        # is painted (edge further out) or unpainted (edge further in), step by the window so
        # the optimizer converges iteratively instead of giving up.
        if m[-rng] <= frac:
            return -rng
        kmax = -rng
        while kmax + 1 <= rng and m[kmax + 1] > frac:
            kmax += 1
        return kmax
    if m[rng] > frac or not matched:  # stroke continues past the window / no stroke in the window
        return None
    return max(matched)


def _resize_hyps(ctx: _Ctx, n: Node) -> list[Hypothesis]:
    if n.type not in _SHAPE_TYPES or n.box.w < 2 or n.box.h < 2:
        return []
    color = _node_color(n)
    if color is None:
        return []
    rng = int(P["refine.critic.resize_max"])
    tol = float(P["refine.critic.fill_tol"])
    d = {s: (_edge_profile(ctx, n, color, s, rng, tol) or 0) for s in ("left", "top", "right", "bottom")}
    if n.type == "line":  # a 1px divider: only its length is searchable
        if n.box.h <= 2:
            d["top"] = d["bottom"] = 0
        if n.box.w <= 2:
            d["left"] = d["right"] = 0
    if not any(d.values()):
        return []
    b = n.box
    new = Box(b.x - d["left"], b.y - d["top"], b.w + d["left"] + d["right"], b.h + d["top"] + d["bottom"])
    if new.w < 1 or new.h < 1:
        return []
    region = b.union(new).expand(1)
    gain = ctx.share(region) * 0.7
    return [Hypothesis("resize", n.id, dict(d), gain, region, _set_box(n.id, new, None))]


# --------------------------------------------------------------------------- recolor / opacity
def _paint_mask(ctx: _Ctx, n: Node, color: Color) -> np.ndarray:
    """Pixels inside the node box (children excluded) that the render paints with ``color``."""
    x0, y0, x1, y1 = _clip(n.box, ctx.W, ctx.H)
    m = np.zeros((ctx.H, ctx.W), bool)
    if x1 <= x0 or y1 <= y0:
        return m
    sub = ctx.rendered[y0:y1, x0:x1]
    m[y0:y1, x0:x1] = _de(sub, color) < float(P["refine.critic.fill_tol"])
    m &= ~_children_mask(n, ctx.W, ctx.H)
    return m


def _recolor_hyps(ctx: _Ctx, n: Node) -> list[Hypothesis]:
    cur = _node_color(n)
    if cur is None:
        return []
    m = _paint_mask(ctx, n, cur)
    if m.sum() < int(P["refine.critic.min_mask_px"]):
        return []
    px = _mask_pixels(ctx.target, m)
    new = dominant_color(px)
    if new is None or new.delta_e(cur) < float(P["refine.critic.recolor_min_de"]):
        return []
    if (not n.children or n.type == "text") and n.id != ctx.doc.root.id:
        behind = _behind_color(ctx.rendered, n.box)
        if behind is not None and new.delta_e(behind) < float(P["refine.critic.fill_tol"]):
            return []  # painting a leaf with its backdrop just hides it: that is an 'extra' node
    consistency = float((_de(px, new) < float(P["refine.critic.fill_tol"])).mean())
    out = [Hypothesis("recolor", n.id, {"from": cur.hex(), "to": new.hex()}, ctx.share(n.box) * consistency, n.box, _set_fill(n.id, new))]
    out += _opacity_hyps(ctx, n, cur, new)
    return out


def _opacity_hyps(ctx: _Ctx, n: Node, cur: Color, sampled: Color) -> list[Hypothesis]:
    """If the sampled target colour is a blend of the fill and what is behind it, propose opacity."""
    if n.type == "text" or n.id == ctx.doc.root.id:
        return []
    out: list[Hypothesis] = []
    if n.opacity < 1.0:
        out.append(Hypothesis("opacity", n.id, {"opacity": 1.0}, ctx.share(n.box) * 0.3, n.box, _set_attr(n.id, opacity=1.0)))
    behind = _behind_color(ctx.rendered, n.box)
    if behind is None:
        return out
    c, b, s = (np.array([v.r, v.g, v.b], np.float32) for v in (cur, behind, sampled))
    seg = c - b
    L2 = float(seg @ seg)
    if L2 < 1.0:
        return out
    t = float(((s - b) @ seg) / L2)
    lo, hi = P["refine.critic.opacity_range"]
    resid = float(np.linalg.norm(s - (b + t * seg)))
    if lo <= t <= hi and resid < float(P["refine.critic.fill_tol"]):
        a = round(t * 20) / 20.0
        out.append(Hypothesis("opacity", n.id, {"opacity": a}, ctx.share(n.box) * 0.5, n.box, _set_attr(n.id, opacity=a)))
    return out


# --------------------------------------------------------------------------- text
def _ocr(rgb: np.ndarray):
    """(lines, paragraphs) of a crop via dt.perceive.ocr, or ([], []) when OCR is unavailable."""
    try:
        from dt.perceive.ocr import group_paragraphs, ocr_lines
    except Exception:
        return [], []
    try:
        lines = ocr_lines(np.ascontiguousarray(rgb))
        return lines, group_paragraphs(rgb, lines)
    except Exception:
        return [], []


def _text_box_from_ink(n: Node, size: float, line_height: Optional[float], ink: Box, baseline: float, n_lines: int) -> Box:
    """Box that makes the renderer put the first baseline at ``baseline`` (see perceive.ocr.paragraph_node)."""
    normal = float(P["perceive.ocr.line_height_ratio"]) * size
    L = line_height if line_height else normal
    ascent = float(P["perceive.ocr.ascent_ratio"]) * size
    lsb = float(P["perceive.ocr.lsb_ratio"]) * size
    y = baseline - ascent - (L - normal) / 2.0
    h = n_lines * L
    w = max(n.box.w, ink.w + 2 * lsb)
    align = n.text_style.align if n.text_style else "left"
    if align == "center":
        x = ink.cx - w / 2
    elif align == "right":
        x = ink.x2 + lsb - w
    else:
        x = ink.x - lsb
    return Box(round(x, 1), round(y, 1), round(w, 1), round(h, 1))


def _doc_sizes(doc: Document) -> list[float]:
    return sorted({t.text_style.size for t in doc.walk() if t.type == "text" and t.text_style})


def _tracking_estimate(ctx: _Ctx, n: Node, target_ink_w: float) -> Optional[float]:
    """Letter spacing that makes the rendered ink as wide as the target ink (single-line text)."""
    if n.text is None or "\n" in n.text or len(n.text) < 2 or n.text_style is None:
        return None
    m = _paint_mask(ctx, n, n.text_style.color)
    cols = np.where(m.any(axis=0))[0]
    if cols.size < 2:
        return None
    rendered_ink_w = float(cols[-1] - cols[0] + 1)
    ls = n.text_style.letter_spacing + (target_ink_w - rendered_ink_w) / (len(n.text) - 1)
    return round(ls * 20) / 20.0


def _snap_size(size: float, doc_sizes: list[float] = ()) -> float:
    """Snap an OCR size estimate to a size the document already uses (typescale consistency),
    else to the nearest integer; otherwise keep one decimal."""
    near = [s for s in doc_sizes if abs(s - size) <= float(P["refine.critic.text_size_snap_doc"])]
    if near:
        return float(min(near, key=lambda s: abs(s - size)))
    snap = float(P["refine.critic.text_size_snap"])
    return float(round(size)) if abs(size - round(size)) <= snap else round(size, 1)


def _text_hyps(ctx: _Ctx, n: Node) -> list[Hypothesis]:
    if n.type != "text" or not n.text:
        return []
    pad = int(P["refine.critic.text_pad"])
    sub = ctx.tcrop(n.box, pad)
    if sub.size == 0:
        return []
    x0, y0, _, _ = n.box.expand(pad).as_int()
    x0, y0 = max(0, x0), max(0, y0)
    _, paras = _ocr(sub)
    if not paras:
        return []
    best, best_ov = None, 0.0
    for p in paras:
        ov = p.box.translate(x0, y0).intersect(n.box).area
        if ov > best_ov:
            best, best_ov = p, ov
    if best is None or best_ov <= 0:
        return []
    out: list[Hypothesis] = []
    share = ctx.share(n.box.expand(pad))
    cur_txt, new_txt = normalize_text(n.text), normalize_text(best.text)
    if new_txt and new_txt != cur_txt and text_cer(new_txt, cur_txt) <= float(P["refine.critic.text_max_cer"]):
        text = "\n".join(l.text for l in best.lines) if "\n" in (n.text or "") or len(best.lines) > 1 else best.text
        out.append(Hypothesis("text", n.id, {"text": text}, share * 0.9, n.box.expand(pad), _set_text(n.id, text, None, None)))
    st = n.text_style or TextStyle()
    fde = n.meta.get("font_de")
    if fde is not None and float(fde) <= float(P["refine.critic.text_verified_de"]):
        return out  # style and box were chosen by rendering against this crop in perception
    size = _snap_size(best.style.size, _doc_sizes(ctx.doc))
    style: dict = {}
    if abs(size - st.size) > float(P["refine.critic.text_size_tol"]):
        style["size"] = size
    if best.style.weight != st.weight:
        style["weight"] = best.style.weight
    ls = _tracking_estimate(ctx, n, best.box.translate(x0, y0).w)
    if ls is not None and abs(ls - st.letter_spacing) > float(P["refine.critic.tracking_min"]):
        reg = n.box.expand(pad)
        out.append(Hypothesis("text", n.id, {"letter_spacing": ls}, ctx.share(reg) * 0.5, reg, _set_text(n.id, None, {"letter_spacing": ls}, None)))
    if style:
        ink = best.box.translate(x0, y0)
        base = best.lines[0].baseline + y0
        box = _text_box_from_ink(n, style.get("size", st.size), st.line_height, ink, base, len(best.lines))
        region = n.box.union(box).expand(pad)
        refit = Hypothesis("text", n.id, dict(style, box=box.to_dict()), ctx.share(region) * 0.8, region, _set_text(n.id, None, style, box))
        keep = Hypothesis("text", n.id, dict(style), ctx.share(region) * 0.6, region, _set_text(n.id, None, style, None))
        # anchor-preserving variant: OCR ink boxes carry sub-pixel noise, the existing box origin is often exact
        snapped = Box(round(box.x), round(box.y), box.w, box.h)
        snap = Hypothesis("text", n.id, dict(style, box=snapped.to_dict()), ctx.share(region) * 0.7, region, _set_text(n.id, None, style, snapped))
        refit.alternatives = [keep, snap]
        out.append(refit)
    return out


# --------------------------------------------------------------------------- raster (as-is fallback)
def _png_data_uri(rgb: np.ndarray) -> str:
    import base64
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8)).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def rasterised_area(doc: Document) -> float:
    return sum(n.box.area for n in doc.walk() if n.type == "image" and n.meta.get("rasterised"))


def _editable(m: Node, logo_id: Optional[str] = None) -> bool:
    """Nodes a crop must never replace or cover: components and real (not tiny, not logo) text."""
    if m.component is not None:
        return True
    return m.type == "text" and bool(m.text and m.text.strip()) and m.id != logo_id and not m.meta.get("logo") \
        and (m.text_style is None or m.text_style.size >= float(P["refine.raster.max_text_px"]))


def _raster_ok(n: Node, doc: Document, logo_id: Optional[str] = None) -> bool:
    if n.id == doc.root.id or n.component is not None or n.meta.get("rasterised"):
        return False
    return not any(_editable(m, logo_id) for m in n.walk())


def _ancestors(doc: Document, node_id: str) -> list[Node]:
    out, p = [], doc.parent_of(node_id)
    while p is not None:
        out.append(p)
        p = doc.parent_of(p.id)
    return out


def _covers_editable(doc: Document, box: Box, inside_ids: set[str], logo_id: Optional[str] = None) -> bool:
    """True when a crop at ``box`` would sit inside a component or hide (bake in) editable text / a component
    that it does not replace: the crop keeps their pixels, so the loss cannot see the editability loss."""
    frac = float(P["refine.raster.overlap_frac"])
    for m in doc.walk():
        if m.id in inside_ids or not m.visible:
            continue
        if m.component is not None and m.box.contains(box, tol=1.0):
            return True  # the crop would land inside the instance (export drops instance children)
        if _editable(m, logo_id) and m.box.intersect(box).area > frac * max(1.0, m.box.area):
            return True
    return False


def _within_budget(doc: Document, box: Box, limit: float, replaced: float = 0.0) -> bool:
    return rasterised_area(doc) - replaced + box.area <= limit + 1e-6


def _glyph_hue_spread(ctx: _Ctx, n: Node) -> float:
    """Degrees spanned by the hues of the strongly coloured glyph pixels of a text node (0 for mono text)."""
    from skimage import color as skcolor
    sub = ctx.tcrop(n.box)
    if sub.size == 0:
        return 0.0
    border = np.concatenate([sub[0], sub[-1], sub[:, 0], sub[:, -1]]).reshape(-1, 3).astype(np.float32)
    bg = np.median(border, axis=0)
    ink = float(P["refine.raster.logo_bg_dist"])
    # glyphs are segmented against the border colour, which only works on a flat backdrop: text over
    # an illustration/photo would take the backdrop's hues for its own
    if float((np.sqrt(((border - bg) ** 2).sum(-1)) < ink).mean()) < float(P["refine.raster.logo_flat_border"]):
        return 0.0
    px = sub[..., :3].reshape(-1, 3).astype(np.float32)
    d = np.sqrt(((px - bg) ** 2).sum(-1))
    if d.size == 0 or d.max() < ink:
        return 0.0
    glyph = px[d > 0.6 * d.max()]
    lab = skcolor.rgb2lab(glyph.reshape(1, -1, 3) / 255.0).reshape(-1, 3)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    col = lab[chroma > 25]
    if len(col) < max(12, 0.15 * len(glyph)):
        return 0.0
    hues = np.degrees(np.arctan2(col[:, 2], col[:, 1])) % 360
    hist, _ = np.histogram(hues, bins=12, range=(0, 360))
    strong = np.where(hist >= 0.08 * len(hues))[0]
    if len(strong) < 2:
        return 0.0
    centres = strong * 30 + 15
    spread = 0.0
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            dd = abs(centres[i] - centres[j]) % 360
            spread = max(spread, min(dd, 360 - dd))
    return float(spread)


def _textured(crop_rgb: np.ndarray) -> bool:
    """True when the crop has real texture (an image/illustration), not a flat fill."""
    if crop_rgb.size == 0:
        return False
    from dt.common.image import dominant_color
    from dt.validate.fidelity import delta_e2000_map
    dom = dominant_color(crop_rgb)
    if dom is None:
        return False
    flat = np.empty_like(crop_rgb[..., :3])
    flat[...] = (dom.r, dom.g, dom.b)
    return float((delta_e2000_map(crop_rgb[..., :3], flat) > 5.0).mean()) >= float(P["refine.raster.min_texture"])


def _remove_and_insert_raster(remove_ids: list[str], parent_id: str, box: Box, uri: str, limit: float) -> Callable[[Document], None]:
    def f(doc: Document) -> None:
        ids = set(remove_ids)
        replaced = sum(m.box.area for m in doc.walk() if m.id in ids and m.type == "image" and m.meta.get("rasterised"))
        if not _within_budget(doc, box, limit, replaced):
            return  # another raster of the same batch used the budget up: no-op (the optimizer drops it)
        for n in list(doc.walk()):
            n.children = [c for c in n.children if c.id not in ids]
        parent = doc.find(parent_id) or doc.root
        img = Node(type="image", name="raster", box=box, image_ref=uri, meta={"rasterised": True, "rasterised_parts": len(ids)})
        _stable_ids(doc, [img], f"raster|{parent_id}")
        parent.children.append(img)
    return f


def _raster_region_hyps(ctx: _Ctx) -> list[Hypothesis]:
    """Residual regions that no single node covers (illustrations split into many siblings): replace every
    node inside the region by one exact crop, if all of them are rasterisable and the region is self-contained."""
    if not bool(P["refine.raster.enabled"]):
        return []
    screen = float(ctx.W * ctx.H)
    used = rasterised_area(ctx.doc)
    out: list[Hypothesis] = []
    regions = _merge_regions(list(ctx.report.residual_regions), float(P["refine.raster.region_gap"]))
    for r in regions:
        x0, y0, x1, y1 = _clip(r, ctx.W, ctx.H)
        box = Box(x0, y0, x1 - x0, y1 - y0)
        if min(box.w, box.h) < int(P["refine.raster.min_side"]) or box.area > float(P["refine.raster.max_frac"]) * screen:
            continue
        if used + box.area > float(P["refine.raster.budget_frac"]) * screen:
            continue
        if _box_sum(ctx.diff_int, box) / max(1.0, box.area) < float(P["refine.raster.min_de"]):
            continue
        inside, ok = [], True
        for n in ctx.doc.walk():
            if n.id == ctx.doc.root.id or not n.visible or n.box.intersect(box).area <= 0:
                continue
            if n.box.contains(box, tol=1.0):
                # a plain container around the region: keep it (never a component: its children are not exported)
                if n.type in ("frame", "rect", "ellipse") and n.component is None:
                    continue
                ok = False  # the region lies inside text / an icon / an image: never paint a crop over it
                break
            if not box.contains(n.box, tol=1.0) or not _raster_ok(n, ctx.doc):
                ok = False
                break
            inside.append(n)
        if not ok:
            continue
        if not inside:  # nothing to replace: a crop here would only paint over what is already there
            continue
        top = [n for n in inside if not any(m is not n and n in list(m.walk()) for m in inside)]
        if not _textured(ctx.target[y0:y1, x0:x1]):
            continue
        parent = _deepest_frame(ctx.doc, box)
        if parent.component is not None or any(a.component is not None for a in _ancestors(ctx.doc, parent.id)):
            continue
        uri = _png_data_uri(ctx.target[y0:y1, x0:x1])
        limit = float(P["refine.raster.budget_frac"]) * screen
        out.append(Hypothesis("raster", None, {"box": box.to_dict(), "replaces": len(top), "region": True},
                              ctx.share(box) * 0.95, box, _remove_and_insert_raster([n.id for n in top], parent.id, box, uri, limit)))
    return out


def _grow(profile: np.ndarray, lo: int, hi: int, reach: int) -> tuple[int, int]:
    """Extend the inked span [lo, hi] of a 1-D ink profile over gaps of at most ``reach`` empty entries."""
    for step in (-1, 1):
        gap, c = 0, (lo if step < 0 else hi) + step
        while 0 <= c < len(profile) and gap <= reach:
            if profile[c]:
                lo, hi, gap = (c, hi, 0) if step < 0 else (lo, c, 0)
            else:
                gap += 1
            c += step
    return lo, hi


def _logo_ink_box(ctx: _Ctx, n: Node) -> Optional[Box]:
    """Box of the wordmark's ink around the text node (+ ``logo_pad``): OCR boxes clip logo letters."""
    sub = ctx.tcrop(n.box)
    if sub.size == 0:
        return None
    border = np.concatenate([sub[0], sub[-1], sub[:, 0], sub[:, -1]]).reshape(-1, 3).astype(np.float32)
    bg = np.median(border, axis=0)
    reach = int(round(float(P["refine.raster.logo_gap"]) * max(1.0, n.box.h)))
    pad = int(P["refine.raster.logo_pad"])
    span = int(np.ceil(n.box.h)) * 4 + reach
    wx0, wy0, wx1, wy1 = _clip(n.box.expand(span), ctx.W, ctx.H)
    win = ctx.target[wy0:wy1, wx0:wx1, :3].astype(np.float32)
    ink = np.sqrt(((win - bg) ** 2).sum(-1)) >= float(P["refine.raster.logo_bg_dist"])
    bx0, by0, bx1, by1 = _clip(n.box, ctx.W, ctx.H)
    rows = ink[by0 - wy0:by1 - wy0]
    cols = np.where(rows.any(axis=0)[bx0 - wx0:bx1 - wx0])[0]
    if len(cols) == 0:
        return None
    lo, hi = _grow(rows.any(axis=0), int(cols[0]) + bx0 - wx0, int(cols[-1]) + bx0 - wx0, reach)
    prof = ink[:, lo:hi + 1].any(axis=1)
    rws = np.where(prof[by0 - wy0:by1 - wy0])[0]
    if len(rws) == 0:
        return None
    top, bot = _grow(prof, int(rws[0]) + by0 - wy0, int(rws[-1]) + by0 - wy0, reach)
    x0, y0, x1, y1 = _clip(Box(wx0 + lo - pad, wy0 + top - pad, hi - lo + 1 + 2 * pad, bot - top + 1 + 2 * pad), ctx.W, ctx.H)
    return Box(x0, y0, x1 - x0, y1 - y0)


def _logo_hyps(ctx: _Ctx, n: Node) -> list[Hypothesis]:
    """Multi-coloured text (a wordmark like the Google logo) is an image asset, not editable text."""
    if not bool(P["refine.raster.enabled"]) or n.type != "text" or n.meta.get("rasterised"):
        return []
    if _glyph_hue_spread(ctx, n) < float(P["refine.raster.logo_hue_spread"]):
        return []
    return _raster_hyps(ctx, n, logo=True)  # never mutate ctx.doc: the logo evidence travels in the edit


def _set_raster(node_id: str, box: Box, uri: str, n_parts: int, limit: float, alt_text: Optional[str] = None) -> Callable[[Document], None]:
    def f(doc: Document) -> None:
        n = doc.find(node_id)
        if n is None or not _within_budget(doc, box, limit):
            return  # gone, or another raster of the same batch used the budget up: no-op (the optimizer drops it)
        n.type, n.box, n.image_ref, n.children = "image", box, uri, []
        n.fills, n.strokes, n.effects, n.text, n.text_style, n.icon_name, n.layout = [], [], [], None, None, None, None
        n.radius = (0.0, 0.0, 0.0, 0.0)
        n.meta = dict(n.meta, rasterised=True, rasterised_parts=n_parts)
        if alt_text is not None:  # a wordmark: the crop still shows this text (refine's text term counts it)
            n.meta.update(logo=True, alt_text=alt_text)
    return f


def _raster_hyps(ctx: _Ctx, n: Node, logo: bool = False) -> list[Hypothesis]:
    """Replace a text-free, component-free subtree that structure explains badly with the exact target crop
    (or, with ``logo``, a multi-coloured wordmark text node with the crop of its ink)."""
    logo_id = n.id if logo else None
    if not bool(P["refine.raster.enabled"]) or not _raster_ok(n, ctx.doc, logo_id):
        return []
    if logo:
        box = _logo_ink_box(ctx, n)
        if box is None:
            return []
    else:
        x0, y0, x1, y1 = _clip(n.box, ctx.W, ctx.H)
        box = Box(x0, y0, x1 - x0, y1 - y0)
    x0, y0, x1, y1 = box.as_int()
    screen = float(ctx.W * ctx.H)
    limit = float(P["refine.raster.budget_frac"]) * screen
    if min(box.w, box.h) < int(P["refine.raster.min_side"]) or box.area > float(P["refine.raster.max_frac"]) * screen:
        return []
    if not _within_budget(ctx.doc, box, limit):
        return []
    if _covers_editable(ctx.doc, box, {m.id for m in n.walk()}, logo_id):
        return []
    inside = _box_sum(ctx.diff_int, box) / max(1.0, box.area)
    parts = [m for m in n.walk() if m is not n]
    unexplained = sum(1 for m in parts if (m.type == "icon" and not m.icon_name) or (m.type in ("rect", "frame") and m.box.area < 400))
    if not logo and inside < float(P["refine.raster.min_de"]) and unexplained < int(P["refine.raster.min_parts"]):
        return []
    if not _textured(ctx.target[y0:y1, x0:x1]):
        return []
    uri = _png_data_uri(ctx.target[y0:y1, x0:x1])
    params = {"box": box.to_dict(), "parts": len(parts), "unexplained": unexplained}
    if logo:
        params["logo"] = True
    region = box.union(n.box)
    return [Hypothesis("raster", n.id, params, ctx.share(region) * 0.95, region,
                       _set_raster(n.id, box, uri, len(parts), limit, n.text if logo else None))]


# --------------------------------------------------------------------------- radius
def _corner_mask(size: int, r: float) -> np.ndarray:
    """Boolean mask of a top-left corner of a rounded rect (pixel-centre sampling)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) + 0.5
    inside = np.ones((size, size), bool)
    if r > 0:
        q = (xx < r) & (yy < r)
        inside[q] = ((xx[q] - r) ** 2 + (yy[q] - r) ** 2) <= r * r
    return inside


def _radius_hyps(ctx: _Ctx, n: Node) -> list[Hypothesis]:
    """Score shape-scale radii by the ΔE between the target corners and a predicted corner
    (fill inside the arc, the behind colour outside); propose the best if it clearly beats the
    current radius. Colour-based (not binary) so low-contrast fills still rank correctly."""
    if n.type not in ("frame", "rect", "instance", "image") or min(n.box.w, n.box.h) < 4 or not (n.fills or n.strokes):
        return []
    color = _node_color(n)
    behind = _behind_color(ctx.rendered, n.box)
    if color is None or behind is None or behind.delta_e(color) < float(P["refine.critic.min_contrast"]):
        return []
    half = min(n.box.w, n.box.h) / 2.0
    shape = [float(c) for c in P["refine.critic.radius_candidates"] if c <= half]
    cands = sorted(set(shape) | {float(int(half))})  # shape scale + 'full' (pill / circle)
    size = int(min((max(shape) if shape else 0) + 2, half))  # window: the shape-scale arcs, not the children
    if size < 2:
        return []
    X0, Y0, X1, Y1 = n.box.as_int()
    kids = _children_mask(n, ctx.W, ctx.H)
    corners = [(X0, Y0, False, False), (X1 - size, Y0, True, False), (X1 - size, Y1 - size, True, True), (X0, Y1 - size, False, True)]
    obs = []
    for cx, cy, fx, fy in corners:
        if cx < 0 or cy < 0 or cx + size > ctx.W or cy + size > ctx.H:
            continue
        sub, km = ctx.target[cy:cy + size, cx:cx + size], kids[cy:cy + size, cx:cx + size]
        if fx:
            sub, km = sub[:, ::-1], km[:, ::-1]
        if fy:
            sub, km = sub[::-1, :], km[::-1, :]
        obs.append((np.ascontiguousarray(sub), np.ascontiguousarray(km)))
    if not obs:
        return []
    fill_rgb = np.array([color.r, color.g, color.b], np.uint8)
    behind_rgb = np.array([behind.r, behind.g, behind.b], np.uint8)

    def cost(r: float) -> float:
        mask = _corner_mask(size, r)
        pred = np.where(mask[..., None], fill_rgb, behind_rgb).astype(np.uint8)
        # children pixels are not the corner's business: predict them as-is (zero cost)
        return float(np.mean([_diff_mean(o, np.where(km[..., None], o, pred)) for o, km in obs]))

    cur = min(n.uniform_radius(), half)
    cur_cost = cost(cur)
    scores = {r: cost(r) for r in cands}
    best = min(scores, key=scores.get)
    rel_gain = (cur_cost - scores[best]) / max(cur_cost, 1e-6)
    if abs(best - cur) < 0.5 or rel_gain < float(P["refine.critic.radius_min_gain"]):
        return []
    region = n.box.expand(1)
    return [Hypothesis("radius", n.id, {"radius": best, "from": cur}, ctx.share(region) * min(1.0, rel_gain),
                       region, _set_attr(n.id, radius=(best, best, best, best)))]


def _diff_mean(a: np.ndarray, b: np.ndarray) -> float:
    """Mean CIE76 ΔE between two equally sized RGB crops."""
    la = cv2.cvtColor(np.ascontiguousarray(a, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lb = cv2.cvtColor(np.ascontiguousarray(b, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    for l in (la, lb):
        l[..., 0] *= 100.0 / 255.0
        l[..., 1:] -= 128.0
    d = la - lb
    return float(np.sqrt((d * d).sum(axis=-1)).mean())


# --------------------------------------------------------------------------- shadow
def _shadow_hyps(ctx: _Ctx, n: Node) -> list[Hypothesis]:
    if n.type not in ("frame", "rect", "instance", "ellipse", "image") or n.id == ctx.doc.root.id or not n.fills:
        return []
    ring = int(P["refine.critic.shadow_ring"])
    b = n.box
    strip = Box(b.x + 2, b.y2, max(1.0, b.w - 4), ring)
    t, r = ctx.tcrop(strip), ctx.rcrop(strip)
    if t.size == 0 or r.size == 0:
        return []
    behind = _behind_color(ctx.rendered, b.expand(ring + 1))
    if behind is None:
        return []
    t_halo, r_halo = float(_de(t, behind).mean()), float(_de(r, behind).mean())
    thr = float(P["refine.critic.shadow_min_de"])
    region = b.expand(ring + 2)
    if n.effects and t_halo < thr <= r_halo:
        return [Hypothesis("shadow", n.id, {"effects": "remove"}, ctx.share(region) * 0.7, region, _set_attr(n.id, effects=[]))]
    if not n.effects and r_halo < thr <= t_halo:
        return [Hypothesis("shadow", n.id, {"effects": "add"}, ctx.share(region) * 0.5, region, _set_attr(n.id, effects=list(_DEFAULT_SHADOW)))]
    return []


# --------------------------------------------------------------------------- extra
def _extra_hyps(ctx: _Ctx, n: Node) -> list[Hypothesis]:
    if n.id == ctx.doc.root.id:
        return []
    t = ctx.tcrop(n.box)
    if t.size == 0 or float(t.reshape(-1, 3).std(axis=0).max()) > float(P["refine.critic.flat_std"]):
        return []
    flat = Color.from_rgb(*t.reshape(-1, 3).mean(axis=0))
    cur = _node_color(n)
    tol = float(P["refine.critic.fill_tol"])
    if cur is not None and flat.delta_e(cur) < tol:
        return []  # the target really is this colour here: not extra
    behind = _behind_color(ctx.rendered, n.box)
    if behind is not None and flat.delta_e(behind) > tol:
        return []  # flat, but not what is behind the node: a recolor case
    return [Hypothesis("extra", n.id, {"type": n.type, "target_color": flat.hex()}, ctx.share(n.box) * 0.9, n.box, _delete(n.id))]


# --------------------------------------------------------------------------- missing
def _deepest_frame(doc: Document, box: Box) -> Node:
    best, best_area = doc.root, float("inf")
    for n in doc.walk():
        if n.type in ("frame", "instance") and n.visible and n.box.contains(box, tol=1.0) and n.box.area < best_area:
            best, best_area = n, n.box.area
    return best


def _missing_hyps(ctx: _Ctx) -> list[Hypothesis]:
    limit = int(P["refine.critic.max_missing"])
    if limit <= 0:
        return []
    try:
        from dt.perceive import perceive_region
    except Exception:
        return []
    leaves = [n.box for n in ctx.doc.root.leaves() if n.visible]
    cover = float(P["refine.critic.missing_cover"])
    pad = int(P["refine.critic.missing_pad"])
    out: list[Hypothesis] = []
    for reg in _merge_regions(ctx.report.residual_regions, float(P["refine.critic.missing_merge"])):
        if len(out) >= limit:
            break
        if reg.area <= 0:
            continue
        covered = max((reg.intersect(b).area for b in leaves), default=0.0) / reg.area
        if covered > cover:
            continue
        t, r = ctx.tcrop(reg), ctx.rcrop(reg)
        flat = float(P["refine.critic.flat_std"])
        if t.size == 0 or float(t.reshape(-1, 3).std(axis=0).max()) <= flat:
            continue  # flat in the target: an 'extra' node, not a missing one
        if r.size and float(r.reshape(-1, 3).std(axis=0).max()) > flat:
            continue  # the render already draws something here: a shift/resize case, not missing
        area = reg.expand(pad)
        try:
            found = perceive_region(ctx.target, area)
        except Exception:
            continue
        # a piece flush with the crop border is the cut edge of a larger (shifted/resized) element
        nodes = [c for c in found.children if c.visible and c.box.area > 0 and (c.type == "text" or not _touches(c.box, area))]
        if not nodes:
            continue
        for nd in nodes:
            _adopt_text_metrics(ctx.doc, nd)
        parent = _deepest_frame(ctx.doc, area)
        out.append(Hypothesis("missing", None, {"parent": parent.id, "n": len(nodes), "types": [c.type for c in nodes]},
                              ctx.share(area) * 0.5, area, _insert(parent.id, nodes)))
    return out


def _touches(inner: Box, outer: Box, tol: float = 1.0) -> bool:
    return inner.x <= outer.x + tol or inner.y <= outer.y + tol or inner.x2 >= outer.x2 - tol or inner.y2 >= outer.y2 - tol


def _merge_regions(regions: list[Box], gap: float) -> list[Box]:
    """Union residual regions whose expanded boxes touch (keeps the input order of the first member)."""
    merged: list[Box] = []
    for r in regions:
        r = Box(r.x, r.y, r.w, r.h)
        changed = True
        while changed:
            changed = False
            for i, m in enumerate(merged):
                if m.expand(gap / 2).intersect(r.expand(gap / 2)).area > 0:
                    r = m.union(r)
                    merged.pop(i)
                    changed = True
                    break
        merged.append(r)
    return merged


def _adopt_text_metrics(doc: Document, node: Node) -> None:
    """Give a perceived text node the line_height and letter_spacing used elsewhere in ``doc``
    for the same size (typescale consistency), re-fitting its box so the measured baseline stays."""
    for n in node.walk():
        if n.type != "text" or n.text_style is None or n.text_style.line_height is not None:
            continue
        size = _snap_size(n.text_style.size, _doc_sizes(doc))
        peer_styles = [t.text_style for t in doc.walk() if t.type == "text" and t.text_style and abs(t.text_style.size - size) < 0.75]
        peers = [ps.line_height for ps in peer_styles if ps.line_height]
        if peer_styles:
            n.text_style.letter_spacing = float(np.median([ps.letter_spacing for ps in peer_styles]))
        normal = float(P["perceive.ocr.line_height_ratio"]) * n.text_style.size
        ascent_old = float(P["perceive.ocr.ascent_ratio"]) * n.text_style.size
        baseline = n.box.y + ascent_old  # paragraph_node: y = baseline - ascent (line_height None)
        n.text_style.size = size
        if peers:
            L = float(np.median(peers))
            n.text_style.line_height = L
            n_lines = (n.text or "").count("\n") + 1
            normal = float(P["perceive.ocr.line_height_ratio"]) * size
            y = baseline - float(P["perceive.ocr.ascent_ratio"]) * size - (L - normal) / 2.0
            n.box = Box(n.box.x, round(y, 1), n.box.w, round(n_lines * L, 1))


# --------------------------------------------------------------------------- entry point
def critique(doc: Document, target_rgb: np.ndarray, rendered_rgb: np.ndarray, report: LossReport) -> list[Hypothesis]:
    """Propose edits to ``doc`` that could bring its render closer to ``target_rgb``.

    Nodes are ranked by their own mean ΔE (``report.per_node``); the worst
    ``refine.critic.max_nodes`` get shift / resize / recolor / text / radius / shadow / extra
    hypotheses, and uncovered residual regions get ``missing`` ones. The list is sorted by
    ``expected_gain`` (descending). Never raises on empty or single-node documents.
    """
    if target_rgb.size == 0 or rendered_rgb.size == 0:
        return []
    ctx = _Ctx(doc, target_rgb, rendered_rgb, report)
    min_de = float(P["refine.critic.min_node_de"])
    ranked: list[tuple[float, Node]] = []
    for n in doc.walk():
        if not n.visible or n.box.area <= 0:
            continue
        own = float(report.per_node.get(n.id, 0.0))
        inside = _box_sum(ctx.diff_int, n.box) / max(1.0, n.box.area)  # children included
        bad_px = _box_sum(ctx.bad_int, n.box)
        mass = _box_sum(ctx.diff_int, n.box)
        if max(own, inside) >= min_de or bad_px >= float(P["refine.critic.min_bad_px"]) or mass >= float(P["refine.critic.min_err_mass"]):
            ranked.append((ctx.share(n.box), n))
    ranked.sort(key=lambda t: -t[0])
    hyps: list[Hypothesis] = []
    for _, n in ranked[: int(P["refine.critic.max_nodes"])]:
        is_root = n.id == doc.root.id
        try:
            if not is_root:
                hyps += _shift_hyps(ctx, n)
                hyps += _resize_hyps(ctx, n)
                hyps += _extra_hyps(ctx, n)
                hyps += _shadow_hyps(ctx, n)
            hyps += _recolor_hyps(ctx, n)
            hyps += _text_hyps(ctx, n)
            if not is_root:
                hyps += _radius_hyps(ctx, n)
                hyps += _logo_hyps(ctx, n) if n.type == "text" else _raster_hyps(ctx, n)
        except Exception:  # a broken proposal must never stop the loop
            continue
    try:
        hyps += _missing_hyps(ctx)
    except Exception:
        pass
    try:
        hyps += _raster_region_hyps(ctx)
    except Exception:
        pass
    hyps.sort(key=lambda h: -h.expected_gain)
    return hyps
