"""Typed perturbation operators on a ground-truth IR document, plus nuisance (noise) operators.

The benchmark needs pairs (target, candidate) whose differences are *known exactly*. We get them
by starting from a ground-truth IR (``fixtures/corpus/*/*.gt.json`` or ``dt.selftest.synth``),
rendering it as the **target**, and applying typed, magnitude-controlled edits to a copy that
becomes the **candidate IR** (its render is the candidate image). Each operator returns the
ground-truth :class:`~dt.adversary.taxonomy.Finding` list: type, union box of the element's
before/after paint extent, and magnitude in the unit of the type.

Magnitudes are drawn on a log scale between a "just visible" floor and a "large" ceiling
(``adversary.perturb.range.<leaf>``), from a uniform ``u`` in [0, 1] so the benchmark can report
recall per magnitude bin.

Noise operators act on the **target** only and must yield NO findings: sub-pixel page offsets
(0.25-0.5 px, rendered by the real browser), font-smoothing / resampling blur differences,
JPEG-like compression, and a different browser build (Playwright's bundled Chromium vs system
Chrome) when one is installed.

Every operator is pure with respect to its ``random.Random`` argument: same doc + same rng state
-> same edit. Text widths after an edit are *measured in the real renderer* (same fonts and CSS as
``dt.render``), so text boxes in the candidate IR and in the ground truth are exact.

Leak-proofing: :func:`sanitize` strips the corpus-only evidence (``tokens``, ``meta``, component
``evidence``, stale names/labels) and renumbers ids in pre-order, so an IR-aware adversary cannot
find a perturbation by spotting a token/colour mismatch, an id gap or a stale ``name``.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
import queue
import random
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from dt.adversary.taxonomy import Finding
from dt.compare.pixel import diff_map
from dt.compare.structural import levenshtein
from dt.ir import Box, Color, Document, Fill, Node, Shadow, rgb_to_lab
from dt.params import P, register

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK_DIR = os.environ.get("DT_ADVERSARY_WORK", os.path.join(ROOT, "out", "adversary", ".work"))

# --------------------------------------------------------------------------- params
register("adversary.perturb.visible_de", 2.3,
         "ΔE (candidate render vs clean render) above which a pixel counts as visibly changed (JND)", (1.0, 6.0))
register("adversary.perturb.min_visible_px", 6,
         "a ground-truth finding must change at least this many pixels by > visible_de, else the case is redrawn", (1, 100))
register("adversary.perturb.max_unexplained_px", 4,
         "visibly changed pixels allowed outside every ground-truth box (padded by check_pad); more = the case is redrawn", (0, 100))
register("adversary.perturb.check_pad", 2,
         "padding (px) around ground-truth boxes for the unexplained-pixel check (anti-aliasing bleeds ~1 px)", (0, 6))
register("adversary.perturb.disjoint_pad", 4,
         "perturbations in one case keep their boxes at least this far apart (px), so ground truth is unambiguous", (0, 24))
register("adversary.perturb.min_side", 4, "nodes thinner than this (px) are never perturbed", (1, 16))
register("adversary.perturb.max_area_frac", 0.35,
         "nodes covering more than this fraction of the page are never removed/moved/faded (page-sized edits are tint_global's job)", (0.05, 1.0))
# magnitude ranges [just visible, large] per leaf type (log-uniform between them)
register("adversary.perturb.range.shift", [1.0, 24.0], "geometry.shift: offset px")
register("adversary.perturb.range.size", [1.0, 24.0], "geometry.size: largest edge displacement px")
register("adversary.perturb.range.radius", [2.0, 20.0], "geometry.radius: |Δradius| px")
register("adversary.perturb.range.fill", [3.0, 40.0], "color.fill: ΔE76")
register("adversary.perturb.range.stroke", [3.0, 40.0], "color.stroke: ΔE76")
register("adversary.perturb.range.tint_global", [2.0, 15.0], "color.tint_global: ΔE76 of the common Lab offset")
register("adversary.perturb.range.split", [1.0, 12.0], "structure.split: seam gap px")
register("adversary.perturb.range.text_size", [1.0, 8.0], "text.style: |Δsize| px")
register("adversary.perturb.range.text_position", [1.0, 16.0], "text.position: offset px")
register("adversary.perturb.range.opacity", [0.1, 0.8], "effect.opacity: |Δopacity|")
register("adversary.perturb.range.spacing", [1.0, 12.0], "layout.spacing: |Δgap| px")
register("adversary.perturb.merge_max_gap", 32.0, "structure.merge: siblings at most this far apart (px) can be merged", (4.0, 96.0))
register("adversary.noise.subpixel", [0.25, 0.5], "noise.subpixel: |offset| px per axis")
register("adversary.noise.blur_sigma", [0.35, 0.6], "noise.antialias (resampling blur): Gaussian sigma px")
register("adversary.noise.jpeg_quality", [70, 95], "noise.compression: JPEG quality")

ICON_POOL = ["search", "more_vert", "settings", "share", "edit", "delete", "add", "filter_list", "refresh", "close",
             "arrow_back", "menu", "star", "archive", "send", "check", "mail", "folder", "description", "image",
             "event", "person", "drafts", "label", "attach_file", "videocam", "place", "note", "home", "chat",
             "notifications", "photo_library", "checklist", "mic", "navigation"]
# M3 elevation levels (dt.selftest.grammar.M3_ELEVATION; duplicated as data to keep this module light)
ELEVATION = {
    1: [Shadow(Color(0, 0, 0, 0.3), 0, 1, 2, 0), Shadow(Color(0, 0, 0, 0.15), 0, 1, 3, 1)],
    2: [Shadow(Color(0, 0, 0, 0.3), 0, 1, 2, 0), Shadow(Color(0, 0, 0, 0.15), 0, 2, 6, 2)],
    3: [Shadow(Color(0, 0, 0, 0.3), 0, 1, 3, 0), Shadow(Color(0, 0, 0, 0.15), 0, 4, 8, 3)],
    4: [Shadow(Color(0, 0, 0, 0.3), 0, 2, 3, 0), Shadow(Color(0, 0, 0, 0.15), 0, 6, 10, 4)],
    5: [Shadow(Color(0, 0, 0, 0.3), 0, 4, 4, 0), Shadow(Color(0, 0, 0, 0.15), 0, 8, 12, 6)],
}
SHAPES = ("frame", "rect", "instance", "ellipse", "line", "image", "vector")


# --------------------------------------------------------------------------- helpers
def log_mag(key: str, u: float) -> float:
    lo, hi = (float(v) for v in P[f"adversary.perturb.range.{key}"])
    u = min(1.0, max(0.0, float(u)))
    return lo * (hi / lo) ** u if lo > 0 else lo + (hi - lo) * u


def magnitude_bin(u: float) -> str:
    return "just_visible" if u < 0.25 else "small" if u < 0.5 else "medium" if u < 0.75 else "large"


def _solid(n: Node) -> Optional[Fill]:
    for f in n.fills:
        if f.kind == "solid" and f.color is not None and f.color.a > 0.02 and f.opacity > 0.02:
            return f
    return None


def paints(n: Node) -> bool:
    """Does the node itself put pixels on screen?"""
    if not n.visible or n.opacity <= 0.02:
        return False
    if n.type == "text":
        return bool((n.text or "").strip()) and n.text_style is not None
    if n.type == "icon":
        return bool(n.icon_name)
    if n.type == "image" and n.image_ref:
        return True
    return _solid(n) is not None or bool(n.strokes) or bool(n.effects) or any(f.kind != "solid" for f in n.fills)


def subtree_paints(n: Node) -> bool:
    return n.visible and any(paints(m) for m in n.walk())


def shadow_reach(effects: list[Shadow]) -> float:
    return max([abs(e.dx) + abs(e.dy) + e.blur + max(0.0, e.spread) for e in effects if not e.inner] or [0.0])


def extent(n: Node, subtree: bool = True) -> Box:
    """Paint extent: node box (and subtree) grown by drop-shadow reach."""
    out: Optional[Box] = None
    for m in (n.walk() if subtree else [n]):
        if not m.visible:
            continue
        b = m.box.expand(shadow_reach(m.effects)) if m.effects else m.box
        out = b if out is None else out.union(b)
    return out or n.box


def _page(doc: Document) -> Box:
    return Box(0, 0, doc.width, doc.height)


def clip_box(b: Box, doc: Document) -> Box:
    return b.intersect(_page(doc))


def _inside(b: Box, doc: Document, margin: float = 0.0) -> bool:
    return b.x >= -margin and b.y >= -margin and b.x2 <= doc.width + margin and b.y2 <= doc.height + margin


def _disjoint(b: Box, taken: list[Box]) -> bool:
    pad = float(P["adversary.perturb.disjoint_pad"])
    g = b.expand(pad)
    return all(g.intersect(t).area <= 0 for t in taken)


def _translate(n: Node, dx: float, dy: float) -> None:
    for m in n.walk():
        m.box = m.box.translate(dx, dy)


def _direction(rng: random.Random, m: float) -> tuple[int, int]:
    ang = rng.choice([0, 45, 90, 135, 180, 225, 270, 315])
    dx, dy = int(round(m * math.cos(math.radians(ang)))), int(round(m * math.sin(math.radians(ang))))
    if dx == 0 and dy == 0:
        dx = 1
    return dx, dy


def _parents(doc: Document) -> dict[str, Node]:
    out = {}
    for n in doc.walk():
        for c in n.children:
            out[c.id] = n
    return out


def _ancestors(doc: Document, node_id: str) -> list[Node]:
    par = _parents(doc)
    out, cur = [], par.get(node_id)
    while cur is not None:
        out.append(cur)
        cur = par.get(cur.id)
    return out


def _ok_node(n: Node, doc: Document, max_frac: Optional[float] = None) -> bool:
    ms = float(P["adversary.perturb.min_side"])
    if n.id == doc.root.id or not n.visible or n.box.w < ms or n.box.h < ms and n.type != "line":
        return False
    if n.type == "line" and min(n.box.w, n.box.h) < 0.5:
        return False
    if not _inside(n.box, doc, 0.5):
        return False
    frac = float(P["adversary.perturb.max_area_frac"]) if max_frac is None else max_frac
    return n.box.area <= frac * doc.width * doc.height


def _lab_to_rgb(L: float, a: float, b: float) -> tuple[int, int, int]:
    from skimage.color import lab2rgb
    rgb = lab2rgb(np.array([[[L, a, b]]], dtype=np.float64))[0, 0]
    return tuple(int(round(float(np.clip(v, 0, 1)) * 255)) for v in rgb)  # type: ignore[return-value]


def shift_color(c: Color, de: float, rng: random.Random, tries: int = 24) -> tuple[Color, float]:
    """A colour ~``de`` ΔE76 away from ``c`` (random Lab direction, gamut-clipped). Returns (colour, actual ΔE)."""
    L, A, B = rgb_to_lab(c.r, c.g, c.b)
    best: Optional[tuple[float, Color, float]] = None
    for _ in range(tries):
        v = np.array([rng.gauss(0, 1) for _ in range(3)])
        v = v / (np.linalg.norm(v) or 1.0) * de
        r, g, b = _lab_to_rgb(L + v[0], A + v[1], B + v[2])
        nc = Color(r, g, b, c.a)
        got = c.delta_e(nc)
        err = abs(got - de)
        if best is None or err < best[0]:
            best = (err, nc, got)
        if err < 0.1 * de:
            break
    assert best is not None
    return best[1], best[2]


def _all_colors(doc: Document) -> list[tuple[object, str]]:
    """(holder, attribute) pairs for every solid colour on the page (fills, strokes, text colours)."""
    out: list[tuple[object, str]] = []
    for n in doc.walk():
        for f in n.fills:
            if f.kind == "solid" and f.color is not None:
                out.append((f, "color"))
            for s in f.stops:
                out.append((s, "color"))
        for s in n.strokes:
            out.append((s, "color"))
        if n.text_style is not None:
            out.append((n.text_style, "color"))
    return out


# --------------------------------------------------------------------------- text measurement (real renderer)
# the span is a stretched flex item: measure the laid-out glyph run with a Range over its contents
_MEASURE_JS = """(() => Array.from(document.querySelectorAll('div[data-type=text] > span')).map(s => {
    const r = document.createRange(); r.selectNodeContents(s); return r.getBoundingClientRect().width; }))()"""


def measure_texts(items: list[tuple[str, Node]]) -> list[float]:
    """Rendered single-line widths (px) of ``text`` set in each node's text style, measured with the
    project renderer (same @font-face set and text CSS as ``dt.render``)."""
    if not items:
        return []
    from dt.ir import Node as _N
    from dt.render.html import render_html
    from dt.render.screenshot import render_url
    rows = []
    y = 0.0
    for i, (s, n) in enumerate(items):
        ts = copy.deepcopy(n.text_style)
        ts.align = "left"
        h = max(8.0, (ts.line_height or ts.size * 1.5) + 4)
        rows.append(_N(id=f"m{i}", type="text", box=Box(0, y, 4000, h), text=s, text_style=ts))
        y += h
    doc = Document(width=800, height=int(min(4000, y + 1)), root=_N(id="root", type="frame", box=Box(0, 0, 800, y + 1), children=rows))
    os.makedirs(WORK_DIR, exist_ok=True)
    tag = hashlib.sha1("\x00".join(s for s, _ in items).encode()).hexdigest()[:10]
    path = os.path.join(WORK_DIR, f"measure_{os.getpid()}_{threading.get_ident()}_{tag}.html")
    with open(path, "w") as f:
        f.write(render_html(doc))
    try:
        _rgb, widths = render_url("file://" + path, 800, 100, wait_ms=0, script=_MEASURE_JS, wait_until="load")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return [float(w) for w in widths]


def _text_box(n: Node, old_measured: float, new_measured: float, scale_h: float = 1.0) -> Box:
    """New text box after its run width changes, keeping the alignment anchor and the IR's box
    convention (width = old box width + measured run delta)."""
    b = n.box
    w = max(1.0, b.w + (new_measured - old_measured))
    h = b.h * scale_h
    align = n.text_style.align if n.text_style else "left"
    if align == "center":
        x = b.cx - w / 2
    elif align == "right":
        x = b.x2 - w
    else:
        x = b.x
    return Box(x, b.y, w, h)


def _single_line_text(n: Node) -> bool:
    return n.type == "text" and n.text_style is not None and bool((n.text or "").strip()) and "\n" not in (n.text or "")


def _retext_refs(doc: Document, node_id: str, old: str, new: str) -> None:
    """Keep component props / names that quoted the old text consistent (no stale label leaks)."""
    for a in _ancestors(doc, node_id):
        if a.component is not None:
            for k, v in list(a.component.props.items()):
                if isinstance(v, str) and old and old in v:
                    a.component.props[k] = v.replace(old, new)


# --------------------------------------------------------------------------- operators
@dataclass
class Ctx:
    doc: Document
    rng: random.Random
    u: float
    taken: list[Box]
    new_ids: list[int] = field(default_factory=lambda: [0])

    def fresh_id(self) -> str:
        self.new_ids[0] += 1
        return f"__new{self.new_ids[0]}"

    def finding(self, type_: str, box: Box, magnitude: Optional[float], node_id: Optional[str], **ev) -> Finding:
        ev.setdefault("u", round(self.u, 4))
        ev.setdefault("bin", magnitude_bin(self.u))
        return Finding(type_, clip_box(box, self.doc), None if magnitude is None else float(magnitude), 1.0, ev, node_id)


Op = Callable[[Ctx], Optional[list[Finding]]]


def _candidates(ctx: Ctx, pred: Callable[[Node], bool]) -> list[Node]:
    nodes = [n for n in ctx.doc.walk() if pred(n)]
    ctx.rng.shuffle(nodes)
    return nodes


def op_shift(ctx: Ctx) -> Optional[list[Finding]]:
    m = log_mag("shift", ctx.u)
    for n in _candidates(ctx, lambda n: n.type != "text" and _ok_node(n, ctx.doc) and subtree_paints(n)):
        dx, dy = _direction(ctx.rng, m)
        before = extent(n)
        after = before.translate(dx, dy)
        if not _inside(n.box.translate(dx, dy), ctx.doc) or not _disjoint(before.union(after), ctx.taken):
            continue
        _translate(n, dx, dy)
        return [ctx.finding("geometry.shift", before.union(after), math.hypot(dx, dy), n.id, dx=dx, dy=dy)]
    return None


def op_size(ctx: Ctx) -> Optional[list[Finding]]:
    d = max(1, int(round(log_mag("size", ctx.u))))
    for n in _candidates(ctx, lambda n: n.type in SHAPES and paints(n) and _ok_node(n, ctx.doc)):
        b = n.box
        mode = ctx.rng.choice(["right", "bottom", "both", "left", "top"]) if n.type != "line" else ctx.rng.choice(["right", "left"])
        grow = ctx.rng.random() < 0.5
        s = d if grow else -d
        x, y, w, h = b.x, b.y, b.w, b.h
        if mode in ("right", "both"):
            w += s
        if mode in ("bottom", "both"):
            h += s
        if mode == "left":
            x, w = x - s, w + s
        if mode == "top":
            y, h = y - s, h + s
        if min(w, h) < (1 if n.type == "line" else float(P["adversary.perturb.min_side"])):
            continue
        nb = Box(x, y, w, h)
        reach = shadow_reach(n.effects)
        union = b.union(nb).expand(reach)
        if n.clip:  # clipped children may be revealed/hidden: include the subtree
            union = union.union(extent(n))
        if not _inside(nb, ctx.doc) or not _disjoint(union, ctx.taken):
            continue
        n.box = nb
        return [ctx.finding("geometry.size", union, float(d), n.id, edge=mode, grow=grow)]
    return None


def op_radius(ctx: Ctx) -> Optional[list[Finding]]:
    m = log_mag("radius", ctx.u)
    for n in _candidates(ctx, lambda n: n.type in ("frame", "rect", "instance", "image") and (_solid(n) or n.strokes)
                         and _ok_node(n, ctx.doc) and min(n.box.w, n.box.h) >= 12):
        cap = min(n.box.w, n.box.h) / 2
        cur = [min(r, cap) for r in n.radius]
        r0 = sum(cur) / 4
        up, down = r0 + m <= cap, r0 - m >= 0
        if up and down:
            new = r0 + m if ctx.rng.random() < 0.5 else r0 - m
        elif up or down:
            new = r0 + m if up else r0 - m
        else:
            new = cap if cap - r0 > r0 else 0.0
        new = round(new)
        delta = sum(abs(new - r) for r in cur) / 4
        if delta < float(P["adversary.perturb.range.radius"][0]) * 0.75:
            continue
        box = extent(n, subtree=bool(n.clip))
        if not _disjoint(box, ctx.taken):
            continue
        n.radius = (new, new, new, new)
        return [ctx.finding("geometry.radius", box, delta, n.id, radius=[r0, new])]
    return None


def _recolor(ctx: Ctx, type_: str, key: str, get: Callable[[Node], Optional[Color]], setc: Callable[[Node, Color], None],
             pred: Callable[[Node], bool]) -> Optional[list[Finding]]:
    m = log_mag(key, ctx.u)
    for n in _candidates(ctx, lambda n: pred(n) and get(n) is not None and _ok_node(n, ctx.doc, 1.0)):
        box = n.box.expand(1.0) if n.type == "text" else n.box  # glyph ink can overhang the line box
        if n.type == "text" or (n.clip and type_ == "color.fill"):
            box = box.union(extent(n, subtree=False))
        if not _disjoint(box, ctx.taken):
            continue
        c = get(n)
        nc, got = shift_color(c, m, ctx.rng)
        if got < 0.6 * m:
            continue
        setc(n, nc)
        return [ctx.finding(type_, box, got, n.id, **{"from": c.hex(), "to": nc.hex()})]
    return None


def op_fill(ctx: Ctx) -> Optional[list[Finding]]:
    def get(n: Node) -> Optional[Color]:
        if n.type == "text":
            return n.text_style.color if n.text_style else None
        f = _solid(n)
        return f.color if f else None

    def setc(n: Node, c: Color) -> None:
        if n.type == "text":
            n.text_style.color = c
        else:
            _solid(n).color = c

    return _recolor(ctx, "color.fill", "fill", get, setc,
                    lambda n: paints(n) and n.box.area <= float(P["adversary.perturb.max_area_frac"]) * ctx.doc.width * ctx.doc.height)


def op_stroke(ctx: Ctx) -> Optional[list[Finding]]:
    def setc(n: Node, c: Color) -> None:
        n.strokes[0].color = c
    return _recolor(ctx, "color.stroke", "stroke", lambda n: n.strokes[0].color if n.strokes else None, setc,
                    lambda n: n.visible and bool(n.strokes) and n.strokes[0].width > 0)


def op_tint_global(ctx: Ctx) -> Optional[list[Finding]]:
    if ctx.taken:
        return None
    m = log_mag("tint_global", ctx.u)
    v = np.array([ctx.rng.gauss(0, 1) for _ in range(3)])
    v = v / (np.linalg.norm(v) or 1.0) * m
    des = []
    cache: dict[tuple, Color] = {}
    for holder, attr in _all_colors(ctx.doc):
        c: Color = getattr(holder, attr)
        k = (c.r, c.g, c.b)
        if k not in cache:
            L, A, B = rgb_to_lab(*k)
            cache[k] = Color(*_lab_to_rgb(L + v[0], A + v[1], B + v[2]))
            des.append(c.delta_e(cache[k]))
        nc = cache[k]
        setattr(holder, attr, Color(nc.r, nc.g, nc.b, c.a))
    if not des:
        return None
    return [ctx.finding("color.tint_global", _page(ctx.doc), float(np.mean(des)), None, exclusive=True,
                        offset_lab=[round(float(x), 3) for x in v])]


def _remove(doc: Document, n: Node) -> None:
    par = doc.parent_of(n.id)
    par.children = [c for c in par.children if c.id != n.id]


def op_missing(ctx: Ctx) -> Optional[list[Finding]]:
    for n in _candidates(ctx, lambda n: n.type != "icon" and subtree_paints(n) and _ok_node(n, ctx.doc)
                         and n.box.area >= 16):
        box = extent(n)
        if not _disjoint(box, ctx.taken):
            continue
        _remove(ctx.doc, n)
        return [ctx.finding("structure.missing", box, n.box.area, None, removed_type=n.type, removed_nodes=n.count())]
    return None


def op_icon_missing(ctx: Ctx) -> Optional[list[Finding]]:
    for n in _candidates(ctx, lambda n: n.type == "icon" and paints(n) and _ok_node(n, ctx.doc)):
        box = extent(n)
        if not _disjoint(box, ctx.taken):
            continue
        _remove(ctx.doc, n)
        return [ctx.finding("icon.missing", box, n.box.area, None, icon=n.icon_name)]
    return None


def op_extra(ctx: Ctx) -> Optional[list[Finding]]:
    page_area = ctx.doc.width * ctx.doc.height
    painted = [m for m in ctx.doc.walk() if paints(m) and m.id != ctx.doc.root.id]
    par_of = _parents(ctx.doc)
    for n in _candidates(ctx, lambda n: not n.children and paints(n) and _ok_node(n, ctx.doc, 0.15) and n.box.area >= 16
                         and n.type in ("rect", "ellipse", "icon", "text", "line", "frame")):
        parent = par_of.get(n.id, ctx.doc.root)
        area = parent.box if parent.box.area < page_area else _page(ctx.doc)
        if area.w < n.box.w + 8 or area.h < n.box.h + 8:
            area = _page(ctx.doc)
            parent = ctx.doc.root
        for _ in range(40):
            x = ctx.rng.uniform(area.x, area.x2 - n.box.w)
            y = ctx.rng.uniform(area.y, area.y2 - n.box.h)
            nb = Box(round(x), round(y), n.box.w, n.box.h)
            reach = shadow_reach(n.effects)
            ext = nb.expand(max(reach, 1.0))
            if not _inside(nb, ctx.doc) or not _disjoint(ext, ctx.taken):
                continue
            # free spot: nothing painted there except backdrops that fully contain it
            if any(m.box.intersect(ext.expand(3)).area > 0 and not m.box.contains(ext) for m in painted):
                continue
            # and it must sit on a backdrop of the clone's own parent chain (so z-order is unambiguous)
            clone = copy.deepcopy(n)
            clone.id = ctx.fresh_id()
            _translate(clone, nb.x - n.box.x, nb.y - n.box.y)
            host = parent if parent.box.contains(nb) else ctx.doc.root
            host.children.append(clone)
            return [ctx.finding("structure.extra", ext if reach else nb, nb.area, clone.id, copy_of_type=n.type)]
    return None


def op_split(ctx: Ctx) -> Optional[list[Finding]]:
    g = max(1, int(round(log_mag("split", ctx.u))))
    par_of = _parents(ctx.doc)
    for n in _candidates(ctx, lambda n: _ok_node(n, ctx.doc) and (
            (n.type in ("rect", "frame", "instance", "line") and not n.children and (_solid(n) or n.strokes) and max(n.box.w, n.box.h) >= 24)
            or (_single_line_text(n) and len((n.text or "").split()) >= 2 and n.text_style.align == "left"))):
        before = extent(n)
        if not _disjoint(before.expand(g), ctx.taken):
            continue
        parent = par_of.get(n.id, ctx.doc.root)
        b = n.box
        a_node, b_node = copy.deepcopy(n), copy.deepcopy(n)
        b_node.id = ctx.fresh_id()
        if n.type == "text":
            words = (n.text or "").split(" ")
            idxs = [i for i in range(1, len(words)) if words[i - 1] and words[i]]
            if not idxs:
                continue
            k = ctx.rng.choice(idxs)
            left, right = " ".join(words[:k]), " ".join(words[k:])
            w_full, w_left_sp = measure_texts([(n.text or "", n), (left + " ", n)])
            w_left = measure_texts([(left, n)])[0]
            slack = b.w - w_full  # the IR box convention (glyph run vs line box) is kept for both parts
            a_node.text, b_node.text = left, right
            a_node.box = Box(b.x, b.y, max(1.0, w_left + slack), b.h)
            b_node.box = Box(b.x + w_left_sp + g, b.y, max(1.0, w_full - w_left_sp + slack), b.h)
            after = b_node.box.union(a_node.box)
            if not _inside(b_node.box, ctx.doc):
                continue
        else:
            horiz = b.w >= b.h
            L = b.w if horiz else b.h
            f = ctx.rng.uniform(0.35, 0.65)
            cut = round(L * f)
            if cut - g / 2 < 4 or L - cut - g / 2 < 4:
                continue
            if horiz:
                a_node.box = Box(b.x, b.y, cut - g / 2, b.h)
                b_node.box = Box(b.x + cut + g / 2, b.y, b.w - cut - g / 2, b.h)
            else:
                a_node.box = Box(b.x, b.y, b.w, cut - g / 2)
                b_node.box = Box(b.x, b.y + cut + g / 2, b.w, b.h - cut - g / 2)
            after = b
        idx = next(i for i, c in enumerate(parent.children) if c.id == n.id)
        parent.children[idx:idx + 1] = [a_node, b_node]
        if n.type == "text":
            a_node.name, b_node.name = f"text:{a_node.text[:24]}", f"text:{b_node.text[:24]}"
        return [ctx.finding("structure.split", before.union(after).expand(1.0 if n.type == "text" else 0.0), float(g),
                            a_node.id, parts=[a_node.id, b_node.id])]
    return None


def _same_style(a: Node, b: Node) -> bool:
    sa, sb = a.text_style, b.text_style
    return (sa is not None and sb is not None and sa.size == sb.size and sa.weight == sb.weight and sa.family == sb.family
            and sa.color.hex() == sb.color.hex() and sa.align == sb.align == "left")


def op_merge(ctx: Ctx) -> Optional[list[Finding]]:
    max_gap = float(P["adversary.perturb.merge_max_gap"])
    pairs: list[tuple[Node, Node, Node, float]] = []
    for parent in ctx.doc.walk():
        kids = [c for c in parent.children if c.visible]
        for a in kids:
            for b in kids:
                if a is b or a.type != b.type or not _ok_node(a, ctx.doc) or not _ok_node(b, ctx.doc):
                    continue
                if a.type == "text":
                    if not (_single_line_text(a) and _single_line_text(b) and _same_style(a, b)):
                        continue
                elif a.type not in ("rect", "frame", "instance") or not (paints(a) and paints(b)):
                    continue
                if abs(a.box.y - b.box.y) > 2 or abs(a.box.h - b.box.h) > 2:
                    continue
                gap = b.box.x - a.box.x2
                if not (1 <= gap <= max_gap):
                    continue
                between = [c for c in kids if c is not a and c is not b and c.box.x < b.box.x and c.box.x2 > a.box.x2
                           and c.box.y < a.box.y2 and c.box.y2 > a.box.y]
                if between:
                    continue
                pairs.append((parent, a, b, gap))
    pairs.sort(key=lambda t: (t[1].box.y, t[1].box.x))
    ctx.rng.shuffle(pairs)
    for parent, a, b, gap in pairs:
        box = extent(a).union(extent(b))
        if not _disjoint(box, ctx.taken):
            continue
        merged = a
        if a.type == "text":
            merged.text = (a.text or "") + " " + (b.text or "")
            merged.name = f"text:{merged.text[:24]}"
            w_new = measure_texts([(merged.text, a)])[0]
            merged.box = Box(a.box.x, min(a.box.y, b.box.y), max(b.box.x2 - a.box.x, w_new), max(a.box.h, b.box.h))
            box = box.union(merged.box).expand(1.0)
        else:
            merged.box = a.box.union(b.box)
            merged.children = a.children + b.children
        parent.children = [c for c in parent.children if c.id != b.id]
        return [ctx.finding("structure.merge", box, float(gap), merged.id, merged_from=[a.id, b.id])]
    return None


def op_text_content(ctx: Ctx) -> Optional[list[Finding]]:
    for n in _candidates(ctx, lambda n: _single_line_text(n) and _ok_node(n, ctx.doc, 1.0)
                         and sum(ch.isalnum() for ch in (n.text or "")) >= 2):
        old = n.text or ""
        positions = [i for i, ch in enumerate(old) if ch.isalnum()]
        max_k = max(1, len(positions) // 2)
        k = 1 + int(ctx.u * (max_k - 1) + 0.5) if max_k > 1 else 1
        chars = list(old)
        for i in ctx.rng.sample(positions, min(k, len(positions))):
            ch = chars[i]
            pool = "0123456789" if ch.isdigit() else ("ABCDEFGHJKMNPQRSTUVWXYZ" if ch.isupper() else "abcdefghjkmnpqrstuvwxyz")
            pool = pool.replace(ch, "")
            r = ctx.rng.random()
            if r < 0.7:
                chars[i] = ctx.rng.choice(pool)
            elif r < 0.85 and len(positions) > 2:
                chars[i] = ""
            else:
                chars[i] = ch + ctx.rng.choice(pool)
        new = "".join(chars)
        if new == old or not new.strip():
            continue
        w_old, w_new = measure_texts([(old, n), (new, n)])
        nb = _text_box(n, w_old, w_new)
        box = n.box.union(nb).expand(1.0)
        if not _inside(nb, ctx.doc, 1) or not _disjoint(box, ctx.taken):
            continue
        n.text, n.box = new, nb
        n.name = f"text:{new[:24]}"
        _retext_refs(ctx.doc, n.id, old, new)
        cer = levenshtein(old, new) / max(1, len(old))
        return [ctx.finding("text.content", box, cer, n.id, **{"from": old, "to": new})]
    return None


def op_text_style(ctx: Ctx) -> Optional[list[Finding]]:
    for n in _candidates(ctx, lambda n: _single_line_text(n) and _ok_node(n, ctx.doc, 1.0)):
        ts = n.text_style
        new_ts = copy.deepcopy(ts)
        if ctx.rng.random() < 0.7:
            d = max(1, int(round(log_mag("text_size", ctx.u))))
            s = ts.size + d if (ctx.rng.random() < 0.5 or ts.size - d < 8) else ts.size - d
            new_ts.size = float(s)
            mag = abs(s - ts.size)
            what = "size"
        else:
            steps = 1 + int(round(ctx.u * 4))
            w = ts.weight + 100 * steps if ts.weight + 100 * steps <= 900 and (ctx.rng.random() < 0.5 or ts.weight - 100 * steps < 100) else ts.weight - 100 * steps
            if not 100 <= w <= 900:
                continue
            new_ts.weight = int(w)
            mag = abs(w - ts.weight) / 100.0
            what = "weight"
        probe = copy.deepcopy(n)
        probe.text_style = new_ts
        w_old, w_new = measure_texts([(n.text or "", n), (n.text or "", probe)])
        scale = new_ts.size / ts.size
        nb = _text_box(n, w_old, w_new, scale if ts.line_height is None else 1.0)
        grow = max(0.0, (scale - 1.0) * n.box.h / 2) + 1.0
        box = n.box.union(nb).expand(grow)
        if not _inside(nb, ctx.doc, 1) or not _disjoint(box, ctx.taken):
            continue
        n.text_style, n.box = new_ts, nb
        return [ctx.finding("text.style", box, mag, n.id, what=what, size=[ts.size, new_ts.size], weight=[ts.weight, new_ts.weight])]
    return None


def op_text_position(ctx: Ctx) -> Optional[list[Finding]]:
    m = log_mag("text_position", ctx.u)
    for n in _candidates(ctx, lambda n: n.type == "text" and paints(n) and _ok_node(n, ctx.doc, 1.0)):
        dx, dy = _direction(ctx.rng, m)
        nb = n.box.translate(dx, dy)
        box = n.box.union(nb).expand(1.0)
        if not _inside(nb, ctx.doc) or not _disjoint(box, ctx.taken):
            continue
        n.box = nb
        return [ctx.finding("text.position", box, math.hypot(dx, dy), n.id, dx=dx, dy=dy)]
    return None


def op_icon_glyph(ctx: Ctx) -> Optional[list[Finding]]:
    for n in _candidates(ctx, lambda n: n.type == "icon" and paints(n) and _ok_node(n, ctx.doc)):
        if not _disjoint(n.box, ctx.taken):
            continue
        old = n.icon_name or ""
        new = ctx.rng.choice([g for g in ICON_POOL if g != old])
        n.icon_name = new
        n.name = f"icon:{new}"
        for a in _ancestors(ctx.doc, n.id):
            if a.component is not None:
                for k, v in list(a.component.props.items()):
                    if v == old:
                        a.component.props[k] = new
        return [ctx.finding("icon.glyph", n.box, None, n.id, **{"from": old, "to": new})]
    return None


def op_shadow(ctx: Ctx) -> Optional[list[Finding]]:
    level = 1 + int(round(ctx.u * 4))
    for n in _candidates(ctx, lambda n: n.type in ("frame", "rect", "instance", "ellipse") and _ok_node(n, ctx.doc)
                         and (bool(n.effects) or (_solid(n) is not None and min(n.box.w, n.box.h) >= 16))):
        old = list(n.effects)
        if old:
            new = [] if ctx.rng.random() < 0.6 else [copy.deepcopy(s) for s in ELEVATION[level]]
        else:
            new = [copy.deepcopy(s) for s in ELEVATION[level]]
        mag = abs(shadow_reach(new) - shadow_reach(old))
        if mag < 1.0:
            continue
        box = n.box.expand(max(shadow_reach(old), shadow_reach(new)))
        if not _disjoint(box, ctx.taken):
            continue
        n.effects = new
        return [ctx.finding("effect.shadow", box, mag, n.id, action="remove" if not new else ("add" if not old else "change"), level=level)]
    return None


def op_opacity(ctx: Ctx) -> Optional[list[Finding]]:
    m = log_mag("opacity", ctx.u)
    for n in _candidates(ctx, lambda n: subtree_paints(n) and n.opacity >= 0.99 and _ok_node(n, ctx.doc)):
        box = extent(n)
        if not _disjoint(box, ctx.taken):
            continue
        n.opacity = round(1.0 - m, 3)
        return [ctx.finding("effect.opacity", box, 1.0 - n.opacity, n.id, opacity=[1.0, n.opacity])]
    return None


def _sequences(doc: Document) -> list[tuple[Node, str, list[Node]]]:
    """Children of a parent that form a strict row (sorted by x, sharing a line) or column (by y, sharing a band)."""
    out = []
    for parent in doc.walk():
        kids = [c for c in parent.children if c.visible and subtree_paints(c)]
        if len(kids) < 3:
            continue
        for axis in ("x", "y"):
            ks = sorted(kids, key=lambda c: c.box.x if axis == "x" else c.box.y)
            if axis == "x":
                ok = all(a.box.x2 <= b.box.x + 0.5 for a, b in zip(ks, ks[1:])) and max(c.box.y for c in ks) < min(c.box.y2 for c in ks)
            else:
                ok = all(a.box.y2 <= b.box.y + 0.5 for a, b in zip(ks, ks[1:])) and max(c.box.x for c in ks) < min(c.box.x2 for c in ks)
            if ok:
                out.append((parent, axis, ks))
    return out


def op_spacing(ctx: Ctx) -> Optional[list[Finding]]:
    m = max(1, int(round(log_mag("spacing", ctx.u))))
    seqs = _sequences(ctx.doc)
    ctx.rng.shuffle(seqs)
    for parent, axis, ks in seqs:
        gaps = [(b.box.x - a.box.x2) if axis == "x" else (b.box.y - a.box.y2) for a, b in zip(ks, ks[1:])]
        d = m if (ctx.rng.random() < 0.5 or min(gaps) < m) else -m
        before = None
        after = None
        ok = True
        for i, c in enumerate(ks[1:], start=1):
            e = extent(c)
            e2 = e.translate(i * d, 0) if axis == "x" else e.translate(0, i * d)
            if not _inside(c.box.translate(*((i * d, 0) if axis == "x" else (0, i * d))), ctx.doc):
                ok = False
                break
            before = e if before is None else before.union(e)
            after = e2 if after is None else after.union(e2)
        if not ok or before is None:
            continue
        box = before.union(after)
        if not _disjoint(box, ctx.taken):
            continue
        for i, c in enumerate(ks[1:], start=1):
            _translate(c, i * d if axis == "x" else 0, i * d if axis == "y" else 0)
        if parent.layout is not None and parent.layout.mode != "none":
            parent.layout.gap = max(0.0, parent.layout.gap + d)
        return [ctx.finding("layout.spacing", box, float(abs(d)), parent.id, axis=axis, delta=d, moved=[c.id for c in ks[1:]])]
    return None


OPERATORS: dict[str, Op] = {
    "geometry.shift": op_shift, "geometry.size": op_size, "geometry.radius": op_radius,
    "color.fill": op_fill, "color.stroke": op_stroke, "color.tint_global": op_tint_global,
    "structure.missing": op_missing, "structure.extra": op_extra, "structure.split": op_split, "structure.merge": op_merge,
    "text.content": op_text_content, "text.style": op_text_style, "text.position": op_text_position,
    "icon.glyph": op_icon_glyph, "icon.missing": op_icon_missing,
    "effect.shadow": op_shadow, "effect.opacity": op_opacity, "layout.spacing": op_spacing,
}


def apply(doc: Document, type_: str, rng: random.Random, u: float, taken: Optional[list[Box]] = None,
          ctx_ids: Optional[list[int]] = None) -> Optional[list[Finding]]:
    """Apply one perturbation of ``type_`` to ``doc`` IN PLACE. Returns its ground truth, or None if the
    document offers no applicable node (doc unchanged in that case)."""
    ctx = Ctx(doc, rng, u, list(taken or []), ctx_ids if ctx_ids is not None else [0])
    return OPERATORS[type_](ctx)


def perturb(doc: Document, plan: list[tuple[str, float]], rng: random.Random) -> tuple[Document, list[Finding]]:
    """Apply a plan of ``(type, u)`` perturbations to a COPY of ``doc``; skips inapplicable items.

    Boxes of one case never overlap (``adversary.perturb.disjoint_pad``), and ``color.tint_global``
    is exclusive. The returned document is sanitized (see :func:`sanitize`) and the findings' node ids
    refer to it.
    """
    work = copy.deepcopy(doc)
    findings: list[Finding] = []
    ids = [0]
    for type_, u in plan:
        if findings and type_ == "color.tint_global":
            continue
        if any(f.evidence.get("exclusive") for f in findings):
            break
        got = apply(work, type_, rng, u, [f.box for f in findings], ids)
        if got:
            findings.extend(got)
    return sanitize(work, findings)


def sanitize(doc: Document, findings: Optional[list[Finding]] = None) -> tuple[Document, list[Finding]]:
    """Strip corpus-only evidence from a candidate IR and renumber ids in pre-order.

    Removes ``tokens``, ``meta`` (except render-relevant ``icon_fill``), component ``evidence`` and
    document ``meta``/``source_image``; names of text/icon nodes are regenerated from content. Finding
    ``node_id``s (and id lists in evidence) are remapped to the new ids.
    """
    from dt.ir import assign_ids
    old_ids = [n.id for n in doc.walk()]
    for n in doc.walk():
        n.tokens = {}
        n.meta = {k: v for k, v in (n.meta or {}).items() if k == "icon_fill"}
        if n.component is not None:
            n.component.evidence = {}
        if n.type == "text" and n.text is not None:
            n.name = f"text:{n.text[:24]}"
        elif n.type == "icon" and n.icon_name:
            n.name = f"icon:{n.icon_name}"
    doc.meta = {}
    doc.source_image = None
    assign_ids(doc.root)
    mapping = dict(zip(old_ids, [n.id for n in doc.walk()]))
    out = []
    for f in findings or []:
        f = copy.deepcopy(f)
        if f.node_id is not None:
            f.node_id = mapping.get(f.node_id)
        for k in ("parts", "merged_from", "moved"):
            if k in f.evidence:
                f.evidence[k] = [mapping.get(i, None) for i in f.evidence[k]]
        out.append(f)
    return doc, out


# --------------------------------------------------------------------------- render-verified ground truth
def verify(clean_rgb: np.ndarray, cand_rgb: np.ndarray, findings: list[Finding]) -> dict:
    """Check ground truth against the real renders (both from the same renderer, no noise).

    Returns ``{"ok", "visible": [px per finding], "unexplained_px", "ink_boxes"}``: every finding must
    change >= ``adversary.perturb.min_visible_px`` pixels inside its box, and at most
    ``adversary.perturb.max_unexplained_px`` changed pixels may lie outside all (padded) boxes.
    """
    thr = float(P["adversary.perturb.visible_de"])
    pad = float(P["adversary.perturb.check_pad"])
    d = diff_map(clean_rgb, cand_rgb)
    bad = d > thr
    H, W = bad.shape
    covered = np.zeros_like(bad)
    visible, inks = [], []
    for f in findings:
        x0, y0, x1, y1 = f.box.expand(pad).as_int()
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        sub = bad[y0:y1, x0:x1]
        visible.append(int(sub.sum()))
        covered[y0:y1, x0:x1] = True
        ys, xs = np.nonzero(sub)
        inks.append(Box(x0 + xs.min(), y0 + ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1).to_dict() if len(xs) else None)
    unexplained = int((bad & ~covered).sum())
    ok = all(v >= int(P["adversary.perturb.min_visible_px"]) for v in visible) and unexplained <= int(P["adversary.perturb.max_unexplained_px"])
    return {"ok": bool(ok), "visible": visible, "unexplained_px": unexplained, "ink_boxes": inks,
            "changed_px": int(bad.sum()), "mean_de": float(d.mean())}


# --------------------------------------------------------------------------- noise (target side, NO findings)
@dataclass
class NoiseRecipe:
    """Nuisance applied to the target render. Render-level knobs go into one browser render; image-level
    ones are applied afterwards in a fixed order (blur, then JPEG)."""
    translate: Optional[tuple[float, float]] = None   # noise.subpixel
    smoothing: Optional[str] = None                   # noise.antialias (CSS -webkit-font-smoothing)
    browser: Optional[str] = None                     # noise.browser (alternate browser build)
    blur_sigma: Optional[float] = None                # noise.antialias (resampling blur)
    jpeg_quality: Optional[int] = None                # noise.compression

    def types(self) -> list[str]:
        t = []
        if self.translate:
            t.append("noise.subpixel")
        if self.smoothing or self.blur_sigma:
            t.append("noise.antialias")
        if self.browser:
            t.append("noise.browser")
        if self.jpeg_quality:
            t.append("noise.compression")
        return t

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items() if v is not None}

    def empty(self) -> bool:
        return not self.to_dict()


NOISE_KINDS = ("subpixel", "smoothing", "blur", "jpeg", "browser")


def sample_noise(rng: random.Random, kinds: list[str], n: int = 1) -> NoiseRecipe:
    """Draw ``n`` distinct nuisance kinds from ``kinds`` with random levels."""
    r = NoiseRecipe()
    for k in rng.sample(list(kinds), min(n, len(kinds))):
        if k == "subpixel":
            lo, hi = (float(v) for v in P["adversary.noise.subpixel"])
            r.translate = (round(rng.choice([-1, 1]) * rng.uniform(lo, hi), 3), round(rng.choice([-1, 1]) * rng.uniform(lo, hi), 3))
        elif k == "smoothing":
            r.smoothing = "none"
        elif k == "blur":
            lo, hi = (float(v) for v in P["adversary.noise.blur_sigma"])
            r.blur_sigma = round(rng.uniform(lo, hi), 3)
        elif k == "jpeg":
            lo, hi = (int(v) for v in P["adversary.noise.jpeg_quality"])
            r.jpeg_quality = rng.randint(lo, hi)
        elif k == "browser":
            r.browser = "alt"
    return r


def _noise_css(r: NoiseRecipe) -> str:
    css = ""
    if r.translate:
        # a fractional *layout* offset (LayoutUnit = 1/64 px): bit-reproducible, unlike a fractional CSS
        # transform, which goes through the compositor and rasterises nondeterministically (~1 in 10 renders)
        css += f"body {{ position: relative; left: {r.translate[0]}px; top: {r.translate[1]}px; }}\n"
    if r.smoothing:
        css += f"* {{ -webkit-font-smoothing: {r.smoothing} !important; }}\n"
    return css


def jpeg(rgb: np.ndarray, quality: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def render_target(doc: Document, recipe: Optional[NoiseRecipe] = None, clean: Optional[np.ndarray] = None) -> np.ndarray:
    """Render ``doc`` as the target, with nuisance ``recipe`` applied (None/empty = the clean render)."""
    from dt.render.screenshot import render_doc
    r = recipe or NoiseRecipe()
    css = _noise_css(r)
    if r.browser:
        rgb = alt_browser().render(doc, css)
    elif css or clean is None:
        rgb = render_doc(doc, extra_css=css)
    else:
        rgb = clean.copy()
    if r.blur_sigma:
        rgb = cv2.GaussianBlur(rgb, (0, 0), float(r.blur_sigma))
    if r.jpeg_quality:
        rgb = jpeg(rgb, r.jpeg_quality)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


# --------------------------------------------------------------------------- alternate browser build
def _alt_candidates() -> list[str]:
    env = os.environ.get("DT_ALT_BROWSER")
    if env:
        return [env]
    import glob
    home = os.path.expanduser("~")
    pats = [os.path.join(home, "Library/Caches/ms-playwright/chromium-*/chrome-mac*/Chromium.app/Contents/MacOS/Chromium"),
            os.path.join(home, ".cache/ms-playwright/chromium-*/chrome-linux*/chrome")]
    found = sorted((p for pat in pats for p in glob.glob(pat)), reverse=True)
    return found + ["chromium"]


class _AltBrowser:
    """A second browser build in its own worker thread (Playwright sync objects are thread-bound and the
    main renderer already owns this thread's driver)."""

    def __init__(self) -> None:
        self.q: "queue.Queue" = queue.Queue()
        self.name: Optional[str] = None
        self.version: Optional[str] = None
        self.error: Optional[str] = None
        ready = threading.Event()
        self.t = threading.Thread(target=self._run, args=(ready,), daemon=True)
        self.t.start()
        ready.wait(120)

    def _run(self, ready: threading.Event) -> None:
        try:
            from playwright.sync_api import sync_playwright
            from dt.render.screenshot import launch_browser
            pw = sync_playwright().start()
            try:
                main = None
                from dt.render import screenshot as S
                main = S.BROWSER_USED
                cands = [c for c in _alt_candidates() if c != main]
                br, self.name = launch_browser(pw, cands)
                self.version = br.version
            except Exception as e:
                pw.stop()
                self.error = str(e).splitlines()[0]
                ready.set()
                return
        except Exception as e:  # playwright missing
            self.error = str(e)
            ready.set()
            return
        ready.set()
        ctx = br.new_context(viewport={"width": 800, "height": 600}, device_scale_factor=1)
        while True:
            item = self.q.get()
            if item is None:
                break
            html, w, h, box = item
            try:
                from io import BytesIO
                from PIL import Image
                from dt.render.screenshot import _tmp_html
                page = ctx.new_page()
                try:
                    page.set_viewport_size({"width": int(w), "height": int(h)})
                    page.goto("file://" + _tmp_html(html), wait_until="load")
                    page.evaluate("""async () => { await Promise.race([document.fonts.ready, new Promise(r => setTimeout(r, 5000))]);
                        const used = new Set();
                        for (const el of document.querySelectorAll('body *')) { const cs = getComputedStyle(el); used.add(cs.fontWeight + ' 16px ' + cs.fontFamily.split(',')[0]); }
                        await Promise.all([...used].map(f => document.fonts.load(f).catch(() => null)));
                        await Promise.race([new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))), new Promise(r => setTimeout(r, 2000))]); return true; }""")
                    data = page.screenshot(type="png", animations="disabled", caret="hide")
                finally:
                    page.close()
                box.append(np.asarray(Image.open(BytesIO(data)).convert("RGB"), dtype=np.uint8).copy())
            except Exception as e:  # pragma: no cover - surfaced to the caller
                box.append(e)
        try:
            ctx.close()
            br.close()
            pw.stop()
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self.error is None and self.name is not None

    def render(self, doc: Document, extra_css: str = "") -> np.ndarray:
        if not self.available:
            raise RuntimeError(f"alternate browser unavailable: {self.error}")
        from dt.render.html import render_html
        box: list = []
        self.q.put((render_html(doc, extra_css=extra_css), doc.width, doc.height, box))
        import time
        t0 = time.time()
        while not box:
            if time.time() - t0 > 120:
                raise TimeoutError("alternate browser render timed out")
            time.sleep(0.005)
        if isinstance(box[0], Exception):
            raise box[0]
        return box[0]


_ALT: Optional[_AltBrowser] = None
_ALT_LOCK = threading.Lock()


def alt_browser() -> _AltBrowser:
    global _ALT
    with _ALT_LOCK:
        if _ALT is None:
            _ALT = _AltBrowser()
        return _ALT


def noise_kinds_available(include_browser: bool = True) -> list[str]:
    kinds = ["subpixel", "smoothing", "blur", "jpeg"]
    if include_browser and alt_browser().available:
        kinds.append("browser")
    return kinds
