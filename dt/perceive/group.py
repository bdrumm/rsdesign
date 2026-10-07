"""Grouping pass: infer the transparent containers that perception cannot see as paint.

Segmentation finds painted regions, OCR finds text and icon glyphs, and ``hierarchy`` nests them
by containment; nothing produces the *unpainted* frames a design tree is made of: list items,
lists, text buttons, standard icon buttons, tabs, top app bars and navigation destinations.
This module adds them deterministically from sibling geometry, one container at a time
(bottom-up, called at the end of :func:`dt.perceive.hierarchy.build_tree`).

Pipeline for one container ``parent``:

1. **Units.** Children become units: ``text``; ``glyph`` (an icon, or a cluster of small glyph
   fragments such as the three dots of ``more_vert`` or an icon OCR'd as ``"="``, merged when
   they are within ``glyph_merge_gap`` px and fit in ``glyph_max``); ``control`` (small painted
   shapes: switch track, checkbox, avatar); ``painted`` (anything larger). ``line`` children are
   separators, not units. A glyph (or small pill holding one) centred above a short label is
   fused into a ``dest`` unit (icon-over-label destination).
2. **Segmentation** (recursive XY-cut). Units are split into horizontal *bands* of siblings whose
   vertical extents overlap. A vertical whitespace stripe of at least ``col_gap`` px is used as a
   column cut first when the two sides have their own row rhythm (the union has clearly fewer
   bands than either side: a navigation rail next to a list), never when the stripe only
   separates the parts of the same rows (leading icon | text | trailing control). Each leaf
   band gets *bounds*: its column's x-range and the y-range halfway to the neighbouring bands
   (or the divider line between them).
3. **Classification of each leaf band** (first match wins):
   * equal-width cells whose labels sit at the cell centres -> ``tab`` frames (+ ``tabs``), or
     ``destination`` frames (+ ``navigation_bar``) when the cells are icon-over-label pairs at the
     bottom of the screen;
   * band inside the top ``appbar_h`` px of the screen -> one 40x40 ``icon_button`` frame per
     glyph, plus a ``top_app_bar`` frame when the bar has no painted container of its own;
   * text + leading glyph/control and/or trailing element near the column edges -> ``list_item``
     frame spanning the column, height snapped to the smallest of ``item_heights`` that leaves
     ``item_min_vpad`` around the content, clipped to the band bounds;
   * otherwise: short single-line labels in an accent colour -> ``text_button`` frames (40 px high,
     ``text_button_pad`` side padding, ``text_button_min_w`` minimum width); a glyph at the right
     edge of a band inside a painted container -> ``icon_button``;
   * a stacked column of icon-over-label pairs -> ``destination`` frames (navigation rail).
4. **Lists.** Runs of >= ``list_min_items`` contiguous list items (same column, frames touching
   within ``list_gap_tol`` px with only dividers between them, i.e. a consistent pitch) are wrapped
   in a ``list`` frame together with those dividers.

Every frame is clipped to its parent, must contain its members, adopts siblings it fully covers,
and never crosses a sibling's boundary (it is clipped away from a partially overlapping sibling,
or dropped when that would cut a member). Frames that would duplicate the parent or an existing
sibling (IoU >= ``dup_iou``) are not emitted. Inferred frames carry ``meta = {"src": "group",
"group_kind": <kind>, "conf": ...}`` and no paint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dt.ir import Box, Node, new_id
from dt.params import P, register

register("perceive.group.enabled", True, "run the grouping pass that infers unpainted container frames (list items, buttons, tabs, app bars)")
register("perceive.group.glyph_max", 28, "px: max side of an icon glyph (cluster) unit", (20, 40))
register("perceive.group.glyph_min", 8, "px: glyph clusters smaller than this on both sides are specks, not icons", (2, 16))
register("perceive.group.glyph_merge_gap", 4, "px: glyph fragments closer than this merge into one icon cluster", (0, 8))
register("perceive.group.glyph_text_chars", 2, "max characters of an OCR text that is really an icon glyph ('=', '>', 'D')", (1, 3))
register("perceive.group.control_max_w", 64, "px: painted shapes up to this width are row controls (switch, checkbox, avatar)", (40, 96))
register("perceive.group.control_max_h", 56, "px: painted shapes up to this height are row controls", (32, 72))
register("perceive.group.row_overlap", 1.0, "px of vertical overlap for two siblings to share a band", (0.0, 6.0))
register("perceive.group.col_gap", 16, "px: min vertical whitespace stripe considered as a column cut", (8, 48))
register("perceive.group.col_band_ratio", 0.6, "a column cut is taken when the union of both sides has fewer than this fraction of the smaller side's bands", (0.3, 0.95))
register("perceive.group.appbar_h", 64, "px: M3 small top app bar height (top-of-screen zone for icon buttons / app bar frame)", (48, 80))
register("perceive.group.appbar_min_w_frac", 0.5, "an app-bar frame must span at least this fraction of the screen width", (0.2, 1.0))
register("perceive.group.icon_button", 40, "px: M3 standard icon button target size centred on a lone icon", (32, 48))
register("perceive.group.edge_max", 32, "px: a leading/trailing row element lies within this distance of its column edge", (8, 64))
register("perceive.group.trail_gap", 24, "px: min gap between a list row's text and its trailing element", (8, 64))
register("perceive.group.item_heights", [56, 72, 88], "M3 list item heights (1/2/3-line); a row frame snaps to the smallest that fits")
register("perceive.group.item_min_vpad", 8, "px: min vertical padding around list item content when snapping to item_heights", (0, 16))
register("perceive.group.item_min_fill", 0.85, "a list item clipped by its neighbours to less than this fraction of its snapped height is not a list item (rows packed tighter than M3 items)", (0.5, 1.0))
register("perceive.group.min_keep_frac", 0.85, "fixed-size frames (icon/text button, tab, destination) clipped below this fraction of their area are dropped", (0.3, 1.0))
register("perceive.group.list_min_items", 3, "min contiguous list items that form a list frame", (2, 6))
register("perceive.group.list_gap_tol", 3, "px: max gap between consecutive list item frames of one list (dividers excepted)", (0, 12))
register("perceive.group.text_button_h", 40, "px: M3 text button height", (32, 48))
register("perceive.group.text_button_pad", 12, "px: M3 text button horizontal padding", (8, 24))
register("perceive.group.text_button_min_w", 64, "px: M3 button minimum width", (40, 96))
register("perceive.group.text_button_icon_gap", 12, "px: max gap between a text button's leading icon and its label", (4, 20))
register("perceive.group.align_tol", 4.0, "px: vertical centre tolerance for icon/label alignment inside a button", (1.0, 8.0))
register("perceive.group.appbar_icon_buttons", True, "wrap app-bar glyphs in standard icon-button frames")
register("perceive.group.text_button_max_chars", 20, "max characters of a text button label", (8, 40))
register("perceive.group.text_button_max_h", 24, "px: max OCR box height of a (single-line) button label", (16, 32))
register("perceive.group.text_button_min_chroma", 20.0, "min CIELab chroma of a label colour to read as an accent (primary) text button", (8.0, 40.0))
register("perceive.group.tab_h", 48, "px: M3 text-only tab height", (40, 56))
register("perceive.group.tab_icon_h", 64, "px: M3 icon+label tab height", (56, 80))
register("perceive.group.cell_center_tol", 6.0, "px: label centre tolerance against an equal-width cell centre (tabs, nav bar)", (2.0, 16.0))
register("perceive.group.cell_max_chars", 24, "max characters of a tab / destination label", (8, 40))
register("perceive.group.dest_max_gap", 16, "px: max gap between an icon and the label below it (destination)", (4, 24))
register("perceive.group.dest_center_tol", 6.0, "px: max horizontal centre offset of icon and label in a destination", (2.0, 12.0))
register("perceive.group.navbar_h", 80, "px: M3 navigation bar height", (64, 96))
register("perceive.group.rail_item_pad", 4, "px: vertical padding of a navigation rail destination around icon+label", (0, 12))
register("perceive.group.destinations", True, "emit destination frames around icon-over-label pairs")
register("perceive.group.dup_iou", 0.85, "IoU with the parent or a sibling above which an inferred frame is a duplicate", (0.6, 0.99))
register("perceive.group.bar_tol", 8, "px: a painted parent within this of appbar_h / navbar_h already is the bar", (2, 16))


@dataclass
class Unit:
    box: Box
    nodes: list[Node]
    kind: str  # text | glyph | control | painted | dest | speck

    @property
    def label(self) -> Optional[Node]:
        return next((n for n in self.nodes if n.type == "text"), None)


@dataclass
class Leaf:
    bounds: Box  # soft bounds: column x-range, y halfway to the neighbouring bands (or the divider)
    units: list[Unit] = field(default_factory=list)
    hard: Optional[Box] = None  # hard bounds: y up to the neighbouring bands' content (or the divider)

    def __post_init__(self) -> None:
        if self.hard is None:
            self.hard = self.bounds


# ----------------------------------------------------------------------------- small helpers
def _union(boxes: list[Box]) -> Box:
    b = boxes[0]
    for o in boxes[1:]:
        b = b.union(o)
    return b


def _gap(a: Box, b: Box) -> float:
    gx = max(0.0, max(a.x, b.x) - min(a.x2, b.x2))
    gy = max(0.0, max(a.y, b.y) - min(a.y2, b.y2))
    return max(gx, gy)


def _painted(n: Node) -> bool:
    return bool(n.fills or n.strokes) or n.type in ("image", "line") or n.meta.get("src") == "group"


def _has_text(n: Node) -> bool:
    return any(c.type == "text" for c in n.walk())


def _txt(n: Node) -> str:
    return "".join((n.text or "").split())


def _glyph_text(n: Node) -> bool:
    """An OCR 'word' that is really an icon glyph ('=', '>', 'D', '<+:')."""
    s = _txt(n)
    if not s or max(n.box.w, n.box.h) > float(P["perceive.group.glyph_max"]) + 6:
        return False
    k = int(P["perceive.group.glyph_text_chars"])
    if len(s) == 1:
        return True
    return len(s) <= k + 1 and not any(ch.isalnum() for ch in s)


def _label_box(n: Node) -> Box:
    """Horizontal extent of a label's ink (OCR text boxes carry side bearings + slack) with the
    vertical extent of its line box (buttons centre the line box, not the ink)."""
    ink = n.meta.get("ink") if n.type == "text" else None
    if not isinstance(ink, dict):
        return n.box
    b = Box.from_dict(ink)
    return Box(b.x, n.box.y, b.w, n.box.h)


def _chroma(n: Node) -> float:
    if n.text_style is None or n.text_style.color is None:
        return 0.0
    _, a, b = n.text_style.color.lab()
    return (a * a + b * b) ** 0.5


# ----------------------------------------------------------------------------- 1. units
def _cluster_glyphs(pieces: list[Node], mg: float, gmax: float) -> list[tuple[Box, list[Node]]]:
    """Agglomerate glyph fragments into icon clusters: repeatedly merge the first pair (i, j) in
    list order whose gap is <= ``mg`` and whose union fits in ``gmax``; the merged cluster takes
    slot i. The pair predicate is kept as a matrix and only the merged row is recomputed, so a
    page with hundreds of fragments (a 60-row list of icons) costs O(n^2) numpy work instead of
    the O(n^3) Python pair scans of restarting after every merge; the result is identical."""
    clusters: list[tuple[Box, list[Node]]] = [(p.box, [p]) for p in pieces]
    if len(clusters) < 2:
        return clusters
    xs = np.array([[b.x, b.y, b.x2, b.y2] for b, _ in clusters], dtype=np.float64)

    def ok(row: np.ndarray, rest: np.ndarray) -> np.ndarray:
        gx = np.maximum(0.0, np.maximum(row[0], rest[:, 0]) - np.minimum(row[2], rest[:, 2]))
        gy = np.maximum(0.0, np.maximum(row[1], rest[:, 1]) - np.minimum(row[3], rest[:, 3]))
        uw = np.maximum(row[2], rest[:, 2]) - np.minimum(row[0], rest[:, 0])
        uh = np.maximum(row[3], rest[:, 3]) - np.minimum(row[1], rest[:, 1])
        return (np.maximum(gx, gy) <= mg) & (np.maximum(uw, uh) <= gmax)

    M = np.stack([ok(xs[i], xs) for i in range(len(xs))])
    np.fill_diagonal(M, False)
    while True:
        T = np.triu(M, 1)
        k = int(np.argmax(T))
        i, j = divmod(k, T.shape[1])
        if not T[i, j]:
            return clusters
        u = clusters[i][0].union(clusters[j][0])
        clusters[i] = (u, clusters[i][1] + clusters[j][1])
        clusters.pop(j)
        xs = np.delete(xs, j, axis=0)
        M = np.delete(np.delete(M, j, axis=0), j, axis=1)
        xs[i] = (u.x, u.y, u.x2, u.y2)
        M[i] = M[:, i] = ok(xs[i], xs)
        M[i, i] = False


def _units(kids: list[Node]) -> list[Unit]:
    gmax = float(P["perceive.group.glyph_max"])
    cw, ch = float(P["perceive.group.control_max_w"]), float(P["perceive.group.control_max_h"])
    pieces: list[Node] = []
    units: list[Unit] = []
    for k in kids:
        if not k.visible or k.type == "line":
            continue
        small = max(k.box.w, k.box.h) <= gmax
        if k.type == "text":
            if _glyph_text(k):
                pieces.append(k)
            else:
                units.append(Unit(k.box, [k], "text"))
        elif k.type == "icon" and small:
            pieces.append(k)
        elif small and k.type in ("rect", "ellipse", "vector", "frame") and not _has_text(k):
            pieces.append(k)
        elif k.box.w <= cw and k.box.h <= ch and all(len(_txt(t)) <= 2 for t in k.walk() if t.type == "text"):
            units.append(Unit(k.box, [k], "control"))  # switch track, checkbox, avatar (initial letter)
        else:
            units.append(Unit(k.box, [k], "painted"))
    clusters = _cluster_glyphs(pieces, float(P["perceive.group.glyph_merge_gap"]), gmax)
    gmin = float(P["perceive.group.glyph_min"])
    for b, ns in clusters:
        units.append(Unit(b, ns, "glyph" if max(b.w, b.h) >= gmin else "speck"))
    return _pair_destinations([u for u in units if u.kind != "speck"]) + [u for u in units if u.kind == "speck"]


def _short_label(u: Unit, max_chars: int) -> bool:
    t = u.label
    return (u.kind == "text" and t is not None and len(_txt(t)) <= max_chars
            and t.box.h <= float(P["perceive.group.text_button_max_h"]) and "\n" not in (t.text or "").strip())


def _pair_destinations(units: list[Unit]) -> list[Unit]:
    """Fuse an icon (or a small pill holding one) centred above a short label into a ``dest`` unit."""
    tol, gap_max = float(P["perceive.group.dest_center_tol"]), float(P["perceive.group.dest_max_gap"])
    maxc = int(P["perceive.group.cell_max_chars"])
    icons = [u for u in units if u.kind == "glyph" or (u.kind == "control" and any(c.type == "icon" for c in u.nodes[0].walk()))]
    labels = [u for u in units if _short_label(u, maxc)]
    used: set[int] = set()
    out: list[Unit] = []
    for g in icons:
        best, bd = None, None
        for t in labels:
            if id(t) in used:
                continue
            dy = t.box.y - g.box.y2
            if 0 <= dy + 1 <= gap_max + 1 and abs(t.box.cx - g.box.cx) <= tol and t.box.w <= 4 * max(g.box.w, 24):
                if bd is None or dy < bd:
                    best, bd = t, dy
        if best is not None:
            used.add(id(g))
            used.add(id(best))
            out.append(Unit(g.box.union(best.box), g.nodes + best.nodes, "dest"))
    return [u for u in units if id(u) not in used] + out


# ----------------------------------------------------------------------------- 2. segmentation
def _ybands(units: list[Unit]) -> list[list[Unit]]:
    ov = float(P["perceive.group.row_overlap"])
    bands: list[list[Unit]] = []
    y2 = None
    for u in sorted(units, key=lambda u: (u.box.y, u.box.x)):
        if bands and u.box.y < y2 - ov:
            bands[-1].append(u)
            y2 = max(y2, u.box.y2)
        else:
            bands.append([u])
            y2 = u.box.y2
    return bands


def _xcut(units: list[Unit]) -> Optional[tuple[float, float]]:
    """Widest vertical whitespace stripe that separates two independent columns, if any."""
    if len(units) < 4:
        return None
    iv = sorted((u.box.x, u.box.x2) for u in units)
    gaps: list[tuple[float, float]] = []
    end = iv[0][1]
    cg = float(P["perceive.group.col_gap"])
    for a, b in iv[1:]:
        if a - end >= cg:
            gaps.append((end, a))
        end = max(end, b)
    n_all = len(_ybands(units))
    ratio = float(P["perceive.group.col_band_ratio"])
    for a, b in sorted(gaps, key=lambda g: -(g[1] - g[0])):
        left = [u for u in units if u.box.x2 <= a + 0.5]
        right = [u for u in units if u.box.x >= b - 0.5]
        nl, nr = len(_ybands(left)), len(_ybands(right))
        if min(nl, nr) >= 2 and n_all < ratio * min(nl, nr):
            return a, b
    return None


def _segment(units: list[Unit], bounds: Box, lines: list[Node], stripes: list[tuple[float, float]],
             hard: Optional[Box] = None, depth: int = 0) -> list[Leaf]:
    """Recursive XY-cut into leaf bands (see module docstring). Column cuts are appended to `stripes`."""
    hard = hard or bounds
    if not units:
        return []
    if depth > 12:
        return [Leaf(bounds, units, hard)]
    cut = _xcut(units)
    if cut is not None:
        a, b = cut
        stripes.append(cut)
        m = (a + b) / 2.0
        left = [u for u in units if u.box.x2 <= a + 0.5]
        right = [u for u in units if u.box.x >= b - 0.5]
        lb = Box(bounds.x, bounds.y, max(0.0, m - bounds.x), bounds.h)
        rb = Box(m, bounds.y, max(0.0, bounds.x2 - m), bounds.h)
        lh = Box(lb.x, hard.y, lb.w, hard.h)
        rh = Box(rb.x, hard.y, rb.w, hard.h)
        return (_segment(left, lb, lines, stripes, lh, depth + 1) + _segment(right, rb, lines, stripes, rh, depth + 1))
    bands = _ybands(units)
    if len(bands) == 1:
        return [Leaf(bounds, units, hard)]
    exts = [(_union([u.box for u in bd])) for bd in bands]
    tops, htops = [bounds.y], [hard.y]
    bots: list[float] = []
    hbots: list[float] = []
    for i in range(len(bands) - 1):
        lo, hi = exts[i].y2, exts[i + 1].y
        seps = [ln for ln in lines if ln.box.y >= lo - 1 and ln.box.y2 <= hi + 1
                and min(ln.box.x2, bounds.x2) - max(ln.box.x, bounds.x) > 0.5 * min(ln.box.w, bounds.w)]
        if seps:
            bots.append(min(ln.box.y for ln in seps))
            tops.append(max(ln.box.y2 for ln in seps))
            hbots.append(bots[-1])
            htops.append(tops[-1])
        else:
            m = (lo + hi) / 2.0
            bots.append(m)
            tops.append(m)
            hbots.append(hi)
            htops.append(lo)
    bots.append(bounds.y2)
    hbots.append(hard.y2)
    out: list[Leaf] = []
    for bd, t, b, ht, hb in zip(bands, tops, bots, htops, hbots):
        out += _segment(bd, Box(bounds.x, t, bounds.w, max(0.0, b - t)), lines, stripes,
                        Box(hard.x, ht, hard.w, max(0.0, hb - ht)), depth + 1)
    return out


def _split_by_stripes(leaves: list[Leaf], stripes: list[tuple[float, float]]) -> list[Leaf]:
    """A column stripe found anywhere in the container also splits other bands it runs through
    cleanly (e.g. a navigation rail FAB that shares a band with the first list row)."""
    out: list[Leaf] = []
    todo = list(leaves)
    while todo:
        lf = todo.pop(0)
        for a, b in stripes:
            m = (a + b) / 2.0
            if not (lf.bounds.x < m < lf.bounds.x2):
                continue
            left = [u for u in lf.units if u.box.x2 <= a + 0.5]
            right = [u for u in lf.units if u.box.x >= b - 0.5]
            if left and right and len(left) + len(right) == len(lf.units):
                bl, br = lf.bounds, lf.hard
                todo.insert(0, Leaf(Box(m, bl.y, bl.x2 - m, bl.h), right, Box(m, br.y, br.x2 - m, br.h)))
                todo.insert(0, Leaf(Box(bl.x, bl.y, m - bl.x, bl.h), left, Box(br.x, br.y, m - br.x, br.h)))
                break
        else:
            out.append(lf)
    return out


# ----------------------------------------------------------------------------- frame emission
class _Ctx:
    def __init__(self, root: Node):
        self.root = root
        self.made = 0


def _clip_away(box: Box, s: Box, keep: Box) -> Optional[Box]:
    """Largest of the four boxes obtained by moving one edge of `box` off `s` that still contains `keep`."""
    cands = [
        Box(s.x2, box.y, box.x2 - s.x2, box.h),     # move left edge right
        Box(box.x, box.y, s.x - box.x, box.h),      # move right edge left
        Box(box.x, s.y2, box.w, box.y2 - s.y2),     # move top edge down
        Box(box.x, box.y, box.w, s.y - box.y),      # move bottom edge up
    ]
    ok = [c for c in cands if c.w > 0 and c.h > 0 and c.contains(keep, tol=0.5)]
    return max(ok, key=lambda c: c.area) if ok else None


_FIXED = ("icon_button", "text_button", "tab", "destination")


def _emit(parent: Node, box: Box, members: list[Node], kind: str, conf: float = 0.6) -> Optional[Node]:
    """Create a transparent frame `box` in `parent` holding `members` (direct children of parent)."""
    if not members:
        return None
    want = box.area
    ids = {id(m) for m in members}
    if any(id(m) not in {id(c) for c in parent.children} for m in members):
        return None
    keep = _union([m.box for m in members])
    box = box.union(keep).intersect(parent.box) if parent.box.contains(keep, tol=0.5) else box.union(keep)
    if box.w <= 0 or box.h <= 0:
        return None
    dup = float(P["perceive.group.dup_iou"])
    if parent.id != "root" and box.iou(parent.box) >= dup:
        return None
    members = list(members)
    for _ in range(3):  # siblings can change the box; settle in a few passes
        changed = False
        for s in parent.children:
            if id(s) in ids or not s.visible:
                continue
            inter = box.intersect(s.box)
            if inter.w <= 0.5 or inter.h <= 0.5:
                continue
            if box.contains(s.box, tol=0.5):
                if s.meta.get("src") == "group" and s.box.iou(box) >= dup:
                    return None
                members.append(s)
                ids.add(id(s))
                changed = True
                continue
            nb = _clip_away(box, s.box, keep)
            if nb is None:
                return None
            box = nb
            changed = True
        if not changed:
            break
    for s in parent.children:
        if id(s) not in ids and s.box.iou(box) >= dup:
            return None
    if kind in _FIXED and box.area < float(P["perceive.group.min_keep_frac"]) * want:
        return None
    f = Node(id=new_id("g"), type="frame", name=kind, box=Box(round(box.x, 1), round(box.y, 1), round(box.w, 1), round(box.h, 1)))
    f.meta = {"src": "group", "group_kind": kind, "conf": conf}
    f.children = [c for c in parent.children if id(c) in ids]
    parent.children = [c for c in parent.children if id(c) not in ids]
    parent.children.append(f)
    return f


def _nodes(us: list[Unit]) -> list[Node]:
    return [n for u in us for n in u.nodes]


# ----------------------------------------------------------------------------- 3. classification
def _equal_cells(cells: list[Unit], bounds: Box) -> bool:
    n = len(cells)
    if n < 2 or bounds.w <= 0:
        return False
    tol = float(P["perceive.group.cell_center_tol"])
    w = bounds.w / n
    for i, u in enumerate(sorted(cells, key=lambda u: u.box.cx)):
        if abs(u.box.cx - (bounds.x + (i + 0.5) * w)) > tol or u.box.w > w:
            return False
    return True


def _cells(parent: Node, leaf: Leaf, ctx: _Ctx) -> bool:
    """Tabs / navigation bar: equal-width cells with centred labels."""
    U = leaf.units
    maxc = int(P["perceive.group.cell_max_chars"])
    if len(U) < 2 or not all(u.kind == "dest" or _short_label(u, maxc) for u in U):
        return False
    b = leaf.bounds
    if not _equal_cells(U, b):
        return False
    U = sorted(U, key=lambda u: u.box.cx)
    n = len(U)
    content = _union([u.box for u in U])
    cy = content.cy
    w = b.w / n
    root = ctx.root
    all_dest = all(u.kind == "dest" for u in U)
    nav_h = float(P["perceive.group.navbar_h"])
    bar_tol = float(P["perceive.group.bar_tol"])
    if all_dest and n >= 3 and content.y2 >= root.box.y2 - nav_h:
        if not P["perceive.group.destinations"]:
            return True
        bar_like = parent.id != "root" and abs(parent.box.h - nav_h) <= bar_tol and _painted(parent)
        host = parent
        if not bar_like:
            y0 = max(b.y, min(cy - nav_h / 2, root.box.y2 - nav_h))
            host = _emit(parent, Box(b.x, y0, b.w, nav_h), _nodes(U), "navigation_bar") or parent
            ys, hh = host.box.y, host.box.h
        else:
            ys, hh = parent.box.y, parent.box.h
        for i, u in enumerate(U):
            _emit(host, Box(b.x + i * w, ys, w, hh), u.nodes, "destination")
        return True
    if content.y2 >= root.box.y2 - nav_h:  # labels of a navigation bar whose icons were not seen
        return False
    h = float(P["perceive.group.tab_icon_h"] if any(u.kind == "dest" for u in U) else P["perceive.group.tab_h"])
    tabs = _emit(parent, Box(b.x, cy - h / 2, b.w, h), _nodes(U), "tabs")
    if tabs is None:
        return False
    # adopt the divider / active indicator lines directly under the tab strip
    under = [c for c in parent.children if c.type == "line" and c.box.y >= tabs.box.y2 - 4 and c.box.y <= tabs.box.y2 + 2
             and c.box.x >= tabs.box.x - 1 and c.box.x2 <= tabs.box.x2 + 1]
    if under:
        ids = {id(c) for c in under}
        tabs.children += under
        parent.children = [c for c in parent.children if id(c) not in ids]
        tabs.box = _union([tabs.box] + [c.box for c in under])
    for i, u in enumerate(U):
        _emit(tabs, Box(b.x + i * w, cy - h / 2, w, h), u.nodes, "tab")
    return True


def _top_bar(parent: Node, leaf: Leaf, ctx: _Ctx) -> bool:
    """Band inside the top app-bar zone: icon buttons for its glyphs (+ an app-bar frame)."""
    root = ctx.root
    ah = float(P["perceive.group.appbar_h"])
    U = leaf.units
    content = _union([u.box for u in U])
    if content.y < root.box.y - 1 or content.y2 > root.box.y + ah + 1:
        return False
    if any(u.kind == "painted" for u in U):
        return False
    glyphs = [u for u in U if u.kind == "glyph"]
    texts = [u for u in U if u.kind == "text"]
    if not glyphs:
        return False
    bar_tol = float(P["perceive.group.bar_tol"])
    bar_like = parent.id != "root" and _painted(parent) and parent.box.y <= root.box.y + bar_tol and abs(parent.box.h - ah) <= bar_tol
    if parent.id != "root" and _painted(parent) and not bar_like:  # inside a button / FAB / card
        return False
    host = parent
    if not bar_like and (texts or len(glyphs) >= 2) and leaf.bounds.w >= float(P["perceive.group.appbar_min_w_frac"]) * root.box.w:
        host = _emit(parent, Box(leaf.bounds.x, root.box.y, leaf.bounds.w, ah), _nodes(U), "top_app_bar") or parent
    if P["perceive.group.appbar_icon_buttons"]:
        _icon_buttons(host, glyphs)
    return True


def _icon_like(g: Unit) -> bool:
    """A glyph cluster drawn as an icon (not a painted control such as a checkbox box)."""
    return not any(n.children or n.strokes or n.type == "frame" for n in g.nodes)


def _icon_buttons(host: Node, glyphs: list[Unit]) -> None:
    s = float(P["perceive.group.icon_button"])
    for g in glyphs:
        _emit(host, Box(g.box.cx - s / 2, g.box.cy - s / 2, s, s), g.nodes, "icon_button")


@dataclass
class _Item:
    leaf: Leaf
    content: Box
    h: float  # snapped M3 item height
    y0: float
    y1: float


def _list_item(parent: Node, leaf: Leaf, ctx: _Ctx) -> Optional[_Item]:
    """List-item candidate for a band (emitted later, after neighbouring items share the space)."""
    U = sorted(leaf.units, key=lambda u: u.box.x)
    if len(U) < 2 or any(u.kind in ("painted", "dest") for u in U):
        return None
    texts = [u for u in U if u.kind == "text"]
    if not texts or all(_text_button_label(u) for u in texts):  # a row of text buttons
        return None
    b = leaf.bounds
    edge = float(P["perceive.group.edge_max"])
    tx0, tx1 = min(t.box.x for t in texts), max(t.box.x2 for t in texts)
    lead = U[0] if U[0].kind in ("glyph", "control") and U[0].box.x2 <= tx0 and U[0].box.x - b.x <= edge else None
    last = U[-1]
    others = [t for t in texts if t is not last]
    trail = None
    if others and b.x2 - last.box.x2 <= edge and last.box.x - max(t.box.x2 for t in others) >= float(P["perceive.group.trail_gap"]):
        trail = last
    elif last.kind != "text" and b.x2 - last.box.x2 <= edge and last.box.x - tx1 >= float(P["perceive.group.trail_gap"]):
        trail = last
    if lead is None and trail is None:
        return None
    content = _union([u.box for u in U])
    pad = float(P["perceive.group.item_min_vpad"])
    hs = sorted(float(h) for h in P["perceive.group.item_heights"])
    h = next((x for x in hs if x >= content.h + 2 * pad), None)
    if h is None:  # taller than a 3-line item: several rows merged, not one item
        return None
    hb = leaf.hard
    return _Item(leaf, content, h, max(hb.y, content.cy - h / 2), min(hb.y2, content.cy + h / 2))


def _emit_items(parent: Node, cands: list[_Item]) -> list[Node]:
    """Neighbouring items in one column split any overlap halfway (never cutting content); an
    item left with less than ``item_min_fill`` of its snapped height is dropped (rows packed
    tighter than M3 list items: radio/checkbox groups, dense tables)."""
    fill = float(P["perceive.group.item_min_fill"])
    cols: dict[tuple[int, int], list[_Item]] = {}
    for c in cands:
        cols.setdefault((round(c.leaf.bounds.x), round(c.leaf.bounds.w)), []).append(c)
    out: list[Node] = []
    for col in cols.values():
        col.sort(key=lambda c: c.content.y)
        packed: set[int] = set()
        for a, b in zip(col, col[1:]):  # centre pitch below an item height: a dense control group
            if b.content.cy - a.content.cy < fill * min(a.h, b.h) and b.y0 <= a.y1 + 1:
                packed.update((id(a), id(b)))
        for a, b in zip(col, col[1:]):
            if a.y1 > b.y0:
                m = min(max((a.y1 + b.y0) / 2.0, a.content.y2), b.content.y)
                a.y1, b.y0 = min(a.y1, m), max(b.y0, m)
        for c in col:
            if id(c) in packed or c.y1 - c.y0 < fill * c.h:
                continue
            b = c.leaf.bounds
            n = _emit(parent, Box(b.x, c.y0, b.w, c.y1 - c.y0), _nodes(c.leaf.units), "list_item")
            if n is not None:
                out.append(n)
    return out


def _text_button_label(u: Unit) -> bool:
    """Short single-line label in an accent colour (M3 text buttons use the primary colour)."""
    return (_short_label(u, int(P["perceive.group.text_button_max_chars"]))
            and _chroma(u.label) >= float(P["perceive.group.text_button_min_chroma"]))


def _button_shaped(u: Unit) -> bool:
    """A painted unit with the height of an M3 button (filled / tonal / outlined / elevated)."""
    bh, tol = float(P["perceive.group.text_button_h"]), float(P["perceive.group.bar_tol"])
    return u.kind in ("painted", "control") and abs(u.box.h - bh) <= tol and u.box.w >= u.box.h


def _buttons(parent: Node, leaf: Leaf, ctx: _Ctx) -> None:
    """Text buttons (accent label, optional leading icon) and icon buttons in a row of buttons."""
    U = leaf.units
    bh, pad, mw = float(P["perceive.group.text_button_h"]), float(P["perceive.group.text_button_pad"]), float(P["perceive.group.text_button_min_w"])
    ig, ctol = float(P["perceive.group.text_button_icon_gap"]), float(P["perceive.group.align_tol"])
    btn_like = parent.id != "root" and _painted(parent) and parent.box.h <= bh + float(P["perceive.group.bar_tol"])
    used: set[int] = set()
    n_text = 0
    labels = [u for u in U if _text_button_label(u)]
    # a lone accent-coloured text on the page is not a button unless it sits among other buttons
    # or inside a painted container (card / dialog / snackbar actions)
    in_card = parent.id != "root" and _painted(parent)
    if not (len(labels) >= 2 or in_card or any(_button_shaped(o) for o in U)):
        labels = []
    for u in ([] if btn_like else labels):
        t = u.label
        if not _text_button_label(u):
            continue
        members = [u]
        lb = _label_box(t)
        icons = [g for g in U if g.kind == "glyph" and id(g) not in used and 0 <= lb.x - g.box.x2 <= ig
                 and abs(g.box.cy - lb.cy) <= ctol]
        if icons:
            members.append(max(icons, key=lambda g: g.box.x2))
        content = _union([lb] + [m.box for m in members if m is not u])
        w = max(content.w + 2 * pad, mw)
        box = Box(content.cx - w / 2, t.box.cy - bh / 2, w, bh)
        if any(all(o is not m for m in members) and o.kind != "speck" and box.intersect(o.box).area > 0 for o in U):
            continue
        if _emit(parent, box, _nodes(members), "text_button") is not None:
            used.update(id(m) for m in members)
            n_text += 1
    if btn_like or not (n_text or any(_button_shaped(o) for o in U)):
        return
    s = float(P["perceive.group.icon_button"])
    for g in U:
        if g.kind != "glyph" or id(g) in used or not _icon_like(g):
            continue
        box = Box(g.box.cx - s / 2, g.box.cy - s / 2, s, s)
        if any(o is not g and o.kind != "speck" and box.intersect(o.box).area > 0 for o in U):
            continue
        _icon_buttons(parent, [g])


def _rail(parent: Node, leaves: list[Leaf]) -> set[int]:
    """Stacked single-destination bands of one column (navigation rail) -> destination frames."""
    done: set[int] = set()
    if not P["perceive.group.destinations"]:
        return done
    pad = float(P["perceive.group.rail_item_pad"])
    singles = [lf for lf in leaves if len(lf.units) == 1 and lf.units[0].kind == "dest"]
    cols: dict[tuple[float, float], list[Leaf]] = {}
    for lf in singles:
        cols.setdefault((round(lf.bounds.x), round(lf.bounds.w)), []).append(lf)
    for lfs in cols.values():
        if len(lfs) < 3:
            continue
        for lf in lfs:
            u = lf.units[0]
            _emit(parent, Box(lf.bounds.x, u.box.y - pad, lf.bounds.w, u.box.h + 2 * pad), u.nodes, "destination")
            done.add(id(lf))
    return done


def _lists(parent: Node, items: list[Node]) -> None:
    """Wrap runs of contiguous list items (consistent pitch, same column) in a list frame."""
    nmin = int(P["perceive.group.list_min_items"])
    tol = float(P["perceive.group.list_gap_tol"])
    items = sorted(items, key=lambda n: n.box.y)
    runs: list[list[Node]] = []
    for it in items:
        if runs:
            prev = runs[-1][-1]
            between = [c for c in parent.children if c is not prev and c is not it and c.type != "line"
                       and c.box.cy > prev.box.y2 - 0.5 and c.box.cy < it.box.y + 0.5
                       and c.box.x < it.box.x2 and c.box.x2 > it.box.x]
            lines_h = sum(c.box.h for c in parent.children if c.type == "line" and c.box.y >= prev.box.y2 - 1 and c.box.y2 <= it.box.y + 1)
            if (abs(prev.box.x - it.box.x) <= 2 and abs(prev.box.w - it.box.w) <= 2 and not between
                    and it.box.y - prev.box.y2 <= tol + lines_h):
                runs[-1].append(it)
                continue
        runs.append([it])
    for run in runs:
        if len(run) < nmin:
            continue
        box = _union([n.box for n in run])
        divs = [c for c in parent.children if c.type == "line" and box.contains(c.box, tol=1.0)]
        _emit(parent, box, run + divs, "list", conf=0.5)


# ----------------------------------------------------------------------------- driver
def group_container(parent: Node, ctx: _Ctx) -> None:
    """Infer grouping frames among the direct children of one container (in place)."""
    kids = [c for c in parent.children if c.visible]
    if len(kids) < 1:
        return
    units = [u for u in _units(kids) if u.kind != "speck"]
    if not units:
        return
    lines = [c for c in kids if c.type == "line"]
    stripes: list[tuple[float, float]] = []
    leaves = _split_by_stripes(_segment(units, parent.box, lines, stripes), stripes)
    rail_done = _rail(parent, leaves)
    cands: list[_Item] = []
    for lf in leaves:
        if id(lf) in rail_done:
            continue
        if _cells(parent, lf, ctx):
            continue
        if _top_bar(parent, lf, ctx):
            continue
        it = _list_item(parent, lf, ctx)
        if it is not None:
            cands.append(it)
            continue
        _buttons(parent, lf, ctx)
    items = _emit_items(parent, cands)
    if items:
        _lists(parent, items)


def group_tree(root: Node) -> Node:
    """Run the grouping pass bottom-up over every container of the tree (in place)."""
    if not P["perceive.group.enabled"]:
        return root
    ctx = _Ctx(root)

    def visit(n: Node) -> None:
        for c in list(n.children):
            if c.children and c.type != "text":
                visit(c)
        if n.type in ("frame", "rect", "ellipse", "image", "instance") or n.id == "root":
            group_container(n, ctx)

    visit(root)
    return root
