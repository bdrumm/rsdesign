"""Auto-layout inference for frames with >= 2 children.

A frame is a ``row`` when its children's x-intervals are pairwise disjoint (and they overlap
in y), a ``column`` when the y-intervals are disjoint; otherwise ``none`` (absolute). Gap is
the median spacing between consecutive children on the main axis (only accepted when the
spacings agree within ``gap_tol``). Padding = frame box minus the children's union (never made
symmetric: the flex content box must equal the union). Cross-axis alignment is start/center/end
by which edge (or centre) the children share; justify is ``center`` when the leading and trailing
padding match, else ``start``. Finally the flex placement of every child is simulated and the
layout is dropped (frame stays absolute) when any child would land more than ``max_drift`` px from
its absolute box, so a flex re-render always reproduces the absolute one.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from dt.ir import Box, Layout, Node
from dt.params import P, register

register("perceive.layout.overlap_tol", 1.0, "px of main-axis overlap tolerated between consecutive children", (0.0, 4.0))
register("perceive.layout.gap_tol", 1.5, "px: all gaps must agree with the median within this for a flex layout", (0.5, 6.0))
register("perceive.layout.align_tol", 1.5, "px tolerance for shared cross-axis edges / centres", (0.5, 6.0))
register("perceive.layout.center_pref_tol", 3.0, "px: centre alignment is preferred whenever centres agree within this (glyph ink boxes are smaller than their layout boxes)", (1.0, 8.0))
register("perceive.layout.center_tol", 1.5, "px tolerance between leading/trailing padding for justify=center", (0.5, 6.0))
register("perceive.layout.max_drift", 1.5, "px: a layout is kept only if a flex render places every child within this of its absolute box (gap deviations accumulate along the main axis)", (0.5, 6.0))


def _intervals(children: list[Node], axis: str) -> list[tuple[float, float]]:
    return [(c.box.x, c.box.x2) if axis == "x" else (c.box.y, c.box.y2) for c in children]


def _disjoint(iv: list[tuple[float, float]], tol: float) -> bool:
    s = sorted(iv)
    return all(s[i + 1][0] >= s[i][1] - tol for i in range(len(s) - 1))


def _gaps(children: list[Node], axis: str) -> list[float]:
    s = sorted(children, key=lambda c: c.box.x if axis == "x" else c.box.y)
    return [(s[i + 1].box.x - s[i].box.x2) if axis == "x" else (s[i + 1].box.y - s[i].box.y2) for i in range(len(s) - 1)]


def _union(children: list[Node]) -> Box:
    b = children[0].box
    for c in children[1:]:
        b = b.union(c.box)
    return b


def _cross_align(children: list[Node], mode: str, tol: float) -> Optional[str]:
    """Cross-axis alignment that displaces no child by more than `tol`, or None.

    Displacement of a child under flex: ``start``/``end`` move every child to the shared edge
    (error = spread of that edge); ``center`` moves every child to the common centre (error = half
    the spread of the centres). Centre is preferred when its centres agree within
    ``center_pref_tol`` (glyph ink boxes are smaller than their layout boxes) and no edge is a
    strictly better fit; when nothing fits the frame must stay absolute (a ``start`` fallback
    collapses indented children onto one edge)."""
    if mode == "row":
        starts, centers, ends = [c.box.y for c in children], [c.box.cy for c in children], [c.box.y2 for c in children]
    else:
        starts, centers, ends = [c.box.x for c in children], [c.box.cx for c in children], [c.box.x2 for c in children]
    spread = lambda v: max(v) - min(v)  # noqa: E731
    errs = {"center": spread(centers) / 2.0, "start": spread(starts), "end": spread(ends)}
    ok = {"center": spread(centers) <= float(P["perceive.layout.center_pref_tol"]),
          "start": errs["start"] <= tol, "end": errs["end"] <= tol}
    for name in sorted(errs, key=lambda k: (errs[k], k != "center")):
        if ok[name]:
            return name
    return None


def _flex_drift(n: Node, kids: list[Node], L: Layout) -> float:
    """Max px distance between where a flex render puts each child (same algorithm as CSS flex with
    ``flex: none`` children) and its absolute box."""
    row = L.mode == "row"
    top, right, bottom, left = L.padding
    m0 = n.box.x + left if row else n.box.y + top
    m1 = n.box.x2 - right if row else n.box.y2 - bottom
    c0 = n.box.y + top if row else n.box.x + left
    c1 = n.box.y2 - bottom if row else n.box.x2 - right
    size = (lambda c: c.box.w) if row else (lambda c: c.box.h)
    csize = (lambda c: c.box.h) if row else (lambda c: c.box.w)
    total = sum(size(c) for c in kids) + L.gap * (len(kids) - 1)
    pos = m0 + ((m1 - m0) - total) / 2.0 if L.justify == "center" else m0
    worst = 0.0
    for c in kids:  # kids are in main-axis order
        main_actual = c.box.x if row else c.box.y
        if L.align_items == "center":
            cpos = c0 + ((c1 - c0) - csize(c)) / 2.0
        elif L.align_items == "end":
            cpos = c1 - csize(c)
        else:
            cpos = c0
        cross_actual = c.box.y if row else c.box.x
        worst = max(worst, abs(pos - main_actual), abs(cpos - cross_actual))
        pos += size(c) + L.gap
    return worst


def infer_frame_layout(n: Node) -> Optional[Layout]:
    """Layout for one frame, or None if its children do not form a row/column that a flex render
    reproduces within ``perceive.layout.max_drift`` px of their absolute boxes."""
    kids = [c for c in n.children if c.visible]
    if len(kids) < 2:  # hidden children are not rendered at all, so they never take part in the flow
        return None
    tol = float(P["perceive.layout.overlap_tol"])
    row_ok = _disjoint(_intervals(kids, "x"), tol)
    col_ok = _disjoint(_intervals(kids, "y"), tol)
    if row_ok and col_ok:  # diagonal: pick the axis with the larger spread
        ux = _union(kids)
        mode = "row" if ux.w >= ux.h else "column"
    elif row_ok:
        mode = "row"
    elif col_ok:
        mode = "column"
    else:
        return None
    kids = sorted(kids, key=lambda c: c.box.x if mode == "row" else c.box.y)
    gaps = _gaps(kids, "x" if mode == "row" else "y")
    gap = float(np.median(gaps))
    if any(abs(g - gap) > float(P["perceive.layout.gap_tol"]) for g in gaps):
        return None
    gap = max(0.0, round(gap, 1))
    align = _cross_align(kids, mode, float(P["perceive.layout.align_tol"]))
    if align is None:
        return None
    u = _union(kids)
    # true padding (frame box minus the children's union) on all four sides: the flex content box is
    # then exactly the union, so centring along either axis keeps every child where it was. Making the
    # padding symmetric re-centres the children in the whole frame instead (a left-aligned card body
    # jumps to the card's middle).
    pad = (u.y - n.box.y, n.box.x2 - u.x2, n.box.y2 - u.y2, u.x - n.box.x)
    pad = tuple(max(0.0, round(v, 1)) for v in pad)
    lead, trail = (pad[3], pad[1]) if mode == "row" else (pad[0], pad[2])
    justify = "center" if abs(lead - trail) <= float(P["perceive.layout.center_tol"]) and lead > 0 else "start"
    L = Layout(mode=mode, gap=gap, padding=pad, align_items=align, justify=justify)
    if _flex_drift(n, kids, L) > float(P["perceive.layout.max_drift"]):
        return None
    return L


def infer_layout(root: Node) -> Node:
    """Set ``node.layout`` on every frame with >= 2 children (in place) and return root.

    Children of a row/column frame are re-ordered along the main axis so a flex render places
    them where the absolute render does.
    """
    for n in root.walk():
        if n.type != "frame" or len(n.children) < 2:
            continue
        L = infer_frame_layout(n)
        if L is None:
            n.layout = Layout(mode="none")
            continue
        n.layout = L
        n.children.sort(key=lambda c: c.box.x if L.mode == "row" else c.box.y)
    return root
