"""Containment hierarchy: regions + text nodes -> IR tree.

Rules:
  * The root frame covers the whole image and carries the page background fill.
  * Regions are placed largest-first; a region becomes a child of the smallest already-placed
    node that contains >= ``perceive.hier.contain_ratio`` of its area (else of the root).
  * Text nodes attach to the deepest container that contains their centre and most of their
    box. Text, icons and divider lines never get children (both the renderer and the Figma
    export drop children of those leaf types): regions inside them are absorbed as fragments
    (``meta["absorbed"]``), texts attach to the leaf's container instead.
  * Siblings are ordered in reading order (rows of ~8px, then x); when two siblings overlap,
    the larger one is painted first (z-order).
  * Finally :func:`dt.perceive.group.group_tree` infers the unpainted containers (list items,
    lists, text/icon buttons, tabs, app bars, destinations) bottom-up over every container.
"""
from __future__ import annotations

from typing import Optional

from dt.ir import Box, Color, Fill, Node, Stroke, new_id
from dt.params import P, register
from dt.perceive.group import group_tree
from dt.perceive.segment import Region

register("perceive.hier.contain_ratio", 0.9, "fraction of a node's area inside a container to be nested in it", (0.6, 1.0))
register("perceive.hier.text_contain_ratio", 0.6, "fraction of a text box inside a container to be nested in it", (0.3, 1.0))
register("perceive.hier.row_quant", 8, "px row quantisation for sibling reading order", (2, 24))
register("perceive.hier.min_ellipse_side", 6, "px: smaller fully-rounded shapes stay rects", (2, 16))


def region_to_node(r: Region) -> Node:
    """IR node for a Region (type/fills/strokes/radius/effects from its evidence)."""
    kind_to_type = {"rect": "rect", "pill": "rect", "ellipse": "ellipse", "divider": "line", "icon": "icon", "image": "image"}
    ntype = kind_to_type.get(r.kind, "rect")
    if ntype == "ellipse" and min(r.box.w, r.box.h) < int(P["perceive.hier.min_ellipse_side"]):
        ntype = "rect"
    n = Node(id=new_id("r"), type=ntype, name=r.kind, box=Box(r.box.x, r.box.y, r.box.w, r.box.h))
    if r.fill is not None:
        n.fills = [Fill.solid(r.fill)]
    if r.stroke is not None:
        n.strokes = [Stroke(color=r.stroke[0], width=float(r.stroke[1]), align="inside")]
    if r.kind == "pill":
        rad = min(r.box.w, r.box.h) / 2.0
        n.radius = (rad, rad, rad, rad)
    elif r.kind in ("rect",) and r.radius > 0:
        n.radius = (round(r.radius), round(r.radius), round(r.radius), round(r.radius))
    if r.has_shadow and r.shadow is not None:
        n.effects = [r.shadow]
    n.meta = {"src": "segment", "kind": r.kind, "conf": r.conf, "depth": r.depth, "bg": r.bg.hex(), **{k: v for k, v in r.meta.items() if k != "mask"}}
    return n


def _inside_frac(inner: Box, outer: Box) -> float:
    if inner.area <= 0:
        return 1.0 if outer.contains_point(inner.cx, inner.cy) else 0.0
    return inner.intersect(outer).area / inner.area


# node types the renderer and the Figma export treat as leaves: children nested in them are never drawn.
# A region inside an icon glyph / divider is a fragment of it (the dot and bar of 'info') and is absorbed;
# a text inside one (the date in a calendar glyph) stays a sibling, or it would vanish from the render.
LEAF_TYPES = ("text", "icon", "line")


def _deepest_container(root: Node, box: Box, ratio: float, candidates: Optional[set[str]] = None,
                       leaves: bool = False) -> Node:
    """Smallest-area node (searched depth-first) that contains >= ratio of box. Text never contains
    anything; icon glyphs and divider lines are only returned when ``leaves`` is set (the caller
    then absorbs the box into the leaf instead of nesting it)."""
    best = root
    stack = [root]
    while stack:
        n = stack.pop()
        for c in n.children:
            if c.type == "text" or (not leaves and c.type in LEAF_TYPES) or (candidates is not None and c.id not in candidates):
                continue
            if _inside_frac(box, c.box) >= ratio and c.box.area < best.box.area + 1e-6 and c.box.area >= box.area * 0.999:
                best = c
                stack.append(c)
    return best


def _sort_children(n: Node) -> None:
    q = int(P["perceive.hier.row_quant"])
    n.children.sort(key=lambda c: (round(c.box.y / q), c.box.x))
    # z-order: among overlapping siblings the larger paints first
    kids = n.children
    changed = True
    while changed:
        changed = False
        for i in range(len(kids) - 1):
            a, b = kids[i], kids[i + 1]
            if a.box.intersect(b.box).area > 0 and a.box.area < b.box.area and not b.box.contains(a.box):
                kids[i], kids[i + 1] = b, a
                changed = True
    for c in kids:
        _sort_children(c)


def build_tree(regions: list[Region], texts: list[Node], W: int, H: int, bg: Optional[Color] = None) -> Node:
    """Nest regions by containment and attach text nodes to their deepest container.

    `texts` are IR text nodes (from ``dt.perceive.ocr.paragraph_node``). Returns the root frame.
    """
    root = Node(id="root", type="frame", name="Screen", box=Box(0, 0, W, H), clip=True)
    root.fills = [Fill.solid(bg or Color(255, 255, 255))]
    ratio = float(P["perceive.hier.contain_ratio"])
    placed: set[str] = set()
    for r in sorted(regions, key=lambda r: -r.box.area):
        n = region_to_node(r)
        n.box = _clamp(n.box, W, H)
        parent = _deepest_container(root, n.box, ratio, placed, leaves=True)
        if parent.type in LEAF_TYPES:
            parent.meta.setdefault("absorbed", []).append({"kind": r.kind, "box": n.box.to_dict()})
            continue
        parent.children.append(n)
        placed.add(n.id)
    tratio = float(P["perceive.hier.text_contain_ratio"])
    for t in texts:
        t.box = _clamp(t.box, W, H)
        parent = _deepest_container(root, t.box, tratio, placed)
        parent.children.append(t)
    for n in root.walk():
        if n.children and n.type in ("rect", "ellipse", "image"):
            n.type = "frame"
            n.clip = n.meta.get("kind") in ("rect", "pill", "ellipse") and n.type == "frame" and False
    group_tree(root)  # transparent containers (list items, buttons, tabs, bars); perceive.group.enabled
    _sort_children(root)
    return root


def _clamp(b: Box, W: int, H: int) -> Box:
    x0, y0 = max(0.0, b.x), max(0.0, b.y)
    x1, y1 = min(float(W), b.x2), min(float(H), b.y2)
    return Box(x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))
