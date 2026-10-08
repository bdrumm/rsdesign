"""Tile-batched trials: score many refine hypotheses with ONE page render.

For each hypothesis the edited document is drawn into its own clipped viewport ("tile") showing
only the pixels the edit can change: the extent of every node whose render inputs differ between
the current and the edited document, plus a margin. Tiles stay at their own screen position in a
page of the screen's size; tiles that overlap go to different layers of the same page, which is
loaded once and screenshotted once per layer. A tile's pixels are the pixels a full render of the
edited document shows there, so pasting the tile into the current render gives the candidate
render, scored exactly like a full render: the loss is a per-pixel sum, so the prediction is
``old_total - region_err_old + region_err_new`` (the text term is computed on the edited IR).

Self-checks keep the prediction honest:
* ring check: the tile margin (outside the changed extent) must be pixel-identical to the current
  render; if the edit reaches further (an overflowing text, a large shadow) the tile is ``leaky``
  and the hypothesis goes through the full-render path;
* edits that are not local (the root's paint, the page size, the z-order of kept top-level nodes,
  an extent above ``refine.tile.max_frac`` of the screen) use the full-render path;
* the optimizer confirms every accepted set with one full render and bisects on a mismatch.
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from dt.ir import Box, Document, Node
from dt.params import P, register

register("refine.tile.enabled", True, "score local hypotheses with one page render of clipped tiles (False: one full render per batch, the pre-tile optimizer)")
register("refine.tile.margin", 6, "px of unchanged context around a hypothesis' changed extent in its tile (ring check: these pixels must not change)", (2, 32))
register("refine.tile.max_frac", 0.45, "a hypothesis whose changed extent exceeds this fraction of the screen is scored by a full render", (0.05, 1.0))
register("refine.tile.raster_max_frac", 0.15, "raster hypotheses larger than this fraction of the screen are scored by a full render", (0.0, 1.0))
register("refine.tile.max_rounds", 3, "tile rounds per iteration (a round re-scores winners that overlapped an accepted move)", (1, 10))
register("refine.tile.min_options", 3, "rounds with fewer options than this skip the tile page (full renders are cheaper)", (1, 16))
register("refine.tile.confirm_tol", 1e-6, "a confirmed set whose full-render loss exceeds its predicted loss by more than this is bisected", (0.0, 0.01))
register("refine.tile.text_char_w", 0.62, "estimated glyph advance (em) for the tight extent of a text node (overflow past the box)", (0.3, 1.2))
register("refine.tile.prune_char_w", 1.25, "conservative glyph advance (em) used when pruning nodes that cannot reach a tile", (0.6, 3.0))


@dataclass
class Tile:
    """A clipped viewport ``rect`` (x0, y0, x1, y1; screen px) of ``doc``; ``core`` is the changed extent."""

    key: int
    doc: Document
    rect: tuple[int, int, int, int]
    core: tuple[int, int, int, int]
    pixels: Optional[np.ndarray] = None
    leaky: bool = False


# --------------------------------------------------------------------------- extents
def _shadow_pad(n: Node) -> float:
    pad = 0.0
    for e in n.effects:
        if not e.inner:
            pad = max(pad, abs(e.dx) + abs(e.dy) + 2.0 * max(0.0, e.blur) + max(0.0, e.spread) + 1.0)
    for s in n.strokes:
        if s.align != "inside":
            pad = max(pad, s.width + 1.0)
    return pad


def _text_overflow(n: Node, char_w: float) -> tuple[float, float]:
    """(horizontal, vertical) px a text node may paint past its box."""
    if n.type != "text" or not n.text:
        return 0.0, 0.0
    size = n.text_style.size if n.text_style else 14.0
    ls = n.text_style.letter_spacing if n.text_style else 0.0
    lines = n.text.split("\n")
    longest = max(len(l) for l in lines)
    est_w = longest * (char_w * size + max(0.0, ls))
    lh = (n.text_style.line_height if n.text_style and n.text_style.line_height else 1.3 * size)
    est_h = len(lines) * max(lh, 1.3 * size)
    over_w = max(0.0, est_w - n.box.w)
    if n.text_style is not None and n.text_style.italic:
        over_w += 0.3 * size
    return over_w + 0.25 * size + 1.0, max(0.0, est_h - n.box.h) + 0.35 * size + 1.0


def node_extent(n: Node, conservative: bool = False) -> Box:
    """Pixels ``n`` itself may paint (box + shadows/outside strokes + text overflow); children excluded."""
    b = n.box
    pad = _shadow_pad(n) + (1.0 if not conservative else 2.0)
    ox, oy = _text_overflow(n, float(P["refine.tile.prune_char_w" if conservative else "refine.tile.text_char_w"]))
    if n.type == "icon":
        ox = oy = max(ox, 2.0)
    return Box(b.x - pad - ox, b.y - pad - oy, b.w + 2 * (pad + ox), b.h + 2 * (pad + oy))


def subtree_extent(n: Node, conservative: bool = False, cache: Optional[dict] = None) -> Box:
    if cache is not None and id(n) in cache:
        return cache[id(n)]
    e = node_extent(n, conservative)
    for c in n.children:
        if c.visible:
            e = e.union(subtree_extent(c, conservative, cache))
    if n.clip:
        e = e.intersect(n.box.expand(_shadow_pad(n) + 1.0)) if e.area > 0 else e
    if cache is not None:
        cache[id(n)] = e
    return e


def _sig(n: Node, parent_id: Optional[str]) -> tuple:
    """Everything about a node that reaches its own pixels (its child list is compared separately)."""
    return (n.type, n.box, n.fills, n.strokes, n.effects, n.radius, n.opacity, n.clip, n.visible, n.text, n.text_style,
            n.icon_name, n.image_ref, n.meta.get("icon_fill") if isinstance(n.meta, dict) else None, parent_id)


def _index(doc: Document) -> dict[str, tuple[Node, Optional[str]]]:
    out: dict[str, tuple[Node, Optional[str]]] = {}

    def rec(n: Node, pid: Optional[str]) -> None:
        out[n.id] = (n, pid)
        for c in n.children:
            rec(c, n.id)

    rec(doc.root, None)
    return out


def changed_extent(base: Document, cand: Document, base_index: Optional[dict] = None) -> Optional[Box]:
    """Union of the (old and new) subtree extents of every node whose render inputs differ, or ``None``
    when the edit is not local (document size, the root's own paint, or a z-order change of kept nodes).
    Inserted / deleted nodes contribute their own extents (appended children paint on top). An empty Box
    means no visible change."""
    if base.width != cand.width or base.height != cand.height or base.root.id != cand.root.id:
        return None
    bi = base_index if base_index is not None else _index(base)
    ci = _index(cand)
    ext: Optional[Box] = None

    def add(n: Node) -> None:
        nonlocal ext
        if not n.visible:
            return
        e = subtree_extent(n)
        ext = e if ext is None else ext.union(e)

    for nid in sorted(set(bi) | set(ci)):
        a, b = bi.get(nid), ci.get(nid)
        if a is not None and b is not None:
            if [c.id for c in a[0].children if c.id in ci] != [c.id for c in b[0].children if c.id in bi]:
                if nid == base.root.id:
                    return None  # kept top-level children reordered: z-order changes wherever they overlap
                add(a[0])
                add(b[0])
                continue
            if _sig(a[0], a[1]) == _sig(b[0], b[1]):
                continue
            if nid == base.root.id:
                return None  # the page background / root paint: not local
            add(a[0])
            add(b[0])
        elif a is not None:
            if a[1] is None or a[1] in ci:
                add(a[0])  # deleted (a whole deleted subtree is covered by its top node)
        else:
            if b[1] is None or b[1] in bi:
                add(b[0])
    return ext if ext is not None else Box(0, 0, 0, 0)


def _shadowed(doc: Document) -> list[tuple[Box, Box]]:
    """(extent, opaque interior) of the nodes that paint an outer box-shadow. The interior (the box inset by
    its corner radius) shows no shadow pixels, so a tile inside it does not cut the shadow."""
    out = []
    for n in doc.walk():
        if n.visible and any(not e.inner for e in n.effects):
            inset = max(n.radius) + 1.0 if n.type != "ellipse" else min(n.box.w, n.box.h) / 2.0
            out.append((node_extent(n), n.box.expand(-inset) if min(n.box.w, n.box.h) > 2 * inset else Box(0, 0, 0, 0)))
    return out


def _groups(doc: Document) -> list[tuple[Box, Box]]:
    """(subtree extent, empty interior) of the visible nodes with opacity < 1. Chrome composites such a group as
    one layer; a tile that cuts it (even fully inside it) rasterises its edges +-1 level differently, which
    the ring check cannot see, so a tile must hold every such group it touches whole."""
    out: list[tuple[Box, Box]] = []

    def rec(n: Node) -> None:
        if not n.visible:
            return
        if n.opacity < 1 and n is not doc.root:
            out.append((subtree_extent(n), Box(0, 0, 0, 0)))
            return  # the group's extent covers its descendants
        for c in n.children:
            rec(c)

    rec(doc.root)
    return out


def _whole_shadows(rect: tuple[int, int, int, int], shadows: list[tuple[Box, Box]], W: int, H: int) -> tuple[int, int, int, int]:
    """Grow ``rect`` until it cuts through no shadow halo: Skia rasterises a blur that a clip cuts with
    slightly different coverage (+-1 level), so a tile must hold every shadow it touches whole."""
    x0, y0, x1, y1 = rect
    changed = True
    while changed:
        changed = False
        for s, inner in shadows:
            sx0, sy0 = max(0, int(math.floor(s.x))), max(0, int(math.floor(s.y)))
            sx1, sy1 = min(W, int(math.ceil(s.x2))), min(H, int(math.ceil(s.y2)))
            if not (sx0 < x1 and x0 < sx1 and sy0 < y1 and y0 < sy1) or (x0 <= sx0 and y0 <= sy0 and sx1 <= x1 and sy1 <= y1):
                continue
            if inner.area > 0 and inner.contains(Box(x0, y0, x1 - x0, y1 - y0), tol=0.0):
                continue
            x0, y0, x1, y1 = min(x0, sx0), min(y0, sy0), max(x1, sx1), max(y1, sy1)
            changed = True
    return x0, y0, x1, y1


def tile_rect(ext: Box, W: int, H: int, doc: Optional[Document] = None
              ) -> Optional[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    """(rect, core) integer screen rectangles for a changed extent, clipped to the screen; None if empty.
    With ``doc`` (the document the tile shows) the rect also holds every shadow and opacity group it would cut."""
    m = int(P["refine.tile.margin"])
    cx0, cy0 = max(0, int(math.floor(ext.x))), max(0, int(math.floor(ext.y)))
    cx1, cy1 = min(W, int(math.ceil(ext.x2))), min(H, int(math.ceil(ext.y2)))
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    rect = (max(0, cx0 - m), max(0, cy0 - m), min(W, cx1 + m), min(H, cy1 + m))
    if doc is not None:
        rect = _whole_shadows(rect, _shadowed(doc) + _groups(doc), W, H)
    return rect, (cx0, cy0, cx1, cy1)


# --------------------------------------------------------------------------- tile pages
def _pruned(n: Node, rect: Box, cache: dict) -> Node:
    """Shallow copy of ``n`` without the child subtrees that cannot paint inside ``rect``."""
    c = copy.copy(n)
    c.children = [_pruned(k, rect, cache) for k in n.children
                  if k.visible and subtree_extent(k, True, cache).intersect(rect).area > 0]
    return c


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def assign_layers(tiles: list[Tile]) -> list[list[Tile]]:
    """Greedy colouring: each layer holds tiles whose rects do not overlap (first fit, in key order)."""
    layers: list[list[Tile]] = []
    for t in sorted(tiles, key=lambda t: t.key):
        for L in layers:
            if not any(_overlaps(t.rect, o.rect) for o in L):
                L.append(t)
                break
        else:
            layers.append([t])
    return layers


_URI = re.compile(r"url\('(data:[^']+)'\)")


def _tile_html(t: Tile) -> str:
    from dt.render.html import _node_html
    x0, y0, x1, y1 = t.rect
    root = _pruned(t.doc.root, Box(x0, y0, x1 - x0, y1 - y0), {})
    inner: list[str] = []
    _node_html(root, None, "absolute", inner)
    # the tile stays at its own screen position in a page of the screen's size: Chrome's anti-aliasing
    # depends on where content falls in its raster tiles (whose size follows the viewport), so a moved
    # tile or a taller page does not reproduce the full render bit for bit
    return (f'<div style="position:absolute;left:{x0}px;top:{y0}px;width:{x1 - x0}px;height:{y1 - y0}px;overflow:hidden;'
            f'background:#fff"><div style="position:absolute;left:{-x0}px;top:{-y0}px;width:{t.doc.width}px;'
            f'height:{t.doc.height}px">' + "\n".join(inner) + "</div></div>")


def page_html(layers: list[list[Tile]], W: int, H: int) -> str:
    """One page holding every layer (one shown at a time); raster crops (data URIs) are declared once."""
    from dt.render.html import _font_face_css
    body = "\n".join(f'<div class="dt-layer" style="position:absolute;left:0;top:0;width:{W}px;height:{H}px;display:none">'
                     + "\n".join(_tile_html(t) for t in L) + "</div>" for L in layers)
    uris: dict[str, str] = {}

    def sub(m: "re.Match[str]") -> str:
        u = m.group(1)
        if u not in uris:
            uris[u] = f"--dt-img{len(uris)}"
        return f"var({uris[u]})"

    body = _URI.sub(sub, body)
    root_vars = "".join(f"{v}: url('{u}');" for u, v in uris.items())
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
{_font_face_css()}
:root {{ {root_vars} }}
html, body {{ margin: 0; padding: 0; width: {W}px; height: {H}px; overflow: hidden; background: #fff; }}
* {{ -webkit-font-smoothing: antialiased; text-rendering: geometricPrecision; }}
div {{ margin: 0; padding: 0; }}
</style></head>
<body>
{body}
</body></html>
"""


_FONTS_JS = """async () => {
    await Promise.race([document.fonts.ready, new Promise(r => setTimeout(r, 5000))]);
    const used = new Set();
    for (const el of document.querySelectorAll('body *')) {
        const cs = getComputedStyle(el); used.add(cs.fontWeight + ' 16px ' + cs.fontFamily.split(',')[0]);
    }
    await Promise.all([...used].map(f => document.fonts.load(f).catch(() => null)));
    return true;
}"""
_SHOW_JS = """async (i) => {
    document.querySelectorAll('.dt-layer').forEach((l, j) => { l.style.display = j === i ? 'block' : 'none'; });
    await Promise.race([new Promise(r => requestAnimationFrame(r)), new Promise(r => setTimeout(r, 2000))]);
    return true;
}"""


def _shoot_layers(html: str, W: int, H: int, n: int) -> list[np.ndarray]:
    """Load ``html`` once in the shared renderer (same browser, fonts and viewport as ``render_doc``) and
    screenshot it with each of its ``n`` layers shown in turn."""
    from io import BytesIO

    from PIL import Image

    from dt.render import screenshot as ss
    out = []
    with ss._lock:
        ss._ensure_browser()
        page = ss._ctx().new_page()
        try:
            page.set_viewport_size({"width": int(W), "height": int(H)})
            page.goto("file://" + ss._tmp_html(html), wait_until="load")
            page.evaluate(_FONTS_JS)
            for i in range(n):
                page.evaluate(_SHOW_JS, i)
                data = page.screenshot(type="png", animations="disabled", caret="hide")
                out.append(np.asarray(Image.open(BytesIO(data)).convert("RGB"), dtype=np.uint8))
        finally:
            page.close()
    return out


def render_tiles(tiles: list[Tile], current: np.ndarray) -> tuple[int, int]:
    """Render every tile with ONE page load (one screenshot per layer of non-overlapping tiles); fills
    ``tile.pixels`` and ``tile.leaky`` (the margin ring differs from ``current``: the edit reaches past its
    extent, or the tile did not reproduce the renderer). Returns ``(pages, layers)``."""
    if not tiles:
        return 0, 0
    H, W = current.shape[:2]
    layers = assign_layers(tiles)
    shots = _shoot_layers(page_html(layers, W, H), W, H, len(layers))
    for L, img in zip(layers, shots):
        for t in L:
            x0, y0, x1, y1 = t.rect
            px = img[y0:y1, x0:x1]
            t.pixels = np.ascontiguousarray(px)
            cx0, cy0, cx1, cy1 = t.core
            ring = np.ones(px.shape[:2], bool)
            ring[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = False
            t.leaky = img.shape[:2] != current.shape[:2] or bool(np.any(px[ring] != current[y0:y1, x0:x1][ring]))
    return 1, len(layers)


def paste(current: np.ndarray, tile: Tile) -> np.ndarray:
    """The candidate full render: ``current`` with the tile pixels in place."""
    out = current.copy()
    x0, y0, x1, y1 = tile.rect
    out[y0:y1, x0:x1] = tile.pixels
    return out
