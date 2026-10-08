"""Font identification by analysis-by-synthesis (docs/SELF_IMPROVEMENT.md, mechanism 5).

Perception estimates text size/weight with a Roboto calibration. Google apps and pages also use
Google Sans / Google Sans Text, whose widths and vertical metrics differ, so a Roboto guess can be
several pixels wide of the target and land on the wrong baseline. Here the renderer decides:

  pass 1  every candidate (family x size) is rendered at the node's box; the rendered ink is
          measured and compared with the target ink -> a corrected size (width ratio) and the
          shift that aligns the ink.
  pass 2  candidates at the corrected size (+-0.5px) and neighbouring weights, shifted onto the
          target ink, are rendered; the one with the lowest mean ΔE2000 against the crop wins.
  pass 3  (only texts pass 2 did not reproduce) the best family/weight with letter-spacing
          candidates (size re-derived from the ink width net of the tracking) and in italic: a
          tracked or slanted line otherwise "wins" pass 2 at a wrong size and weight.

The current style is always in the pool, so a node is only changed when the change reproduces
the target better. Cells are rendered through dt.render (render_doc), so what is measured here is
exactly what the pipeline will render.

    summary = annotate_fonts(doc, rgb)   # in place: text_style.family/size/weight, box shift, meta['font_de']
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dt.ir import Box, Color, Document, Fill, Node, TextStyle
from dt.params import P, register

register("perceive.fonts.enabled", True, "identify text font family/size/weight by rendering candidates")
register("perceive.fonts.families", ["Roboto", "Google Sans", "Google Sans Text"], "candidate families (must be self-hosted)")
register("perceive.fonts.pad", 3, "px of context around a text box used for comparison", (1, 8))
register("perceive.fonts.max_texts", 120, "cap on text nodes identified per screen (largest first); the rest keep their style", (10, 1000))
register("perceive.fonts.min_gain", 0.15, "a new style must lower the cell's mean ΔE by at least this much to replace the current one", (0.0, 2.0))
register("perceive.fonts.refine_de", 1.0, "texts whose best pass-2 cell still has mean ΔE above this get pass 3 (letter-spacing and italic candidates)", (0.0, 10.0))
register("perceive.fonts.tracking", [-0.25, 0.1, 0.25, 0.4, 0.5], "letter-spacing (px) candidates tried in pass 3 (M3 type-scale tracking values, plus tight)")
register("perceive.fonts.p3_rel_gain", 0.25, "a pass-3 (tracking/italic) style must lower the best pass-2 ΔE by at least this fraction; on real screens small gains are rasteriser noise (spurious italics)", (0.0, 0.9))
register("perceive.fonts.realign_k", 3, "best distinct styles per text re-placed so their rendered ink lands on the target ink (fixes the ~1px baseline error of the scaled-ink estimate)", (0, 10))
register("perceive.fonts.max_page_h", 8000, "max height (px) of one batch page; more cells go to further pages (Chrome cannot screenshot > ~16k px)", (1000, 16000))
register("perceive.fonts.page_width", 3000, "width (px) of the batch page cells are packed into", (1000, 8000))

_WEIGHTS = (400, 500, 700)


@dataclass
class _Cell:
    qi: int  # query index
    style: TextStyle
    box: Box  # text box in crop coordinates
    x: float = 0.0  # cell origin on the batch page
    y: float = 0.0
    de: float = float("inf")
    ink: Optional[Box] = None
    page: int = 0  # batch page index (set by _render_cells)
    p3: bool = False  # pass-3 candidate (tracking / italic)


@dataclass
class _Query:
    node: Node
    crop: np.ndarray
    origin: tuple[int, int]
    bg: Color
    ink: Box  # target ink in crop coordinates
    cells: list[_Cell] = field(default_factory=list)
    meas: dict = field(default_factory=dict)  # family -> pass-1 cell whose ink width is closest to the target


def _ink_box(crop: np.ndarray, bg: Color, rel: float = 0.35) -> Box:
    px = crop[..., :3].astype(np.float32)
    d = np.sqrt(((px - np.array([bg.r, bg.g, bg.b], np.float32)) ** 2).sum(-1))
    mx = float(d.max()) if d.size else 0.0
    if mx < 20:
        return Box()
    ys, xs = np.where(d > rel * mx)
    return Box(int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def _border_bg(crop: np.ndarray) -> Color:
    b = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]]).reshape(-1, 3)
    return Color.from_rgb(*np.median(b, axis=0))


def _render_cells(queries: list[_Query], cells: list[_Cell]) -> list[np.ndarray]:
    """Pack cells on pages (each cell a clip frame of the query's crop size filled with its bg, holding a
    clone of the text node at the candidate style/box) and render with the pipeline renderer. A page never
    exceeds perceive.fonts.max_page_h (Chrome refuses screenshots taller than ~16k px); sets c.page."""
    from dt.render.screenshot import render_doc

    page_w = max([int(P["perceive.fonts.page_width"])] + [q.crop.shape[1] for q in queries])
    max_h = int(P["perceive.fonts.max_page_h"])
    page = x = y = row_h = 0
    heights: list[int] = []
    for c in cells:
        h, w = queries[c.qi].crop.shape[:2]
        if x + w > page_w and x > 0:
            x, y, row_h = 0, y + row_h + 2, 0
        if y + h > max_h and y > 0:  # start a new page
            heights.append(y + row_h if x > 0 else y)
            page, x, y, row_h = page + 1, 0, 0, 0
        c.x, c.y, c.page = x, y, page
        x += w + 2
        row_h = max(row_h, h)
    heights.append(y + row_h)
    docs = [Document.blank(page_w, max(1, hh), "#ffffff") for hh in heights]
    for i, c in enumerate(cells):
        q = queries[c.qi]
        h, w = q.crop.shape[:2]
        frame = Node(id=f"c{i}", type="frame", box=Box(c.x, c.y, w, h), fills=[Fill.solid(q.bg)], clip=True)
        t = copy.copy(q.node)  # shallow: the renderer only reads it (deep copies of 10k cells cost seconds)
        t.id, t.children, t.component = f"t{i}", [], None
        t.text_style = c.style
        t.box = c.box.translate(c.x, c.y)
        t.fills, t.strokes, t.effects = [], [], []
        frame.children = [t]
        docs[c.page].root.children.append(frame)
    return [render_doc(d) for d in docs]


def _score(queries: list[_Query], cells: list[_Cell], pages: list[np.ndarray], measure_ink: bool) -> None:
    """c.de = mean ΔE2000 of the cell vs the query crop. Only pixels whose RGB differs are converted and
    compared (ΔE of identical colours is 0, so the mean is exact), in one batch for all cells: converting
    whole pages and every background pixel cell by cell dominated the run time (60 texts: ~45 s of ~75 s)."""
    tgt, ren, owners, sizes = [], [], [], []
    for ci, c in enumerate(cells):
        q = queries[c.qi]
        h, w = q.crop.shape[:2]
        x0, y0 = int(c.x), int(c.y)
        reg = pages[c.page][y0:y0 + h, x0:x0 + w, :3]
        if reg.shape[:2] != (h, w):
            continue
        diff = np.any(reg != q.crop[..., :3], axis=-1)
        tgt.append(q.crop[..., :3][diff])
        ren.append(reg[diff])
        owners.append(ci)
        sizes.append(len(tgt[-1]))
        c.de = 0.0
        if measure_ink:
            c.ink = _ink_box(reg, q.bg)
    total = int(sum(sizes))
    if total == 0:
        return
    from dt import accel  # GPU (MLX) on Apple Silicon, NumPy elsewhere; cell means average out float32 rounding
    de = accel.delta_e2000(np.concatenate(tgt)[None], np.concatenate(ren)[None])[0].astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(de, dtype=np.float64)])
    off = 0
    for ci, n in zip(owners, sizes):
        c = cells[ci]
        h, w = queries[c.qi].crop.shape[:2]
        c.de = float(csum[off + n] - csum[off]) / float(h * w)
        off += n


def _style(base: TextStyle, family: str, size: float, weight: int, letter_spacing: Optional[float] = None,
           italic: Optional[bool] = None) -> TextStyle:
    s = copy.deepcopy(base)
    s.family, s.size, s.weight = family, float(size), int(weight)
    if letter_spacing is not None:
        s.letter_spacing = float(letter_spacing)
    if italic is not None:
        s.italic = bool(italic)
    if base.line_height:
        s.line_height = float(base.line_height) * (size / base.size if base.size else 1.0)
    return s


_SHIFTS = ((0, 0), (0.5, 0), (0, 0.5), (-0.5, 0), (0, -0.5))


def _aligned_cells(qi: int, q: _Query, m: _Cell, sty: TextStyle, shifts=_SHIFTS) -> list[_Cell]:
    """Cells for style `sty`, placed so the ink measured on pass-1 cell `m` (scaled to sty.size) starts where
    the target ink starts (x) and ends on the target baseline-ish bottom (y), plus half-pixel shifts."""
    k = sty.size / m.style.size
    dx = q.ink.x - (m.box.x + (m.ink.x - m.box.x) * k)
    dy = (q.ink.y + q.ink.h) - (m.box.y + (m.ink.y + m.ink.h - m.box.y) * k)
    extra = max(0.0, sty.letter_spacing) * max(len(l) for l in (q.node.text or " ").split("\n"))
    box = Box(m.box.x + dx, m.box.y + dy, m.box.w * k + 2 + extra, m.box.h * k)
    return [_Cell(qi, sty, Box(round((box.x + sx) * 2) / 2, round((box.y + sy) * 2) / 2, box.w, box.h)) for sx, sy in shifts]


def _realign(queries: list[_Query], cells: list[_Cell]) -> list[_Cell]:
    """Every distinct style per query, moved by its measured ink offset (left edge; bottom and top edge; the
    best `realign_k` styles also +-0.5px in x and y). The ink-scaling estimate of pass 2/3 is off by up to ~1px in y
    (integer ink boxes, baseline snapping), beyond the +-0.5px shifts it tries: Google Sans 12px was then
    matched by an 11.5px/500 style at ΔE 4.7 while its own style reproduces the line exactly."""
    k = int(P["perceive.fonts.realign_k"])
    per: dict[int, list[_Cell]] = {}
    for c in cells:
        if c.ink is not None and c.ink.w > 0 and c.de != float("inf"):
            per.setdefault(c.qi, []).append(c)
    out: list[_Cell] = []
    for qi, cs in per.items():
        q = queries[qi]
        seen: set = set()
        for c in sorted(cs, key=lambda c: c.de):  # best placement of every distinct style
            key = (c.style.family, c.style.size, c.style.weight, c.style.letter_spacing, c.style.italic)
            if key in seen:
                continue
            seen.add(key)
            dx = q.ink.x - c.ink.x
            dyb = (q.ink.y + q.ink.h) - (c.ink.y + c.ink.h)
            moves = {(dx, dyb), (dx, q.ink.y - c.ink.y)}
            if len(seen) <= k:  # the best few styles also get half-pixel neighbours (text x is sub-pixel positioned)
                moves |= {(dx, dyb - 0.5), (dx, dyb + 0.5), (dx - 0.5, dyb), (dx + 0.5, dyb)}
            out.extend(_Cell(qi, c.style, c.box.translate(mx, my), p3=c.p3) for mx, my in sorted(moves - {(0, 0)}))
    return out


def _render_and_score(queries: list[_Query], cells: list[_Cell]) -> None:
    if cells:
        _score(queries, cells, _render_cells(queries, cells), measure_ink=True)


def _family_installed(family: str) -> bool:
    """A candidate family is only useful if the renderer can draw it (self-hosted CSS present);
    Google Sans Text, for one, is local-only and absent from public clones."""
    import os
    from dt.render.html import FONTS_DIR
    css = {"Roboto": "roboto.local.css", "Google Sans": "gsans.local.css", "Google Sans Text": "gsans_text.local.css",
           "Google Sans Flex": "gsans_flex.local.css"}.get(family)
    return css is None or os.path.exists(os.path.join(FONTS_DIR, css))


def annotate_fonts(doc: Document, rgb: np.ndarray) -> dict:
    """Identify family/size/weight of text nodes in place. Returns a summary."""
    if not bool(P["perceive.fonts.enabled"]):
        return {"enabled": False}
    texts = [n for n in doc.walk() if n.type == "text" and n.text and n.text.strip() and n.text_style]
    texts.sort(key=lambda n: -n.box.area)
    texts = texts[: int(P["perceive.fonts.max_texts"])]
    if not texts:
        return {"texts": 0, "changed": 0}
    H, W = rgb.shape[:2]
    pad = int(P["perceive.fonts.pad"])
    queries: list[_Query] = []
    for n in texts:
        x0, y0, x1, y1 = n.box.expand(pad).as_int()
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1, :3])
        if crop.size == 0:
            continue
        bg = _border_bg(crop)
        ink = _ink_box(crop, bg)
        if ink.w <= 0:
            continue
        queries.append(_Query(n, crop, (x0, y0), bg, ink))
    if not queries:
        return {"texts": 0, "changed": 0}
    fams = [f for f in P["perceive.fonts.families"] if _family_installed(f)]
    # ---- pass 1: candidates at the node box, measure their ink
    p1: list[_Cell] = []
    for qi, q in enumerate(queries):
        st = q.node.text_style
        box = q.node.box.translate(-q.origin[0], -q.origin[1])
        p1.append(_Cell(qi, copy.deepcopy(st), box))  # the current style (reference)
        for fam in fams:
            for ds in (-1.0, 0.0, 1.0):
                p1.append(_Cell(qi, _style(st, fam, max(6.0, st.size + ds), st.weight), box))
    page = _render_cells(queries, p1)
    _score(queries, p1, page, measure_ink=True)
    # ---- pass 2: size corrected by ink width, shifted onto the target ink, neighbouring weights
    p2: list[_Cell] = []
    for qi, q in enumerate(queries):
        mine = [c for c in p1 if c.qi == qi]
        q.cells = mine
        per_fam: dict[str, list[_Cell]] = {}
        for c in mine[1:]:
            if c.ink is not None and c.ink.w > 0:
                per_fam.setdefault(c.style.family, []).append(c)
        for fam, cs in per_fam.items():
            best = min(cs, key=lambda c: abs(c.ink.w - q.ink.w))
            q.meas[fam] = best
            ratio = q.ink.w / best.ink.w if best.ink.w else 1.0
            size = round(best.style.size * ratio * 2) / 2.0
            wts = sorted({best.style.weight, *[w for w in _WEIGHTS if abs(w - best.style.weight) <= 200]})
            for s in (size - 0.5, size, size + 0.5):
                if s < 6:
                    continue
                for wt in wts:
                    p2.extend(_aligned_cells(qi, q, best, _style(q.node.text_style, fam, s, wt)))
    _render_and_score(queries, p2)
    r2 = _realign(queries, p1 + p2)
    _render_and_score(queries, r2)
    p2 = p2 + r2
    # ---- pass 3: letter-spacing and italic for texts pass 2 did not reproduce
    p3: list[_Cell] = []
    refine_de = float(P["perceive.fonts.refine_de"])
    by_q: dict[int, list[_Cell]] = {}
    for c in p2:
        by_q.setdefault(c.qi, []).append(c)
    for qi, q in enumerate(queries):
        pool = sorted(q.cells + by_q.get(qi, []), key=lambda c: c.de)
        top = pool[0]
        m = q.meas.get(top.style.family)
        if top.de <= refine_de or m is None or not m.ink or m.ink.w <= 0:
            continue
        chars = max(len(l) for l in (q.node.text or " ").split("\n"))
        fams_seen: list[_Cell] = []  # every family's best cell: tracking/slant also mislead the family ranking
        for c in pool:
            if c.style.family in q.meas and all(c.style.family != f.style.family for f in fams_seen):
                fams_seen.append(c)
        for rank, c in enumerate(fams_seen):
            mf = q.meas[c.style.family]
            # a tracked line is matched in pass 2 by a bolder/larger untracked one: retry the perceived weight too
            wts = sorted({c.style.weight, q.node.text_style.weight})
            for ls in P["perceive.fonts.tracking"]:
                ls = float(ls)
                # ink width ~ natural width (scales with size) + ls per glyph gap
                size = round((q.ink.w - ls * max(0, chars - 1)) * mf.style.size / mf.ink.w * 2) / 2.0
                for s in (size - 0.5, size, size + 0.5):
                    for wt in wts:
                        if s >= 6:  # placed once; _realign moves the promising ones onto the target ink
                            p3.extend(_aligned_cells(qi, q, mf, _style(q.node.text_style, c.style.family, s, wt, letter_spacing=ls),
                                                     shifts=((0, 0),)))
            for s in (c.style.size - 0.5, c.style.size, c.style.size + 0.5):
                if s >= 6 and rank < 2:  # italic for the two best families
                    p3.extend(_aligned_cells(qi, q, mf, _style(c.style, c.style.family, s, c.style.weight, italic=not c.style.italic),
                                             shifts=((0, 0),)))
    for c in p3:
        c.p3 = True
    _render_and_score(queries, p3)
    r3 = _realign(queries, p3)
    _render_and_score(queries, r3)
    p3 = p3 + r3
    for c in p3:
        by_q.setdefault(c.qi, []).append(c)
    changed = 0
    gain_min = float(P["perceive.fonts.min_gain"])
    fam_count: dict[str, int] = {}
    for qi, q in enumerate(queries):
        ref = q.cells[0]
        pool = q.cells + by_q.get(qi, [])
        best = min(pool, key=lambda c: c.de)
        best12 = min((c for c in pool if not c.p3), key=lambda c: c.de)
        if best.p3 and best.de > (1.0 - float(P["perceive.fonts.p3_rel_gain"])) * best12.de:
            best = best12  # tracking/italic only when it clearly explains the pixels better
        q.node.meta["font_de_before"] = round(ref.de, 3)
        if best is not ref and best.de < ref.de - gain_min:
            q.node.text_style = best.style
            q.node.box = best.box.translate(q.origin[0], q.origin[1])
            q.node.meta["font_de"] = round(best.de, 3)
            changed += 1
        else:
            q.node.meta["font_de"] = round(ref.de, 3)
        fam_count[q.node.text_style.family] = fam_count.get(q.node.text_style.family, 0) + 1
    if fam_count:
        doc.fonts = sorted(fam_count, key=lambda f: -fam_count[f])
    return {"texts": len(queries), "changed": changed, "families": fam_count, "cells": len(p1) + len(p2) + len(p3)}
