"""A small model of Figma's auto-layout engine, operating on build-plan / fake-figma node dicts.

Figma ignores the x/y the plugin writes on children of an auto-layout frame: it re-positions them
from the frame's padding / itemSpacing / alignment, re-sizes hugging frames and FILL children, and
sizes auto-width text from its glyphs. Neither the plan (pure JSON) nor fake_figma.js (stores what
it is told) reproduces that, so without this model an auto-layout plan *looks* lossless while the
real import moves things. Used by:

  * ``dt.export.figma_json.to_build_plan``: frames whose inferred layout would not put the children
    back at their IR boxes are exported without auto-layout (absolute children) -- see
    ``demote_inexact_layouts``.
  * ``dt.validate.roundtrip``: simulate the import of the fake-figma tree before converting it back.

Field names are the plan's / Figma's (``layoutMode``, ``itemSpacing``, ``padding*``,
``primaryAxisAlignItems``, ``counterAxisAlignItems``, ``primaryAxisSizingMode``,
``counterAxisSizingMode``, ``layoutWrap``, ``counterAxisSpacing``, ``layoutSizingHorizontal/Vertical``,
``layoutPositioning``, ``textAutoResize``). Coordinates are parent-relative, as in the plan.
"""
from __future__ import annotations

import copy
from typing import Callable, Optional

from dt.params import P, register

register("export.figma.layout_exact_tol", 0.5,
         "px: an auto-layout frame is exported only if Figma's layout puts every child within this of its IR box.",
         (0.05, 3.0))
register("export.figma.layout_max_passes", 32, "demote_inexact_layouts: max simulate/demote passes.", (1, 128))
register("export.figma.layout_text_probe", 4.0,
         "px added to auto-width text (w and h) in the second layout simulation: auto-layout is kept only if no child "
         "moves, i.e. the layout does not depend on Figma's glyph metrics.", (0.5, 16.0))
register("export.figma.verify_layout", True,
         "to_build_plan: drop auto-layout from frames whose Figma layout would not reproduce the IR boxes.")

TextSize = Callable[[dict], tuple[float, float]]

LAYOUT_KEYS = ("layoutMode", "itemSpacing", "paddingTop", "paddingRight", "paddingBottom", "paddingLeft",
               "primaryAxisAlignItems", "counterAxisAlignItems", "primaryAxisSizingMode", "counterAxisSizingMode",
               "layoutWrap", "counterAxisSpacing")


def is_auto(n: dict) -> bool:
    return n.get("type") in ("FRAME", "INSTANCE", "COMPONENT") and n.get("layoutMode") not in (None, "NONE")


def _flow(n: dict) -> list[dict]:
    """Children that take part in the flow (hidden and absolutely positioned ones do not)."""
    return [c for c in n.get("children") or [] if c.get("visible", True) is not False
            and c.get("layoutPositioning", "AUTO") != "ABSOLUTE"]


def _axes(n: dict) -> tuple[str, str, str, str, str, str, str, str]:
    """(main pos, main dim, cross pos, cross dim, main pad lead/trail, cross pad lead/trail)."""
    if n.get("layoutMode") == "HORIZONTAL":
        return "x", "width", "y", "height", "paddingLeft", "paddingRight", "paddingTop", "paddingBottom"
    return "y", "height", "x", "width", "paddingTop", "paddingBottom", "paddingLeft", "paddingRight"


def _hugs(n: dict, horizontal: bool, parent_auto: bool) -> bool:
    """Does `n` hug its content on this axis? A child's layoutSizing* (set last by the plugin) wins."""
    key = "layoutSizingHorizontal" if horizontal else "layoutSizingVertical"
    if parent_auto and n.get(key) in ("HUG", "FIXED", "FILL"):
        return n[key] == "HUG"
    if not is_auto(n):
        return False
    main_is_h = n.get("layoutMode") == "HORIZONTAL"
    mode = n.get("primaryAxisSizingMode") if horizontal == main_is_h else n.get("counterAxisSizingMode")
    return mode == "AUTO"


def _num(n: dict, k: str) -> float:
    v = n.get(k)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def _measure(n: dict, text_size: Optional[TextSize], parent_auto: bool) -> None:
    """Bottom-up: text auto-resize, then hug sizes of auto-layout frames."""
    for c in n.get("children") or []:
        _measure(c, text_size, is_auto(n))
    if n.get("type") == "TEXT" and text_size is not None and n.get("textAutoResize") in ("WIDTH_AND_HEIGHT", "HEIGHT"):
        w, h = text_size(n)
        if n["textAutoResize"] == "WIDTH_AND_HEIGHT":
            n["width"] = w
        n["height"] = h
    if not is_auto(n):
        return
    mp, md, cp, cd, ml, mt, cl, ct = _axes(n)
    kids = _flow(n)
    row = n.get("layoutMode") == "HORIZONTAL"
    if kids:  # a hugging wrap frame grows rather than wraps
        main = sum(_num(c, md) for c in kids) + _num(n, "itemSpacing") * (len(kids) - 1)
        cross = max(_num(c, cd) for c in kids)
    else:
        main = cross = 0.0
    if _hugs(n, row, parent_auto):
        n[md] = _num(n, ml) + main + _num(n, mt)
    if _hugs(n, not row, parent_auto):
        n[cd] = _num(n, cl) + cross + _num(n, ct)


def _place(n: dict) -> None:
    """Top-down: FILL sizes, then child positions of every auto-layout frame."""
    if is_auto(n):
        mp, md, cp, cd, ml, mt, cl, ct = _axes(n)
        row = n.get("layoutMode") == "HORIZONTAL"
        main_key = "layoutSizingHorizontal" if row else "layoutSizingVertical"
        cross_key = "layoutSizingVertical" if row else "layoutSizingHorizontal"
        kids = _flow(n)
        gap = _num(n, "itemSpacing")
        inner_main = _num(n, md) - _num(n, ml) - _num(n, mt)
        inner_cross = _num(n, cd) - _num(n, cl) - _num(n, ct)
        lines: list[list[dict]] = [kids]
        if row and n.get("layoutWrap") == "WRAP" and kids:
            lines, cur, used = [], [], 0.0
            for c in kids:
                w = _num(c, md)
                if cur and used + gap + w > inner_main + 1e-6:
                    lines.append(cur)
                    cur, used = [], 0.0
                used = w if not cur else used + gap + w
                cur.append(c)
            lines.append(cur)
        cross_gap = _num(n, "counterAxisSpacing")
        cross_pos = _num(n, cl)
        for line in lines:
            fills = [c for c in line if c.get(main_key) == "FILL"]
            if fills:
                rest = inner_main - sum(_num(c, md) for c in line if c not in fills) - gap * (len(line) - 1)
                share = max(0.0, rest) / len(fills)
                for c in fills:
                    c[md] = share
            line_cross = inner_cross if len(lines) == 1 else max(_num(c, cd) for c in line)
            for c in line:
                if c.get(cross_key) == "FILL":
                    c[cd] = line_cross
            content = sum(_num(c, md) for c in line) + gap * (len(line) - 1)
            justify = n.get("primaryAxisAlignItems", "MIN")
            step = gap
            start = _num(n, ml)
            if justify == "CENTER":
                start += (inner_main - content) / 2.0
            elif justify == "MAX":
                start += inner_main - content
            elif justify == "SPACE_BETWEEN" and len(line) > 1:
                step = (inner_main - sum(_num(c, md) for c in line)) / (len(line) - 1)
            align = n.get("counterAxisAlignItems", "MIN")
            pos = start
            for c in line:
                c[mp] = pos
                off = 0.0
                if align == "CENTER":
                    off = (line_cross - _num(c, cd)) / 2.0
                elif align == "MAX":
                    off = line_cross - _num(c, cd)
                c[cp] = cross_pos + off
                pos += _num(c, md) + step
            cross_pos += line_cross + cross_gap
    for c in n.get("children") or []:
        _place(c)


def apply_auto_layout(root: dict, text_size: Optional[TextSize] = None) -> dict:
    """Run the Figma auto-layout model over `root` in place (and return it).

    `text_size(node) -> (w, h)` gives the glyph box of auto-resizing text; when None, text keeps its size.
    """
    _measure(root, text_size, False)
    _place(root)
    return root


def _child_lists(n: dict) -> list[list[dict]]:
    out = []
    if n.get("children"):
        out.append(n["children"])
    if isinstance(n.get("fallback"), dict):
        out.append([n["fallback"]])
    return out


def _walk(n: dict):
    yield n
    for lst in _child_lists(n):
        for c in lst:
            yield from _walk(c)


def _strip_layout(n: dict) -> None:
    for k in LAYOUT_KEYS:
        n.pop(k, None)
    for c in n.get("children") or []:
        c.pop("layoutSizingHorizontal", None)
        c.pop("layoutSizingVertical", None)


def _deviates(orig: dict, sim: dict, tol: float, keys: tuple[str, ...] = ("x", "y", "width", "height")) -> bool:
    for k in keys:
        if abs(_num(orig, k) - _num(sim, k)) > tol:
            return True
    return False


def predicted_text_height(n: dict) -> Optional[float]:
    """Figma's height of auto-resizing text when it is knowable without glyph metrics (pixel line height)."""
    lh = n.get("lineHeight") or {}
    if lh.get("unit") != "PIXELS" or not isinstance(lh.get("value"), (int, float)):
        return None
    return float(lh["value"]) * (str(n.get("characters", "")).count("\n") + 1)


def _sim(plan_root: dict, dw: float, dh: float) -> dict:
    """Simulate Figma on a copy; auto-resizing text gets its plan width + dw and its predicted height + dh."""
    def size(n: dict) -> tuple[float, float]:
        h = predicted_text_height(n)  # known exactly with a pixel line height: only the width is probed then
        return _num(n, "width") + dw, h if h is not None else _num(n, "height") + dh
    sim = copy.deepcopy(plan_root)
    apply_auto_layout(sim, size)
    for f in [s for s in _walk(sim) if isinstance(s.get("fallback"), dict)]:
        apply_auto_layout(f["fallback"], size)  # missing component: the plugin builds the fallback frame
    return sim


def _frame_ok(o: dict, a: dict, b: dict, tol: float) -> bool:
    """`o` (plan) is reproduced by simulation `a`, and simulation `b` (perturbed glyph metrics) agrees with `a`."""
    if _deviates(o, a, tol, ("width", "height")) or _deviates(a, b, tol, ("width", "height")):
        return False
    for co, ca, cb in zip(o.get("children") or [], a.get("children") or [], b.get("children") or []):
        keys = ("x", "y") if co.get("type") == "TEXT" else ("x", "y", "width", "height")
        if _deviates(co, ca, tol, keys) or _deviates(ca, cb, tol, keys):
            return False
    return True


def demote_inexact_layouts(plan_root: dict, warnings: Optional[list[str]] = None) -> int:
    """Remove auto-layout from every frame whose Figma layout would not reproduce the plan's (IR) geometry,
    so an import looks like the IR render. Returns how many frames were demoted (children stay absolute).

    A frame keeps auto-layout only if (a) simulating Figma puts every child back at its plan box and (b) the
    result does not depend on glyph metrics: Figma sizes auto-width text from its own font rendering, which
    the exporter cannot know, so the simulation is repeated with text grown by ``layout_text_probe`` px.
    Deepest offending frames are demoted first (a parent may be fine once a child frame is fixed).
    """
    tol = float(P["export.figma.layout_exact_tol"])
    probe = float(P["export.figma.layout_text_probe"])
    demoted = 0
    for _ in range(int(P["export.figma.layout_max_passes"])):
        a, b = _sim(plan_root, 0.0, 0.0), _sim(plan_root, probe, probe)
        bad: set[int] = set()
        nodes = list(zip(_walk(plan_root), _walk(a), _walk(b)))
        for o, na, nb in nodes:
            if is_auto(o) and not _frame_ok(o, na, nb, tol):
                bad.add(id(o))
        if not bad:
            break
        # innermost first: skip a bad frame when one of its descendants is also bad
        for o, _a, _b in nodes:
            if id(o) not in bad or any(id(d) in bad for d in list(_walk(o))[1:]):
                continue
            if warnings is not None:
                warnings.append(f"{o.get('source', {}).get('irType', 'frame')}:{o.get('id')}: Figma auto-layout would not "
                                f"reproduce the IR geometry; exported with absolute children")
            _strip_layout(o)
            demoted += 1
    return demoted
