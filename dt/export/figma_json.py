"""IR Document -> Figma build plan (JSON).

The build plan is a plain, versioned JSON tree (schema: ``dt/export/BUILD_PLAN.md``) that the
Figma plugin in ``dt/export/figma_plugin`` turns into real nodes. Everything here is a pure,
deterministic transformation of the IR: no I/O, no network, no pixels.

Public API:
    to_build_plan(doc) -> dict          IR -> plan (parent-relative coords, Figma paints, ...)
    validate_plan(plan) -> list[str]    schema + numeric sanity; [] means valid
    save_plan(plan, path) -> str        json dump helper
    font_style(weight, italic) -> str   Figma font style name from CSS weight/italic
    gradient_handles(angle, w, h)       CSS angle -> Figma gradientHandlePositions

Figma conventions reproduced here (so the plugin stays dumb):
  * colors are 0..1 floats; SOLID paint opacity = color alpha * fill opacity
  * gradient handles are in normalized node space (0..1 x 0..1), start/end/width
  * strokes: one weight + align per node; per-side weights via strokeTopWeight etc.
  * auto-layout vocabulary: HORIZONTAL/VERTICAL, MIN/CENTER/MAX/SPACE_BETWEEN, FIXED/AUTO
  * left-aligned text is auto-width (WIDTH_AND_HEIGHT) with explicit line breaks, so it can never wrap;
    centred / right-aligned text keeps a fixed box (textAutoResize NONE) so its alignment frame is preserved
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from dt.export.figma_layout import demote_inexact_layouts
from dt.ir import Color, Document, Fill, Layout, Node, Shadow, Stroke, TextStyle
from dt.params import P, register

PLAN_VERSION = "1.0"
GENERATOR = "designtranslator"

register("export.figma.coord_precision", 3, "Decimal places kept for plan coordinates/sizes (2 shifts sub-pixel "
         "text by up to 0.005px, which changes glyph rasterisation: export round-trip ΔE > 0).", (0, 4))
register("export.figma.max_dimension", 65536.0, "validate_plan: largest allowed node width/height (px).", (4096, 200000))
register("export.figma.max_nodes", 20000, "validate_plan: largest allowed node count in one plan.", (100, 100000))
register("export.figma.min_font_size", 1.0, "validate_plan: smallest allowed fontSize (px).", (0.5, 8))
register("export.figma.single_line_autowidth", True,
         "Left-aligned text exports as auto-width (WIDTH_AND_HEIGHT) so Figma never wraps it; line breaks are "
         "explicit, mirroring dt.render where text never wraps (a wrap turns a 1px width error into a line jump).")
register("export.figma.line_mode", "rect", "IR 'line' nodes -> 'rect' (filled RECTANGLE, pixel-exact) or 'line' (Figma LINE for thin horizontal lines).")
register("export.figma.line_max_thickness", 2.0, "line_mode='line': lines thicker than this become rectangles (px).", (0.5, 8))
register("export.figma.default_font_family", "Roboto", "Font family used when a text node has no style.")
register("export.figma.fallback_font_style", "Regular", "Font style the plugin falls back to when a (family, style) is missing.")
register("export.figma.icon_font_family", "Material Symbols Outlined", "Font family used for icon text nodes.")
register("export.figma.image_placeholder", "#e0e0e0", "Solid color the plugin uses for IMAGE fills it cannot resolve.")
register("export.figma.embed_local_images", True, "Embed local image files as PNG data URIs (the plugin cannot read files).")
register("export.figma.embed_max_bytes", 8_000_000, "Largest local image file embedded into the plan (bytes).", (100_000, 50_000_000))
register("export.figma.max_text_name_len", 40, "Characters of text used to auto-name unnamed text nodes.", (8, 120))
register("export.figma.synth_label_size", 14.0, "fontSize of the label synthesized for instance fallbacks without children (px).", (8, 24))

FigmaNodeType = str  # FRAME | RECTANGLE | ELLIPSE | LINE | TEXT | INSTANCE
NODE_TYPES = ("FRAME", "RECTANGLE", "ELLIPSE", "LINE", "TEXT", "INSTANCE")
PAINT_TYPES = ("SOLID", "GRADIENT_LINEAR", "GRADIENT_RADIAL", "IMAGE")
EFFECT_TYPES = ("DROP_SHADOW", "INNER_SHADOW")
STROKE_ALIGNS = ("INSIDE", "CENTER", "OUTSIDE")
LAYOUT_MODES = ("NONE", "HORIZONTAL", "VERTICAL")
PRIMARY_ALIGNS = ("MIN", "CENTER", "MAX", "SPACE_BETWEEN")
COUNTER_ALIGNS = ("MIN", "CENTER", "MAX")
SIZING_MODES = ("FIXED", "AUTO")
CHILD_SIZINGS = ("FIXED", "HUG", "FILL")
TEXT_ALIGN_H = ("LEFT", "CENTER", "RIGHT")
TEXT_ALIGN_V = ("TOP", "CENTER", "BOTTOM")

WEIGHT_STYLES: dict[int, str] = {
    100: "Thin", 200: "ExtraLight", 300: "Light", 400: "Regular", 500: "Medium",
    600: "SemiBold", 700: "Bold", 800: "ExtraBold", 900: "Black",
}

# IR token keys (node.tokens) -> Figma bindable fields. Unknown keys are kept in `tokenRefs`.
TOKEN_FIELDS: dict[str, tuple[str, ...]] = {
    "fill": ("fills",), "background": ("fills",), "color": ("fills",), "text_color": ("fills",),
    "stroke": ("strokes",), "border": ("strokes",), "outline": ("strokes",),
    "radius": ("cornerRadius",), "corner_radius": ("cornerRadius",),
    "gap": ("itemSpacing",), "spacing": ("itemSpacing",),
    "padding": ("paddingTop", "paddingRight", "paddingBottom", "paddingLeft"),
    "padding_top": ("paddingTop",), "padding_right": ("paddingRight",),
    "padding_bottom": ("paddingBottom",), "padding_left": ("paddingLeft",),
    "font_size": ("fontSize",), "line_height": ("lineHeight",), "letter_spacing": ("letterSpacing",),
    "opacity": ("opacity",), "width": ("width",), "height": ("height",),
}


# --------------------------------------------------------------------------- small converters
def _num(v: float) -> float:
    """Round to the registered precision; integers come back as ints for compact JSON."""
    r = round(float(v), int(P["export.figma.coord_precision"]))
    return int(r) if r == int(r) else r


def _rgb(c: Color) -> dict[str, float]:
    return {"r": round(c.r / 255.0, 6), "g": round(c.g / 255.0, 6), "b": round(c.b / 255.0, 6)}


def _rgba(c: Color) -> dict[str, float]:
    d = _rgb(c)
    d["a"] = round(float(c.a), 6)
    return d


def font_style(weight: int, italic: bool = False) -> str:
    """CSS weight (100..900) + italic -> Figma style name ("Regular", "Medium", "Bold Italic"...)."""
    w = int(round(max(100, min(900, int(weight))) / 100.0)) * 100
    base = WEIGHT_STYLES[w]
    if not italic:
        return base
    return "Italic" if base == "Regular" else f"{base} Italic"


def gradient_handles(angle_deg: float, w: float, h: float) -> list[dict[str, float]]:
    """CSS linear-gradient angle -> Figma gradientHandlePositions [start, end, width] (normalized).

    CSS: 0deg points up, 90deg right, 180deg down; the gradient line goes through the box center
    and has length |w sin a| + |h cos a| (so the first/last stops touch the corners).
    Figma: handles live in node space where (0,0)/(1,1) are the top-left/bottom-right corners.
    """
    w, h = max(float(w), 1e-6), max(float(h), 1e-6)
    a = math.radians(float(angle_deg))
    dx, dy = math.sin(a), -math.cos(a)  # unit direction in pixel space (y down)
    length = abs(w * dx) + abs(h * dy)
    cx, cy = w / 2.0, h / 2.0
    sx, sy = cx - dx * length / 2.0, cy - dy * length / 2.0
    ex, ey = cx + dx * length / 2.0, cy + dy * length / 2.0
    start = {"x": sx / w, "y": sy / h}
    end = {"x": ex / w, "y": ey / h}
    # third handle: perpendicular to start->end, half its length (Figma's default convention)
    vx, vy = end["x"] - start["x"], end["y"] - start["y"]
    width_h = {"x": start["x"] + vy * 0.5, "y": start["y"] - vx * 0.5}
    return [{k: round(v, 6) for k, v in p.items()} for p in (start, end, width_h)]


def radial_handles(w: float, h: float) -> list[dict[str, float]]:
    """CSS radial-gradient(circle, ...) (farthest-corner) -> Figma handles [center, x-radius, y-radius]."""
    w, h = max(float(w), 1e-6), max(float(h), 1e-6)
    r = math.hypot(w / 2.0, h / 2.0)
    return [
        {"x": 0.5, "y": 0.5},
        {"x": round(0.5 + r / w, 6), "y": 0.5},
        {"x": 0.5, "y": round(0.5 + r / h, 6)},
    ]


def embed_image_ref(ref: str, warnings: Optional[list[str]] = None, owner: str = "") -> str:
    """Local image files (path or file:// URL) -> PNG data URI, which the plugin can turn into a Figma image
    (it cannot read the user's disk: a path would become a grey placeholder). Other refs pass through."""
    if not bool(P["export.figma.embed_local_images"]) or ref.startswith("data:"):
        return ref
    path = ref[len("file://"):] if ref.startswith("file://") else ref
    if "://" in path or not os.path.isfile(path):
        return ref
    try:
        if os.path.getsize(path) > int(P["export.figma.embed_max_bytes"]):
            raise ValueError(f"larger than export.figma.embed_max_bytes")
        from PIL import Image
        buf = io.BytesIO()
        Image.open(path).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:  # unreadable / huge: keep the ref (plugin uses its placeholder) and say so
        if warnings is not None:
            warnings.append(f"{owner}: image {ref!r} not embedded ({e})")
        return ref


def fill_to_paint(f: Fill, w: float, h: float, warnings: list[str], owner: str = "") -> Optional[dict]:
    """IR Fill -> Figma Paint dict (None when the fill carries nothing paintable)."""
    if f.kind == "solid":
        if f.color is None:
            return None
        return {"type": "SOLID", "color": _rgb(f.color), "opacity": round(float(f.color.a) * float(f.opacity), 6)}
    if f.kind in ("linear", "radial"):
        if len(f.stops) < 2:
            warnings.append(f"{owner}: {f.kind} gradient with <2 stops dropped")
            return None
        stops = [{"position": round(float(s.pos), 6), "color": _rgba(s.color)} for s in f.stops]
        if f.kind == "linear":
            return {"type": "GRADIENT_LINEAR", "gradientStops": stops,
                    "gradientHandlePositions": gradient_handles(f.angle, w, h), "opacity": round(float(f.opacity), 6)}
        return {"type": "GRADIENT_RADIAL", "gradientStops": stops,
                "gradientHandlePositions": radial_handles(w, h), "opacity": round(float(f.opacity), 6)}
    if f.kind == "image":
        if not f.image_ref:
            return None
        return {"type": "IMAGE", "scaleMode": "FILL", "imageRef": embed_image_ref(f.image_ref, warnings, owner),
                "opacity": round(float(f.opacity), 6)}
    warnings.append(f"{owner}: unknown fill kind {f.kind!r} dropped")
    return None


def strokes_to_props(strokes: list[Stroke], per_side_ok: bool, warnings: list[str], owner: str = "") -> dict:
    """IR strokes -> {strokes, strokeWeight, strokeAlign[, strokeTopWeight...]}. Figma has ONE weight/align per node."""
    if not strokes:
        return {}
    first = strokes[0]
    paints = [{"type": "SOLID", "color": _rgb(s.color), "opacity": round(float(s.color.a), 6)} for s in strokes]
    out: dict[str, Any] = {"strokes": paints, "strokeAlign": first.align.upper()}
    if len(strokes) > 1 and any(s.width != first.width or s.align != first.align for s in strokes[1:]):
        warnings.append(f"{owner}: multiple strokes with different width/align; using the first one's")
    if first.sides is not None:
        t, r, b, l = first.sides
        if per_side_ok:
            out.update({"strokeWeight": _num(max(t, r, b, l)), "strokeTopWeight": _num(t), "strokeRightWeight": _num(r),
                        "strokeBottomWeight": _num(b), "strokeLeftWeight": _num(l), "strokeAlign": "INSIDE"})
        else:
            warnings.append(f"{owner}: per-side stroke not supported on this node type; using max side")
            out["strokeWeight"] = _num(max(t, r, b, l))
    else:
        out["strokeWeight"] = _num(first.width)
    return out


def effect_to_figma(s: Shadow) -> dict:
    return {
        "type": "INNER_SHADOW" if s.inner else "DROP_SHADOW",
        "color": _rgba(s.color),
        "offset": {"x": _num(s.dx), "y": _num(s.dy)},
        "radius": _num(s.blur),
        "spread": _num(s.spread),
        "visible": True,
        "blendMode": "NORMAL",
    }


def radius_to_props(radius: tuple[float, float, float, float]) -> dict:
    tl, tr, br, bl = radius
    if tl == tr == br == bl:
        return {"cornerRadius": _num(tl)} if tl > 0 else {}
    return {"topLeftRadius": _num(tl), "topRightRadius": _num(tr), "bottomRightRadius": _num(br), "bottomLeftRadius": _num(bl)}


_JUSTIFY = {"start": "MIN", "center": "CENTER", "end": "MAX", "space-between": "SPACE_BETWEEN"}
_ALIGN = {"start": "MIN", "center": "CENTER", "end": "MAX", "stretch": "MIN"}
_SIZING = {"fixed": "FIXED", "hug": "AUTO", "fill": "FIXED"}
_CHILD_SIZING = {"fixed": "FIXED", "hug": "HUG", "fill": "FILL"}


def _enum(table: dict, value: Any, default: str, what: str, warnings: Optional[list[str]], owner: str = "") -> str:
    """Tolerant enum lookup: values outside the IR contract (e.g. a hand-edited ir.json with
    justify='space-around') fall back to `default` with a warning instead of raising."""
    if value in table:
        return table[value]
    if warnings is not None:
        warnings.append(f"{owner}: unsupported {what} {value!r}; using {default}")
    return default


def layout_to_props(L: Layout, warnings: Optional[list[str]] = None, owner: str = "") -> dict:
    """IR Layout -> Figma auto-layout properties (empty dict when mode == 'none')."""
    if L.mode == "none":
        return {}
    t, r, b, l = L.padding
    row = L.mode == "row"
    main_sizing, cross_sizing = (L.sizing_h, L.sizing_v) if row else (L.sizing_v, L.sizing_h)
    return {
        "layoutMode": "HORIZONTAL" if row else "VERTICAL",
        "itemSpacing": _num(L.gap),
        "paddingTop": _num(t), "paddingRight": _num(r), "paddingBottom": _num(b), "paddingLeft": _num(l),
        "primaryAxisAlignItems": _enum(_JUSTIFY, L.justify, "MIN", "layout.justify", warnings, owner),
        "counterAxisAlignItems": _enum(_ALIGN, L.align_items, "MIN", "layout.align_items", warnings, owner),
        "primaryAxisSizingMode": _enum(_SIZING, main_sizing, "FIXED", "layout.sizing", warnings, owner),
        "counterAxisSizingMode": _enum(_SIZING, cross_sizing, "FIXED", "layout.sizing", warnings, owner),
        "layoutWrap": "WRAP" if L.wrap else "NO_WRAP",
    }


def is_autowidth_text(n: Node) -> bool:
    """True when `n` exports as auto-width text (left aligned, policy enabled). Line breaks are explicit
    ("\n" per perceived line) and auto-width text never wraps in Figma, exactly like dt.render."""
    if n.type != "text" or not bool(P["export.figma.single_line_autowidth"]):
        return False
    ts = n.text_style
    return ts is None or ts.align == "left"


def child_sizing_props(child: Node, parent: Optional[Node]) -> dict:
    """layoutSizingHorizontal/Vertical for a child of an auto-layout parent."""
    if parent is None or parent.layout is None or parent.layout.mode == "none":
        return {}
    pl = parent.layout
    sh = _CHILD_SIZING[child.layout.sizing_h] if child.layout else ("HUG" if is_autowidth_text(child) else "FIXED")
    sv = _CHILD_SIZING[child.layout.sizing_v] if child.layout else ("HUG" if is_autowidth_text(child) else "FIXED")
    if pl.align_items == "stretch":  # cross-axis stretch == FILL on the cross axis
        if pl.mode == "row":
            sv = "FILL"
        else:
            sh = "FILL"
    # HUG is only legal on auto-layout frames and text nodes
    if sh == "HUG" and not (child.type == "text" or (child.layout and child.layout.mode != "none")):
        sh = "FIXED"
    if sv == "HUG" and not (child.type == "text" or (child.layout and child.layout.mode != "none")):
        sv = "FIXED"
    return {"layoutSizingHorizontal": sh, "layoutSizingVertical": sv}


_TEXT_ALIGN = {"left": "LEFT", "center": "CENTER", "right": "RIGHT"}
_TEXT_VALIGN = {"top": "TOP", "middle": "CENTER", "bottom": "BOTTOM"}
_TEXT_DECORATION = {"none": "NONE", "underline": "UNDERLINE", "line-through": "STRIKETHROUGH"}
_TEXT_CASE = {"none": "ORIGINAL", "uppercase": "UPPER", "lowercase": "LOWER", "capitalize": "TITLE"}


def text_style_props(ts: TextStyle, fonts: set[tuple[str, str]], warnings: Optional[list[str]] = None, owner: str = "") -> dict:
    """TextStyle -> Figma text properties (registers the font in `fonts`)."""
    style = font_style(ts.weight, ts.italic)
    fonts.add((ts.family, style))
    out = {
        "fontName": {"family": ts.family, "style": style},
        "fontSize": _num(ts.size),
        "lineHeight": {"value": _num(ts.line_height), "unit": "PIXELS"} if ts.line_height else {"unit": "AUTO"},
        "letterSpacing": {"value": _num(ts.letter_spacing), "unit": "PIXELS"},
        "textAlignHorizontal": _enum(_TEXT_ALIGN, ts.align, "LEFT", "text align", warnings, owner),
        "textAlignVertical": _enum(_TEXT_VALIGN, ts.valign, "TOP", "text valign", warnings, owner),
        "textDecoration": _enum(_TEXT_DECORATION, ts.decoration, "NONE", "text decoration", warnings, owner),
        "textCase": _enum(_TEXT_CASE, ts.transform, "ORIGINAL", "text transform", warnings, owner),
        "textAutoResize": "NONE",
        "fills": [{"type": "SOLID", "color": _rgb(ts.color), "opacity": round(float(ts.color.a), 6)}],
    }
    return out


def token_bindings(tokens: dict[str, str]) -> tuple[dict, dict]:
    """node.tokens -> (variableBindings {figmaField: variableName}, tokenRefs {irKey: name} for the rest)."""
    bindings: dict[str, str] = {}
    refs: dict[str, str] = {}
    for key, name in tokens.items():
        fields = TOKEN_FIELDS.get(key)
        if fields:
            for f in fields:
                bindings[f] = name
        else:
            refs[key] = name
    return bindings, refs


def node_display_name(n: Node) -> str:
    if n.name:
        return n.name
    if n.type == "text" and n.text:
        return n.text.strip()[: int(P["export.figma.max_text_name_len"])] or "Text"
    if n.type == "icon" and n.icon_name:
        return f"icon/{n.icon_name}"
    if n.type == "instance" and n.component:
        variant = "/".join(str(v) for v in n.component.variant.values())
        return f"{n.component.name}/{variant}" if variant else n.component.name
    return n.type.capitalize()


# --------------------------------------------------------------------------- tree conversion
@dataclass
class _Ctx:
    fonts: set[tuple[str, str]] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)
    count: int = 0


def _geometry(n: Node, parent: Optional[Node]) -> dict:
    px, py = (parent.box.x, parent.box.y) if parent is not None else (0.0, 0.0)
    return {"x": _num(n.box.x - px), "y": _num(n.box.y - py), "width": _num(n.box.w), "height": _num(n.box.h)}


def _base(n: Node, parent: Optional[Node], figma_type: str, ctx: _Ctx) -> dict:
    """Fields shared by every plan node."""
    d: dict[str, Any] = {"id": n.id, "name": node_display_name(n), "type": figma_type}
    d.update(_geometry(n, parent))
    d["visible"] = bool(n.visible)
    d["opacity"] = round(float(n.opacity), 6)
    bindings, refs = token_bindings(n.tokens)
    if bindings:
        d["variableBindings"] = bindings
    if refs:
        d["tokenRefs"] = refs
    d.update(child_sizing_props(n, parent))
    d["source"] = {"irId": n.id, "irType": n.type}
    return d


def _paint_props(n: Node, ctx: _Ctx, per_side_ok: bool, figma_type: str) -> dict:
    owner = f"{n.type}:{n.id}"
    d: dict[str, Any] = {}
    if figma_type != "LINE":
        paints = [p for p in (fill_to_paint(f, n.box.w, n.box.h, ctx.warnings, owner) for f in n.fills) if p]
        d["fills"] = paints
    d.update(strokes_to_props(n.strokes, per_side_ok, ctx.warnings, owner))
    if n.effects:
        d["effects"] = [effect_to_figma(e) for e in n.effects]
    return d


def _shape_node(n: Node, parent: Optional[Node], ctx: _Ctx, figma_type: str) -> dict:
    d = _base(n, parent, figma_type, ctx)
    d.update(_paint_props(n, ctx, per_side_ok=figma_type in ("FRAME", "RECTANGLE"), figma_type=figma_type))
    if figma_type in ("FRAME", "RECTANGLE"):
        d.update(radius_to_props(n.radius))
    if figma_type == "FRAME":
        d["clipsContent"] = bool(n.clip)
        if n.layout is not None:
            d.update(layout_to_props(n.layout, ctx.warnings, f"{n.type}:{n.id}"))
        d["children"] = [c for c in (node_to_plan(ch, n, ctx) for ch in n.children) if c is not None]
    elif n.children:
        ctx.warnings.append(f"{n.type}:{n.id}: {len(n.children)} children of a non-frame dropped")
    return d


def _line_node(n: Node, parent: Optional[Node], ctx: _Ctx) -> dict:
    """IR line -> filled RECTANGLE (default) or Figma LINE (thin horizontal only, stroke centered on y)."""
    color = n.fill_color or (n.strokes[0].color if n.strokes else Color(0, 0, 0))
    thick = min(n.box.w, n.box.h)
    use_line = (P["export.figma.line_mode"] == "line" and n.box.w >= n.box.h and thick <= P["export.figma.line_max_thickness"])
    if not use_line:
        d = _base(n, parent, "RECTANGLE", ctx)
        d["fills"] = [{"type": "SOLID", "color": _rgb(color), "opacity": round(float(color.a), 6)}]
        d.update(strokes_to_props(n.strokes, True, ctx.warnings, f"line:{n.id}"))
        d.update(radius_to_props(n.radius))  # rounded dividers / drag handles (e.g. 32x4, r=2)
        if n.effects:
            d["effects"] = [effect_to_figma(e) for e in n.effects]
        return d
    d = _base(n, parent, "LINE", ctx)
    d["y"] = _num(d["y"] + n.box.h / 2.0)
    d["height"] = 0
    d["strokes"] = [{"type": "SOLID", "color": _rgb(color), "opacity": round(float(color.a), 6)}]
    d["strokeWeight"] = _num(max(thick, 0.01))
    d["strokeAlign"] = "CENTER"
    return d


def _text_node(n: Node, parent: Optional[Node], ctx: _Ctx) -> dict:
    d = _base(n, parent, "TEXT", ctx)
    ts = n.text_style or TextStyle(family=P["export.figma.default_font_family"])
    d["characters"] = n.text or ""
    d.update(text_style_props(ts, ctx.fonts, ctx.warnings, f"text:{n.id}"))
    if is_autowidth_text(n):
        _autowidth_valign(n, ts, d)
    d.update(strokes_to_props(n.strokes, False, ctx.warnings, f"text:{n.id}"))
    if n.effects:
        d["effects"] = [effect_to_figma(e) for e in n.effects]
    return d


def _autowidth_valign(n: Node, ts: TextStyle, d: dict) -> None:
    """Auto-width text has no vertical alignment frame in Figma (its box shrinks to the lines, top-left fixed),
    so a middle/bottom-aligned IR text box must be converted to the equivalent content box:
      * pixel line height: move y by the slack, height = lines x line height (exact)
      * one line, normal line height, middle: line height := box height (CSS centres the line box exactly
        like the half-leading of a taller line)
      * otherwise keep a fixed box (textAutoResize NONE), like centred/right-aligned text."""
    d["textAutoResize"] = "WIDTH_AND_HEIGHT"
    if ts.valign == "top":
        return
    lines = (n.text or "").count("\n") + 1
    h = n.box.h
    if ts.line_height:
        content = ts.line_height * lines
        slack = h - content
        d["y"] = _num(d["y"] + (slack / 2.0 if ts.valign == "middle" else slack))
        d["height"] = _num(content)
    elif lines == 1 and ts.valign == "middle":
        d["lineHeight"] = {"value": _num(h), "unit": "PIXELS"}
    else:
        d["textAutoResize"] = "NONE"
        for k in ("layoutSizingHorizontal", "layoutSizingVertical"):  # HUG would switch it back to auto-width
            if d.get(k) == "HUG":
                d[k] = "FIXED"
    # textAlignVertical stays as stated: an auto-width box has no vertical slack, so it is inert there


def _icon_node(n: Node, parent: Optional[Node], ctx: _Ctx) -> dict:
    """Icons are Material Symbols ligature text nodes: characters == icon name, size == min(w, h)."""
    d = _base(n, parent, "TEXT", ctx)
    family = P["export.figma.icon_font_family"]
    ctx.fonts.add((family, "Regular"))
    color = n.fill_color or (n.text_style.color if n.text_style else Color(0, 0, 0))
    d.update({
        "characters": n.icon_name or "",
        "isIcon": True,
        "fontName": {"family": family, "style": "Regular"},
        "fontSize": _num(min(n.box.w, n.box.h)),
        "lineHeight": {"unit": "AUTO"},
        "letterSpacing": {"value": 0, "unit": "PIXELS"},
        "textAlignHorizontal": "CENTER",
        "textAlignVertical": "CENTER",
        "textDecoration": "NONE",
        "textCase": "ORIGINAL",
        "textAutoResize": "NONE",
        "fills": [{"type": "SOLID", "color": _rgb(color), "opacity": round(float(color.a), 6)}],
    })
    fill_axis = n.meta.get("icon_fill") if isinstance(n.meta, dict) else None
    if fill_axis:  # filled Material Symbol (FILL axis 1): dt.render draws it filled; keep it in the plan
        d["iconFill"] = int(fill_axis)
    if not n.icon_name:
        ctx.warnings.append(f"icon:{n.id}: missing icon_name")
    return d


def _synth_label(n: Node, label: str) -> Node:
    """A centered text child used when an instance fallback has no children but a label prop."""
    ts = TextStyle(family=P["export.figma.default_font_family"], size=P["export.figma.synth_label_size"], weight=500,
                   align="center", valign="middle")
    return Node(id=f"{n.id}_label", type="text", name="label", box=n.box, text=label, text_style=ts,
                meta={"synthetic": True})  # not an IR node: excluded from meta.nodeCount


def _instance_node(n: Node, parent: Optional[Node], ctx: _Ctx) -> dict:
    c = n.component
    fb_src = n
    collapsed = n.meta.get("collapsed_children") if isinstance(n.meta, dict) else None
    if not n.children and collapsed:  # `dt map --collapse`: the real subtree beats a synthesized label
        fb_src = Node(**{**n.__dict__, "children": [Node.from_dict(cd) for cd in collapsed]})
    elif not n.children and c is not None and isinstance(c.props.get("label"), str):
        fb_src = Node(**{**n.__dict__, "children": [_synth_label(n, c.props["label"])]})
        ctx.warnings.append(f"instance:{n.id}: synthesized label text for fallback")
    fallback = _shape_node(fb_src, parent, ctx, "FRAME")
    fallback["name"] = node_display_name(n) + " (fallback)"
    if c is None:
        ctx.warnings.append(f"instance:{n.id}: no component ref; exported as frame")
        return fallback
    d = _base(n, parent, "INSTANCE", ctx)
    d.update({
        "componentKey": c.key,
        "componentName": c.name,
        "library": c.library,
        "variantProperties": {str(k): str(v) for k, v in c.variant.items()},
        "props": dict(c.props),
        "confidence": round(float(c.confidence), 4),
        "fallback": fallback,
    })
    return d


def node_to_plan(n: Node, parent: Optional[Node], ctx: _Ctx) -> Optional[dict]:
    """Convert one IR node (and its subtree) to a plan node. Invisible nodes are kept (visible=false)."""
    if not n.meta.get("synthetic"):
        ctx.count += 1
    t = n.type
    if t in ("text", "icon", "line") and n.children:  # leaf types: children have nowhere to go
        ctx.warnings.append(f"{t}:{n.id}: {len(n.children)} children of a {t} node dropped")
    if t == "text":
        return _text_node(n, parent, ctx)
    if t == "icon":
        return _icon_node(n, parent, ctx)
    if t == "line":
        return _line_node(n, parent, ctx)
    if t == "instance":
        return _instance_node(n, parent, ctx)
    if t == "ellipse":
        return _shape_node(n, parent, ctx, "ELLIPSE")
    if t == "frame" or n.children or t == "image":
        # images: RECTANGLE when leaf, FRAME when they have children; either way keep the image paint
        d = _shape_node(n, parent, ctx, "FRAME" if (t == "frame" or n.children) else "RECTANGLE")
        if t == "image" and n.image_ref and not any(p.get("type") == "IMAGE" for p in d.get("fills", [])):
            d["fills"].append({"type": "IMAGE", "scaleMode": "FILL", "opacity": 1,
                               "imageRef": embed_image_ref(n.image_ref, ctx.warnings, f"image:{n.id}")})
        return d
    if t == "vector":
        ctx.warnings.append(f"vector:{n.id}: no path data in IR; exported as rectangle")
    return _shape_node(n, parent, ctx, "RECTANGLE")


def to_build_plan(doc: Document) -> dict:
    """IR Document -> build plan dict (see BUILD_PLAN.md). Deterministic; never raises on odd input."""
    ctx = _Ctx()
    root = node_to_plan(doc.root, None, ctx)
    assert root is not None
    root["x"], root["y"] = 0, 0
    if bool(P["export.figma.verify_layout"]):
        demote_inexact_layouts(root, ctx.warnings)
    fonts = sorted(ctx.fonts)
    return {
        "version": PLAN_VERSION,
        "generator": GENERATOR,
        "width": int(doc.width),
        "height": int(doc.height),
        "fonts": [{"family": f, "style": s} for f, s in fonts],
        "fallbackFont": {"family": P["export.figma.default_font_family"], "style": P["export.figma.fallback_font_style"]},
        "imagePlaceholder": _rgb(Color.from_hex(P["export.figma.image_placeholder"])),
        "root": root,
        "warnings": list(ctx.warnings),
        "meta": {
            "sourceImage": doc.source_image,
            "designSystem": doc.design_system,
            "dpr": doc.dpr,
            "nodeCount": ctx.count,
        },
    }


def save_plan(plan: dict, path: str) -> str:
    with open(path, "w") as f:
        json.dump(plan, f, indent=2)
    return path


# --------------------------------------------------------------------------- validation
def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _check_color(c: Any, where: str, errs: list[str], alpha: bool = False) -> None:
    if not isinstance(c, dict):
        errs.append(f"{where}: color must be an object")
        return
    keys = ("r", "g", "b", "a") if alpha else ("r", "g", "b")
    for k in keys:
        v = c.get(k)
        if not _is_num(v) or not 0.0 <= v <= 1.0:
            errs.append(f"{where}: color.{k} must be a number in [0,1]")


def _check_paint(p: Any, where: str, errs: list[str]) -> None:
    if not isinstance(p, dict) or p.get("type") not in PAINT_TYPES:
        errs.append(f"{where}: paint type must be one of {PAINT_TYPES}")
        return
    if "opacity" in p and not (_is_num(p["opacity"]) and 0 <= p["opacity"] <= 1):
        errs.append(f"{where}: paint opacity must be in [0,1]")
    if p["type"] == "SOLID":
        _check_color(p.get("color"), where, errs)
    elif p["type"].startswith("GRADIENT"):
        stops = p.get("gradientStops")
        if not isinstance(stops, list) or len(stops) < 2:
            errs.append(f"{where}: gradient needs >=2 stops")
        else:
            for i, s in enumerate(stops):
                if not isinstance(s, dict) or not _is_num(s.get("position")) or not 0 <= s["position"] <= 1:
                    errs.append(f"{where}: stop[{i}].position must be in [0,1]")
                else:
                    _check_color(s.get("color"), f"{where}.stop[{i}]", errs, alpha=True)
        hp = p.get("gradientHandlePositions")
        if not isinstance(hp, list) or len(hp) != 3 or not all(isinstance(h, dict) and _is_num(h.get("x")) and _is_num(h.get("y")) for h in hp):
            errs.append(f"{where}: gradientHandlePositions must be 3 {{x,y}} points")
    elif p["type"] == "IMAGE":
        if not isinstance(p.get("imageRef"), str) and not isinstance(p.get("imageHash"), str):
            errs.append(f"{where}: IMAGE paint needs imageRef or imageHash")


def _check_enum(d: dict, key: str, allowed: tuple[str, ...], where: str, errs: list[str]) -> None:
    if key in d and d[key] not in allowed:
        errs.append(f"{where}: {key} must be one of {allowed}, got {d[key]!r}")


def _check_node(d: Any, where: str, errs: list[str], ids: set[str], state: dict, parent: Optional[dict]) -> None:
    state["count"] += 1
    if state["count"] > int(P["export.figma.max_nodes"]):
        if not state.get("overflow"):
            errs.append(f"plan exceeds max_nodes={P['export.figma.max_nodes']}")
            state["overflow"] = True
        return
    if not isinstance(d, dict):
        errs.append(f"{where}: node must be an object")
        return
    t = d.get("type")
    if t not in NODE_TYPES:
        errs.append(f"{where}: type must be one of {NODE_TYPES}, got {t!r}")
        return
    nid = d.get("id")
    if not isinstance(nid, str) or not nid:
        errs.append(f"{where}: id must be a non-empty string")
    elif nid in ids:
        errs.append(f"{where}: duplicate id {nid!r}")
    else:
        ids.add(nid)
    if not isinstance(d.get("name"), str):
        errs.append(f"{where}: name must be a string")
    maxdim = float(P["export.figma.max_dimension"])
    for k in ("x", "y"):
        if not _is_num(d.get(k)) or abs(d[k]) > maxdim:
            errs.append(f"{where}: {k} must be a finite number within +-{maxdim}")
    for k in ("width", "height"):
        v = d.get(k)
        if not _is_num(v) or v < 0 or v > maxdim:
            errs.append(f"{where}: {k} must be in [0,{maxdim}]")
    if "opacity" in d and not (_is_num(d["opacity"]) and 0 <= d["opacity"] <= 1):
        errs.append(f"{where}: opacity must be in [0,1]")
    for key in ("fills", "strokes"):
        if key in d:
            if not isinstance(d[key], list):
                errs.append(f"{where}: {key} must be a list")
            else:
                for i, p in enumerate(d[key]):
                    _check_paint(p, f"{where}.{key}[{i}]", errs)
    if "strokeWeight" in d and not (_is_num(d["strokeWeight"]) and d["strokeWeight"] >= 0):
        errs.append(f"{where}: strokeWeight must be >= 0")
    _check_enum(d, "strokeAlign", STROKE_ALIGNS, where, errs)
    for k in ("cornerRadius", "topLeftRadius", "topRightRadius", "bottomRightRadius", "bottomLeftRadius",
              "itemSpacing", "paddingTop", "paddingRight", "paddingBottom", "paddingLeft"):
        if k in d and not (_is_num(d[k]) and d[k] >= 0):
            errs.append(f"{where}: {k} must be >= 0")
    if "effects" in d:
        for i, e in enumerate(d["effects"]):
            ew = f"{where}.effects[{i}]"
            if not isinstance(e, dict) or e.get("type") not in EFFECT_TYPES:
                errs.append(f"{ew}: effect type must be one of {EFFECT_TYPES}")
                continue
            _check_color(e.get("color"), ew, errs, alpha=True)
            off = e.get("offset")
            if not isinstance(off, dict) or not _is_num(off.get("x")) or not _is_num(off.get("y")):
                errs.append(f"{ew}: offset must be {{x,y}}")
            if not _is_num(e.get("radius")) or e["radius"] < 0:
                errs.append(f"{ew}: radius must be >= 0")
    _check_enum(d, "layoutMode", LAYOUT_MODES, where, errs)
    _check_enum(d, "primaryAxisAlignItems", PRIMARY_ALIGNS, where, errs)
    _check_enum(d, "counterAxisAlignItems", COUNTER_ALIGNS, where, errs)
    _check_enum(d, "primaryAxisSizingMode", SIZING_MODES, where, errs)
    _check_enum(d, "counterAxisSizingMode", SIZING_MODES, where, errs)
    _check_enum(d, "layoutWrap", ("WRAP", "NO_WRAP"), where, errs)
    _check_enum(d, "layoutSizingHorizontal", CHILD_SIZINGS, where, errs)
    _check_enum(d, "layoutSizingVertical", CHILD_SIZINGS, where, errs)
    parent_auto = bool(parent and parent.get("layoutMode") not in (None, "NONE"))
    for k in ("layoutSizingHorizontal", "layoutSizingVertical"):
        if d.get(k) == "FILL" and not parent_auto:
            errs.append(f"{where}: {k}=FILL requires an auto-layout parent")
    if "variableBindings" in d and not (isinstance(d["variableBindings"], dict) and all(isinstance(v, str) for v in d["variableBindings"].values())):
        errs.append(f"{where}: variableBindings must map field -> variable name")
    if t == "TEXT":
        if not isinstance(d.get("characters"), str):
            errs.append(f"{where}: TEXT needs characters (string)")
        fn = d.get("fontName")
        if not isinstance(fn, dict) or not isinstance(fn.get("family"), str) or not isinstance(fn.get("style"), str):
            errs.append(f"{where}: TEXT needs fontName {{family, style}}")
        if not _is_num(d.get("fontSize")) or d["fontSize"] < float(P["export.figma.min_font_size"]):
            errs.append(f"{where}: fontSize must be >= {P['export.figma.min_font_size']}")
        _check_enum(d, "textAlignHorizontal", TEXT_ALIGN_H, where, errs)
        _check_enum(d, "textAlignVertical", TEXT_ALIGN_V, where, errs)
        lh = d.get("lineHeight")
        if lh is not None and not (isinstance(lh, dict) and (lh.get("unit") == "AUTO" or (lh.get("unit") in ("PIXELS", "PERCENT") and _is_num(lh.get("value"))))):
            errs.append(f"{where}: lineHeight must be {{unit:AUTO}} or {{value, unit:PIXELS|PERCENT}}")
        ls = d.get("letterSpacing")
        if ls is not None and not (isinstance(ls, dict) and ls.get("unit") in ("PIXELS", "PERCENT") and _is_num(ls.get("value"))):
            errs.append(f"{where}: letterSpacing must be {{value, unit:PIXELS|PERCENT}}")
    if t == "INSTANCE":
        if not isinstance(d.get("componentKey"), str) or not d["componentKey"]:
            errs.append(f"{where}: INSTANCE needs componentKey")
        if not isinstance(d.get("fallback"), dict):
            errs.append(f"{where}: INSTANCE needs a fallback frame")
        else:
            if d["fallback"].get("type") != "FRAME":
                errs.append(f"{where}: fallback must be a FRAME")
            _check_node(d["fallback"], f"{where}.fallback", errs, set(), state, parent)
        for k in ("variantProperties", "props"):
            if k in d and not isinstance(d[k], dict):
                errs.append(f"{where}: {k} must be an object")
    children = d.get("children")
    if children is not None:
        if t != "FRAME":
            errs.append(f"{where}: only FRAME nodes may have children")
        elif not isinstance(children, list):
            errs.append(f"{where}: children must be a list")
        else:
            for i, c in enumerate(children):
                _check_node(c, f"{where}.children[{i}]", errs, ids, state, d)


def validate_plan(plan: Any) -> list[str]:
    """Schema + numeric sanity check. Returns a list of human-readable errors ([] == valid)."""
    errs: list[str] = []
    if not isinstance(plan, dict):
        return ["plan must be an object"]
    v = str(plan.get("version", ""))
    if v.split(".")[0] != PLAN_VERSION.split(".")[0]:
        errs.append(f"unsupported plan version {v!r} (expected {PLAN_VERSION.split('.')[0]}.x)")
    for k in ("width", "height"):
        if not _is_num(plan.get(k)) or plan[k] <= 0:
            errs.append(f"{k} must be a positive number")
    fonts = plan.get("fonts", [])
    if not isinstance(fonts, list) or not all(isinstance(f, dict) and isinstance(f.get("family"), str) and isinstance(f.get("style"), str) for f in fonts):
        errs.append("fonts must be a list of {family, style}")
    if "root" not in plan:
        errs.append("plan needs a root node")
        return errs
    state = {"count": 0}
    _check_node(plan["root"], "root", errs, set(), state, None)
    if isinstance(plan["root"], dict) and plan["root"].get("type") != "FRAME":
        errs.append("root must be a FRAME")
    return errs
