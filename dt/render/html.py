"""IR -> HTML/CSS. The "code" layer.

Two modes:
  * absolute (default): every node absolutely positioned at its IR box. Deterministic and
    pixel-faithful; used for the adversarial compare/refine loop.
  * flex: nodes with `layout.mode != 'none'` render as CSS flex containers using the inferred
    gap/padding/alignment. Used to VERIFY that inferred auto-layout reproduces the absolute
    geometry (if flex render != absolute render, the layout inference is wrong).

Fonts: Roboto + Material Symbols are self-hosted in fixtures/fonts so renders are deterministic
and offline. Icons render as Material Symbols ligatures (`icon_name`).
"""
from __future__ import annotations

import html as _html
import os
from typing import Iterable

from dt.ir import Document, Fill, Node, Stroke, Shadow, TextStyle
from dt.params import P, register

register("render.text.wrap", False, "let CSS wrap text inside its box (False: line breaks are explicit, nothing ever wraps)")

FONTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "fonts"))

FONT_STACK = {
    "Roboto": "'Roboto', 'Helvetica Neue', Arial, sans-serif",
    "Google Sans": "'Google Sans', 'Roboto', sans-serif",
    "Google Sans Text": "'Google Sans Text', 'Google Sans', 'Roboto', sans-serif",
    "Google Sans Flex": "'Google Sans Flex', 'Google Sans', 'Roboto', sans-serif",
    "Inter": "'Inter', 'Roboto', sans-serif",
    "system": "-apple-system, BlinkMacSystemFont, 'Roboto', sans-serif",
}


ICON_VARIABLE_FONT = os.path.abspath(os.path.join(FONTS_DIR, "..", "icons", "MaterialSymbolsOutlined-variable.woff2"))


def _font_face_css() -> str:
    parts = []
    if os.path.exists(ICON_VARIABLE_FONT):
        # full-axis Material Symbols (FILL/wght/GRAD/opsz); at FILL 0 / wght 400 / opsz 24 it is pixel-identical
        # to Google Fonts' static 'Material Symbols Outlined' (verified), and it can also draw filled icons
        parts.append("@font-face { font-family: 'Material Symbols Outlined Variable'; font-style: normal; "
                     f"font-weight: 100 700; src: url(file://{ICON_VARIABLE_FONT}) format('woff2'); }}")
    for name in ("roboto.local.css", "material-symbols.local.css", "gsans.local.css", "gsans_text.local.css", "gsans_flex.local.css"):
        p = os.path.join(FONTS_DIR, name)
        if os.path.exists(p):
            css = open(p).read()
            # make url() absolute file paths so set_content() works without a base URL
            css = css.replace("url(", f"url(file://{FONTS_DIR}/")
            parts.append(css)
    return "\n".join(parts)


def _fmt(v: float) -> str:
    """Compact px formatting: 12 -> '12px', 12.5 -> '12.5px'."""
    if abs(v - round(v)) < 1e-6:
        return f"{int(round(v))}px"
    return f"{round(v, 3)}px"


def _fill_css(fills: list[Fill]) -> list[str]:
    layers = []
    for f in reversed(fills):  # IR: first fill is bottom; CSS: first layer is top
        if f.kind == "solid" and f.color is not None:
            c = Fill(color=f.color).color
            a = c.a * f.opacity
            layers.append(f"linear-gradient(rgba({c.r},{c.g},{c.b},{round(a,3)}), rgba({c.r},{c.g},{c.b},{round(a,3)}))")
        elif f.kind == "linear" and f.stops:
            stops = ", ".join(f"{s.color.css()} {round(s.pos*100,2)}%" for s in f.stops)
            layers.append(f"linear-gradient({f.angle}deg, {stops})")
        elif f.kind == "radial" and f.stops:
            stops = ", ".join(f"{s.color.css()} {round(s.pos*100,2)}%" for s in f.stops)
            layers.append(f"radial-gradient(circle, {stops})")
        elif f.kind == "image" and f.image_ref:
            layers.append(f"url('{f.image_ref}') center/cover no-repeat")
    return layers


def _stroke_css(strokes: list[Stroke], radius: tuple[float, float, float, float]) -> list[str]:
    """Inside-aligned strokes via inset box-shadow (doesn't affect layout box). Center/outside via outline."""
    out = []
    for s in strokes:
        if s.sides is not None:
            t, r, b, l = s.sides
            out.append(f"box-shadow: inset 0 {_fmt(t)} 0 0 {s.color.css()}, inset -{_fmt(r)} 0 0 0 {s.color.css()}, inset 0 -{_fmt(b)} 0 0 {s.color.css()}, inset {_fmt(l)} 0 0 0 {s.color.css()}")
        elif s.align == "inside":
            out.append(f"box-shadow: inset 0 0 0 {_fmt(s.width)} {s.color.css()}")
        elif s.align == "outside":
            out.append(f"outline: {_fmt(s.width)} solid {s.color.css()}; outline-offset: 0px")
        else:  # center
            out.append(f"outline: {_fmt(s.width)} solid {s.color.css()}; outline-offset: -{_fmt(s.width/2)}")
    return out


def _shadow_css(effects: list[Shadow]) -> str:
    parts = []
    for e in effects:
        inner = "inset " if e.inner else ""
        parts.append(f"{inner}{_fmt(e.dx)} {_fmt(e.dy)} {_fmt(e.blur)} {_fmt(e.spread)} {e.color.css()}")
    return ", ".join(parts)


def _text_css(ts: TextStyle) -> str:
    fam = FONT_STACK.get(ts.family, f"'{ts.family}', 'Roboto', sans-serif")
    css = [
        f"font-family: {fam}",
        f"font-size: {_fmt(ts.size)}",
        f"font-weight: {ts.weight}",
        f"color: {ts.color.css()}",
        f"text-align: {ts.align}",
        f"letter-spacing: {_fmt(ts.letter_spacing)}",
        f"font-style: {'italic' if ts.italic else 'normal'}",
        f"text-decoration: {ts.decoration}",
        f"text-transform: {ts.transform}",
        f"line-height: {_fmt(ts.line_height) if ts.line_height else 'normal'}",
    ]
    return "; ".join(css)


def node_style(n: Node, parent: Node | None, mode: str) -> str:
    b = n.box
    css: list[str] = []
    if mode == "absolute" or parent is None or not (parent.layout and parent.layout.mode != "none"):
        px, py = (parent.box.x, parent.box.y) if parent else (0.0, 0.0)
        css += ["position: absolute", f"left: {_fmt(b.x - px)}", f"top: {_fmt(b.y - py)}"]
    else:
        css += ["position: relative", "flex: none"]
    css += [f"width: {_fmt(b.w)}", f"height: {_fmt(b.h)}", "box-sizing: border-box"]
    if n.type == "ellipse":
        css.append("border-radius: 50%")
    elif any(r > 0 for r in n.radius):
        tl, tr, br, bl = n.radius
        css.append(f"border-radius: {_fmt(tl)} {_fmt(tr)} {_fmt(br)} {_fmt(bl)}")
    layers = _fill_css(n.fills)
    if layers:
        css.append("background: " + ", ".join(layers))
    css += _stroke_css(n.strokes, n.radius)
    sh = _shadow_css(n.effects)
    if sh:
        # merge with inset stroke shadows if both present
        existing = [i for i, c in enumerate(css) if c.startswith("box-shadow:")]
        if existing:
            css[existing[0]] = css[existing[0]] + ", " + sh
        else:
            css.append(f"box-shadow: {sh}")
    if n.opacity < 1:
        css.append(f"opacity: {n.opacity}")
    if n.clip:
        css.append("overflow: hidden")
    if not n.visible:
        css.append("visibility: hidden")
    if mode == "flex" and n.layout and n.layout.mode != "none":
        L = n.layout
        css += [
            "display: flex",
            f"flex-direction: {'row' if L.mode == 'row' else 'column'}",
            f"gap: {_fmt(L.gap)}",
            f"padding: {_fmt(L.padding[0])} {_fmt(L.padding[1])} {_fmt(L.padding[2])} {_fmt(L.padding[3])}",
            "align-items: " + {"start": "flex-start", "center": "center", "end": "flex-end", "stretch": "stretch"}[L.align_items],
            "justify-content: " + {"start": "flex-start", "center": "center", "end": "flex-end", "space-between": "space-between"}[L.justify],
            f"flex-wrap: {'wrap' if L.wrap else 'nowrap'}",
        ]
    if n.type == "text" and n.text_style:
        css.append(_text_css(n.text_style))
        # Text never wraps in the renderer: line breaks are explicit ("\n" per perceived line). Letting the
        # renderer re-wrap turns a 1px width error (or a font-metric difference) into a whole-line jump.
        # P["render.text.wrap"] = True restores CSS wrapping for hand-authored IR that relies on it.
        if bool(P["render.text.wrap"]):
            css.append("white-space: pre-wrap")
            css.append("overflow-wrap: anywhere")
        else:
            css.append("white-space: pre")
            css.append("overflow: visible")
        css.append("display: flex")
        css.append("flex-direction: column")
        css.append("justify-content: " + {"top": "flex-start", "middle": "center", "bottom": "flex-end"}[n.text_style.valign])
    if n.type == "icon":
        css.append("font-family: 'Material Symbols Outlined Variable', 'Material Symbols Outlined'")
        css.append(f"font-size: {_fmt(min(b.w, b.h))}")
        css.append("line-height: 1")
        css.append("display: flex; align-items: center; justify-content: center")
        fill_axis = int(n.meta.get("icon_fill", 0)) if isinstance(n.meta, dict) else 0
        css.append(f"font-variation-settings: 'FILL' {fill_axis}, 'wght' 400, 'GRAD' 0, 'opsz' 24")
        c = n.fill_color
        if c is not None:
            css.append(f"color: {c.css()}")
            css = [x for x in css if not x.startswith("background:")]
    if n.type == "line":
        c = n.fill_color or (n.strokes[0].color if n.strokes else None)
        if c is not None:
            css.append(f"background: {c.css()}")
    return "; ".join(css)


def _node_html(n: Node, parent: Node | None, mode: str, out: list[str]) -> None:
    if not n.visible:
        return
    attrs = f'id="{_html.escape(n.id)}" data-type="{n.type}" data-name="{_html.escape(n.name)}"'
    style = node_style(n, parent, mode)
    if n.type == "text":
        txt = _html.escape(n.text or "")
        out.append(f'<div {attrs} style="{style}"><span>{txt}</span></div>')
        return
    if n.type == "icon":
        out.append(f'<div {attrs} style="{style}">{_html.escape(n.icon_name or "")}</div>')
        return
    if n.type == "image" and n.image_ref and not n.fills:
        out.append(f'<div {attrs} style="{style}; background: url(\'{n.image_ref}\') center/cover no-repeat"></div>')
        return
    out.append(f'<div {attrs} style="{style}">')
    for c in n.children:
        _node_html(c, n, mode, out)
    out.append("</div>")


def render_html(doc: Document, mode: str = "absolute", extra_css: str = "", fonts: bool = True) -> str:
    """Return a complete HTML document reproducing `doc`."""
    assert mode in ("absolute", "flex")
    body: list[str] = []
    _node_html(doc.root, None, mode, body)
    font_css = _font_face_css() if fonts else ""
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="dt-render" content="1">
<style>
{font_css}
html, body {{ margin: 0; padding: 0; width: {doc.width}px; height: {doc.height}px; overflow: hidden; background: #fff; }}
* {{ -webkit-font-smoothing: antialiased; text-rendering: geometricPrecision; }}
div {{ margin: 0; padding: 0; }}
{extra_css}
</style></head>
<body>
{chr(10).join(body)}
</body></html>
"""


def render_fragment(nodes: Iterable[Node], mode: str = "absolute") -> str:
    out: list[str] = []
    for n in nodes:
        _node_html(n, None, mode, out)
    return "\n".join(out)
