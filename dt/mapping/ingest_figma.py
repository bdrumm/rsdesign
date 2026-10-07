"""Figma REST API ingestion -> DesignSystem.

    from_figma_json(file_json, components_json, styles_json, variables_json) -> DesignSystem   (pure)
    from_figma_file(file_key, token=os.environ["FIGMA_TOKEN"]) -> DesignSystem                (HTTP)

Endpoints (https://www.figma.com/developers/api):
    GET /v1/files/:key            -> file_json       {document, components, componentSets, styles}
    GET /v1/files/:key/components -> components_json {meta: {components: [{key, name, node_id, ...}]}}
    GET /v1/files/:key/styles     -> styles_json     {meta: {styles: [{key, name, style_type, node_id}]}}
    GET /v1/files/:key/variables/local -> variables_json {meta: {variables: {...}, variableCollections: {...}}}

What we extract:
  * COMPONENT / COMPONENT_SET nodes -> `ComponentSpec` (key = Figma component key so the exporter
    can `importComponentByKeyAsync`).  A set's children "Prop=Value, ..." names become `Variant`s.
    Each component is converted to an IR exemplar (`figma_node_to_ir`) and its signature is
    *derived* from that exemplar (`matcher.derive_signature`), so matching uses the same
    constraint machinery as the hand-written catalogs.
  * Styles (FILL/TEXT/EFFECT) -> tokens; values are read from the first node that uses the style.
  * Local variables (COLOR/FLOAT/STRING/BOOLEAN) -> tokens (default mode; aliases resolved).

Figma colors are 0..1 floats; geometry is absolute (`absoluteBoundingBox`) and converted to the
component's own origin for the exemplar.  No network access in `from_figma_json`.
"""
from __future__ import annotations

import os
import re
from typing import Any, Iterator, Optional

from dt.ir import Box, Color, Fill, Layout, Node, Shadow, Stroke, TextStyle
from dt.mapping.design_system import ComponentSpec, DesignSystem, Signature, Slot, Token, Variant
from dt.mapping.ingest_library import kebab
from dt.mapping.matcher import derive_signature

FIGMA_API = "https://api.figma.com"
_CONTAINER_TYPES = {"FRAME", "GROUP", "COMPONENT", "COMPONENT_SET", "INSTANCE", "SECTION"}


# --------------------------------------------------------------------------- figma -> IR
def figma_color(c: dict, opacity: float = 1.0) -> Color:
    return Color.from_rgb(float(c.get("r", 0)) * 255, float(c.get("g", 0)) * 255, float(c.get("b", 0)) * 255, float(c.get("a", 1.0)) * opacity)


def _paints(paints: list[dict] | None) -> list[Fill]:
    out: list[Fill] = []
    for p in paints or []:
        if p.get("visible", True) is False:
            continue
        if p.get("type") == "SOLID" and p.get("color"):
            out.append(Fill(kind="solid", color=figma_color(p["color"]), opacity=float(p.get("opacity", 1.0))))
        elif p.get("type", "").startswith("GRADIENT") and p.get("gradientStops"):
            from dt.ir import GradientStop

            stops = [GradientStop(float(s["position"]), figma_color(s["color"])) for s in p["gradientStops"]]
            out.append(Fill(kind="linear" if p["type"] == "GRADIENT_LINEAR" else "radial", stops=stops))
        elif p.get("type") == "IMAGE":
            out.append(Fill(kind="image", image_ref=p.get("imageRef")))
    return out


def _radius(n: dict) -> tuple[float, float, float, float]:
    rr = n.get("rectangleCornerRadii")
    if rr and len(rr) == 4:
        return tuple(float(v) for v in rr)  # type: ignore[return-value]
    r = float(n.get("cornerRadius", 0) or 0)
    return (r, r, r, r)


def _text_style(n: dict) -> TextStyle:
    s = n.get("style", {})
    fills = _paints(n.get("fills"))
    color = fills[0].color if fills and fills[0].color is not None else Color(0, 0, 0)
    align = {"LEFT": "left", "CENTER": "center", "RIGHT": "right"}.get(s.get("textAlignHorizontal", "LEFT"), "left")
    valign = {"TOP": "top", "CENTER": "middle", "BOTTOM": "bottom"}.get(s.get("textAlignVertical", "TOP"), "top")
    return TextStyle(
        family=s.get("fontFamily", "Roboto"), size=float(s.get("fontSize", 14)), weight=int(s.get("fontWeight", 400)),
        line_height=float(s["lineHeightPx"]) if s.get("lineHeightPx") else None, letter_spacing=float(s.get("letterSpacing", 0) or 0),
        color=color, align=align, valign=valign, italic=bool(s.get("italic", False)),
        transform={"UPPER": "uppercase", "LOWER": "lowercase", "TITLE": "capitalize"}.get(s.get("textCase", ""), "none"),
    )


def _layout(n: dict) -> Optional[Layout]:
    mode = n.get("layoutMode")
    if mode not in ("HORIZONTAL", "VERTICAL"):
        return None
    align = {"MIN": "start", "CENTER": "center", "MAX": "end", "STRETCH": "stretch"}
    just = {"MIN": "start", "CENTER": "center", "MAX": "end", "SPACE_BETWEEN": "space-between"}
    return Layout(
        mode="row" if mode == "HORIZONTAL" else "column", gap=float(n.get("itemSpacing", 0) or 0),
        padding=(float(n.get("paddingTop", 0) or 0), float(n.get("paddingRight", 0) or 0), float(n.get("paddingBottom", 0) or 0), float(n.get("paddingLeft", 0) or 0)),
        align_items=align.get(n.get("counterAxisAlignItems", "MIN"), "start"), justify=just.get(n.get("primaryAxisAlignItems", "MIN"), "start"),
        wrap=n.get("layoutWrap") == "WRAP",
    )


def _is_icon(n: dict) -> bool:
    name = (n.get("name") or "").lower()
    t = n.get("type")
    if t in ("VECTOR", "BOOLEAN_OPERATION", "STAR", "REGULAR_POLYGON"):
        return True
    return t in ("INSTANCE", "FRAME", "GROUP") and ("icon" in name or name.startswith("ic_") or name.startswith("ic/"))


def figma_node_to_ir(n: dict, origin: tuple[float, float] = (0.0, 0.0)) -> Node:
    """Convert a Figma REST node (and its subtree) to an IR Node with coordinates relative to `origin`."""
    bb = n.get("absoluteBoundingBox") or n.get("absoluteRenderBounds") or {}
    box = Box(float(bb.get("x", 0)) - origin[0], float(bb.get("y", 0)) - origin[1], float(bb.get("width", 0)), float(bb.get("height", 0)))
    t = n.get("type", "FRAME")
    node = Node(id=n.get("id", ""), name=n.get("name", ""), box=box, visible=n.get("visible", True) is not False,
                opacity=float(n.get("opacity", 1.0)), radius=_radius(n), clip=bool(n.get("clipsContent", False)))
    node.fills = _paints(n.get("fills")) if t != "TEXT" else []
    sw = float(n.get("strokeWeight", 0) or 0)
    strokes = _paints(n.get("strokes"))
    if strokes and sw > 0:
        c = strokes[0].color or Color(0, 0, 0)
        node.strokes = [Stroke(color=c, width=sw, align={"INSIDE": "inside", "OUTSIDE": "outside", "CENTER": "center"}.get(n.get("strokeAlign", "INSIDE"), "inside"))]
    for e in n.get("effects", []) or []:
        if e.get("type") in ("DROP_SHADOW", "INNER_SHADOW") and e.get("visible", True) is not False:
            off = e.get("offset", {})
            node.effects.append(Shadow(color=figma_color(e.get("color", {"r": 0, "g": 0, "b": 0, "a": 0.3})), dx=float(off.get("x", 0)), dy=float(off.get("y", 0)),
                                       blur=float(e.get("radius", 0)), spread=float(e.get("spread", 0)), inner=e["type"] == "INNER_SHADOW"))
    if t == "TEXT":
        node.type = "text"
        node.text = n.get("characters", "")
        node.text_style = _text_style(n)
    elif t == "ELLIPSE":
        node.type = "ellipse"
    elif t == "LINE":
        node.type = "line"
    elif t == "RECTANGLE":
        node.type = "image" if any(f.kind == "image" for f in node.fills) else "rect"
    elif _is_icon(n):
        node.type = "icon"
        node.icon_name = kebab(re.sub(r"^(icon[s]?[/_-]?)", "", n.get("name", ""), flags=re.I)) or "icon"
        if not node.fills:
            inner = [c for c in n.get("children", []) if c.get("fills")]
            if inner:
                node.fills = _paints(inner[0]["fills"])
    else:
        node.type = "frame"
        node.layout = _layout(n)
        node.children = [figma_node_to_ir(c, origin) for c in n.get("children", []) if c.get("visible", True) is not False]
    if n.get("styles"):
        node.meta["figma_styles"] = dict(n["styles"])
    if n.get("boundVariables"):
        node.meta["figma_variables"] = n["boundVariables"]
    return node


# --------------------------------------------------------------------------- tree helpers
def walk_figma(n: dict, depth: int = 0) -> Iterator[tuple[dict, int]]:
    yield n, depth
    for c in n.get("children", []) or []:
        yield from walk_figma(c, depth + 1)


def parse_variant_name(name: str) -> dict[str, str]:
    """'Style=Filled, State=Enabled' -> {'style': 'filled', 'state': 'enabled'}."""
    props: dict[str, str] = {}
    for part in name.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            props[kebab(k)] = kebab(v)
    return props


def _default_variant(children: list[dict]) -> Optional[dict]:
    """Prefer the 'enabled/default' state variant as the exemplar of a component set."""
    for c in children:
        props = parse_variant_name(c.get("name", ""))
        if props.get("state") in (None, "enabled", "default", "rest"):
            return c
    return children[0] if children else None


# --------------------------------------------------------------------------- styles / variables
def _style_value(node: dict, style_type: str) -> Any:
    if style_type == "FILL":
        fills = _paints(node.get("fills"))
        return fills[0].color.hex() if fills and fills[0].color is not None else None
    if style_type == "TEXT":
        ts = _text_style(node)
        return {"size": ts.size, "line_height": ts.line_height, "weight": ts.weight, "tracking": ts.letter_spacing, "family": ts.family}
    if style_type == "EFFECT":
        ir = figma_node_to_ir(node)
        if ir.effects:
            e = ir.effects[0]
            return {"dy": e.dy, "dx": e.dx, "blur": e.blur, "spread": e.spread, "color": e.color.hex()}
        return None
    if style_type == "STROKE":
        strokes = _paints(node.get("strokes"))
        return strokes[0].color.hex() if strokes and strokes[0].color is not None else None
    return None


_STYLE_CATEGORY = {"FILL": "color", "TEXT": "typescale", "EFFECT": "elevation", "STROKE": "color", "GRID": "other"}
_STYLE_PROP = {"FILL": "fill", "TEXT": "text", "EFFECT": "effect", "STROKE": "stroke", "GRID": "grid"}


def tokens_from_styles(file_json: dict, styles_json: Optional[dict]) -> list[Token]:
    """FILL/TEXT/EFFECT styles -> tokens. The value comes from the first node referencing the style."""
    styles: dict[str, dict] = dict(file_json.get("styles", {}))
    for s in ((styles_json or {}).get("meta", {}).get("styles", []) or []):
        sid = s.get("node_id") or s.get("key")
        styles.setdefault(sid, {"key": s.get("key"), "name": s.get("name"), "styleType": s.get("style_type")})
    usage: dict[str, dict] = {}
    for n, _ in walk_figma(file_json.get("document", {})):
        for prop, sid in (n.get("styles") or {}).items():
            usage.setdefault(sid, n)
    toks: list[Token] = []
    for sid, s in styles.items():
        st = (s.get("styleType") or s.get("style_type") or "").upper()
        node = usage.get(sid)
        if node is None or st not in _STYLE_CATEGORY:
            continue
        val = _style_value(node, st)
        if val is None:
            continue
        name = kebab(s.get("name", sid)).replace("-/-", ".").replace("/", ".")
        toks.append(Token(name=f"figma.style.{name}", category=_STYLE_CATEGORY[st], value=val,
                          meta={"figma_key": s.get("key"), "figma_id": sid, "style_type": st, "figma_name": s.get("name")}))
    return toks


def tokens_from_variables(variables_json: Optional[dict]) -> list[Token]:
    """Local variables (default mode) -> tokens. COLOR -> hex, FLOAT -> dimension, else 'other'."""
    if not variables_json:
        return []
    meta = variables_json.get("meta", variables_json)
    variables: dict[str, dict] = meta.get("variables", {}) or {}
    collections: dict[str, dict] = meta.get("variableCollections", {}) or {}

    def resolve(v: Any, depth: int = 0) -> Any:
        if isinstance(v, dict) and v.get("type") == "VARIABLE_ALIAS" and depth < 16:
            target = variables.get(v.get("id"))
            return resolve(_default_value(target), depth + 1) if target else None
        return v

    def _default_value(var: dict) -> Any:
        coll = collections.get(var.get("variableCollectionId"), {})
        mode = coll.get("defaultModeId") or next(iter(var.get("valuesByMode", {})), None)
        return var.get("valuesByMode", {}).get(mode)

    toks: list[Token] = []
    for vid, var in variables.items():
        rt = var.get("resolvedType", "STRING")
        val = resolve(_default_value(var))
        name = "figma.var." + kebab(var.get("name", vid)).replace("/", ".")
        coll = collections.get(var.get("variableCollectionId"), {})
        m = {"figma_id": vid, "figma_key": var.get("key"), "collection": coll.get("name"), "resolved_type": rt}
        if rt == "COLOR" and isinstance(val, dict):
            toks.append(Token(name, "color", figma_color(val).hex(), m))
        elif rt == "FLOAT" and isinstance(val, (int, float)):
            cat = "spacing" if "spac" in name or "gap" in name or "pad" in name else ("shape" if "radius" in name or "corner" in name else "dimension")
            toks.append(Token(name, cat, float(val), m))
        elif val is not None:
            toks.append(Token(name, "other", val, m))
    return toks


# --------------------------------------------------------------------------- components
def _component_meta(components_json: Optional[dict], file_json: dict) -> dict[str, dict]:
    """node_id -> {key, name, description, containing_frame}."""
    out: dict[str, dict] = {}
    for nid, c in (file_json.get("components") or {}).items():
        out[nid] = dict(c)
    for nid, c in (file_json.get("componentSets") or {}).items():
        out.setdefault(nid, dict(c))
    for c in ((components_json or {}).get("meta", {}).get("components", []) or []):
        out.setdefault(c.get("node_id"), {}).update({k: v for k, v in c.items() if k != "node_id"})
    return out


def _slots_from_exemplar(node: Node) -> list[Slot]:
    slots: list[Slot] = []
    texts = [n for n in node.walk() if n.type == "text"]
    icons = [n for n in node.walk() if n.type == "icon"]
    for i, t in enumerate(texts):
        slots.append(Slot(name=kebab(t.name) or ("label" if i == 0 else f"text{i}"), type="text"))
    for i, ic in enumerate(icons):
        slots.append(Slot(name=kebab(ic.name) or ("icon" if i == 0 else f"icon{i}"), type="icon"))
    return slots


def _style_token_map(ds: DesignSystem) -> dict[str, str]:
    """figma style node id -> token name (built from tokens_from_styles meta)."""
    return {t.meta["figma_id"]: t.name for t in ds.tokens if t.meta.get("figma_id") and t.meta.get("style_type")}


def _apply_style_hints(sig: Signature, ex: Node, style_tokens: dict[str, str]) -> None:
    """A node bound to a Figma style should match on that style's token, not on nearest ΔE."""
    own = ex.meta.get("figma_styles", {})
    if own.get("fill") in style_tokens:
        sig.fills = [style_tokens[own["fill"]]]
    if own.get("stroke") in style_tokens and isinstance(sig.stroke, dict):
        sig.stroke = {**sig.stroke, "tokens": [style_tokens[own["stroke"]]]}
    texts = [t for t in ex.walk() if t.type == "text"]
    roles = [style_tokens[t.meta["figma_styles"]["text"]] for t in texts if t.meta.get("figma_styles", {}).get("text") in style_tokens]
    if roles:
        sig.text_roles = list(dict.fromkeys(roles))
    if texts:
        primary = max(texts, key=lambda t: t.text_style.size if t.text_style else 0)
        fill_style = primary.meta.get("figma_styles", {}).get("fill")
        if fill_style in style_tokens:
            sig.text_colors = [style_tokens[fill_style]]


def _spec_from_component(n: dict, meta: dict, ds_tokens: DesignSystem, library: str) -> ComponentSpec:
    bb = n.get("absoluteBoundingBox", {})
    ex = figma_node_to_ir(n, (float(bb.get("x", 0)), float(bb.get("y", 0))))
    ex.component = None
    sig = derive_signature(ex, ds_tokens, order="row" if (ex.layout and ex.layout.mode == "row") or not ex.layout else "column")
    _apply_style_hints(sig, ex, _style_token_map(ds_tokens))
    return ComponentSpec(key=meta.get("key") or n["id"], name=n.get("name", n["id"]), library=library, signature=sig,
                         exemplar=ex.to_dict(), slots=_slots_from_exemplar(ex),
                         meta={"figma_id": n["id"], "description": meta.get("description", ""), "containing_frame": meta.get("containing_frame", {})})


def _spec_from_component_set(n: dict, meta: dict, all_meta: dict[str, dict], ds_tokens: DesignSystem, library: str) -> ComponentSpec:
    children = [c for c in n.get("children", []) if c.get("type") == "COMPONENT"]
    default = _default_variant(children) or n
    base_spec = _spec_from_component(default, all_meta.get(default.get("id"), {}), ds_tokens, library)
    variants: list[Variant] = []
    heights: list[float] = []
    for c in children:
        cm = all_meta.get(c["id"], {})
        vs = _spec_from_component(c, cm, ds_tokens, library)
        heights.append(float(c.get("absoluteBoundingBox", {}).get("height", 0)))
        vsig = vs.signature
        vsig.types = None  # inherit base
        variants.append(Variant(name=c.get("name", c["id"]), props={**parse_variant_name(c.get("name", "")), "figma_key": cm.get("key", "")}, signature=vsig))
    base = base_spec.signature
    if heights:
        base.height = (min(heights) - 2, max(heights) + 2)
    props = {k: v.get("variantOptions") for k, v in (n.get("componentPropertyDefinitions") or {}).items() if v.get("type") == "VARIANT"}
    return ComponentSpec(key=meta.get("key") or n["id"], name=n.get("name", n["id"]), library=library, signature=base, variants=variants,
                         exemplar=base_spec.exemplar, slots=base_spec.slots,
                         meta={"figma_id": n["id"], "component_set": True, "variant_properties": props, "description": meta.get("description", "")})


def from_figma_json(file_json: dict, components_json: Optional[dict] = None, styles_json: Optional[dict] = None,
                    variables_json: Optional[dict] = None, name: Optional[str] = None) -> DesignSystem:
    """Pure conversion of Figma REST payloads into a DesignSystem (no network)."""
    library = name or kebab(file_json.get("name", "figma")) or "figma"
    tokens = tokens_from_styles(file_json, styles_json) + tokens_from_variables(variables_json)
    ds = DesignSystem(name=library, tokens=tokens, meta={"source": "figma", "figma_name": file_json.get("name"), "version": file_json.get("version"),
                                                        "last_modified": file_json.get("lastModified")})
    all_meta = _component_meta(components_json, file_json)
    doc = file_json.get("document", {})
    in_set: set[str] = set()
    for n, _ in walk_figma(doc):
        if n.get("type") == "COMPONENT_SET":
            ds.add_component(_spec_from_component_set(n, all_meta.get(n["id"], {}), all_meta, ds, library))
            in_set.update(c["id"] for c in n.get("children", []))
    for n, _ in walk_figma(doc):
        if n.get("type") == "COMPONENT" and n["id"] not in in_set:
            ds.add_component(_spec_from_component(n, all_meta.get(n["id"], {}), ds, library))
    fonts: set[str] = set()
    for spec in ds.components:
        ex = spec.exemplar_node()
        if ex:
            fonts.update(t.text_style.family for t in ex.walk() if t.type == "text" and t.text_style)
    ds.fonts = sorted(fonts)
    return ds


# --------------------------------------------------------------------------- HTTP
def _get(path: str, token: str, params: Optional[dict] = None) -> dict:
    import requests

    r = requests.get(FIGMA_API + path, headers={"X-Figma-Token": token}, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def from_figma_file(file_key: str, token: Optional[str] = None, with_variables: bool = True) -> DesignSystem:
    """Fetch a file (+ components, styles, local variables) from the Figma REST API and ingest it.
    Variables need an Enterprise plan; a 4xx there is ignored."""
    token = token or os.environ.get("FIGMA_TOKEN")
    if not token:
        raise RuntimeError("FIGMA_TOKEN not set")
    file_json = _get(f"/v1/files/{file_key}", token, {"geometry": "paths"})
    components_json = _get(f"/v1/files/{file_key}/components", token)
    styles_json = _get(f"/v1/files/{file_key}/styles", token)
    variables_json = None
    if with_variables:
        try:
            variables_json = _get(f"/v1/files/{file_key}/variables/local", token)
        except Exception:
            variables_json = None
    return from_figma_json(file_json, components_json, styles_json, variables_json)
