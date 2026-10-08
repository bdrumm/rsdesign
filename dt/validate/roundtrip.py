"""Export round-trip check (docs/VALIDATION.md, "Export round-trip"): export must be lossless.

    IR --to_build_plan--> plan --code.js on fake_figma (node run_fake.js)--> Figma node tree
       --figma semantics (auto-layout, text auto-resize)--> plan_tree_to_ir --> IR' --render--> PNG'
    mean ΔE2000(render(IR), render(IR'))  ->  > P["validate.roundtrip.max_mean_de"] means a lossy export

fake_figma.js stores whatever the plugin writes; real Figma then re-lays-out auto-layout frames and sizes
auto-width text from its glyphs. ``dt.export.figma_layout`` models that (text measured with the same
browser + fonts that render the IR), so a plan that only *looks* right in the fake is still caught.

Public API:
    run_plugin(plan, fonts=None, components=None) -> dict          run_fake.js output (tree, images, summary)
    plan_tree_to_ir(tree_root, width, height, images=None, measure=True) -> Document
    roundtrip(doc, out_dir=None) -> RoundTripReport
"""
from __future__ import annotations

import copy
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from dt.export import RUN_FAKE_JS, to_build_plan
from dt.export.figma_layout import apply_auto_layout
from dt.ir import Box, Color, Document, Fill, GradientStop, Node, Shadow, Stroke, TextStyle
from dt.params import P, register

register("validate.roundtrip.max_mean_de", 0.5, "mean ΔE2000 between render(IR) and render(export round-trip IR) above which the export is lossy", (0.05, 5.0))
register("validate.roundtrip.bad_de", 2.3, "ΔE2000 per pixel counted as a round-trip difference (JND)", (0.5, 10.0))
register("validate.roundtrip.timeout_s", 120.0, "seconds allowed for the headless plugin run", (10.0, 600.0))

_DEFAULT_FONTS = [("Roboto", "Regular"), ("Inter", "Regular")]
_STYLE_WEIGHT = {"thin": 100, "extralight": 200, "light": 300, "regular": 400, "medium": 500,
                 "semibold": 600, "bold": 700, "extrabold": 800, "black": 900}
_ALIGN_H = {"LEFT": "left", "CENTER": "center", "RIGHT": "right", "JUSTIFIED": "left"}
_ALIGN_V = {"TOP": "top", "CENTER": "middle", "BOTTOM": "bottom"}
_DECO = {"NONE": "none", "UNDERLINE": "underline", "STRIKETHROUGH": "line-through"}
_CASE = {"ORIGINAL": "none", "UPPER": "uppercase", "LOWER": "lowercase", "TITLE": "capitalize"}


# --------------------------------------------------------------------------- plugin
def run_plugin(plan: dict, fonts: Optional[list[tuple[str, str]]] = None, components: Optional[list[str]] = None,
               code: Optional[str] = None) -> dict:
    """Run code.js headlessly on fake_figma.js; returns the parsed run_fake.js JSON (raises on failure)."""
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("node not found on PATH (needed to run the Figma plugin headlessly)")
    with tempfile.TemporaryDirectory() as td:
        pp = os.path.join(td, "plan.json")
        with open(pp, "w") as f:
            json.dump(plan, f)
        cmd = [node, RUN_FAKE_JS, "--plan", pp]
        if code:  # another build of the plugin (e.g. an older code.js) against the same fake
            cmd += ["--code", code]
        if components:
            cmd += ["--components", ",".join(components)]
        if fonts:
            cmd += ["--fonts", ";".join(f"{a}:{b}" for a, b in fonts)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=float(P["validate.roundtrip.timeout_s"]))
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:  # pragma: no cover - surfaced to the caller
        raise RuntimeError(f"run_fake.js produced no JSON (rc={r.returncode}): {r.stdout[-2000:]} {r.stderr[-2000:]}") from e
    if not out.get("ok"):
        raise RuntimeError(f"plugin failed: {out.get('error')}")
    return out


# --------------------------------------------------------------------------- tree -> IR
def _color(c: dict, opacity: float = 1.0) -> Color:
    return Color(int(round(float(c.get("r", 0)) * 255)), int(round(float(c.get("g", 0)) * 255)),
                 int(round(float(c.get("b", 0)) * 255)), float(c.get("a", 1.0)) * float(opacity))


def _paint_to_fill(p: dict, images: dict[str, str]) -> Optional[Fill]:
    t = p.get("type")
    op = float(p.get("opacity", 1.0))
    if p.get("visible") is False:
        return None
    if t == "SOLID":
        return Fill(kind="solid", color=_color(p.get("color", {}), op))
    if t in ("GRADIENT_LINEAR", "GRADIENT_RADIAL"):
        stops = [GradientStop(float(s["position"]), _color(s["color"])) for s in p.get("gradientStops", [])]
        if t == "GRADIENT_RADIAL":
            return Fill(kind="radial", stops=stops, opacity=op)
        return Fill(kind="linear", stops=stops, angle=_handles_to_angle(p), opacity=op)
    if t == "IMAGE":
        h = p.get("imageHash")
        b64 = images.get(h) if h else None
        if b64 is None:
            return None
        return Fill(kind="image", image_ref="data:image/png;base64," + b64, opacity=op)
    return None


_GRAD_SIZE: list[tuple[float, float]] = [(1.0, 1.0)]


def _handles_to_angle(p: dict) -> float:
    w, h = _GRAD_SIZE[-1]
    hp = p.get("gradientHandlePositions") or [{"x": 0.5, "y": 0}, {"x": 0.5, "y": 1}]
    dx = (float(hp[1]["x"]) - float(hp[0]["x"])) * w
    dy = (float(hp[1]["y"]) - float(hp[0]["y"])) * h
    return round(math.degrees(math.atan2(dx, -dy)) % 360.0, 4)


def _style_to_weight(style: str) -> tuple[int, bool]:
    s = style.replace(" ", "").lower()
    italic = "italic" in s
    s = s.replace("italic", "") or "regular"
    return _STYLE_WEIGHT.get(s, 400), italic


def _strokes(fn: dict) -> list[Stroke]:
    paints = [p for p in fn.get("strokes") or [] if p.get("type") == "SOLID" and p.get("visible", True) is not False]
    if not paints:
        return []
    raw = fn.get("strokeWeight", 1)
    w = float(raw) if isinstance(raw, (int, float)) else 1.0
    align = str(fn.get("strokeAlign", "INSIDE")).lower()
    sides = None
    if fn.get("type") in ("FRAME", "RECTANGLE", "INSTANCE", "COMPONENT"):
        # Figma: strokeWeight writes all four sides; differing sides read back as figma.mixed
        sw = tuple(float(fn.get(k, w)) for k in ("strokeTopWeight", "strokeRightWeight", "strokeBottomWeight", "strokeLeftWeight"))
        if not isinstance(raw, (int, float)) or any(abs(v - w) > 1e-6 for v in sw):
            sides, w = sw, max(sw)
    return [Stroke(color=_color(p["color"], float(p.get("opacity", 1))), width=w, align=align, sides=sides) for p in paints]


def _effects(fn: dict) -> list[Shadow]:
    out = []
    for e in fn.get("effects") or []:
        if e.get("visible") is False or e.get("type") not in ("DROP_SHADOW", "INNER_SHADOW"):
            continue
        off = e.get("offset", {})
        out.append(Shadow(color=_color(e.get("color", {})), dx=float(off.get("x", 0)), dy=float(off.get("y", 0)),
                          blur=float(e.get("radius", 0)), spread=float(e.get("spread", 0)), inner=e["type"] == "INNER_SHADOW"))
    return out


def _radius(fn: dict) -> tuple[float, float, float, float]:
    if fn.get("type") not in ("FRAME", "RECTANGLE", "INSTANCE", "COMPONENT"):
        return (0.0, 0.0, 0.0, 0.0)
    keys = ("topLeftRadius", "topRightRadius", "bottomRightRadius", "bottomLeftRadius")
    if all(isinstance(fn.get(k), (int, float)) for k in keys):  # Figma keeps the four corners authoritative
        return tuple(float(fn[k]) for k in keys)  # type: ignore[return-value]
    r = fn.get("cornerRadius")
    r = float(r) if isinstance(r, (int, float)) else 0.0
    return (r, r, r, r)


def _plugin_data(fn: dict) -> dict:
    d = fn.get("pluginData")
    return d if isinstance(d, dict) else {}


def _convert(fn: dict, ox: float, oy: float, images: dict[str, str], icon_family: str) -> Optional[Node]:
    t = fn.get("type")
    x, y = ox + float(fn.get("x", 0)), oy + float(fn.get("y", 0))
    w, h = float(fn.get("width", 0)), float(fn.get("height", 0))
    pid = _plugin_data(fn).get("dt.id")
    common: dict[str, Any] = dict(name=str(fn.get("name", "")), opacity=float(fn.get("opacity", 1)),
                                  visible=fn.get("visible", True) is not False, effects=_effects(fn))
    if pid:  # plan id stamped by the plugin (setPluginData) -> per-node comparison with the source IR
        common["id"] = pid
    if t == "TEXT":
        font = fn.get("fontName") or {}
        fills = [f for f in (_paint_to_fill(p, images) for p in fn.get("fills") or []) if f is not None]
        color = fills[0].color if fills and fills[0].color is not None else Color(0, 0, 0)
        if font.get("family") == icon_family:
            n = Node(type="icon", box=Box(x, y, w, h), icon_name=str(fn.get("characters", "")),
                     fills=[Fill.solid(color)], **common)
            fill_axis = _plugin_data(fn).get("dt.iconFill")
            if fill_axis not in (None, "", "0"):
                n.meta["icon_fill"] = int(float(fill_axis))
            return n
        weight, italic = _style_to_weight(str(font.get("style", "Regular")))
        lh = fn.get("lineHeight") or {}
        size = float(fn.get("fontSize", 14))
        line_height = float(lh["value"]) if lh.get("unit") == "PIXELS" else (
            float(lh["value"]) * size / 100.0 if lh.get("unit") == "PERCENT" else None)
        ls = fn.get("letterSpacing") or {}
        letter = float(ls.get("value", 0)) * (size / 100.0 if ls.get("unit") == "PERCENT" else 1.0)
        auto = fn.get("textAutoResize") == "WIDTH_AND_HEIGHT"
        ts = TextStyle(family=str(font.get("family", "Roboto")), size=size, weight=weight, italic=italic,
                       line_height=line_height, letter_spacing=letter, color=color,
                       align=_ALIGN_H.get(fn.get("textAlignHorizontal", "LEFT"), "left"),
                       valign="top" if auto else _ALIGN_V.get(fn.get("textAlignVertical", "TOP"), "top"),
                       decoration=_DECO.get(fn.get("textDecoration", "NONE"), "none"),
                       transform=_CASE.get(fn.get("textCase", "ORIGINAL"), "none"))
        return Node(type="text", box=Box(x, y, w, h), text=str(fn.get("characters", "")), text_style=ts,
                    strokes=_strokes(fn), **common)
    if t == "LINE":
        sw = float(fn.get("strokeWeight", 1))
        paints = [p for p in fn.get("strokes") or [] if p.get("type") == "SOLID"]
        fills = [Fill.solid(_color(paints[0]["color"], float(paints[0].get("opacity", 1))))] if paints else []
        return Node(type="line", box=Box(x, y - sw / 2.0, w, sw), fills=fills, **common)
    if t not in ("FRAME", "RECTANGLE", "ELLIPSE", "INSTANCE", "COMPONENT"):
        return None
    _GRAD_SIZE.append((w, h))
    try:
        fills = [f for f in (_paint_to_fill(p, images) for p in fn.get("fills") or []) if f is not None]
    finally:
        _GRAD_SIZE.pop()
    irtype = {"FRAME": "frame", "RECTANGLE": "rect", "ELLIPSE": "ellipse"}.get(t, "frame")
    n = Node(type=irtype, box=Box(x, y, w, h), fills=fills, strokes=_strokes(fn), radius=_radius(fn),
             clip=bool(fn.get("clipsContent", False)) if t != "RECTANGLE" and t != "ELLIPSE" else False, **common)
    for c in fn.get("children") or []:
        cn = _convert(c, x, y, images, icon_family)
        if cn is not None:
            n.children.append(cn)
    return n


def measure_text_nodes(fnodes: list[dict]) -> dict[int, tuple[float, float]]:
    """Glyph box (max-content width, content height) of fake-figma TEXT nodes, measured in the IR renderer."""
    if not fnodes:
        return {}
    from dt.render.html import render_html
    from dt.render.screenshot import _tmp_html, render_url
    icon_family = P["export.figma.icon_font_family"]
    root = Node(id="root", type="frame", box=Box(0, 0, 10, 10))
    for i, fn in enumerate(fnodes):
        tn = _convert(fn, 0.0, 0.0, {}, icon_family)
        if tn is None or tn.type != "text":
            continue
        tn.id = f"m{i}"
        root.children.append(tn)
    doc = Document(width=10, height=10, root=root)
    path = _tmp_html(render_html(doc))
    script = """() => [...document.querySelectorAll('[data-type=text]')].map(el => {
        el.style.width = 'max-content'; el.style.height = 'auto';
        const r = el.getBoundingClientRect(); return [el.id, r.width, r.height]; })"""
    _, res = render_url("file://" + path, 10, 10, wait_ms=0, script="""async () => {
        await Promise.race([document.fonts.ready, new Promise(r => setTimeout(r, 5000))]);
        const used = new Set();
        for (const el of document.querySelectorAll('body *')) { const cs = getComputedStyle(el); used.add(cs.fontWeight + ' 16px ' + cs.fontFamily.split(',')[0]); }
        await Promise.all([...used].map(f => document.fonts.load(f).catch(() => null)));
        return (""" + script + """)(); }""", wait_until="load")
    return {int(i[1:]): (float(w), float(h)) for i, w, h in (res or [])}


def plan_tree_to_ir(tree_root: dict, width: int, height: int, images: Optional[dict[str, str]] = None,
                    measure: bool = True) -> Document:
    """Fake-figma node tree (run_fake.js `tree[0]`) -> IR Document, after modelling Figma's own layout:
    auto-width text is re-sized to its glyphs and auto-layout frames re-position/re-size their children."""
    tree = copy.deepcopy(tree_root)
    icon_family = P["export.figma.icon_font_family"]
    sizes: dict[int, tuple[float, float]] = {}
    if measure:
        auto_texts = [n for n in _walk(tree) if n.get("type") == "TEXT" and n.get("textAutoResize") in ("WIDTH_AND_HEIGHT", "HEIGHT")
                      and (n.get("fontName") or {}).get("family") != icon_family]
        measured = measure_text_nodes(auto_texts)
        sizes = {id(auto_texts[i]): wh for i, wh in measured.items()}
    apply_auto_layout(tree, (lambda n: sizes.get(id(n), (float(n.get("width", 0)), float(n.get("height", 0))))) if measure else None)
    root = _convert(tree, -float(tree.get("x", 0)), -float(tree.get("y", 0)), images or {}, icon_family)
    assert root is not None
    return Document(width=int(width), height=int(height), root=root)


def _walk(n: dict):
    yield n
    for c in n.get("children") or []:
        yield from _walk(c)


# --------------------------------------------------------------------------- the check
@dataclass
class RoundTripReport:
    mean_de: float
    max_de: float
    frac_bad: float
    lossy: bool
    nodes_ir: int
    nodes_figma: int
    warnings: list[str] = field(default_factory=list)
    worst_box: Optional[list[int]] = None
    files: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def roundtrip(doc: Document, out_dir: Optional[str] = None, plan: Optional[dict] = None,
              plugin_code: Optional[str] = None) -> RoundTripReport:
    """Export `doc`, build it on the fake Figma, convert back, render both and compare (ΔE2000)."""
    from dt.render.screenshot import render_doc
    from dt.validate.fidelity import delta_e2000_map
    plan = plan if plan is not None else to_build_plan(doc)
    # the check is about the export, not about which fonts this machine has: install every plan font in the
    # fake (font substitution in a real Figma file is reported by the plugin's own warnings)
    fonts = sorted({(f["family"], f["style"]) for f in plan.get("fonts", [])} | set(_DEFAULT_FONTS))
    out = run_plugin(plan, fonts=fonts, code=plugin_code)
    tree = out["tree"][0]
    rt = plan_tree_to_ir(tree, doc.width, doc.height, out.get("images") or {})
    # reference = the design the plan describes: instances collapsed by `dt map --collapse` keep their real
    # subtree in meta["collapsed_children"], which the export uses for the fallback frame
    from dt.mapping.matcher import expand_instances
    a = render_doc(expand_instances(Document.from_dict(doc.to_dict())))
    b = render_doc(rt)
    de = delta_e2000_map(a, b)
    bad = de > float(P["validate.roundtrip.bad_de"])
    worst = None
    if bad.any():
        ys, xs = np.nonzero(bad)
        worst = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    rep = RoundTripReport(mean_de=float(de.mean()), max_de=float(de.max()), frac_bad=float(bad.mean()),
                          lossy=float(de.mean()) > float(P["validate.roundtrip.max_mean_de"]),
                          nodes_ir=doc.root.count(), nodes_figma=sum(1 for _ in _walk(tree)),
                          warnings=list(plan.get("warnings", [])) + list((out.get("summary") or {}).get("warnings", [])),
                          worst_box=worst)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        from PIL import Image
        Image.fromarray(a).save(os.path.join(out_dir, "direct.png"))
        Image.fromarray(b).save(os.path.join(out_dir, "roundtrip.png"))
        heat = np.clip(de * 25.0, 0, 255).astype(np.uint8)
        Image.fromarray(heat).save(os.path.join(out_dir, "roundtrip_heat.png"))
        rt.save(os.path.join(out_dir, "roundtrip.ir.json"))
        rep.files = {k: os.path.join(out_dir, f) for k, f in (("direct", "direct.png"), ("roundtrip", "roundtrip.png"),
                                                                ("heat", "roundtrip_heat.png"), ("ir", "roundtrip.ir.json"))}
        with open(os.path.join(out_dir, "roundtrip.json"), "w") as f:
            json.dump(rep.to_dict(), f, indent=2)
    return rep


def main(argv: Optional[list[str]] = None) -> int:
    """python -m dt.validate.roundtrip ir.json [...] [-o out_dir] [--json]; exit 1 when an export is lossy."""
    import argparse
    ap = argparse.ArgumentParser(prog="python -m dt.validate.roundtrip", description=main.__doc__)
    ap.add_argument("ir", nargs="+")
    ap.add_argument("-o", "--out")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    lossy = False
    for path in a.ir:
        out = os.path.join(a.out, os.path.splitext(os.path.basename(path))[0]) if a.out else None
        rep = roundtrip(Document.load(path), out_dir=out)
        lossy |= rep.lossy
        if a.json:
            print(json.dumps({"ir": path, **{k: v for k, v in rep.to_dict().items() if k != "warnings"}}))
        else:
            print(f"{'LOSSY' if rep.lossy else 'ok   '} mean dE {rep.mean_de:.4f}  max {rep.max_de:.1f}  "
                  f">JND {100 * rep.frac_bad:.3f}%  worst {rep.worst_box}  {path}")
    return 1 if lossy else 0


if __name__ == "__main__":
    raise SystemExit(main())
