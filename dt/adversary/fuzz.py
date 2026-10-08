"""Boundary fuzzing of perception: the robustness envelope of every component type.

For each component type (filled/outlined button, chip, filled/outlined/elevated card, FAB, icon,
avatar, divider, body text, checkbox) a minimal scene is built as IR, rendered with the real
renderer and perceived. Three axes are searched for the boundary at which perception loses the
component:

* ``contrast`` -- every colour of the component is interpolated towards the page background
  (``t = 0`` invisible, ``t = 1`` the Material 3 baseline colours); reported as the ΔE76 between the
  component's key colour and the background at the boundary;
* ``size``     -- every dimension (box, radius, text and icon size) scaled by ``s``; reported as the
  component's key dimension in px (height, or text size for text);
* ``spacing``  -- two instances side by side ``g`` px apart; the boundary is the smallest gap at which
  they are still perceived as two (below it they merge).

A grid pass finds the stable region (detected at the grid value and at every larger one), then
bisection refines the boundary. Islands below the boundary are reported (non-monotonic perception).
Each boundary is a scenario family (``fuzz.<component>.<axis>``): :func:`scene` regenerates a case
from ``(component, axis, value)``, so a lost component becomes a standing test.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from dt.adversary.taxonomy import Finding
from dt.ir import Box, Color, Document, Fill, Node, Shadow, Stroke, TextStyle
from dt.params import P, register

register("adversary.fuzz.canvas", [360, 160], "fuzz scene size (w, h) px")
register("adversary.fuzz.shape_iou", 0.5, "a shape part is found when a perceived non-text node overlaps it at IoU >= this", (0.1, 0.95))
register("adversary.fuzz.text_cer", 0.25, "a label is found when a perceived text node near it reads with CER <= this", (0.0, 1.0))
register("adversary.fuzz.bisect_steps", 2, "bisection steps after the grid pass", (0, 6))

BG = Color.from_hex("#fef7ff")  # md.sys.color.surface
PRIMARY = Color.from_hex("#6750a4")
ON_PRIMARY = Color.from_hex("#ffffff")
OUTLINE = Color.from_hex("#79747e")
OUTLINE_VARIANT = Color.from_hex("#cac4d0")
ON_SURFACE_VARIANT = Color.from_hex("#49454f")
PRIMARY_CONTAINER = Color.from_hex("#eaddff")
ON_PRIMARY_CONTAINER = Color.from_hex("#21005d")
SC_LOW = Color.from_hex("#f7f2fa")
SC_HIGHEST = Color.from_hex("#e6e0e9")
ELEV1 = [Shadow(Color(0, 0, 0, 0.3), 0, 1, 2, 0), Shadow(Color(0, 0, 0, 0.15), 0, 1, 3, 1)]

GRIDS = {
    "contrast": [0.03, 0.06, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65, 1.0],
    "size": [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0],
    "spacing": [0, 1, 2, 3, 4, 6, 8, 12, 16, 24],
}
UNITS = {"contrast": "ΔE76 key colour vs background", "size": "px (key dimension)", "spacing": "px gap"}


def _lerp(c: Color, t: float, bg: Color = BG) -> Color:
    return Color(int(round(bg.r + (c.r - bg.r) * t)), int(round(bg.g + (c.g - bg.g) * t)), int(round(bg.b + (c.b - bg.b) * t)), c.a)


@dataclass
class Spec:
    """A component type: nominal geometry and colours at scale 1."""
    name: str
    w: float
    h: float
    shape: Optional[str] = "frame"      # frame | ellipse | line | None (no container)
    radius: float = 0.0
    fill: Optional[Color] = None
    stroke: Optional[Color] = None
    stroke_w: float = 1.0
    shadow: bool = False
    label: Optional[str] = None
    label_color: Color = field(default_factory=lambda: ON_SURFACE_VARIANT)
    label_size: float = 14.0
    label_weight: int = 500
    icon: Optional[str] = None
    icon_color: Color = field(default_factory=lambda: ON_SURFACE_VARIANT)
    icon_size: float = 24.0
    key: str = "fill"                    # which colour defines contrast: fill | stroke | label | icon

    def key_color(self) -> Color:
        return {"fill": self.fill, "stroke": self.stroke, "label": self.label_color, "icon": self.icon_color}[self.key] or ON_SURFACE_VARIANT

    def key_dim(self, s: float) -> float:
        if self.shape is None and self.label:
            return self.label_size * s
        if self.shape is None and self.icon:
            return self.icon_size * s
        return (self.w if self.shape == "line" else self.h) * s


COMPONENTS: dict[str, Spec] = {s.name: s for s in [
    Spec("filled_button", 104, 40, radius=20, fill=PRIMARY, label="Button", label_color=ON_PRIMARY),
    Spec("outlined_button", 104, 40, radius=20, stroke=OUTLINE, label="Button", label_color=PRIMARY, key="stroke"),
    Spec("chip", 88, 32, radius=8, stroke=OUTLINE, label="Filter", label_color=ON_SURFACE_VARIANT, key="stroke"),
    Spec("filled_card", 160, 80, radius=12, fill=SC_HIGHEST),
    Spec("outlined_card", 160, 80, radius=12, stroke=OUTLINE_VARIANT, key="stroke"),
    Spec("elevated_card", 160, 80, radius=12, fill=SC_LOW, shadow=True),
    Spec("fab", 56, 56, radius=16, fill=PRIMARY_CONTAINER, icon="add", icon_color=ON_PRIMARY_CONTAINER),
    Spec("icon", 24, 24, shape=None, icon="settings", icon_color=ON_SURFACE_VARIANT, key="icon"),
    Spec("avatar", 40, 40, shape="ellipse", fill=PRIMARY),
    Spec("divider", 200, 1, shape="line", fill=OUTLINE_VARIANT),
    Spec("body_text", 120, 20, shape=None, label="Supporting text", label_color=ON_SURFACE_VARIANT, label_weight=400, key="label"),
    Spec("checkbox", 18, 18, radius=2, stroke=ON_SURFACE_VARIANT, stroke_w=2.0, key="stroke"),
]}
AXES = ("contrast", "size", "spacing")


def _instance(sp: Spec, x: float, y: float, t: float, s: float, tag: str) -> tuple[list[Node], list[tuple[str, Box, Optional[str]]]]:
    """Nodes of one instance at (x, y) and its parts ``[(role, box, text)]``."""
    w, h = max(1.0, sp.w * s), max(1.0, sp.h * s) if sp.shape != "line" else 1.0
    box = Box(round(x), round(y), round(w), max(1, round(h)))
    nodes: list[Node] = []
    parts: list[tuple[str, Box, Optional[str]]] = []
    if sp.shape is not None:
        n = Node(id=f"{tag}_shape", type="frame" if sp.shape in ("frame",) else sp.shape, name=sp.name, box=box)
        if sp.shape == "ellipse":
            n.radius = (box.w / 2,) * 4
        else:
            r = min(sp.radius * s, box.h / 2)
            n.radius = (r, r, r, r)
        if sp.fill is not None:
            n.fills = [Fill.solid(_lerp(sp.fill, t))]
        if sp.stroke is not None:
            n.strokes = [Stroke(color=_lerp(sp.stroke, t), width=max(1.0, round(sp.stroke_w * s)), align="inside")]
        if sp.shadow:
            n.effects = [Shadow(Color(0, 0, 0, round(e.color.a * t, 3)), e.dx, e.dy, e.blur, e.spread) for e in ELEV1]
        nodes.append(n)
        parts.append(("shape", box, None))
    if sp.label:
        size = max(4.0, round(sp.label_size * s, 1))
        ts = TextStyle(family="Roboto", size=size, weight=sp.label_weight, line_height=round(size * 1.43, 1), color=_lerp(sp.label_color, t),
                       align="center", valign="middle")
        tb = box if sp.shape is not None else Box(round(x), round(y), round(sp.w * s), round(size * 1.43))
        nodes.append(Node(id=f"{tag}_label", type="text", name=f"text:{sp.label}", box=tb, text=sp.label, text_style=ts))
        parts.append(("label", tb, sp.label))
    if sp.icon:
        isz = max(4.0, round(sp.icon_size * s))
        ib = Box(round(box.cx - isz / 2), round(box.cy - isz / 2), isz, isz) if sp.shape is not None else Box(round(x), round(y), isz, isz)
        ic = Node(id=f"{tag}_icon", type="icon", name=f"icon:{sp.icon}", box=ib, icon_name=sp.icon, fills=[Fill.solid(_lerp(sp.icon_color, t))])
        nodes.append(ic)
        parts.append(("icon", ib, sp.icon))
    return nodes, parts


def scene(component: str, contrast: float = 1.0, size: float = 1.0, spacing: Optional[float] = None) -> tuple[Document, list[list[tuple[str, Box, Optional[str]]]]]:
    """A fuzz scene: one instance centred (or two, ``spacing`` px apart). Returns (doc, parts per instance)."""
    sp = COMPONENTS[component]
    W, H = (int(v) for v in P["adversary.fuzz.canvas"])
    w = sp.w * size
    h = (sp.h if sp.shape != "line" else 1) * size if sp.shape != "line" else 1
    if sp.shape is None and sp.label:
        h = sp.label_size * size * 1.43
    insts = []
    if spacing is None:
        insts.append((W / 2 - w / 2, H / 2 - h / 2))
    else:
        tot = 2 * w + spacing
        x0 = W / 2 - tot / 2
        insts += [(x0, H / 2 - h / 2), (x0 + round(w) + spacing, H / 2 - h / 2)]
    root = Node(id="root", type="frame", name="Screen", box=Box(0, 0, W, H), fills=[Fill.solid(BG)])
    all_parts = []
    for k, (x, y) in enumerate(insts):
        nodes, parts = _instance(sp, x, y, contrast, size, f"i{k}")
        if sp.shape is not None and len(nodes) > 1:
            nodes[0].children = nodes[1:]
            root.children.append(nodes[0])
        else:
            root.children.extend(nodes)
        all_parts.append(parts)
    return Document(width=W, height=H, root=root), all_parts


def detect(perceived: Document, parts: list[list[tuple[str, Box, Optional[str]]]]) -> tuple[bool, dict]:
    """Are all parts of all instances found as distinct perceived nodes?"""
    from dt.adversary.metamorphic import _cer
    nodes = [n for n in perceived.walk() if n.id != perceived.root.id and n.visible]
    used: set[str] = set()
    report = []
    ok = True
    thr = float(P["adversary.fuzz.shape_iou"])
    for parts_i in parts:
        for role, box, text in parts_i:
            best, score = None, 0.0
            for n in nodes:
                if n.id in used:
                    continue
                if role == "label":
                    if n.type != "text" or not box.expand(4).contains_point(n.box.cx, n.box.cy):
                        continue
                    sc = 1.0 - _cer(n.text, text)
                    if sc >= 1.0 - float(P["adversary.fuzz.text_cer"]) and sc > score:
                        best, score = n, sc
                else:
                    if n.type == "text":
                        continue
                    if box.h <= 2 or box.w <= 2:  # hairlines: overlap along the long axis
                        inter = n.box.expand(1).intersect(box.expand(1))
                        sc = inter.area / max(box.expand(1).area, 1e-6) if n.box.h <= 4 or n.box.w <= 4 else 0.0
                    else:
                        sc = n.box.iou(box)
                    if sc >= (0.3 if role == "icon" else thr) and sc > score:
                        best, score = n, sc
            if best is None:
                ok = False
                report.append({"role": role, "found": False})
            else:
                used.add(best.id)
                report.append({"role": role, "found": True, "type": best.type, "score": round(score, 3)})
    types = sorted({n.type for n in nodes})
    return ok, {"parts": report, "perceived_types": types, "n_nodes": len(nodes)}


def _value_args(axis: str, v: float) -> dict:
    return {"contrast": {"contrast": v}, "size": {"size": v}, "spacing": {"spacing": v}}[axis]


def _measure(component: str, axis: str, v: float) -> float:
    sp = COMPONENTS[component]
    if axis == "contrast":
        return round(_lerp(sp.key_color(), v).delta_e(BG), 2)
    if axis == "size":
        return round(sp.key_dim(v), 2)
    return float(v)


def _evaluate(jobs: list[tuple[str, str, float]], workers: int) -> dict[tuple[str, str, float], tuple[bool, dict]]:
    from dt.adversary import metamorphic as M
    from dt.render.screenshot import render_doc
    imgs = {}
    for comp, axis, v in jobs:
        doc, parts = scene(comp, **_value_args(axis, v))
        imgs[(comp, axis, v)] = (render_doc(doc), parts)
    M.precompute([(im, "perceive", 1.0) for im, _ in imgs.values()], workers)
    out = {}
    for k, (im, parts) in imgs.items():
        out[k] = detect(M.translate_ir(im, "perceive"), parts)
    return out


def envelope(components: Optional[list[str]] = None, axes: tuple[str, ...] = AXES, workers: int = 3,
             log: Optional[Callable[[str], None]] = None) -> list[dict]:
    """Grid + bisection search of the perception boundary for every (component, axis)."""
    comps = components or list(COMPONENTS)
    jobs = [(c, a, v) for c in comps for a in axes for v in GRIDS[a]]
    t0 = time.time()
    res = _evaluate(jobs, workers)
    if log:
        log(f"fuzz grid: {len(jobs)} scenes in {time.time() - t0:.0f}s")
    rows = []
    for c in comps:
        for a in axes:
            grid = GRIDS[a]
            hits = [res[(c, a, v)][0] for v in grid]
            k = len(grid)
            while k > 0 and hits[k - 1]:
                k -= 1
            rows.append({"component": c, "axis": a, "grid": grid, "hits": hits, "k": k,
                         "islands": [grid[i] for i in range(k) if hits[i]],
                         "nominal_ok": hits[-1], "detail_below": res[(c, a, grid[k - 1])][1] if 0 < k <= len(grid) else None})
    # bisection between the last miss and the first stable hit
    for _ in range(int(P["adversary.fuzz.bisect_steps"])):
        pend = []
        for r in rows:
            if not r["nominal_ok"] or r["k"] == 0:
                continue
            lo, hi = r.get("lo", r["grid"][r["k"] - 1]), r.get("hi", r["grid"][r["k"]])
            if r["axis"] == "spacing" and hi - lo <= 1:
                continue
            mid = (lo + hi) / 2 if r["axis"] != "spacing" else float(int((lo + hi) / 2))
            r["lo"], r["hi"], r["mid"] = lo, hi, mid
            pend.append((r["component"], r["axis"], mid))
        if not pend:
            break
        got = _evaluate(pend, workers)
        for r in rows:
            if "mid" in r:
                ok, det = got[(r["component"], r["axis"], r.pop("mid"))]
                mid = (r["lo"] + r["hi"]) / 2 if r["axis"] != "spacing" else float(int((r["lo"] + r["hi"]) / 2))
                if ok:
                    r["hi"] = mid
                else:
                    r["lo"] = mid
                    r["detail_below"] = det
    for r in rows:
        if not r["nominal_ok"]:
            r["boundary"] = None
        elif r["k"] == 0:
            r["boundary"] = r["grid"][0]
            r["below_grid"] = True
        else:
            r["boundary"] = r.get("hi", r["grid"][r["k"]])
        r["boundary_value"] = None if r["boundary"] is None else _measure(r["component"], r["axis"], r["boundary"])
        r["lost_at"] = None if r["boundary"] is None or r["k"] == 0 else _measure(r["component"], r["axis"], r.get("lo", r["grid"][r["k"] - 1]))
        r["unit"] = UNITS[r["axis"]]
        r["monotonic"] = not r["islands"]
    return rows


def findings(rows: list[dict]) -> list[Finding]:
    """One typed finding per boundary (a robustness limit of the translator) and per nominal failure."""
    out = []
    for r in rows:
        doc, parts = scene(r["component"], **_value_args(r["axis"], r["boundary"] if r["boundary"] is not None else 1.0))
        box = None
        for ps in parts:
            for _, b, _ in ps:
                box = b if box is None else box.union(b)
        sp = COMPONENTS[r["component"]]
        t = "structure.merge" if r["axis"] == "spacing" else ("icon.missing" if sp.shape is None and sp.icon else "structure.missing")
        if r["boundary"] is None:
            out.append(Finding(t, box, None, 1.0, {"component": r["component"], "axis": r["axis"], "nominal_fail": True,
                                                   "family": f"fuzz.{r['component']}.{r['axis']}"}))
            continue
        out.append(Finding(t, box, r["boundary_value"], 0.5, {"component": r["component"], "axis": r["axis"], "boundary": r["boundary"],
                                                              "lost_at": r["lost_at"], "unit": r["unit"], "monotonic": r["monotonic"],
                                                              "family": f"fuzz.{r['component']}.{r['axis']}"}))
    return out


def scenario_specs(rows: list[dict]) -> list[dict]:
    from dt.adversary.metamorphic import translator_version
    out = []
    for r in rows:
        lo = r.get("lo", r["grid"][r["k"] - 1] if r["k"] > 0 else None)
        out.append({"family": f"fuzz.{r['component']}.{r['axis']}", "generator": "dt.adversary.fuzz:scene",
                    "params": {"component": r["component"], "axis": r["axis"], "bracket": [lo, r["boundary"]]},
                    "oracle": "detect(perceive(render(scene)), parts) is True above the bracket", "boundary_value": r["boundary_value"],
                    "unit": r["unit"], "nominal_ok": r["nominal_ok"], "monotonic": r["monotonic"], "translator": translator_version("perceive")})
    return out
