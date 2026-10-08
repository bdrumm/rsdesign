"""Map an IR Document onto a DesignSystem: component matching + token snapping.

`map_document(doc, ds)` does three things, all deterministic and explainable:

1. **Subtree pattern matching** (bottom-up).  Every non-root container node is turned into
   `NodeFeatures` (size, radius, paint, texts, icons, ordered child "atoms") and scored against
   every `ComponentSpec` × `Variant` signature with `score_signature`.  The score is a weighted
   mean of soft per-constraint scores in [0, 1]; the best (spec, variant) above
   `P["map.min_conf"]` becomes `node.component` (a `ComponentRef` with variant, slot props,
   confidence and evidence).  Matched subtrees are *atomic* for their ancestors (collapse), so a
   list item sees its trailing switch as `instance:Switch`, not as a pill + circle.

2. **Token snapping** for every node: fills/strokes/text colors -> nearest color role by ΔE,
   radius -> shape scale (pill detection first), text style -> typescale role by
   (size, weight, line-height) distance, shadows -> elevation level, layout gap/padding -> the
   spacing grid.  Results go to `node.tokens` and the evidence to `node.meta["tokens_evidence"]`.

3. **As-is preference**: matched nodes get `type = "instance"` and their content is summarised
   in `component.props` (label/icon/...).  Children are kept (renderers and the Figma exporter's
   fallback need them) unless `collapse=True`, which moves them to `meta["collapsed_children"]`
   (restore with `expand_instances`).

Everything numeric is a registered parameter (`dt.params`, prefix `map.`).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from dt.ir import Color, ComponentRef, Document, Node, TextStyle
from dt.mapping.design_system import ChildPattern, ComponentSpec, DesignSystem, Signature, Slot, Variant
from dt.params import P, register

# --------------------------------------------------------------------------- params
register("map.color_de", 6.0, "max CIE76 ΔE for a color to snap to a role token", (2.0, 15.0))
register("map.min_conf", 0.6, "min signature score to assign a ComponentRef", (0.3, 0.95))
register("map.candidate_conf", 0.35, "min score to record a component candidate in node.meta (decision queue)", (0.1, 0.6))
register("map.radius_tol", 2.0, "px tolerance when matching / snapping corner radius", (0.5, 6.0))
register("map.pill_ratio", 0.42, "radius / min(w,h) above which a corner counts as 'pill' (full)", (0.3, 0.5))
register("map.grid", 4.0, "spacing grid in px (M3 = 4)", (2.0, 8.0))
register("map.grid_tol", 1.0, "px tolerance for snapping gaps/padding to the grid", (0.0, 2.0))
register("map.height_tol", 6.0, "px slack outside a height range before the score reaches 0", (1.0, 16.0))
register("map.width_tol_frac", 0.25, "fraction of the width range edge used as soft slack", (0.05, 0.5))
register("map.aspect_tol", 0.3, "soft slack (ratio units) outside an aspect range", (0.05, 1.0))
register("map.stroke_width_tol", 1.0, "px slack for stroke width constraints", (0.25, 3.0))
register("map.type.size_tol", 1.5, "px of font-size that counts as one unit of typescale distance", (0.5, 4.0))
register("map.type.weight_tol", 150.0, "font-weight difference that counts as one unit of typescale distance", (50.0, 300.0))
register("map.type.lh_tol", 4.0, "px of line-height that counts as one unit of typescale distance", (1.0, 8.0))
register("map.type.max_dist", 1.5, "max typescale distance (units) to snap a text style to a role", (0.5, 2.5))
register("map.derive_color_de", 2.0, "ΔE below which an exemplar's exact color is expressed as a token when deriving a signature", (0.5, 6.0))
register("map.extra_child_penalty", 0.25, "score penalty per child not explained by the children pattern", (0.0, 1.0))
register("map.bg_child_iou", 0.95, "IoU above which a first child rect is treated as the container's own paint", (0.8, 1.0))
register("map.elevation_tol", 3.0, "px slack when matching a shadow (dy+blur) to an elevation level", (1.0, 8.0))
register("map.zero_penalty", 0.5, "multiplier applied per *critical* constraint (size/paint) that scores 0", (0.1, 1.0))
register("map.w.height", 2.0, "weight: height constraint", (0.0, 4.0))
register("map.w.width", 1.0, "weight: width constraint", (0.0, 4.0))
register("map.w.aspect", 1.0, "weight: aspect constraint", (0.0, 4.0))
register("map.w.radius", 1.5, "weight: radius constraint", (0.0, 4.0))
register("map.w.fill", 2.0, "weight: fill constraint", (0.0, 4.0))
register("map.w.stroke", 1.5, "weight: stroke constraint", (0.0, 4.0))
register("map.w.shadow", 1.0, "weight: shadow constraint", (0.0, 4.0))
register("map.w.text", 0.75, "weight: text role constraint", (0.0, 4.0))
register("map.w.text_color", 1.0, "weight: text color constraint", (0.0, 4.0))
register("map.w.children", 1.5, "weight: children pattern constraint", (0.0, 4.0))
register("map.w.icon_size", 0.5, "weight: icon size constraint", (0.0, 4.0))
register("map.w.counts", 1.0, "weight: text/icon count constraints", (0.0, 4.0))
register("map.w.cells", 1.5, "weight: equal-width cells constraint (tab strips)", (0.0, 4.0))
register("map.w.indicator", 1.5, "weight: indicator line constraint (tab strips)", (0.0, 4.0))
register("map.w.title_align", 1.0, "weight: title alignment constraint (top app bars)", (0.0, 4.0))
register("map.cells.center_tol", 0.25,
         "max |cell centre - equal-slot centre| as a fraction of the slot width before the cells score reaches 0",
         (0.05, 0.5))
register("map.cells.merge_px", 6.0, "px: atoms whose horizontal centres are this close share one cell (icon over label)", (1.0, 16.0))
register("map.indicator.thick_tol", 1.0, "px slack around a signature's indicator thickness range", (0.0, 3.0))
register("map.indicator.min_aspect", 4.0, "min width/height of a line/rect to count as a horizontal indicator", (2.0, 20.0))
register("map.indicator.bottom_frac", 0.35,
         "an indicator must lie in this bottom fraction of the container's height", (0.1, 0.6))
register("map.title.center_tol", 12.0,
         "px: distance of the title centre from the bar centre (or from the centre of the free span between "
         "its neighbours) up to which a title counts as centred", (4.0, 40.0))
register("map.title.start_gap", 32.0,
         "px: max gap between a start-aligned title and its left neighbour (or the bar edge)", (8.0, 80.0))
register("map.keep_geom_types", "",
         "comma list of IR node types that keep their geometric type when matched to a component "
         "(the component is then recorded in node.component only); all others become 'instance' "
         "(the documented contract, relied on by the CLI/export). 'line' would let matched dividers "
         "pair with gt 'line' nodes in compare.structural (TYPE_COMPAT has no line<->instance): "
         "+0.13 component_acc / +0.04 node_recall on the bench, see docs/VARIANTS.md", None)

# Confidence calibration. The *assignment* uses the raw signature score (best guess, unchanged);
# `ComponentRef.confidence` additionally says how sure that guess is, so that wrong or undecidable
# matches fall below `pipeline.decisions.component_conf` and are queued in decisions.json.
register("map.conf.margin_full", 0.15,
         "score margin over the best *different* (component, variant) at or above which confidence equals the "
         "score; below it confidence shrinks linearly towards map.conf.tie_factor x score at a tie", (0.02, 0.5))
register("map.conf.tie_factor", 0.7,
         "confidence multiplier at a zero margin (two different components / variants score the same)", (0.3, 1.0))
register("map.conf.inferred_src", "group",
         "comma list of node.meta['src'] values marking unpainted frames *inferred* by perceive (not observed)", None)
register("map.conf.imposed", "height,width,aspect,radius,fill,stroke,shadow",
         "comma list of constraints an inferred unpainted frame satisfies by construction (its size and 'no paint' "
         "were imposed by the inference); confidence on such a node is capped by the score of the other, observed "
         "constraints", None)
register("map.conf.inferred_critical", "text_color",
         "comma list of observed constraints that take the role of paint on an inferred unpainted frame (its only "
         "colour evidence): a zero score there applies map.zero_penalty to the observed-evidence cap", None)

CONTROL_NAMES = {"Switch", "Checkbox", "Radio", "IconButton"}
CRITICAL = {"height", "width", "aspect", "fill", "radius", "stroke", "shadow"}
_TIE = 0.05  # distances within this are considered equal (priority order decides)
GRAPHIC_TYPES = {"icon", "image", "ellipse", "vector"}


# --------------------------------------------------------------------------- features
@dataclass
class NodeFeatures:
    """Everything the scorer needs to know about a subtree, computed once per node."""

    node: Node
    type: str
    w: float
    h: float
    radius: float
    is_pill: bool
    fill: Optional[Color]
    stroke_color: Optional[Color]
    stroke_width: float
    has_shadow: bool
    atoms: list[Node] = field(default_factory=list)  # ordered direct content (wrappers flattened)
    texts: list[Node] = field(default_factory=list)
    icons: list[Node] = field(default_factory=list)

    @property
    def aspect(self) -> float:
        return self.w / self.h if self.h > 0 else 0.0

    @property
    def primary_text(self) -> Optional[Node]:
        if not self.texts:
            return None
        return max(self.texts, key=lambda t: (t.text_style.size if t.text_style else 0, -t.box.y))


def _is_atomic(n: Node) -> bool:
    """A child that is content in its own right (not a transparent wrapper frame)."""
    if n.component is not None or n.type != "frame":
        return True
    return bool(n.fills or n.strokes or n.effects or not n.children)


def _flatten_atoms(n: Node) -> list[Node]:
    out: list[Node] = []
    for c in n.children:
        if not c.visible:
            continue
        if _is_atomic(c):
            out.append(c)
        else:
            out.extend(_flatten_atoms(c))
    return out


def order_atoms(atoms: list[Node], order: str) -> list[Node]:
    if order == "column":
        return sorted(atoms, key=lambda a: (round(a.box.y), round(a.box.x)))
    return sorted(atoms, key=lambda a: (round(a.box.x), round(a.box.y)))


def _icon_size(n: Node) -> float:
    return min(n.box.w, n.box.h)


def features(node: Node, order: str = "row") -> NodeFeatures:
    """Compute match features for `node`. A first child covering the node (IoU ≥ P["map.bg_child_iou"])
    with no fill on the node itself is treated as the node's own paint (common 'background' rect)."""
    atoms = _flatten_atoms(node)
    paint = node
    if node.fill_color is None and not node.strokes and atoms:
        first = atoms[0]
        if first.type in ("rect", "frame") and first.component is None and first.box.iou(node.box) >= P["map.bg_child_iou"] and (first.fills or first.strokes):
            paint = first
            atoms = atoms[1:] + _flatten_atoms(first)
    atoms = order_atoms(atoms, order)
    r = paint.uniform_radius() if paint is not node or any(node.radius) else node.uniform_radius()
    if node.type == "ellipse":
        r = min(node.box.w, node.box.h) / 2
    short = max(1.0, min(node.box.w, node.box.h))
    stroke = paint.strokes[0] if paint.strokes else (node.strokes[0] if node.strokes else None)
    fill = paint.fill_color
    if fill is not None and fill.a <= 0.01:
        fill = None
    # A stroke indistinguishable from the fill it borders (anti-aliasing / shadow halo read as an
    # outline by perceive) is not evidence of an outline: drop it rather than fail `stroke: none`.
    if stroke is not None and fill is not None and stroke.color.delta_e(fill) <= P["map.color_de"]:
        stroke = None
    return NodeFeatures(
        node=node, type=node.type, w=node.box.w, h=node.box.h, radius=r,
        is_pill=node.type == "ellipse" or r >= P["map.pill_ratio"] * short,
        fill=fill, stroke_color=stroke.color if stroke else None, stroke_width=stroke.width if stroke else 0.0,
        has_shadow=bool(paint.effects or node.effects),
        atoms=atoms, texts=[a for a in atoms if a.type == "text" and (a.text or "").strip()],
        icons=[a for a in atoms if a.type == "icon"],
    )


# --------------------------------------------------------------------------- soft scores
def soft_range(v: float, lo: float, hi: float, tol: float) -> float:
    """1 inside [lo, hi]; linear fall-off to 0 at distance `tol` outside."""
    if lo <= v <= hi:
        return 1.0
    over = lo - v if v < lo else v - hi
    return max(0.0, 1.0 - over / max(tol, 1e-6))


def color_score(c: Optional[Color], candidates: list[str], ds: DesignSystem) -> tuple[float, dict]:
    """Score a color against candidate tokens ('none' allows absence). Soft on ΔE."""
    de_max = P["map.color_de"]
    allow_none = "none" in candidates
    if c is None:
        return (1.0, {"observed": None}) if allow_none else (0.0, {"observed": None})
    # A catalog may carry alternate scheme values for its roles (e.g. material3 stores the dark
    # scheme under meta["dark_scheme"] = {short role: hex}); a role then matches in either scheme.
    alt_scheme = ds.meta.get("dark_scheme") or {}
    best_de, best_name, best_scheme = math.inf, None, None
    for name in candidates:
        if name == "none":
            continue
        refs: list[tuple[Optional[Color], Optional[str]]] = [(ds.resolve_color(name), None)]
        alt = alt_scheme.get(name.rsplit(".", 1)[-1]) if not name.startswith("#") else None
        if alt:
            refs.append((Color.from_hex(alt), "dark"))
        for ref, scheme in refs:
            if ref is None:
                continue
            de = c.delta_e(ref)
            if de < best_de:
                best_de, best_name, best_scheme = de, name, scheme
    if best_name is None:
        return (0.0, {"observed": c.hex()})
    s = 1.0 if best_de <= de_max else max(0.0, 1.0 - (best_de - de_max) / de_max)
    info = {"observed": c.hex(), "nearest": best_name, "de": round(best_de, 2)}
    if best_scheme:
        info["scheme"] = best_scheme
    return s, info


def radius_score(f: NodeFeatures, rule: Any) -> tuple[float, dict]:
    tol = P["map.radius_tol"]
    short = max(1.0, min(f.w, f.h))
    info = {"observed": round(f.radius, 2), "pill": f.is_pill}
    if rule == "any":
        return 1.0, info
    if rule == "pill":
        if f.is_pill:
            return 1.0, info
        return max(0.0, f.radius / (short / 2)), info
    if rule == "none":
        return (1.0 if f.radius <= tol else max(0.0, 1.0 - (f.radius - tol) / (2 * tol))), info
    if isinstance(rule, (int, float)):
        if f.is_pill and rule < short / 2 - tol:
            return 0.0, info
        return soft_range(f.radius, rule - tol, rule + tol, 2 * tol), info
    lo, hi = rule
    return soft_range(f.radius, lo, hi, 2 * tol), info


def stroke_score(f: NodeFeatures, rule: Any, ds: DesignSystem) -> tuple[float, dict]:
    has = f.stroke_color is not None and f.stroke_width > 0
    info = {"observed": f.stroke_color.hex() if f.stroke_color else None, "width": f.stroke_width}
    if rule == "any":
        return 1.0, info
    if rule == "none":
        return (0.0 if has else 1.0), info
    if not has:
        return 0.0, info
    cs, cinfo = color_score(f.stroke_color, list(rule.get("tokens", [])), ds) if rule.get("tokens") else (1.0, {})
    ws = 1.0
    if rule.get("width"):
        lo, hi = rule["width"]
        ws = soft_range(f.stroke_width, lo, hi, P["map.stroke_width_tol"])
    info.update(cinfo)
    return 0.7 * cs + 0.3 * ws, info


def typescale_distance(ts: TextStyle, role: dict) -> float:
    """Distance in 'tolerance units' between a text style and a typescale role."""
    d = abs(ts.size - float(role.get("size", ts.size))) / P["map.type.size_tol"]
    d += abs(ts.weight - int(role.get("weight", ts.weight))) / P["map.type.weight_tol"]
    if ts.line_height and role.get("line_height"):
        d += abs(ts.line_height - float(role["line_height"])) / P["map.type.lh_tol"]
    return d


def _priority(name: str, ds: DesignSystem, key: str) -> int:
    """Rank of a token in the design system's tie-break list (`meta[key]`, short names); unknown = last."""
    order = ds.meta.get(key) or []
    short = name.rsplit(".", 1)[-1]
    return order.index(short) if short in order else len(order)


def nearest_typescale(ts: TextStyle, ds: DesignSystem) -> tuple[Optional[str], float]:
    """Nearest typescale role by (size, weight, line-height) distance; ties broken by `meta['typescale_priority']`."""
    best, best_d, best_p = None, math.inf, math.inf
    for name, role in ds.typescale().items():
        d = typescale_distance(ts, role)
        p = _priority(name, ds, "typescale_priority")
        if d < best_d - _TIE or (abs(d - best_d) <= _TIE and p < best_p):
            best, best_d, best_p = name, d, p
    return best, best_d


def text_role_score(f: NodeFeatures, roles: list[str], ds: DesignSystem) -> tuple[float, dict]:
    if not f.texts:
        return 1.0, {"observed": None}
    scale = ds.typescale()
    best = 0.0
    info: dict = {}
    for t in f.texts:
        if not t.text_style:
            continue
        for r in roles:
            role = scale.get(r) or scale.get(f"md.sys.typescale.{r}") or next((v for k, v in scale.items() if k.endswith("." + r)), None)
            if role is None:
                continue
            d = typescale_distance(t.text_style, role)
            s = 1.0 if d <= 0.5 else max(0.0, 1.0 - (d - 0.5))
            if s > best:
                best, info = s, {"text": (t.text or "")[:20], "role": r, "dist": round(d, 2)}
    return best, info


def _ptype_matches(a: Node, ptype: str) -> bool:
    for alt in ptype.split("|"):
        if alt == "any":
            return True
        if alt.startswith("instance:"):
            if a.component is not None and a.component.name == alt.split(":", 1)[1]:
                return True
            continue
        if alt == "instance":
            if a.component is not None or a.type == "instance":
                return True
            continue
        if alt == "control":
            if a.component is not None and a.component.name in CONTROL_NAMES:
                return True
            continue
        if alt == "graphic":
            if a.type in GRAPHIC_TYPES:
                return True
            continue
        if a.type == alt and a.component is None:
            return True
        if a.type == alt and alt in ("frame", "rect", "ellipse"):
            return True
    return False


def children_score(atoms: list[Node], pattern: list[ChildPattern]) -> tuple[float, dict]:
    """Greedy ordered alignment of `atoms` to `pattern` (items may be optional / repeated)."""
    pen = P["map.extra_child_penalty"]
    j = 0
    req_total = sum(1 for p in pattern if not p.optional)
    req_matched = 0
    matched = 0
    assignment: list[str] = []
    for p in pattern:
        count = 0
        while j < len(atoms) and count < p.max and _ptype_matches(atoms[j], p.type):
            count += 1
            j += 1
            assignment.append(p.name or p.type)
        if count:
            matched += count
            if not p.optional:
                req_matched += 1
    extra = len(atoms) - j
    base = (req_matched / req_total) if req_total else 1.0
    score = max(0.0, base - extra * pen)
    return score, {"matched": matched, "extra": extra, "assignment": assignment, "required": f"{req_matched}/{req_total}"}


def cells_score(f: NodeFeatures) -> tuple[float, dict]:
    """Do the content atoms sit in >= 2 equal-width cells spanning the container (a tab strip)?

    Atoms (text / icon / frame / instance, thin indicator lines excluded) are grouped into cells by
    horizontal centre (``map.cells.merge_px``); with k cells the i-th centre should sit at the
    centre of the i-th of k equal slots. Score falls off linearly with the mean offset, reaching
    0 at ``map.cells.center_tol`` x slot width.
    """
    thin = float(P["map.indicator.min_aspect"])
    atoms = [a for a in f.atoms if a.type in ("text", "icon", "frame", "instance", "vector")
             and not (a.box.h > 0 and a.box.w / a.box.h >= thin and a.box.h <= 4)]
    if len(atoms) < 2 or f.w <= 0:
        return 0.0, {"cells": len(atoms)}
    merge = float(P["map.cells.merge_px"])
    centres: list[list[float]] = []
    for c in sorted(a.box.cx for a in atoms):
        if centres and c - centres[-1][-1] <= merge:
            centres[-1].append(c)
        else:
            centres.append([c])
    k = len(centres)
    if k < 2:
        return 0.0, {"cells": k}
    slot = f.w / k
    x0 = f.node.box.x
    offs = [abs(sum(g) / len(g) - (x0 + slot * (i + 0.5))) for i, g in enumerate(centres)]
    mean_off = sum(offs) / k
    tol = float(P["map.cells.center_tol"]) * slot
    return max(0.0, 1.0 - mean_off / max(tol, 1e-6)), {"cells": k, "mean_offset": round(mean_off, 1), "slot": round(slot, 1)}


def indicator_score(f: NodeFeatures, rng: tuple[float, float]) -> tuple[float, dict]:
    """Best soft score (thickness in ``rng``, linear fall-off over ``map.indicator.thick_tol``) of a
    painted, thin, horizontal line/rect in the bottom part of the container (anywhere in its
    subtree); 0 when there is none. Primary tabs use a 3px indicator, secondary tabs 2px."""
    lo, hi = rng
    tol = float(P["map.indicator.thick_tol"])
    box = f.node.box
    y_min = box.y2 - float(P["map.indicator.bottom_frac"]) * box.h
    best, info = 0.0, {"indicator": None}
    for d in f.node.walk():
        if d is f.node or not d.visible or d.type not in ("line", "rect", "frame", "instance") or d.children:
            continue
        b = d.box
        if b.h <= 0 or b.w < float(P["map.indicator.min_aspect"]) * b.h or b.y < y_min - tol or b.y2 > box.y2 + tol:
            continue
        if d.fill_color is None and not d.strokes:
            continue
        s = soft_range(b.h, lo, hi, tol)
        if s > best:
            best, info = s, {"indicator": [round(b.x, 1), round(b.y, 1), round(b.w, 1), round(b.h, 1)]}
    return best, info


def title_align_score(f: NodeFeatures, want: str) -> tuple[float, dict]:
    """Horizontal placement of the primary (largest) text.

    * ``center``: the title centre is within ``map.title.center_tol`` of the bar centre, or of the
      centre of the free span between its left and right neighbours (a title centred in the space
      left over by asymmetric actions);
    * ``start``: the title starts within ``map.title.start_gap`` of its left neighbour (or the bar
      edge) and is not centred on the bar.
    """
    pt = f.primary_text
    if pt is None:
        return 0.0, {"observed": None}
    box = f.node.box
    others = [a for a in f.atoms if a is not pt and a.box.y < pt.box.y2 and a.box.y2 > pt.box.y]
    left = max([a.box.x2 for a in others if a.box.x2 <= pt.box.x + 1] + [box.x])
    right = min([a.box.x for a in others if a.box.x >= pt.box.x2 - 1] + [box.x2])
    tol = float(P["map.title.center_tol"])
    off = min(abs(pt.box.cx - box.cx), abs(pt.box.cx - (left + right) / 2))
    centred = soft_range(off, 0.0, tol, tol)
    gap = pt.box.x - left
    info = {"center_offset": round(off, 1), "start_gap": round(gap, 1)}
    if want == "center":
        return centred, info
    g = float(P["map.title.start_gap"])
    on_bar_centre = soft_range(abs(pt.box.cx - box.cx), 0.0, tol, tol)
    return soft_range(gap, 0.0, g, g) * (1.0 - on_bar_centre), info


def count_score(n: int, rng: tuple[int, int]) -> float:
    lo, hi = rng
    if lo <= n <= hi:
        return 1.0
    return max(0.0, 1.0 - 0.5 * (lo - n if n < lo else n - hi))


# --------------------------------------------------------------------------- signature scoring
def _weight(name: str, sig: Signature) -> float:
    if sig.weights and name in sig.weights:
        return float(sig.weights[name])
    return float(P[f"map.w.{name}"])


def score_signature(f: NodeFeatures, sig: Signature, ds: DesignSystem) -> tuple[float, dict]:
    """Weighted mean of per-constraint soft scores. Returns (score, per-constraint details).
    A type mismatch (`sig.types`) is a hard 0."""
    if sig.types is not None and f.type not in sig.types and not (f.type == "instance" and "frame" in sig.types):
        return 0.0, {"types": {"score": 0.0, "observed": f.type, "expected": sig.types}}
    parts: list[tuple[str, float, float, dict]] = []  # name, weight, score, info

    def add(name: str, wkey: str, s: float, info: dict, expected: Any) -> None:
        info = dict(info)
        info["expected"] = expected
        parts.append((name, _weight(wkey, sig), s, info))

    if sig.height is not None:
        lo, hi = sig.height
        add("height", "height", soft_range(f.h, lo, hi, P["map.height_tol"]), {"observed": f.h}, list(sig.height))
    if sig.width is not None:
        lo, hi = sig.width
        add("width", "width", soft_range(f.w, lo, hi, max(4.0, P["map.width_tol_frac"] * max(lo, 1))), {"observed": f.w}, list(sig.width))
    if sig.aspect is not None:
        lo, hi = sig.aspect
        add("aspect", "aspect", soft_range(f.aspect, lo, hi, P["map.aspect_tol"] * max(lo, 1)), {"observed": round(f.aspect, 2)}, list(sig.aspect))
    if sig.radius is not None and sig.radius != "any":
        s, info = radius_score(f, sig.radius)
        add("radius", "radius", s, info, sig.radius)
    if sig.fills is not None:
        s, info = color_score(f.fill, sig.fills, ds)
        add("fill", "fill", s, info, sig.fills)
    if sig.stroke is not None and sig.stroke != "any":
        s, info = stroke_score(f, sig.stroke, ds)
        add("stroke", "stroke", s, info, sig.stroke)
    if sig.shadow is not None and sig.shadow != "any":
        want = sig.shadow == "required"
        add("shadow", "shadow", 1.0 if f.has_shadow == want else 0.0, {"observed": f.has_shadow}, sig.shadow)
    if sig.text_roles is not None and f.texts:
        s, info = text_role_score(f, sig.text_roles, ds)
        add("text", "text", s, info, sig.text_roles)
    if sig.text_colors is not None and f.texts:
        pt = f.primary_text
        c = pt.text_style.color if pt is not None and pt.text_style else None
        s, info = color_score(c, sig.text_colors, ds)
        add("text_color", "text_color", s, info, sig.text_colors)
    if sig.text_count is not None:
        add("text_count", "counts", count_score(len(f.texts), sig.text_count), {"observed": len(f.texts)}, list(sig.text_count))
    if sig.icon_count is not None:
        add("icon_count", "counts", count_score(len(f.icons), sig.icon_count), {"observed": len(f.icons)}, list(sig.icon_count))
    if sig.icon_size is not None and f.icons:
        lo, hi = sig.icon_size
        sizes = [_icon_size(i) for i in f.icons]
        add("icon_size", "icon_size", min(soft_range(s, lo, hi, P["map.radius_tol"] * 2) for s in sizes), {"observed": sizes}, list(sig.icon_size))
    if sig.children is not None:
        atoms = order_atoms(f.atoms, sig.order or "row") if sig.order else f.atoms
        s, info = children_score(atoms, sig.children)
        add("children", "children", s, info, [c.to_dict() for c in sig.children])
    if sig.cells == "equal":
        s, info = cells_score(f)
        add("cells", "cells", s, info, sig.cells)
    if sig.indicator is not None:
        s, info = indicator_score(f, sig.indicator)
        add("indicator", "indicator", s, info, list(sig.indicator))
    if sig.title_align is not None:
        s, info = title_align_score(f, sig.title_align)
        add("title_align", "title_align", s, info, sig.title_align)
    total_w = sum(w for _, w, _, _ in parts)
    if total_w <= 0:
        return 0.0, {}
    score = sum(w * s for _, w, s, _ in parts) / total_w
    critical = CRITICAL | set(sig.critical or ())
    n_zero = sum(1 for name, w, s, _ in parts if s <= 0 and w > 0 and name in critical)
    score *= P["map.zero_penalty"] ** n_zero
    details = {name: {"score": round(s, 3), "weight": w, **info} for name, w, s, info in parts}
    if n_zero:
        details["_zero_critical"] = n_zero
    return score, details


@dataclass
class Candidate:
    spec: ComponentSpec
    variant: Optional[Variant]
    score: float
    details: dict
    # best-scoring variant of the same spec whose (normalised) variant differs from this one's:
    # a near-tie here means the pixels cannot tell the variants apart (see `calibrate`)
    alt: Optional["Candidate"] = None

    def summary(self) -> dict:
        return {"name": self.spec.name, "key": self.spec.key, "variant": dict(self.variant.props) if self.variant else {},
                "score": round(self.score, 3)}


def _variant_key(c: Candidate, ds: DesignSystem) -> tuple:
    return tuple(sorted(ds.normalize_variant(c.spec.name, dict(c.variant.props) if c.variant else {}).items()))


def match_node(node: Node, ds: DesignSystem, top: int = 3) -> list[Candidate]:
    """Score `node` against the whole catalog; best candidate per spec, sorted desc. Each
    candidate's `alt` is the best different variant of the same spec (for calibration)."""
    cands: list[Candidate] = []
    feat_cache: dict[str, NodeFeatures] = {}
    for spec in ds.components:
        scored: list[Candidate] = []
        for variant, sig in spec.effective_signatures():
            order = sig.order or "row"
            f = feat_cache.get(order)
            if f is None:
                f = feat_cache[order] = features(node, order)
            s, det = score_signature(f, sig, ds)
            scored.append(Candidate(spec, variant, s, det))
        if not scored:
            continue
        best = scored[0]
        for c in scored[1:]:
            if c.score > best.score:
                best = c
        if best.score <= 0:
            continue
        key = _variant_key(best, ds)
        others = [c for c in scored if c is not best and c.score > 0 and _variant_key(c, ds) != key]
        best.alt = max(others, key=lambda c: c.score) if others else None
        cands.append(best)
    cands.sort(key=lambda c: -c.score)
    return cands[:top]


def calibrate(node: Node, cand: Candidate, cands: list[Candidate]) -> tuple[float, dict]:
    """Confidence of assigning `cand` to `node` (<= its score) and the evidence behind it.

    * **margin**: the score gap to the runner-up, i.e. the best *different* answer (another
      component, or another variant of the same one). Below ``map.conf.margin_full`` the
      confidence shrinks linearly to ``map.conf.tie_factor`` x score at a tie.
    * **observed evidence**: on an unpainted frame inferred by perceive (``meta.src`` in
      ``map.conf.inferred_src``) the ``map.conf.imposed`` constraints hold by construction, so
      the confidence is capped by the weighted score of the remaining constraints.
    """
    others = [c for c in cands if c is not cand and c.spec.key != cand.spec.key]
    pool = others[:1] + ([cand.alt] if cand.alt is not None else [])
    ru = max(pool, key=lambda c: c.score) if pool else None
    margin = cand.score - ru.score if ru is not None else cand.score
    full = max(float(P["map.conf.margin_full"]), 1e-6)
    tie = float(P["map.conf.tie_factor"])
    factor = tie + (1.0 - tie) * min(1.0, max(0.0, margin) / full)
    info: dict[str, Any] = {"margin": round(margin, 3)}
    if ru is not None:
        info["runner_up"] = ru.summary()
    base = cand.score
    inferred = {s.strip() for s in str(P["map.conf.inferred_src"]).split(",") if s.strip()}
    if node.meta.get("src") in inferred and not (node.fills or node.strokes or node.effects):
        imposed = {s.strip() for s in str(P["map.conf.imposed"]).split(",") if s.strip()}
        crit = {s.strip() for s in str(P["map.conf.inferred_critical"]).split(",") if s.strip()} | set(CRITICAL)
        obs = [(k, d["weight"], d["score"]) for k, d in cand.details.items()
               if isinstance(d, dict) and not k.startswith("_") and k not in imposed and "weight" in d]
        tw = sum(w for _, w, _ in obs)
        if tw > 0:
            o = sum(w * s for _, w, s in obs) / tw
            o *= float(P["map.zero_penalty"]) ** sum(1 for k, w, s in obs if s <= 0 and w > 0 and k in crit)
            info["observed_score"] = round(o, 3)
            base = min(base, o)
    return round(base * factor, 3), info


# --------------------------------------------------------------------------- props from slots
def _slot_fill(slots: list[Slot], f: NodeFeatures) -> dict[str, Any]:
    """Assign atoms to named slots in order (texts to text slots, icons to icon slots, ...)."""
    props: dict[str, Any] = {}
    used: set[str] = set()
    for slot in slots:
        vals: list[Any] = []
        for a in f.atoms:
            if a.id in used or not _ptype_matches(a, slot.type):
                continue
            if a.type == "text":
                vals.append(a.text)
            elif a.type == "icon":
                vals.append(a.icon_name or "")
            elif a.component is not None:
                vals.append({"instance": a.component.name, "variant": a.component.variant, "props": a.component.props})
            else:
                vals.append({"node": a.id, "type": a.type})
            used.add(a.id)
            if not slot.multiple:
                break
        if vals:
            props[slot.name] = vals if slot.multiple else vals[0]
    if "label" not in props and f.texts and not any(s.name in ("headline", "title", "message", "value") for s in slots):
        props.setdefault("label", f.primary_text.text if f.primary_text else f.texts[0].text)
    if f.texts:
        props["texts"] = [t.text for t in f.texts]
    if f.icons:
        props["icons"] = [i.icon_name or "" for i in f.icons]
    return props


# --------------------------------------------------------------------------- token snapping
def nearest_color_token(c: Color, ds: DesignSystem, categories: tuple[str, ...] = ("color",), exclude_ref: bool = True) -> tuple[Optional[str], float]:
    """Nearest color token by CIE76 ΔE; exact ties (many roles share a hex) are broken by
    `ds.meta['color_priority']` (list of short names, most common roles first)."""
    best, best_de, best_p = None, math.inf, math.inf
    for t in ds.tokens:
        if t.category not in categories or (exclude_ref and t.meta.get("ref")):
            continue
        ref = t.color()
        if ref is None:
            continue
        de = c.delta_e(ref)
        p = _priority(t.name, ds, "color_priority")
        if de < best_de - _TIE or (abs(de - best_de) <= _TIE and p < best_p):
            best, best_de, best_p = t.name, de, p
    return best, best_de


def snap_color(c: Optional[Color], ds: DesignSystem) -> tuple[Optional[str], dict]:
    if c is None or c.a <= 0.01:
        return None, {}
    name, de = nearest_color_token(c, ds)
    if name is None:
        return None, {}
    ev = {"observed": c.hex(), "nearest": name, "de": round(de, 2)}
    return (name if de <= P["map.color_de"] else None), ev


def snap_radius(r: float, w: float, h: float, ds: DesignSystem) -> tuple[Optional[str], dict]:
    short = max(1.0, min(w, h))
    shapes = ds.shape_scale()
    if not shapes:
        return None, {}
    ev: dict = {"observed": r}
    if r >= P["map.pill_ratio"] * short:
        full = next((k for k, v in shapes.items() if v == "full"), None)
        if full:
            ev["pill"] = True
            return full, ev
    best, best_d = None, math.inf
    for k, v in shapes.items():
        if v == "full":
            continue
        d = abs(float(v) - r)
        if d < best_d:
            best, best_d = k, d
    ev["dist"] = best_d
    return (best if best_d <= P["map.radius_tol"] else None), ev


def snap_text(ts: TextStyle, ds: DesignSystem) -> tuple[Optional[str], dict]:
    name, d = nearest_typescale(ts, ds)
    if name is None:
        return None, {}
    ev = {"observed": {"size": ts.size, "weight": ts.weight, "line_height": ts.line_height}, "nearest": name, "dist": round(d, 2)}
    return (name if d <= P["map.type.max_dist"] else None), ev


def snap_spacing(v: float, ds: DesignSystem) -> tuple[Optional[str], dict]:
    grid = ds.spacing_grid() or P["map.grid"]
    snapped = round(v / grid) * grid
    ev = {"observed": v, "grid": grid, "snapped": snapped}
    if abs(snapped - v) > P["map.grid_tol"]:
        return None, ev
    tok = next((t.name for t in ds.tokens_by_category("spacing") if float(t.value) == snapped), None)
    return (tok or f"spacing.{int(snapped)}"), ev


def snap_elevation(node: Node, ds: DesignSystem) -> tuple[Optional[str], dict]:
    if not node.effects:
        return None, {}
    e = max(node.effects, key=lambda s: s.blur + s.dy)
    best, best_d = None, math.inf
    for t in ds.tokens_by_category("elevation"):
        v = t.value if isinstance(t.value, dict) else {"dy": 0, "blur": 0}
        d = abs(float(v.get("dy", 0)) - e.dy) + abs(float(v.get("blur", 0)) - e.blur)
        if d < best_d:
            best, best_d = t.name, d
    ev = {"observed": {"dy": e.dy, "blur": e.blur}, "nearest": best, "dist": round(best_d, 2)}
    return (best if best_d <= P["map.elevation_tol"] else None), ev


def snap_node_tokens(node: Node, ds: DesignSystem) -> None:
    """Fill `node.tokens` / `node.meta['tokens_evidence']` for one node."""
    ev: dict = {}
    c = node.fill_color
    if c is not None:
        tok, e = snap_color(c, ds)
        ev["fill"] = e
        if tok:
            node.tokens["fill"] = tok
    if node.strokes:
        tok, e = snap_color(node.strokes[0].color, ds)
        ev["stroke"] = e
        if tok:
            node.tokens["stroke"] = tok
    if node.type == "text" and node.text_style:
        tok, e = snap_color(node.text_style.color, ds)
        ev["text_color"] = e
        if tok:
            node.tokens["text_color"] = tok
        tok, e = snap_text(node.text_style, ds)
        ev["typescale"] = e
        if tok:
            node.tokens["typescale"] = tok
    if node.type != "text" and (any(node.radius) or node.type == "ellipse"):
        r = min(node.box.w, node.box.h) / 2 if node.type == "ellipse" else node.uniform_radius()
        tok, e = snap_radius(r, node.box.w, node.box.h, ds)
        ev["shape"] = e
        if tok:
            node.tokens["shape"] = tok
    if node.effects:
        tok, e = snap_elevation(node, ds)
        ev["elevation"] = e
        if tok:
            node.tokens["elevation"] = tok
    if node.layout and node.layout.mode != "none":
        tok, e = snap_spacing(node.layout.gap, ds)
        ev["gap"] = e
        if tok:
            node.tokens["gap"] = tok
        for i, side in enumerate(("top", "right", "bottom", "left")):
            tok, e = snap_spacing(node.layout.padding[i], ds)
            ev[f"padding_{side}"] = e
            if tok:
                node.tokens[f"padding_{side}"] = tok
    if ev:
        node.meta["tokens_evidence"] = ev


# --------------------------------------------------------------------------- learned rules (user-informed tuning)
# A learned rule records that nodes with a given *signature* (categorical features + an observed size
# range) were corrected to a component by people using the engine (dt/feedback, docs/FEEDBACK.md).
# Rules are data, not code: map_document consults them as one more piece of evidence, never as an
# unconditional override (see `_apply_learned_rule`).
register("map.learned_rules", "global,local",
         "comma list of learned matcher-rule files map_document applies: 'global' = knowledge/learned_rules.json "
         "(shipped; promoted from contributed feedback through the bench gate), 'local' = $DT_HOME/learned_rules.json "
         "(this user's `dt feedback learn`), or explicit paths; '' disables learned rules", None)
register("map.learned.h_tol", 6.0, "px a node's height may lie outside a learned rule's observed height range and still match it",
         (0.0, 16.0))
register("map.learned.aspect_tol", 0.35, "relative slack on a learned rule's observed aspect (w/h) range", (0.0, 1.0))
register("map.learned.boost", 0.25,
         "score added to the rule's component candidate, times the rule confidence (boost mode: the engine still decides)",
         (0.0, 1.0))
register("map.learned.override_conf", 0.75,
         "rule confidence at or above which the rule's component is assigned even below map.min_conf (override mode)",
         (0.5, 1.0))
register("map.learned.veto_margin", 0.4,
         "evidence against a rule = best score of a *different* component minus the rule component's own signature "
         "score; above this margin the rule is vetoed and the engine's match stands", (0.0, 1.0))
register("map.learned.veto_score", 0.85,
         "a 'not a component' rule is vetoed when the engine's calibrated confidence in its best match (score, margin "
         "to the runner-up and observed rather than imposed evidence, see calibrate) is at least this", (0.6, 1.0))
register("map.learned.conf_cap", 0.9, "highest ComponentRef.confidence a learned rule can grant", (0.5, 1.0))

LEARNED_RULES_SCHEMA = "rsdesign.learned-rules/1"
_RULES_CACHE: dict[tuple, dict[str, list[dict]]] = {}


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def learned_rule_paths(spec: Optional[str] = None) -> list[str]:
    """Rule files named by ``P['map.learned_rules']`` (or ``spec``), in priority order."""
    spec = str(P["map.learned_rules"] if spec is None else spec)
    out: list[str] = []
    for tok in (t.strip() for t in spec.split(",")):
        if not tok:
            continue
        if tok == "global":
            out.append(os.path.join(_repo_root(), "knowledge", "learned_rules.json"))
        elif tok == "local":
            home = os.environ.get("DT_HOME") or os.path.join(os.path.expanduser("~"), ".rsdesign")
            out.append(os.path.join(home, "learned_rules.json"))
        else:
            out.append(os.path.abspath(os.path.expanduser(tok)))
    return out


def load_learned_rules(spec: Optional[str] = None) -> dict[str, list[dict]]:
    """Active rules indexed by signature hash (cached per file mtime). Missing / unreadable files are skipped."""
    stamp = []
    for p in learned_rule_paths(spec):
        try:
            stamp.append((p, os.path.getmtime(p), os.path.getsize(p)))
        except OSError:
            stamp.append((p, None, None))
    key = tuple(stamp)
    if key in _RULES_CACHE:
        return _RULES_CACHE[key]
    glob_path = os.path.join(_repo_root(), "knowledge", "learned_rules.json")
    index: dict[str, list[dict]] = {}
    for p, mt, _sz in stamp:
        if mt is None:
            continue
        try:
            with open(p) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        for r in data.get("rules", []) if isinstance(data, dict) else []:
            if isinstance(r, dict) and r.get("active") and r.get("signature_hash"):
                r = dict(r)
                r.setdefault("scope", "global" if os.path.abspath(p) == glob_path else "local")
                index.setdefault(r["signature_hash"], []).append(r)
    if len(_RULES_CACHE) > 32:
        _RULES_CACHE.clear()
    _RULES_CACHE[key] = index
    return index


register("map.learned.neutral_chroma", 12.0,
         "Lab chroma below which a colour counts as neutral in a learned signature (colour classes, not roles: the "
         "same outline is read as `outline` on one screen and `on-surface-variant` on another)", (4.0, 30.0))
register("map.learned.light_l", 75.0, "Lab L at or above which a fill counts as light in a learned signature", (50.0, 95.0))
register("map.learned.dark_l", 35.0, "Lab L below which a neutral fill counts as dark in a learned signature", (10.0, 50.0))


def _color_class(c: Optional[Color], fine: bool = True) -> str:
    """Coarse, perception-robust colour class: none | neutral[-light|-mid|-dark] | accent[-light]. Strokes use
    ``fine=False`` (thin anti-aliased lines: only none / neutral / accent survive re-perception)."""
    if c is None or c.a <= 0.01:
        return "none"
    L, a, b = c.lab()
    if math.hypot(a, b) < float(P["map.learned.neutral_chroma"]):
        if not fine:
            return "neutral"
        return "neutral-light" if L >= float(P["map.learned.light_l"]) else ("neutral-dark" if L < float(P["map.learned.dark_l"]) else "neutral-mid")
    if not fine:
        return "accent"
    return "accent-light" if L >= float(P["map.learned.light_l"]) else "accent"


def _atom_kind(a: Node) -> str:
    if a.component is not None:
        return "instance:" + a.component.name
    return a.meta.get("orig_type", a.type) if a.type == "instance" else a.type


def learned_signature(node: Node, ds: DesignSystem) -> tuple[str, dict]:
    """``(signature_hash, features)`` of a container node for learned rules.

    The hash covers only categorical features that survive re-perception of the same design (IR type,
    corner class, coarse fill / stroke / label-ink colour classes, shadow, the ordered kinds of its content atoms
    with repeated runs collapsed). Size stays out of the hash and is matched with a tolerance (``map.learned.h_tol``,
    ``map.learned.aspect_tol``), so a rule learned on an "Undo" button also covers a "Calendar" button
    of another width, and not a 72 px list row with the same content kinds."""
    f = features(node, "row")
    kinds: list[str] = []
    for a in f.atoms:
        k = _atom_kind(a)
        if kinds and kinds[-1].rstrip("+") == k:
            kinds[-1] = k + "+"
        else:
            kinds.append(k)
    cat = {
        "type": node.meta.get("orig_type", node.type) if node.type == "instance" else node.type,
        "radius": "pill" if f.is_pill else ("none" if f.radius <= P["map.radius_tol"] else "rounded"),
        "fill": _color_class(f.fill),
        "stroke": _color_class(f.stroke_color, fine=False) if f.stroke_width > 0 else "none",
        "shadow": bool(f.has_shadow),
        "atoms": kinds,
        # label ink: what tells a text button (accent label) from a navigation link (neutral label)
        "ink": _color_class(f.primary_text.text_style.color, fine=False)
        if f.primary_text is not None and f.primary_text.text_style is not None else "none",
    }
    h = hashlib.sha1(json.dumps(cat, sort_keys=True).encode()).hexdigest()[:16]
    return h, {**cat, "h": round(f.h, 1), "w": round(f.w, 1), "aspect": round(f.aspect, 3)}


def find_learned_rule(node: Node, ds: DesignSystem, index: dict[str, list[dict]]) -> Optional[dict]:
    """The best active rule for ``node``: same signature hash, height and aspect within tolerance of the
    range the rule was learned on; ties go to higher confidence, then support, then local scope."""
    if not index:
        return None
    h, feat = learned_signature(node, ds)
    rules = index.get(h)
    if not rules:
        return None
    htol, atol = float(P["map.learned.h_tol"]), float(P["map.learned.aspect_tol"])
    best, best_key = None, None
    for r in rules:
        rf = r.get("features") or {}
        lo, hi = rf.get("h_range") or [rf.get("h", feat["h"])] * 2
        if not (float(lo) - htol <= feat["h"] <= float(hi) + htol):
            continue
        alo, ahi = rf.get("aspect_range") or [rf.get("aspect", feat["aspect"])] * 2
        if not (float(alo) * (1 - atol) <= feat["aspect"] <= float(ahi) * (1 + atol)):
            continue
        key = (float(r.get("confidence", 0)), int(r.get("support", 0)), r.get("scope") == "local")
        if best_key is None or key > best_key:
            best, best_key = r, key
    return best


def _rule_candidate(node: Node, rule: dict, ds: DesignSystem) -> Optional[Candidate]:
    """The rule's component as a Candidate scored by its own signature (the rule's variant when the
    catalog has it, else the best-scoring variant); None when the catalog does not know the component."""
    name = str(rule.get("component") or "")
    spec = next((s for s in ds.components if s.name.lower() == name.lower()), None)
    if spec is None:
        return None
    want = ds.normalize_variant(spec.name, dict(rule.get("variant") or {})) if rule.get("variant") else None
    best: Optional[Candidate] = None
    for variant, sig in spec.effective_signatures():
        s, det = score_signature(features(node, sig.order or "row"), sig, ds)
        c = Candidate(spec, variant, s, det)
        if want is not None and ds.normalize_variant(spec.name, dict(variant.props) if variant else {}) == want:
            return c
        if best is None or c.score > best.score:
            best = c
    return best


def _apply_learned_rule(node: Node, rule: dict, cands: list[Candidate], ds: DesignSystem) -> tuple[list[Candidate], dict]:
    """Fold a learned rule into the candidate list. Returns ``(cands, info)``; ``info['mode']`` is one of

    * ``override`` -- rule confidence >= ``map.learned.override_conf``: the rule's component is assigned;
    * ``boost``    -- its candidate gains ``map.learned.boost`` x confidence; the usual threshold decides;
    * ``reject``   -- the rule says "not a component": no match, no decision question;
    * ``vetoed``   -- the engine's evidence against the rule (best *other* component's score minus the
      rule component's own score) exceeds ``map.learned.veto_margin`` (for "not a component": the best
      match's calibrated confidence is at least ``map.learned.veto_score``), or the component cannot have this node type:
      the engine's own match stands;
    * ``unknown_component`` -- the design system has no such component (rule ignored).
    """
    conf = float(rule.get("confidence", 0.0))
    info: dict[str, Any] = {"rule": rule.get("id"), "scope": rule.get("scope"), "support": rule.get("support"),
                            "confidence": round(conf, 3), "component": rule.get("component")}
    veto = float(P["map.learned.veto_margin"])
    if not rule.get("component"):  # "this is not a component"
        against = calibrate(node, cands[0], cands)[0] if cands else 0.0
        info["against"] = round(against, 3)
        if against >= float(P["map.learned.veto_score"]):
            info["mode"] = "vetoed"
            return cands, info
        info["mode"] = "reject"
        return [], info
    rc = _rule_candidate(node, rule, ds)
    if rc is None:
        info["mode"] = "unknown_component"
        return cands, info
    if rc.details.get("types", {}).get("score") == 0.0:  # the component can never be this IR type
        info["mode"], info["against"] = "vetoed", 1.0
        return cands, info
    others = [c for c in cands if c.spec.key != rc.spec.key]
    against = (others[0].score - rc.score) if others else 0.0
    info["against"] = round(against, 3)
    info["signature_score"] = round(rc.score, 3)
    if against > veto:
        info["mode"] = "vetoed"
        return cands, info
    override = conf >= float(P["map.learned.override_conf"])
    score = min(1.0, rc.score + float(P["map.learned.boost"]) * conf)
    if override:  # rank first and clear the assignment threshold
        score = max(score, float(P["map.min_conf"]), (others[0].score + 1e-6) if others else 0.0)
    rc.score = score
    rc.details = {**rc.details, "_learned": dict(info)}
    # runner-up for calibration / the decision queue: the engine's own pick when the rule changed only the variant
    orig = next((c for c in cands if c.spec.key == rc.spec.key), None)
    if orig is not None:
        rc.alt = orig.alt if _variant_key(orig, ds) == _variant_key(rc, ds) else orig
    info["mode"] = "override" if override else "boost"
    return sorted([rc] + others, key=lambda c: -c.score), info


# --------------------------------------------------------------------------- document mapping
def _assign(node: Node, cand: Candidate, cands: list[Candidate], ds: DesignSystem) -> None:
    f = features(node, (cand.variant.signature.order if cand.variant and cand.variant.signature and cand.variant.signature.order else cand.spec.signature.order) or "row")
    conf, cal = calibrate(node, cand, cands)
    summaries = [c.summary() for c in cands]
    alt = cand.alt
    if alt is not None and cand.score - alt.score < float(P["map.conf.margin_full"]):
        # an undecidable variant is a choice the decision queue should offer
        summaries.insert(1, alt.summary())
    node.component = ComponentRef(
        key=cand.spec.key, name=cand.spec.name, library=cand.spec.library or ds.name,
        # exactly the design system's variant vocabulary for this component (docs/VARIANTS.md)
        variant=ds.normalize_variant(cand.spec.name, dict(cand.variant.props) if cand.variant else {}),
        props=_slot_fill(cand.spec.slots, f),
        confidence=conf,
        evidence={"score": round(cand.score, 3), "variant": cand.variant.name if cand.variant else None,
                  "constraints": cand.details, "candidates": summaries, **cal},
    )
    keep = {t.strip() for t in str(P["map.keep_geom_types"]).split(",") if t.strip()}
    if node.type not in keep:
        if node.type != "instance":  # keep the original IR type across remap=True passes
            node.meta["orig_type"] = node.type
        node.type = "instance"
    _refine_text_tokens(f, cand, ds)


def _refine_text_tokens(f: NodeFeatures, cand: Candidate, ds: DesignSystem) -> None:
    """A matched component knows which typescale roles its texts use: when a text child's nearest
    role is tied with one of them, prefer the component's role (e.g. label-large inside a Button)."""
    sig = cand.spec.signature.merged(cand.variant.signature if cand.variant else None)
    if not sig.text_roles:
        return
    scale = ds.typescale()
    for t in f.texts:
        if not t.text_style:
            continue
        _, best_d = nearest_typescale(t.text_style, ds)
        for r in sig.text_roles:
            full = next((k for k in scale if k == r or k.endswith("." + r)), None)
            if full and typescale_distance(t.text_style, scale[full]) <= best_d + _TIE:
                t.tokens["typescale"] = full
                t.meta.setdefault("tokens_evidence", {}).setdefault("typescale", {})["from_component"] = cand.spec.name
                break


def _is_background(node: Node, doc: Document) -> bool:
    return node.box.w >= doc.width * 0.98 and node.box.h >= doc.height * 0.98


def map_document(doc: Document, ds: DesignSystem, collapse: bool = False, remap: bool = False,
                 learned: Optional[dict[str, list[dict]]] = None) -> Document:
    """Match components bottom-up and snap tokens for every node. Mutates and returns `doc`.

    ``learned``: learned-rule index (:func:`load_learned_rules`); default = the files named by
    ``P['map.learned_rules']`` (none exist in a fresh checkout, so mapping is then unchanged)."""
    min_conf, cand_conf = P["map.min_conf"], P["map.candidate_conf"]
    rules = load_learned_rules() if learned is None else learned
    stats = {"matched": 0, "applied": 0, "vetoed": 0}
    for n in doc.walk():
        snap_node_tokens(n, ds)

    def visit(node: Node, is_root: bool) -> None:
        if node.component is not None and not remap:
            return
        for c in node.children:
            visit(c, False)
        if is_root or node.type in ("text", "icon", "image") or _is_background(node, doc):
            return
        cands = match_node(node, ds)
        info: Optional[dict] = None
        rule = find_learned_rule(node, ds, rules) if rules else None
        if rule is not None:
            cands, info = _apply_learned_rule(node, rule, cands, ds)
            node.meta["learned_rule"] = info
            stats["matched"] += 1
            if info["mode"] == "vetoed":
                stats["vetoed"] += 1
            elif info["mode"] != "unknown_component":
                stats["applied"] += 1
        if info is not None and info["mode"] == "reject":
            return
        forced = info is not None and info["mode"] == "override"
        if cands and (cands[0].score >= min_conf or forced):
            _assign(node, cands[0], cands, ds)
            if info is not None and info["mode"] in ("override", "boost") and node.component is not None \
                    and node.component.name.lower() == str(info.get("component", "")).lower():
                cap = min(float(info["confidence"]), float(P["map.learned.conf_cap"]))
                node.component.confidence = round(max(node.component.confidence, cap), 3)
                node.component.evidence["learned_rule"] = info
        elif cands and cands[0].score >= cand_conf:
            node.meta["component_candidates"] = [c.summary() for c in cands]

    visit(doc.root, True)
    if collapse:
        collapse_instances(doc)
    doc.design_system = ds.name
    doc.meta.setdefault("mapping", {})["components"] = sum(1 for n in doc.walk() if n.component is not None)
    if stats["matched"]:
        doc.meta["mapping"]["learned_rules"] = stats
    return doc


def collapse_instances(doc: Document) -> Document:
    """Move children of matched instances into meta['collapsed_children'] (props keep the content)."""
    for n in doc.walk():
        if n.component is not None and n.children:
            n.meta["collapsed_children"] = [c.to_dict() for c in n.children]
            n.children = []
    return doc


def expand_instances(doc: Document) -> Document:
    """Inverse of `collapse_instances`."""
    for n in doc.walk():
        if n.meta.get("collapsed_children") and not n.children:
            n.children = [Node.from_dict(d) for d in n.meta.pop("collapsed_children")]
    return doc


def instances(doc: Document) -> list[Node]:
    return [n for n in doc.walk() if n.component is not None]


# --------------------------------------------------------------------------- signature derivation (ingesters)
def derive_signature(node: Node, ds: Optional[DesignSystem] = None, order: str = "row") -> Signature:
    """Build a signature from a concrete exemplar subtree (used by the Figma / screenshot ingesters).
    Colors become the nearest token of `ds` when within ΔE, else literal hex."""
    f = features(node, order)
    tol = P["map.radius_tol"]

    def color_ref(c: Optional[Color]) -> Optional[str]:
        if c is None:
            return None
        if ds is not None:
            name, de = nearest_color_token(c, ds)
            if name and de <= P["map.derive_color_de"]:
                return name
        return c.hex()

    sig = Signature(types=[f.type if f.type != "instance" else "frame"], height=(max(0.0, f.h - tol), f.h + tol))
    if f.w > 0 and f.h > 0:
        a = f.aspect
        sig.aspect = (round(a * 0.85, 3), round(a * 1.15, 3)) if a < 1.2 else (round(a * 0.6, 3), round(a * 1.6, 3))
    sig.radius = "pill" if f.is_pill else ("none" if f.radius <= tol else round(f.radius, 1))
    sig.fills = [color_ref(f.fill)] if f.fill is not None else ["none"]
    if f.stroke_color is not None and f.stroke_width > 0:
        sig.stroke = {"tokens": [color_ref(f.stroke_color)], "width": (max(0.0, f.stroke_width - 0.5), f.stroke_width + 0.5)}
    else:
        sig.stroke = "none"
    sig.shadow = "required" if f.has_shadow else "none"
    sig.text_count = (len(f.texts), len(f.texts))
    sig.icon_count = (len(f.icons), len(f.icons))
    if f.icons:
        sizes = [_icon_size(i) for i in f.icons]
        sig.icon_size = (min(sizes) - tol, max(sizes) + tol)
    if f.texts and ds is not None and ds.typescale():
        roles = []
        for t in f.texts:
            if t.text_style:
                name, d = nearest_typescale(t.text_style, ds)
                if name and d <= P["map.type.max_dist"] and name not in roles:
                    roles.append(name)
        if roles:
            sig.text_roles = roles
        pt = f.primary_text
        if pt is not None and pt.text_style:
            sig.text_colors = [color_ref(pt.text_style.color)]
    sig.children = [ChildPattern(type=(f"instance:{a.component.name}" if a.component is not None else a.type)) for a in f.atoms]
    sig.order = order
    return sig
