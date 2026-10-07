"""Learn a DesignSystem from reference screenshots (no catalog needed).

    from_reference_screenshots(paths) -> DesignSystem        # perceive() each image, then learn
    learn_from_documents(docs, name="learned") -> DesignSystem   # pure core, also for GT corpora

Algorithm (all deterministic, parameters under `map.learn.*`):
  1. Candidate subtrees: every non-root container (frame/rect/ellipse/instance) that is painted
     (fill/stroke/shadow) or has children, not a page-sized background, at least
     `map.learn.min_size` px.
  2. Signature vector per candidate: (h, w/h, radius ratio, fill L*a*b*, has_stroke, n_text,
     n_icon, primary text size), each dimension divided by its own tolerance so that a distance
     of 1 means "one tolerance apart".
  3. Greedy threshold clustering (area-descending order, assign to first centroid within
     `map.learn.cluster_thr`, else open a new cluster) followed by one reassignment pass.
  4. Clusters with >= `map.learn.min_members` members become `ComponentSpec`s: the medoid is the
     exemplar, the signature is derived from it and widened to the members' observed ranges.
  5. Tokens by frequency analysis: colors merged within `map.color_de` (role names guessed:
     surface = root background, on-surface = most common text color, primary = most common
     saturated fill), typescale roles from (size, weight) with median line-height, shape radii.
The result plugs into `matcher.map_document` like any other DesignSystem.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from dt.ir import Color, Document, Node
from dt.mapping.design_system import ComponentSpec, DesignSystem, Signature, Slot, Token
from dt.mapping.matcher import NodeFeatures, derive_signature, features
from dt.params import P, register

register("map.learn.cluster_thr", 1.5, "max weighted distance (tolerance units) for a node to join a cluster", (0.5, 4.0))
register("map.learn.min_members", 2, "min cluster size to emit a learned ComponentSpec", (2, 6))
register("map.learn.min_size", 8.0, "px: ignore candidate nodes smaller than this in either dimension", (2.0, 24.0))
register("map.learn.h_tol", 16.0, "px of height that counts as one unit of cluster distance", (4.0, 48.0))
register("map.learn.aspect_tol", 1.0, "aspect-ratio difference that counts as one unit", (0.25, 3.0))
register("map.learn.radius_tol", 0.5, "radius/short-side ratio difference that counts as one unit", (0.1, 1.0))
register("map.learn.de_tol", 10.0, "fill ΔE that counts as one unit", (3.0, 30.0))
register("map.learn.text_size_tol", 4.0, "px of primary text size that counts as one unit", (1.0, 12.0))
register("map.learn.max_colors", 16, "max learned color tokens", (4, 64))
register("map.learn.saturation_min", 0.15, "min HSV saturation for a fill to be a 'primary' candidate", (0.05, 0.5))
register("map.learn.range_pad", 2.0, "px added around observed member ranges in learned signatures", (0.0, 8.0))

PerceiveFn = Callable[[str], Document]


# --------------------------------------------------------------------------- signature vectors
@dataclass
class Candidate:
    doc_index: int
    node: Node
    feat: NodeFeatures
    vec: list[float]


def _lab(c: Optional[Color]) -> tuple[float, float, float]:
    return c.lab() if c is not None else (100.0, 0.0, 0.0)


def signature_vector(f: NodeFeatures) -> list[float]:
    """Dimension-normalised feature vector (see module docstring)."""
    short = max(1.0, min(f.w, f.h))
    L, a, b = _lab(f.fill)
    pt = f.primary_text
    tsize = pt.text_style.size if pt is not None and pt.text_style else 0.0
    return [
        f.h / P["map.learn.h_tol"],
        min(f.aspect, 20.0) / P["map.learn.aspect_tol"],
        min(1.0, f.radius / (short / 2)) / P["map.learn.radius_tol"],
        L / P["map.learn.de_tol"], a / P["map.learn.de_tol"], b / P["map.learn.de_tol"],
        0.0 if f.fill is not None else 1.0,
        1.0 if f.stroke_color is not None else 0.0,
        1.0 if f.has_shadow else 0.0,
        float(len(f.texts)), float(len(f.icons)),
        tsize / P["map.learn.text_size_tol"],
    ]


def _dist(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def _candidates(docs: list[Document]) -> list[Candidate]:
    out: list[Candidate] = []
    min_size = P["map.learn.min_size"]
    for di, doc in enumerate(docs):
        for n in doc.walk():
            if n is doc.root or n.type in ("text", "icon", "image", "line", "vector"):
                continue
            if n.box.w < min_size or n.box.h < min_size:
                continue
            if n.box.w >= doc.width * 0.98 and n.box.h >= doc.height * 0.98:
                continue
            if not (n.fills or n.strokes or n.effects or n.children):
                continue
            f = features(n)
            out.append(Candidate(di, n, f, signature_vector(f)))
    out.sort(key=lambda c: -c.node.box.area)
    return out


# --------------------------------------------------------------------------- clustering
def greedy_cluster(vecs: list[list[float]], thr: float, passes: int = 1) -> list[int]:
    """Greedy threshold clustering + `passes` reassignment rounds. Returns cluster id per vector."""
    labels = [-1] * len(vecs)
    centroids: list[list[float]] = []
    counts: list[int] = []
    for i, v in enumerate(vecs):
        best, best_d = -1, math.inf
        for k, c in enumerate(centroids):
            d = _dist(v, c)
            if d < best_d:
                best, best_d = k, d
        if best >= 0 and best_d <= thr:
            labels[i] = best
            n = counts[best]
            centroids[best] = [(c * n + x) / (n + 1) for c, x in zip(centroids[best], v)]
            counts[best] = n + 1
        else:
            labels[i] = len(centroids)
            centroids.append(list(v))
            counts.append(1)
    for _ in range(passes):
        for i, v in enumerate(vecs):
            best = min(range(len(centroids)), key=lambda k: _dist(v, centroids[k]))
            if _dist(v, centroids[best]) <= thr:
                labels[i] = best
        for k in range(len(centroids)):
            members = [vecs[i] for i in range(len(vecs)) if labels[i] == k]
            if members:
                centroids[k] = [sum(m[j] for m in members) / len(members) for j in range(len(members[0]))]
    return labels


def _medoid(members: list[Candidate]) -> Candidate:
    return min(members, key=lambda m: sum(_dist(m.vec, o.vec) for o in members))


def _widen(sig: Signature, members: list[Candidate]) -> Signature:
    """Replace point ranges derived from the medoid by the members' observed ranges (+ pad)."""
    pad = P["map.learn.range_pad"]
    hs = [m.feat.h for m in members]
    sig.height = (min(hs) - pad, max(hs) + pad)
    asp = [m.feat.aspect for m in members if m.feat.h > 0]
    if asp:
        sig.aspect = (round(min(asp) * 0.9, 3), round(max(asp) * 1.1, 3))
    nt = [len(m.feat.texts) for m in members]
    ni = [len(m.feat.icons) for m in members]
    sig.text_count = (min(nt), max(nt))
    sig.icon_count = (min(ni), max(ni))
    rads = [m.feat.radius for m in members]
    if sig.radius not in ("pill", "none") and isinstance(sig.radius, (int, float)):
        sig.radius = (min(rads) - P["map.radius_tol"], max(rads) + P["map.radius_tol"])
    return sig


# --------------------------------------------------------------------------- tokens by frequency
def _hsv_sat(c: Color) -> float:
    mx, mn = max(c.r, c.g, c.b), min(c.r, c.g, c.b)
    return 0.0 if mx == 0 else (mx - mn) / mx


def _merge_colors(colors: Iterable[tuple[Color, str]]) -> list[tuple[Color, int, dict[str, int]]]:
    """Merge colors within ΔE P['map.color_de']; returns (color, count, {kind: count}) by frequency."""
    de_max = P["map.color_de"]
    groups: list[tuple[Color, int, dict[str, int]]] = []
    for c, kind in colors:
        for i, (gc, n, kinds) in enumerate(groups):
            if c.delta_e(gc) <= de_max:
                kinds[kind] = kinds.get(kind, 0) + 1
                groups[i] = (gc, n + 1, kinds)
                break
        else:
            groups.append((c, 1, {kind: 1}))
    groups.sort(key=lambda g: -g[1])
    return groups


def learn_color_tokens(docs: list[Document], prefix: str = "learned") -> list[Token]:
    samples: list[tuple[Color, str]] = []
    root_bg: Optional[Color] = None
    for doc in docs:
        if root_bg is None:
            root_bg = doc.root.fill_color
        for n in doc.walk():
            c = n.fill_color
            if c is not None and c.a > 0.01 and n.type != "icon":
                samples.append((c, "fill"))
            if n.type == "icon" and c is not None:
                samples.append((c, "icon"))
            for s in n.strokes:
                samples.append((s.color, "stroke"))
            if n.type == "text" and n.text_style:
                samples.append((n.text_style.color, "text"))
    groups = _merge_colors(samples)[: int(P["map.learn.max_colors"])]
    toks: list[Token] = []
    named: set[str] = set()

    def name_for(c: Color, kinds: dict[str, int], idx: int) -> str:
        if root_bg is not None and c.delta_e(root_bg) <= P["map.color_de"] and "surface" not in named:
            named.add("surface")
            return "surface"
        if kinds.get("text", 0) >= max(kinds.get("fill", 0), 1) and "on-surface" not in named and c.luminance() < 0.3:
            named.add("on-surface")
            return "on-surface"
        if kinds.get("fill", 0) > 0 and _hsv_sat(c) >= P["map.learn.saturation_min"] and "primary" not in named:
            named.add("primary")
            return "primary"
        if kinds.get("stroke", 0) >= max(kinds.get("fill", 0), kinds.get("text", 0), 1) and "outline" not in named:
            named.add("outline")
            return "outline"
        return f"color-{idx}"

    for i, (c, n, kinds) in enumerate(groups):
        toks.append(Token(f"{prefix}.color.{name_for(c, kinds, i)}", "color", c.hex(), {"count": n, "kinds": kinds}))
    return toks


def learn_typescale_tokens(docs: list[Document], prefix: str = "learned") -> list[Token]:
    groups: dict[tuple[int, int], list[Node]] = {}
    for doc in docs:
        for n in doc.walk():
            if n.type == "text" and n.text_style:
                groups.setdefault((int(round(n.text_style.size)), int(n.text_style.weight)), []).append(n)
    toks: list[Token] = []
    for (size, weight), nodes in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lhs = [t.text_style.line_height for t in nodes if t.text_style and t.text_style.line_height]
        fam = statistics.mode([t.text_style.family for t in nodes if t.text_style]) if nodes else "Roboto"
        toks.append(Token(f"{prefix}.type.{size}-{weight}", "typescale",
                          {"size": float(size), "weight": weight, "line_height": float(statistics.median(lhs)) if lhs else None, "tracking": 0.0, "family": fam},
                          {"count": len(nodes)}))
    return toks


def learn_shape_tokens(docs: list[Document], prefix: str = "learned") -> list[Token]:
    counts: dict[float, int] = {}
    has_pill = False
    for doc in docs:
        for n in doc.walk():
            if n.type in ("text", "icon") or not any(n.radius):
                continue
            r = n.uniform_radius()
            if r >= P["map.pill_ratio"] * max(1.0, min(n.box.w, n.box.h)):
                has_pill = True
                continue
            r = float(round(r))
            counts[r] = counts.get(r, 0) + 1
    toks = [Token(f"{prefix}.shape.r{int(r)}", "shape", r, {"count": n}) for r, n in sorted(counts.items())]
    toks.append(Token(f"{prefix}.shape.none", "shape", 0, {}))
    if has_pill:
        toks.append(Token(f"{prefix}.shape.full", "shape", "full", {}))
    return toks


# --------------------------------------------------------------------------- entry points
def learn_from_documents(docs: list[Document], name: str = "learned") -> DesignSystem:
    """Pure core: cluster recurring subtrees across IR documents into learned ComponentSpecs + tokens."""
    docs = list(docs)
    tokens = learn_color_tokens(docs, name) + learn_typescale_tokens(docs, name) + learn_shape_tokens(docs, name)
    tokens.append(Token(f"{name}.spacing.{int(P['map.grid'])}", "spacing", float(P["map.grid"])))
    fonts = sorted({n.text_style.family for d in docs for n in d.walk() if n.type == "text" and n.text_style})
    ds = DesignSystem(name=name, tokens=tokens, fonts=fonts, meta={"source": "screenshots", "n_docs": len(docs)})
    cands = _candidates(docs)
    if not cands:
        return ds
    labels = greedy_cluster([c.vec for c in cands], P["map.learn.cluster_thr"])
    clusters: dict[int, list[Candidate]] = {}
    for c, k in zip(cands, labels):
        clusters.setdefault(k, []).append(c)
    idx = 0
    for k in sorted(clusters, key=lambda k: (-len(clusters[k]), k)):
        members = clusters[k]
        if len(members) < int(P["map.learn.min_members"]):
            continue
        med = _medoid(members)
        sig = _widen(derive_signature(med.node, ds), members)
        ex = Node.from_dict(med.node.to_dict())
        _rebase(ex)
        slots = [Slot("label", "text")] if med.feat.texts else []
        if med.feat.icons:
            slots.append(Slot("icon", "icon"))
        ds.add_component(ComponentSpec(
            key=f"{name}.component-{idx}", name=f"Component {idx}", library=name, signature=sig, exemplar=ex.to_dict(), slots=slots,
            meta={"members": len(members), "docs": sorted({m.doc_index for m in members}), "member_ids": [m.node.id for m in members]},
        ))
        idx += 1
    return ds


def _rebase(node: Node) -> None:
    """Translate a subtree so its root box is at the origin (exemplars are origin-relative)."""
    dx, dy = node.box.x, node.box.y
    for n in node.walk():
        n.box = n.box.translate(-dx, -dy)


def from_reference_screenshots(paths: Iterable[str], name: str = "learned", perceive_fn: Optional[PerceiveFn] = None) -> DesignSystem:
    """Perceive each screenshot (dt.perceive.perceive unless `perceive_fn` is given) and learn a DesignSystem."""
    if perceive_fn is None:
        from dt.perceive import perceive  # provided by module A

        perceive_fn = perceive
    docs = [perceive_fn(p) for p in paths]
    ds = learn_from_documents(docs, name)
    ds.meta["sources"] = list(paths)
    return ds
