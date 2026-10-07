"""Structural comparison of two IR trees (predicted vs ground truth).

Nodes are matched one-to-one by box IoU with the Hungarian algorithm, gated by type
compatibility (a text node never matches a rectangle, but a detected ``rect`` may match a
ground-truth ``frame``). Metrics are then computed over the matched pairs.

Fairness rules (the metrics must measure what matters, not representation choices):

* **Instances are matched by geometry.** The mapper turns a recognised node into
  ``type="instance"`` and keeps the perceived type in ``meta["orig_type"]``; matching and
  ``type_acc`` use that geometric type (:func:`geom_type`). An instance without ``orig_type``
  may match any non-text shape (frame/rect/line/icon/ellipse/vector/image), so recognising a
  divider as ``Divider`` can never lower node recall.
* **Icons are scored against the ink box too.** When a gt icon/vector carries
  ``meta["ink_box"]`` (tight glyph box, as opposed to the 24 px em box), the pair IoU is the
  max of the IoU against the element box and the IoU against the ink box (:func:`pair_iou`).
* **Text recall is line-fair.** A predicted multi-line paragraph credits every gt line it
  holds: its text is split on newlines and a gt line counts as recovered when one part has
  CER <= ``compare.struct.text_match_cer`` and that part's line band (the paragraph box split
  evenly by its line count) overlaps the gt line vertically (:func:`text_recovered`).
  ``text_cer`` is unchanged (mean CER over one-to-one matched text pairs).

Metrics that are undefined for a pair of documents (e.g. ``text_cer`` when there is no text)
are reported as ``None`` rather than a fake number, so aggregators can skip them.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from dt.ir import Box, Document, Node
from dt.params import P, register

register("compare.struct.iou_thr", 0.5,
         "Minimum box IoU for a pred/gt node pair to count as matched.", (0.2, 0.9))
register("compare.struct.type_penalty", 2.0,
         "Extra assignment cost for type-incompatible pairs (on top of 1-IoU); pairs are "
         "never accepted as matches regardless, this only steers the assignment.", (0.5, 10.0))
register("compare.struct.text_match_cer", 0.2,
         "Max character error rate for a gt text line to count as recovered (text_recall).", (0.0, 0.5))
register("compare.struct.layout_de_norm", 50.0,
         "ΔE that maps to layout_consistency == 0 (mean ΔE between absolute and flex renders).", (10.0, 100.0))
register("compare.struct.line_band_overlap", 0.0,
         "Line-fair text recall: minimum vertical overlap (px) between a gt text line and the "
         "line band of a predicted paragraph part (paragraph box split evenly per line); 0 = any "
         "positive overlap.", (0.0, 8.0))
register("compare.struct.min_paint_alpha", 0.02,
         "Nodes whose cumulative opacity (and text whose colour alpha x opacity) is at or below this paint "
         "nothing and are left out of the structural comparison (with their subtrees).", (0.0, 0.2))

# Which node types may be matched to each other (keyed by the *pred* geometric type). Perception
# can only see geometry, so containers (frame/rect) are interchangeable; icons may be detected as
# vectors; fully rounded containers may be read as ellipses; dividers as thin rects. ``instance``
# (a mapped node without ``meta.orig_type``) is a component of unknown geometry: any shape.
_SHAPES = frozenset({"frame", "rect", "instance", "ellipse", "line", "image", "icon", "vector"})
TYPE_COMPAT: dict[str, frozenset[str]] = {
    "text": frozenset({"text"}),
    "frame": frozenset({"frame", "rect", "instance", "ellipse", "line", "image"}),
    "rect": frozenset({"frame", "rect", "instance", "ellipse", "line", "image"}),
    "instance": _SHAPES,
    "ellipse": frozenset({"ellipse", "frame", "rect", "instance"}),
    "line": frozenset({"line", "frame", "rect", "instance"}),
    "image": frozenset({"image", "frame", "rect", "instance"}),
    "icon": frozenset({"icon", "vector", "instance"}),
    "vector": frozenset({"icon", "vector", "instance"}),
}


def types_compatible(a: str, b: str) -> bool:
    """True when a node of type ``a`` may be matched to a node of type ``b``."""
    return b in TYPE_COMPAT.get(a, frozenset({a}))


def geom_type(n: Node) -> str:
    """Geometric type of a node: ``meta["orig_type"]`` for a mapped instance (the type it had
    before the mapper replaced it), else ``node.type``."""
    if n.type == "instance" and isinstance(n.meta, dict):
        ot = n.meta.get("orig_type")
        if isinstance(ot, str) and ot and ot != "instance":
            return ot
    return n.type


def _as_box(v) -> Optional[Box]:
    """A ``Box`` from a Box / ``{x,y,w,h}`` dict / ``[x,y,w,h]`` list (``None`` if malformed)."""
    if v is None or isinstance(v, Box):
        return v
    try:
        if isinstance(v, dict):
            return Box(float(v["x"]), float(v["y"]), float(v["w"]), float(v["h"]))
        if isinstance(v, (list, tuple)) and len(v) == 4:
            return Box(*(float(x) for x in v))
    except (TypeError, ValueError, KeyError):
        return None
    return None


def pair_iou(p: Node, g: Node) -> float:
    """Box IoU of a pred/gt pair. For a gt icon/vector with ``meta["ink_box"]`` this is the max
    of the IoU against the element (em) box and the IoU against the ink box."""
    iou = p.box.iou(g.box)
    if g.type in ("icon", "vector") and isinstance(g.meta, dict):
        ink = _as_box(g.meta.get("ink_box"))
        if ink is not None and ink.w > 0 and ink.h > 0:
            iou = max(iou, p.box.iou(ink))
    return float(iou)


def _painted_text(n: Node, opacity: float) -> bool:
    """False for a text node whose glyphs cannot be seen (fully transparent colour / opacity)."""
    if n.type != "text":
        return True
    a = n.text_style.color.a if (n.text_style is not None and n.text_style.color is not None) else 1.0
    return a * opacity > float(P["compare.struct.min_paint_alpha"])


def _nodes(x: Document | Node | Iterable[Node]) -> list[Node]:
    """Flatten a Document / subtree / iterable into the list of nodes that can be seen.

    A tree is walked with its paint semantics: a hidden (``visible=False``) or fully transparent
    (cumulative opacity <= ``compare.struct.min_paint_alpha``) node hides its whole subtree, and a
    text node with a transparent colour is not text anyone can read. Otherwise an output could
    claim structure it does not show (e.g. invisible gt-shaped frames + text under a screenshot
    crop would score perfect structure *and* perfect pixels)."""
    if isinstance(x, (Document, Node)):
        min_a = float(P["compare.struct.min_paint_alpha"])
        out: list[Node] = []

        def rec(n: Node, op: float) -> None:
            op = op * float(n.opacity if n.opacity is not None else 1.0)
            if not n.visible or op <= min_a:
                return
            if _painted_text(n, op):
                out.append(n)
            for c in n.children:
                rec(c, op)
        rec(x.root if isinstance(x, Document) else x, 1.0)
        return out
    return [n for n in x if n.visible]


# --------------------------------------------------------------------------- matching
def match_nodes(pred: Document | Node | Iterable[Node], gt: Document | Node | Iterable[Node],
                iou_thr: float | None = None) -> list[tuple[str, str, float]]:
    """One-to-one node matching by box IoU (Hungarian) with type-compatibility gating.

    Returns ``[(pred_id, gt_id, iou), ...]`` for pairs whose geometric types
    (:func:`geom_type`) are compatible and whose :func:`pair_iou` is ``>= iou_thr`` (default
    ``compare.struct.iou_thr``). Invisible nodes are ignored.
    """
    thr = float(P["compare.struct.iou_thr"] if iou_thr is None else iou_thr)
    penalty = float(P["compare.struct.type_penalty"])
    pn, gn = _nodes(pred), _nodes(gt)
    if not pn or not gn:
        return []
    ptypes = [geom_type(p) for p in pn]
    gtypes = [geom_type(g) for g in gn]
    iou = np.zeros((len(pn), len(gn)), dtype=np.float64)
    compat = np.zeros_like(iou, dtype=bool)
    for i, p in enumerate(pn):
        for j, g in enumerate(gn):
            iou[i, j] = pair_iou(p, g)
            compat[i, j] = types_compatible(ptypes[i], gtypes[j])
    cost = 1.0 - iou + np.where(compat, 0.0, penalty)
    rows, cols = linear_sum_assignment(cost)
    out: list[tuple[str, str, float]] = []
    for i, j in zip(rows, cols):
        if compat[i, j] and iou[i, j] >= thr:
            out.append((pn[i].id, gn[j].id, float(iou[i, j])))
    return out


# --------------------------------------------------------------------------- text distance
def levenshtein(a: str, b: str) -> int:
    """Edit distance (insert/delete/substitute, unit costs) between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize_text(s: Optional[str]) -> str:
    """Whitespace-collapsed text used for comparisons (case preserved)."""
    return " ".join((s or "").split())


def text_cer(pred: Optional[str], gt: Optional[str]) -> float:
    """Character error rate of ``pred`` against ``gt`` (0 = identical). Empty gt: 0 if pred empty, else 1."""
    p, g = normalize_text(pred), normalize_text(gt)
    if not g:
        return 0.0 if not p else 1.0
    return min(1.0, levenshtein(p, g) / len(g))


def text_parts(p: Node) -> list[tuple[str, float, float]]:
    """``[(text, y0, y1), ...]``: the whole text of ``p`` over its full box, plus — for a
    multi-line node — each non-empty line with its band (box split evenly by line count)."""
    raw = p.text or ""
    out = [(normalize_text(raw), p.box.y, p.box.y2)]
    lines = raw.split("\n")
    if len(lines) > 1:
        h = p.box.h / len(lines)
        for k, ln in enumerate(lines):
            t = normalize_text(ln)
            if t:
                out.append((t, p.box.y + k * h, p.box.y + (k + 1) * h))
    return out


def text_recovered(g: Node, pred_texts: Iterable[Node], cer_thr: Optional[float] = None) -> bool:
    """Line-fair recovery of gt text line ``g``: some overlapping pred text node has a part
    (whole text or one of its lines, :func:`text_parts`) with CER <= ``cer_thr`` (default
    ``compare.struct.text_match_cer``) whose band overlaps ``g`` vertically by more than
    ``compare.struct.line_band_overlap`` px."""
    thr = float(P["compare.struct.text_match_cer"] if cer_thr is None else cer_thr)
    min_ov = float(P["compare.struct.line_band_overlap"])
    for p in pred_texts:
        if p.box.iou(g.box) <= 0:
            continue
        for t, y0, y1 in text_parts(p):
            if not t:
                continue
            ov = min(y1, g.box.y2) - max(y0, g.box.y)
            if ov > min_ov and text_cer(t, g.text) <= thr:
                return True
    return False


# --------------------------------------------------------------------------- metrics
def _mean(vals: list[float]) -> Optional[float]:
    return float(np.mean(vals)) if vals else None


def _component_match(p: Node, g: Node) -> bool:
    if p.component is None or g.component is None:
        return False
    if p.component.name.strip().lower() != g.component.name.strip().lower():
        return False
    return all(str(p.component.variant.get(k, "")).lower() == str(v).lower() for k, v in g.component.variant.items())


def structural_metrics(pred: Document | Node, gt: Document | Node, iou_thr: float | None = None) -> dict:
    """Compare a predicted IR tree against ground truth.

    Keys (``None`` when undefined for this pair of documents):
      * ``n_pred``, ``n_gt``, ``n_matched``
      * ``node_precision`` = matched / n_pred, ``node_recall`` = matched / n_gt (geometry-fair
        matching, see module docstring)
      * ``mean_iou`` over matched pairs
      * ``color_de``: mean ΔE between solid fill colours of matched pairs that both have one
      * ``text_cer``: mean character error rate over matched text pairs
      * ``text_recall``: fraction of gt text lines recovered (line-fair, :func:`text_recovered`)
        — independent of the one-to-one matching
      * ``radius_mae``: mean |Δ uniform radius| over matched non-text pairs
      * ``type_acc``: fraction of matched pairs with identical geometric type (:func:`geom_type`)
      * ``component_acc``: over gt nodes with a component ref, fraction whose match has the same
        component name and all gt variant values
      * ``token_acc``: over gt (node, prop → token) bindings, fraction reproduced on the match
      * ``matches``: the ``match_nodes`` result
    """
    pn, gn = _nodes(pred), _nodes(gt)
    pmap = {n.id: n for n in pn}
    gmap = {n.id: n for n in gn}
    matches = match_nodes(pn, gn, iou_thr)
    n_m = len(matches)
    pairs = [(pmap[pi], gmap[gi]) for pi, gi, _ in matches]

    color = [p.fill_color.delta_e(g.fill_color) for p, g in pairs
             if p.fill_color is not None and g.fill_color is not None]
    cers = [text_cer(p.text, g.text) for p, g in pairs if g.type == "text" and p.type == "text"]
    radii = [abs(p.uniform_radius() - g.uniform_radius()) for p, g in pairs if g.type != "text"]
    types = [geom_type(p) == geom_type(g) for p, g in pairs]

    gt_texts = [g for g in gn if g.type == "text" and normalize_text(g.text)]
    pred_texts = [p for p in pn if p.type == "text" and normalize_text(p.text)]
    recovered = sum(1 for g in gt_texts if text_recovered(g, pred_texts))

    matched_pred_of = {gi: pmap[pi] for pi, gi, _ in matches}
    gt_comp = [g for g in gn if g.component is not None]
    comp_ok = [1.0 if (g.id in matched_pred_of and _component_match(matched_pred_of[g.id], g)) else 0.0 for g in gt_comp]
    tok_ok: list[float] = []
    for g in gn:
        if not g.tokens:
            continue
        p = matched_pred_of.get(g.id)
        for k, v in g.tokens.items():
            tok_ok.append(1.0 if (p is not None and p.tokens.get(k) == v) else 0.0)

    return {
        "n_pred": len(pn),
        "n_gt": len(gn),
        "n_matched": n_m,
        "node_precision": (n_m / len(pn)) if pn else None,
        "node_recall": (n_m / len(gn)) if gn else None,
        "mean_iou": _mean([iou for _, _, iou in matches]) if gn else None,
        "color_de": _mean(color),
        "text_cer": _mean(cers),
        "text_recall": (recovered / len(gt_texts)) if gt_texts else None,
        "radius_mae": _mean(radii),
        "type_acc": _mean([1.0 if t else 0.0 for t in types]),
        "component_acc": _mean(comp_ok),
        "token_acc": _mean(tok_ok),
        "matches": matches,
    }


# --------------------------------------------------------------------------- layout check
def has_layouts(doc: Document) -> bool:
    """True when any visible node carries an inferred auto-layout (mode != 'none')."""
    return any(n.layout is not None and n.layout.mode != "none" for n in doc.walk() if n.visible)


def layout_consistency(doc: Document, de_norm: float | None = None) -> Optional[float]:
    """How well the inferred auto-layout reproduces the absolute geometry, in 0..1.

    Renders ``doc`` in ``absolute`` and ``flex`` modes and returns
    ``1 - clamp(mean ΔE / de_norm)`` (``de_norm`` defaults to ``compare.struct.layout_de_norm``).
    Returns ``None`` when the document has no layouts (nothing to verify). Requires the renderer.
    """
    if not has_layouts(doc):
        return None
    from dt.compare.pixel import diff_map
    from dt.render.screenshot import render_doc

    norm = float(P["compare.struct.layout_de_norm"] if de_norm is None else de_norm)
    a = render_doc(doc, mode="absolute")
    b = render_doc(doc, mode="flex")
    d = diff_map(a, b)
    return float(1.0 - min(1.0, max(0.0, float(d.mean()) / norm)))
