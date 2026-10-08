"""Versioned taxonomy of render-vs-target discrepancies, and the ``Finding`` an adversary reports.

An *adversary* compares a target screenshot with a candidate render (optionally with the candidate
IR) and reports typed, localised findings. This module is the shared vocabulary that the
perturbation operators (``dt.adversary.perturb``) produce as ground truth and that the scorer
(``dt.adversary.benchmark``) grades against.

Types are two-level, ``<parent>.<leaf>``. Every type records:

* ``definition``   what the discrepancy is, from the point of view of the candidate (our render)
* ``ir_property``  the IR field(s) of ``dt.ir`` that carry the error
* ``refine_kinds`` the ``dt.refine.critic`` hypothesis kinds that typically repair it
* ``magnitude``    the unit of ``Finding.magnitude`` and whether the scorer grades it

The ``noise`` parent is **not an error**: it names nuisance differences (anti-aliasing, sub-pixel
offsets, compression, browser builds) that a good adversary must stay silent on. A finding typed
``noise.*`` is treated as an explicit abstention by the scorer (never a false finding).

Versioning: additive changes (a new leaf) bump the minor version; renaming or re-parenting a leaf
bumps the major version. Benchmarks record the taxonomy version they were built with.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from dt.ir import Box

TAXONOMY_VERSION = "1.0"

# hypothesis kinds produced by dt.refine.critic (kept here as data so the taxonomy has no import-time
# dependency on the refine stage; tests check this list against critic.py)
CRITIC_KINDS = ("shift", "resize", "recolor", "text", "radius", "missing", "extra", "opacity", "shadow", "raster")


@dataclass(frozen=True)
class TypeSpec:
    key: str                      # "<parent>.<leaf>"
    definition: str
    ir_property: str
    refine_kinds: tuple[str, ...]
    magnitude_unit: Optional[str]  # None = magnitude undefined for this type
    magnitude_scored: bool         # graded by the scorer (only when the unit is a continuous quantity)
    is_error: bool = True

    @property
    def parent(self) -> str:
        return self.key.split(".", 1)[0]

    @property
    def leaf(self) -> str:
        return self.key.split(".", 1)[1]


def _t(key, definition, ir_property, refine_kinds, unit, scored, is_error=True) -> TypeSpec:
    return TypeSpec(key, definition, ir_property, tuple(refine_kinds), unit, scored, is_error)


TYPES: dict[str, TypeSpec] = {t.key: t for t in [
    # ---- geometry
    _t("geometry.shift", "a shape, icon or subtree is drawn at the wrong position (whole-pixel offset, size unchanged)",
       "Node.box.x/y (whole subtree)", ["shift"], "px (euclidean offset)", True),
    _t("geometry.size", "a shape has the wrong width and/or height (one or more edges displaced)",
       "Node.box.w/h", ["resize"], "px (largest edge displacement)", True),
    _t("geometry.radius", "a shape's corner radius differs (corner arcs too round or too square)",
       "Node.radius", ["radius"], "px (mean |Δradius| over corners)", True),
    # ---- colour
    _t("color.fill", "a shape fill, icon colour or glyph colour differs",
       "Node.fills[].color / TextStyle.color", ["recolor"], "ΔE76", True),
    _t("color.stroke", "a border/outline colour differs",
       "Node.strokes[].color", ["recolor"], "ΔE76", True),
    _t("color.tint_global", "every colour on the page is offset the same way (theme / colour-space / white-balance error)",
       "Document palette (all fills, strokes and text colours)", ["recolor"], "ΔE76 (mean over colours)", True),
    # ---- structure
    _t("structure.missing", "an element painted in the target is absent from the candidate",
       "Node (subtree absent)", ["missing", "raster"], "px² (area)", False),
    _t("structure.extra", "the candidate paints an element the target does not have",
       "Node (spurious subtree)", ["extra"], "px² (area)", False),
    _t("structure.split", "one target element is represented as two candidate elements (a visible seam/gap appears)",
       "Node (one node became two siblings)", ["extra", "resize"], "px (gap)", True),
    _t("structure.merge", "two target elements are represented as one candidate element (the gap between them is filled/lost)",
       "Node (two siblings became one)", ["missing", "resize"], "px (gap lost)", True),
    # ---- text
    _t("text.content", "the characters of a text run differ (substitution, insertion, deletion)",
       "Node.text", ["text"], "CER (edits / target length)", True),
    _t("text.style", "a text run has the wrong size or weight (glyphs differ in scale or stroke)",
       "Node.text_style.size/weight", ["text"], "style steps (|Δsize| px + |Δweight|/100)", True),
    _t("text.position", "a text run is offset inside an otherwise correct container",
       "Node.box.x/y (text node only)", ["shift", "text"], "px (euclidean offset)", True),
    # ---- icon
    _t("icon.glyph", "an icon shows the wrong glyph (same place, same size, different symbol)",
       "Node.icon_name", ["raster", "missing"], None, False),
    _t("icon.missing", "an icon painted in the target is absent from the candidate",
       "Node (icon node absent)", ["missing"], "px² (area)", False),
    # ---- effect
    _t("effect.shadow", "a drop shadow is missing, spurious or of the wrong elevation",
       "Node.effects", ["shadow"], "px (|Δ(blur + |dy|)| of the largest shadow)", True),
    _t("effect.opacity", "an element is drawn with the wrong opacity (blended with its backdrop)",
       "Node.opacity", ["opacity"], "|Δopacity| (0..1)", True),
    # ---- layout
    _t("layout.spacing", "the gap between siblings of a row/column differs (later siblings drift cumulatively)",
       "Node.layout.gap / sibling Node.box positions", ["shift"], "px (Δgap)", True),
    # ---- noise (NOT errors: a good adversary stays silent)
    _t("noise.antialias", "anti-aliasing / font-smoothing / resampling-blur differences at edges",
       "none (rasteriser)", [], "relative level", False, is_error=False),
    _t("noise.subpixel", "the whole page is offset by a fraction of a pixel (0.25-0.5 px)",
       "none (capture offset)", [], "px", False, is_error=False),
    _t("noise.compression", "lossy (JPEG-like) compression artefacts",
       "none (encoding)", [], "1 - quality/100", False, is_error=False),
    _t("noise.browser", "the target was rasterised by a different browser build",
       "none (rasteriser)", [], None, False, is_error=False),
]}

PARENTS: tuple[str, ...] = tuple(dict.fromkeys(t.parent for t in TYPES.values()))
ERROR_TYPES: tuple[str, ...] = tuple(k for k, t in TYPES.items() if t.is_error)
NOISE_TYPES: tuple[str, ...] = tuple(k for k, t in TYPES.items() if not t.is_error)
# types whose ground truth spans the whole page (localisation uses IoU only, never centre-in-box)
GLOBAL_TYPES: tuple[str, ...] = ("color.tint_global",)


def parent_of(type_key: str) -> str:
    """Parent of ``type_key`` (``"geometry.shift" -> "geometry"``; a bare parent maps to itself)."""
    return (type_key or "").split(".", 1)[0]


def is_noise(type_key: str) -> bool:
    return parent_of(type_key) == "noise"


def is_known(type_key: str) -> bool:
    return type_key in TYPES or type_key in PARENTS


def spec(type_key: str) -> TypeSpec:
    return TYPES[type_key]


def describe() -> dict:
    """The taxonomy as JSON-able data (for reports and for agents choosing what to emit)."""
    return {"version": TAXONOMY_VERSION, "parents": list(PARENTS), "types": {
        k: {"parent": t.parent, "definition": t.definition, "ir_property": t.ir_property,
            "refine_kinds": list(t.refine_kinds), "magnitude_unit": t.magnitude_unit,
            "magnitude_scored": t.magnitude_scored, "is_error": t.is_error} for k, t in TYPES.items()}}


# --------------------------------------------------------------------------- finding
@dataclass
class Finding:
    """One typed, localised discrepancy of the candidate w.r.t. the target.

    ``box``: where the discrepancy is visible, in target pixel space (ground truth uses the union of
    the element's before/after paint extent). ``magnitude``: in the unit of the type (see
    ``TYPES[type].magnitude_unit``), None when unknown/undefined. ``confidence``: 0..1, used to rank
    findings (average precision). ``evidence``: free-form, JSON-able. ``node_id``: the candidate IR
    node at fault, when one exists (None for e.g. a missing element).
    """

    type: str
    box: Box
    magnitude: Optional[float] = None
    confidence: float = 1.0
    evidence: dict = field(default_factory=dict)
    node_id: Optional[str] = None

    @property
    def parent(self) -> str:
        return parent_of(self.type)

    def to_dict(self) -> dict:
        return {"type": self.type, "box": self.box.to_dict(), "magnitude": self.magnitude,
                "confidence": float(self.confidence), "evidence": self.evidence, "node_id": self.node_id}

    @staticmethod
    def from_dict(d: dict) -> "Finding":
        b = d.get("box", {})
        box = Box(*b) if isinstance(b, (list, tuple)) else Box.from_dict(b)
        m = d.get("magnitude")
        return Finding(type=str(d["type"]), box=box, magnitude=None if m is None else float(m),
                       confidence=float(d.get("confidence", 1.0)), evidence=dict(d.get("evidence", {})),
                       node_id=d.get("node_id"))
