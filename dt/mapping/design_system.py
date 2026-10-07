"""Design-system model: tokens, component specs and *structured* match signatures.

A `DesignSystem` is the thing the matcher (`dt.mapping.matcher`) evaluates an IR `Document`
against.  It is pure data (JSON round-trippable, see `fixtures/design_systems/*.json`) and
comes from one of several ingesters:

    material3.py            -> hand-derived from Material Web's token files
    ingest_figma.py         -> Figma REST JSON (components / styles / variables)
    ingest_library.py       -> W3C DTCG token JSON, Material Theme Builder export
    ingest_screenshots.py   -> learned from reference screenshots (dt.perceive)

Signatures are *constraints*, not prose.  Every field is optional; an unset field is not
evaluated.  The matcher turns each set field into a soft score in [0, 1] and combines them
with per-constraint weights (see `matcher.score_signature`).

Token naming conventions (used by the matcher when resolving short names):
    md.sys.color.primary          category "color"      value "#6750a4"
    md.sys.typescale.label-large  category "typescale"  value {size, line_height, weight, tracking, family}
    md.sys.shape.corner-small     category "shape"      value 8  (px) or "full"
    md.sys.elevation.level1       category "elevation"  value {"level": 1, "dy": 1, "blur": 3}
    md.sys.spacing.8              category "spacing"    value 8
A signature may refer to a color token by its full name, by its last segment ("primary"),
or by a literal hex ("#6750a4").
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from dt.ir import Color, Node

TokenCategory = str  # "color" | "typescale" | "shape" | "elevation" | "spacing" | "font" | "dimension" | "other"

FIXTURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "design_systems"))


# --------------------------------------------------------------------------- tokens
@dataclass
class Token:
    """A named design decision. `value` depends on `category` (see module docstring)."""

    name: str
    category: TokenCategory
    value: Any
    meta: dict = field(default_factory=dict)

    @property
    def short(self) -> str:
        """Last dotted segment, e.g. 'primary' for 'md.sys.color.primary'."""
        return self.name.rsplit(".", 1)[-1]

    def color(self) -> Optional[Color]:
        if self.category != "color":
            return None
        v = self.value
        if isinstance(v, str):
            return Color.from_hex(v)
        if isinstance(v, dict):
            return Color.from_dict(v)
        return None

    def to_dict(self) -> dict:
        return {"name": self.name, "category": self.category, "value": self.value, "meta": self.meta}

    @staticmethod
    def from_dict(d: dict) -> "Token":
        return Token(name=d["name"], category=d.get("category", "other"), value=d.get("value"), meta=dict(d.get("meta", {})))


# --------------------------------------------------------------------------- signatures
@dataclass
class ChildPattern:
    """One element of an ordered children pattern.

    `type` is one of: text, icon, image, rect, ellipse, line, frame, vector, instance,
    "instance:<ComponentName>", "control" (switch/checkbox/radio instance), "graphic"
    (icon|image|ellipse|vector), "any", or a union "a|b".  `max` allows repeats (e.g. up to
    3 text lines); `optional` means the item may be absent.
    """

    type: str
    optional: bool = False
    max: int = 1
    name: str = ""

    def to_dict(self) -> dict:
        d: dict = {"type": self.type}
        if self.optional:
            d["optional"] = True
        if self.max != 1:
            d["max"] = self.max
        if self.name:
            d["name"] = self.name
        return d

    @staticmethod
    def from_dict(d: dict | str) -> "ChildPattern":
        if isinstance(d, str):
            return ChildPattern(type=d)
        return ChildPattern(type=d["type"], optional=bool(d.get("optional", False)), max=int(d.get("max", 1)), name=d.get("name", ""))


RadiusRule = Any  # "pill" | "none" | "any" | float | (lo, hi)
StrokeRule = Any  # "none" | "any" | {"tokens": [...], "width": (lo, hi)}

_SIG_FIELDS = (
    "types", "height", "width", "aspect", "radius", "fills", "stroke", "shadow", "text_roles",
    "text_colors", "text_count", "icon_count", "icon_size", "children", "order", "weights",
    "critical", "cells", "indicator", "title_align",
)


@dataclass
class Signature:
    """Structured constraints a subtree must satisfy to be an instance of a component.

    All fields are optional; `None` means "not evaluated".  Ranges are inclusive (lo, hi).
    """

    types: Optional[list[str]] = None  # allowed IR node types of the container
    height: Optional[tuple[float, float]] = None
    width: Optional[tuple[float, float]] = None
    aspect: Optional[tuple[float, float]] = None  # w / h
    radius: RadiusRule = None  # "pill" | "none" | "any" | px | (lo, hi)
    fills: Optional[list[str]] = None  # color token candidates; "none" = no fill allowed
    stroke: StrokeRule = None  # "none" | "any" | {"tokens": [...], "width": (lo, hi)}
    shadow: Optional[str] = None  # "none" | "required" | "any"
    text_roles: Optional[list[str]] = None  # typescale token candidates for any text
    text_colors: Optional[list[str]] = None  # color token candidates for the primary text
    text_count: Optional[tuple[int, int]] = None
    icon_count: Optional[tuple[int, int]] = None
    icon_size: Optional[tuple[float, float]] = None
    children: Optional[list[ChildPattern]] = None
    order: Optional[str] = None  # "row" | "column" (ordering used for the children pattern)
    weights: Optional[dict[str, float]] = None  # per-constraint weight overrides
    # constraint names (beyond the size/paint ones) whose 0 score multiplies by map.zero_penalty
    critical: Optional[list[str]] = None
    cells: Optional[str] = None  # "equal": content sits in >= 2 equal-width cells spanning the width
    indicator: Optional[tuple[float, float]] = None  # thickness range (px) of a required indicator line/rect
    title_align: Optional[str] = None  # "start" | "center": horizontal placement of the primary text

    def merged(self, override: Optional["Signature"]) -> "Signature":
        """Return a copy with every *set* field of `override` replacing this one's."""
        out = Signature(**{k: getattr(self, k) for k in _SIG_FIELDS})
        if override is None:
            return out
        for k in _SIG_FIELDS:
            v = getattr(override, k)
            if v is not None:
                setattr(out, k, v)
        return out

    def is_empty(self) -> bool:
        return all(getattr(self, k) is None for k in _SIG_FIELDS if k != "weights")

    def to_dict(self) -> dict:
        d: dict = {}
        for k in _SIG_FIELDS:
            v = getattr(self, k)
            if v is None:
                continue
            if k == "children":
                v = [c.to_dict() for c in v]
            elif isinstance(v, tuple):
                v = list(v)
            elif k == "stroke" and isinstance(v, dict):
                v = {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
            d[k] = v
        return d

    @staticmethod
    def from_dict(d: dict | None) -> "Signature":
        if not d:
            return Signature()
        s = Signature()
        for k in _SIG_FIELDS:
            if k not in d or d[k] is None:
                continue
            v = d[k]
            if k == "children":
                v = [ChildPattern.from_dict(c) for c in v]
            elif k in ("height", "width", "aspect", "icon_size", "indicator") and isinstance(v, list):
                v = (float(v[0]), float(v[1]))
            elif k in ("text_count", "icon_count") and isinstance(v, list):
                v = (int(v[0]), int(v[1]))
            elif k == "radius" and isinstance(v, list):
                v = (float(v[0]), float(v[1]))
            elif k == "stroke" and isinstance(v, dict) and isinstance(v.get("width"), list):
                v = dict(v)
                v["width"] = (float(v["width"][0]), float(v["width"][1]))
            setattr(s, k, v)
        return s


# --------------------------------------------------------------------------- components
@dataclass
class Variant:
    """A named variant of a component: `props` is what goes into ComponentRef.variant,
    `signature` is a partial override on top of the component's base signature."""

    name: str
    props: dict[str, str] = field(default_factory=dict)
    signature: Optional[Signature] = None

    def to_dict(self) -> dict:
        return {"name": self.name, "props": self.props, "signature": self.signature.to_dict() if self.signature else {}}

    @staticmethod
    def from_dict(d: dict) -> "Variant":
        return Variant(name=d["name"], props=dict(d.get("props", {})), signature=Signature.from_dict(d.get("signature")) or None)


@dataclass
class Slot:
    """A named content slot filled from matched children (e.g. label <- text)."""

    name: str
    type: str = "text"  # same vocabulary as ChildPattern.type
    multiple: bool = False

    def to_dict(self) -> dict:
        return {"name": self.name, "type": self.type, "multiple": self.multiple}

    @staticmethod
    def from_dict(d: dict | str) -> "Slot":
        if isinstance(d, str):
            return Slot(name=d)
        return Slot(name=d["name"], type=d.get("type", "text"), multiple=bool(d.get("multiple", False)))


@dataclass
class ComponentSpec:
    """A component of a design system with its match signature and variants."""

    key: str  # stable id (Figma component key when known)
    name: str  # human name, e.g. "Button"
    library: str = ""
    variants: list[Variant] = field(default_factory=list)
    signature: Signature = field(default_factory=Signature)
    exemplar: Optional[dict] = None  # IR Node dict (origin at 0,0) of a canonical instance
    slots: list[Slot] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def effective_signatures(self) -> list[tuple[Optional[Variant], Signature]]:
        """Base signature merged with each variant override (or just the base if none)."""
        if not self.variants:
            return [(None, self.signature)]
        return [(v, self.signature.merged(v.signature)) for v in self.variants]

    def exemplar_node(self) -> Optional[Node]:
        return Node.from_dict(self.exemplar) if self.exemplar else None

    def to_dict(self) -> dict:
        return {
            "key": self.key, "name": self.name, "library": self.library,
            "signature": self.signature.to_dict(),
            "variants": [v.to_dict() for v in self.variants],
            "slots": [s.to_dict() for s in self.slots],
            "exemplar": self.exemplar, "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "ComponentSpec":
        return ComponentSpec(
            key=d["key"], name=d.get("name", d["key"]), library=d.get("library", ""),
            variants=[Variant.from_dict(v) for v in d.get("variants", [])],
            signature=Signature.from_dict(d.get("signature")),
            exemplar=d.get("exemplar"), slots=[Slot.from_dict(s) for s in d.get("slots", [])],
            meta=dict(d.get("meta", {})),
        )


# --------------------------------------------------------------------------- design system
@dataclass
class DesignSystem:
    name: str
    tokens: list[Token] = field(default_factory=list)
    components: list[ComponentSpec] = field(default_factory=list)
    fonts: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ---- token access
    def tokens_by_category(self, category: TokenCategory) -> list[Token]:
        return [t for t in self.tokens if t.category == category]

    def token(self, name: str) -> Optional[Token]:
        for t in self.tokens:
            if t.name == name:
                return t
        for t in self.tokens:
            if t.name.endswith("." + name):
                return t
        return None

    def add_token(self, tok: Token, replace: bool = True) -> None:
        for i, t in enumerate(self.tokens):
            if t.name == tok.name:
                if replace:
                    self.tokens[i] = tok
                return
        self.tokens.append(tok)

    def color_roles(self) -> dict[str, Color]:
        """name -> Color for every color token."""
        out: dict[str, Color] = {}
        for t in self.tokens_by_category("color"):
            c = t.color()
            if c is not None:
                out[t.name] = c
        return out

    def resolve_color(self, ref: str) -> Optional[Color]:
        """Token name (full or short) or literal '#hex' -> Color."""
        if ref.startswith("#"):
            return Color.from_hex(ref)
        t = self.token(ref)
        return t.color() if t is not None and t.category == "color" else None

    def typescale(self) -> dict[str, dict]:
        return {t.name: dict(t.value) for t in self.tokens_by_category("typescale") if isinstance(t.value, dict)}

    def shape_scale(self) -> dict[str, Any]:
        return {t.name: t.value for t in self.tokens_by_category("shape")}

    def spacing_grid(self) -> float:
        """Base spacing unit (px). Defaults to 4 when no spacing tokens exist."""
        vals = sorted({float(t.value) for t in self.tokens_by_category("spacing") if isinstance(t.value, (int, float)) and t.value > 0})
        return vals[0] if vals else 4.0

    # ---- component access
    def component(self, key_or_name: str) -> Optional[ComponentSpec]:
        for c in self.components:
            if c.key == key_or_name:
                return c
        for c in self.components:
            if c.name == key_or_name:
                return c
        return None

    def component_names(self) -> list[str]:
        return [c.name for c in self.components]

    # ---- variant vocabulary
    def variant_vocabulary(self) -> dict[str, dict[str, list[str]]]:
        """``{component name: {variant key: [allowed values, default first]}}`` from
        ``meta['variants']`` (empty when the design system does not declare one)."""
        return dict(self.meta.get("variants") or {})

    def normalize_variant(self, name: str, variant: dict) -> dict[str, str]:
        """Project a variant dict onto the declared vocabulary of component ``name``: exactly the
        vocabulary keys, values lower-cased strings, unknown or missing values -> the key's
        default. Components without a declared vocabulary pass through (values as strings)."""
        vocab = self.variant_vocabulary().get(name)
        if vocab is None:
            return {k: str(v) for k, v in variant.items()}
        out: dict[str, str] = {}
        for key, allowed in vocab.items():
            v = variant.get(key)
            v = None if v is None else (("true" if v else "false") if isinstance(v, bool) else str(v).lower())
            out[key] = v if v in allowed else allowed[0]
        return out

    def add_component(self, spec: ComponentSpec, replace: bool = True) -> None:
        for i, c in enumerate(self.components):
            if c.key == spec.key:
                if replace:
                    self.components[i] = spec
                return
        self.components.append(spec)

    # ---- (de)serialisation
    def to_dict(self) -> dict:
        return {
            "name": self.name, "fonts": list(self.fonts), "meta": self.meta,
            "tokens": [t.to_dict() for t in self.tokens],
            "components": [c.to_dict() for c in self.components],
        }

    @staticmethod
    def from_dict(d: dict) -> "DesignSystem":
        return DesignSystem(
            name=d["name"], tokens=[Token.from_dict(t) for t in d.get("tokens", [])],
            components=[ComponentSpec.from_dict(c) for c in d.get("components", [])],
            fonts=list(d.get("fonts", [])), meta=dict(d.get("meta", {})),
        )

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1, sort_keys=False)
        return path

    @staticmethod
    def load(path: str) -> "DesignSystem":
        with open(path) as f:
            return DesignSystem.from_dict(json.load(f))

    @staticmethod
    def load_named(name: str) -> "DesignSystem":
        """Load `fixtures/design_systems/<name>.json`; 'material3' falls back to the code catalog."""
        path = os.path.join(FIXTURES_DIR, f"{name}.json")
        if os.path.exists(path):
            return DesignSystem.load(path)
        if name == "material3":
            from dt.mapping.material3 import material3

            return material3()
        raise FileNotFoundError(path)


def merge_tokens(base: DesignSystem, tokens: Iterable[Token], name: Optional[str] = None) -> DesignSystem:
    """Copy of `base` with `tokens` added/replaced (re-theming a catalog with new colors etc.)."""
    out = DesignSystem.from_dict(base.to_dict())
    if name:
        out.name = name
    for t in tokens:
        out.add_token(t, replace=True)
    return out
