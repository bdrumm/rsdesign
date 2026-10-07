"""Material 3 baseline design system, derived from Material Web's token files.

Source of truth: `node_modules/@material/web/tokens/versions/v0_192/` (Material Web 2.5):
  * `_md-ref-palette.scss`   -> REF_PALETTE
  * `_md-sys-color.scss`     -> LIGHT_SCHEME / DARK_SCHEME (role -> palette key)
  * `_md-sys-typescale.scss` -> TYPESCALE (rem values x 16px)
  * `_md-sys-shape.scss`     -> SHAPE
  * `_md-sys-elevation.scss` -> ELEVATION levels (dp)
  * `_md-comp-*.scss`        -> component metrics used in the signatures below
    (container-height, container-shape, container-color, outline-*, icon-size ...)

The values are embedded so the catalog works without node_modules; `verify_against_material_web()`
re-parses the scss files when they are present and reports any drift (used by the tests).
Component metrics were also verified against DOM boxes of `out/mwc_test.html` rendered with
the real Material Web bundle (filled button h=40 pill, FAB 56x56 r=16, switch 52x32,
checkbox 18x18, outlined text field h=56 r=4, chips h=32 r=8).

Run `python -m dt.mapping.material3` to regenerate `fixtures/design_systems/material3.json`.
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional

from dt.ir import Color, ComponentRef
from dt.mapping.design_system import (
    ChildPattern, ComponentSpec, DesignSystem, Signature, Slot, Token, Variant, FIXTURES_DIR,
)

LIBRARY = "material3"
MW_TOKENS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "node_modules", "@material", "web", "tokens", "versions", "v0_192"))

# --------------------------------------------------------------------------- md.ref.palette
REF_PALETTE: dict[str, str] = {
    "black": "#000000", "white": "#ffffff",
    "primary0": "#000000", "primary10": "#21005d", "primary20": "#381e72", "primary30": "#4f378b",
    "primary40": "#6750a4", "primary50": "#7f67be", "primary60": "#9a82db", "primary70": "#b69df8",
    "primary80": "#d0bcff", "primary90": "#eaddff", "primary95": "#f6edff", "primary99": "#fffbfe", "primary100": "#ffffff",
    "secondary0": "#000000", "secondary10": "#1d192b", "secondary20": "#332d41", "secondary30": "#4a4458",
    "secondary40": "#625b71", "secondary50": "#7a7289", "secondary60": "#958da5", "secondary70": "#b0a7c0",
    "secondary80": "#ccc2dc", "secondary90": "#e8def8", "secondary95": "#f6edff", "secondary99": "#fffbfe", "secondary100": "#ffffff",
    "tertiary0": "#000000", "tertiary10": "#31111d", "tertiary20": "#492532", "tertiary30": "#633b48",
    "tertiary40": "#7d5260", "tertiary50": "#986977", "tertiary60": "#b58392", "tertiary70": "#d29dac",
    "tertiary80": "#efb8c8", "tertiary90": "#ffd8e4", "tertiary95": "#ffecf1", "tertiary99": "#fffbfa", "tertiary100": "#ffffff",
    "error0": "#000000", "error10": "#410e0b", "error20": "#601410", "error30": "#8c1d18", "error40": "#b3261e",
    "error50": "#dc362e", "error60": "#e46962", "error70": "#ec928e", "error80": "#f2b8b5", "error90": "#f9dedc",
    "error95": "#fceeee", "error99": "#fffbf9", "error100": "#ffffff",
    "neutral0": "#000000", "neutral4": "#0f0d13", "neutral6": "#141218", "neutral10": "#1d1b20", "neutral12": "#211f26",
    "neutral17": "#2b2930", "neutral20": "#322f35", "neutral22": "#36343b", "neutral24": "#3b383e", "neutral30": "#48464c",
    "neutral40": "#605d64", "neutral50": "#79767d", "neutral60": "#938f96", "neutral70": "#aea9b1", "neutral80": "#cac5cd",
    "neutral87": "#ded8e1", "neutral90": "#e6e0e9", "neutral92": "#ece6f0", "neutral94": "#f3edf7", "neutral95": "#f5eff7",
    "neutral96": "#f7f2fa", "neutral98": "#fef7ff", "neutral99": "#fffbff", "neutral100": "#ffffff",
    "neutral-variant0": "#000000", "neutral-variant10": "#1d1a22", "neutral-variant20": "#322f37",
    "neutral-variant30": "#49454f", "neutral-variant40": "#605d66", "neutral-variant50": "#79747e",
    "neutral-variant60": "#938f99", "neutral-variant70": "#aea9b4", "neutral-variant80": "#cac4d0",
    "neutral-variant90": "#e7e0ec", "neutral-variant95": "#f5eefa", "neutral-variant99": "#fffbfe", "neutral-variant100": "#ffffff",
}

# md.sys.color role -> md.ref.palette key (from _md-sys-color.scss values-light / values-dark)
LIGHT_SCHEME: dict[str, str] = {
    "background": "neutral98", "error": "error40", "error-container": "error90",
    "inverse-on-surface": "neutral95", "inverse-primary": "primary80", "inverse-surface": "neutral20",
    "on-background": "neutral10", "on-error": "error100", "on-error-container": "error10",
    "on-primary": "primary100", "on-primary-container": "primary10", "on-primary-fixed": "primary10",
    "on-primary-fixed-variant": "primary30", "on-secondary": "secondary100", "on-secondary-container": "secondary10",
    "on-secondary-fixed": "secondary10", "on-secondary-fixed-variant": "secondary30", "on-surface": "neutral10",
    "on-surface-variant": "neutral-variant30", "on-tertiary": "tertiary100", "on-tertiary-container": "tertiary10",
    "on-tertiary-fixed": "tertiary10", "on-tertiary-fixed-variant": "tertiary30", "outline": "neutral-variant50",
    "outline-variant": "neutral-variant80", "primary": "primary40", "primary-container": "primary90",
    "primary-fixed": "primary90", "primary-fixed-dim": "primary80", "scrim": "neutral0", "secondary": "secondary40",
    "secondary-container": "secondary90", "secondary-fixed": "secondary90", "secondary-fixed-dim": "secondary80",
    "shadow": "neutral0", "surface": "neutral98", "surface-bright": "neutral98", "surface-container": "neutral94",
    "surface-container-high": "neutral92", "surface-container-highest": "neutral90", "surface-container-low": "neutral96",
    "surface-container-lowest": "neutral100", "surface-dim": "neutral87", "surface-tint": "primary40",
    "surface-variant": "neutral-variant90", "tertiary": "tertiary40", "tertiary-container": "tertiary90",
    "tertiary-fixed": "tertiary90", "tertiary-fixed-dim": "tertiary80",
}
DARK_SCHEME: dict[str, str] = {
    "background": "neutral6", "error": "error80", "error-container": "error30", "inverse-on-surface": "neutral20",
    "inverse-primary": "primary40", "inverse-surface": "neutral90", "on-background": "neutral90", "on-error": "error20",
    "on-error-container": "error90", "on-primary": "primary20", "on-primary-container": "primary90",
    "on-primary-fixed": "primary10", "on-primary-fixed-variant": "primary30", "on-secondary": "secondary20",
    "on-secondary-container": "secondary90", "on-secondary-fixed": "secondary10", "on-secondary-fixed-variant": "secondary30",
    "on-surface": "neutral90", "on-surface-variant": "neutral-variant80", "on-tertiary": "tertiary20",
    "on-tertiary-container": "tertiary90", "on-tertiary-fixed": "tertiary10", "on-tertiary-fixed-variant": "tertiary30",
    "outline": "neutral-variant60", "outline-variant": "neutral-variant30", "primary": "primary80",
    "primary-container": "primary30", "primary-fixed": "primary90", "primary-fixed-dim": "primary80", "scrim": "neutral0",
    "secondary": "secondary80", "secondary-container": "secondary30", "secondary-fixed": "secondary90",
    "secondary-fixed-dim": "secondary80", "shadow": "neutral0", "surface": "neutral6", "surface-bright": "neutral24",
    "surface-container": "neutral12", "surface-container-high": "neutral17", "surface-container-highest": "neutral22",
    "surface-container-low": "neutral10", "surface-container-lowest": "neutral4", "surface-dim": "neutral6",
    "surface-tint": "primary80", "surface-variant": "neutral-variant30", "tertiary": "tertiary80",
    "tertiary-container": "tertiary30", "tertiary-fixed": "tertiary90", "tertiary-fixed-dim": "tertiary80",
}

# md.sys.typescale: role -> (size px, line-height px, weight, tracking px, typeface)
TYPESCALE: dict[str, tuple[float, float, int, float, str]] = {
    "display-large": (57, 64, 400, -0.25, "brand"), "display-medium": (45, 52, 400, 0.0, "brand"),
    "display-small": (36, 44, 400, 0.0, "brand"), "headline-large": (32, 40, 400, 0.0, "brand"),
    "headline-medium": (28, 36, 400, 0.0, "brand"), "headline-small": (24, 32, 400, 0.0, "brand"),
    "title-large": (22, 28, 400, 0.0, "brand"), "title-medium": (16, 24, 500, 0.15, "plain"),
    "title-small": (14, 20, 500, 0.1, "plain"), "body-large": (16, 24, 400, 0.5, "plain"),
    "body-medium": (14, 20, 400, 0.25, "plain"), "body-small": (12, 16, 400, 0.4, "plain"),
    "label-large": (14, 20, 500, 0.1, "plain"), "label-medium": (12, 16, 500, 0.5, "plain"),
    "label-small": (11, 16, 500, 0.5, "plain"),
}
TYPEFACE = {"brand": "Roboto", "plain": "Roboto", "weight-regular": 400, "weight-medium": 500, "weight-bold": 700}

# md.sys.shape: corner-* -> px ("full" = 9999px pill)
SHAPE: dict[str, float | str] = {
    "corner-none": 0, "corner-extra-small": 4, "corner-small": 8, "corner-medium": 12,
    "corner-large": 16, "corner-extra-large": 28, "corner-full": "full",
}
# md.sys.elevation levels (dp) and the umbra shadow Material Web renders for each level
ELEVATION: dict[str, dict] = {
    "level0": {"level": 0, "dp": 0, "dy": 0, "blur": 0},
    "level1": {"level": 1, "dp": 1, "dy": 1, "blur": 3},
    "level2": {"level": 2, "dp": 3, "dy": 1, "blur": 5},
    "level3": {"level": 3, "dp": 6, "dy": 2, "blur": 10},
    "level4": {"level": 4, "dp": 8, "dy": 3, "blur": 14},
    "level5": {"level": 5, "dp": 12, "dy": 5, "blur": 22},
}
SPACING_GRID = 4
SPACING_STEPS = tuple(range(0, 65, 4))
# Tie-break order when several roles share a hex (e.g. on-primary/on-error/#fff): most common UI role first.
COLOR_PRIORITY = [
    "surface", "on-surface", "primary", "on-primary", "on-surface-variant", "primary-container", "on-primary-container",
    "secondary-container", "on-secondary-container", "surface-container-low", "surface-container", "surface-container-high",
    "surface-container-highest", "surface-container-lowest", "outline", "outline-variant", "secondary", "on-secondary",
    "tertiary", "tertiary-container", "error", "on-error", "error-container", "inverse-surface", "inverse-on-surface",
    "surface-variant", "background", "on-background", "surface-bright", "surface-dim", "surface-tint", "inverse-primary",
    "scrim", "shadow",
]
TYPESCALE_PRIORITY = ["body-large", "body-medium", "label-large", "title-medium", "body-small", "label-medium", "title-large",
                      "title-small", "label-small", "headline-small", "headline-medium", "headline-large",
                      "display-small", "display-medium", "display-large"]


# --------------------------------------------------------------------------- variant vocabulary
#: The ONE variant vocabulary per component (documented in docs/VARIANTS.md). Every
#: ``ComponentRef.variant`` emitted by the matcher (``dt.mapping.matcher``), the Material Web
#: corpus (``dt.selftest.mwc_corpus``) and the synth corpus (``dt.selftest.synth``) carries exactly
#: these keys, with one of these values (always strings). The first value of a key is its default.
#: Content (label, icon, text, value ...) is never a variant: it goes to ``ComponentRef.props``.
VARIANTS: dict[str, dict[str, tuple[str, ...]]] = {
    "Button": {"style": ("filled", "outlined", "text", "elevated", "tonal")},
    "FAB": {"size": ("regular", "small", "large"), "extended": ("false", "true")},
    "IconButton": {"style": ("standard", "filled", "tonal", "outlined")},
    "Chip": {"type": ("assist", "filter", "input", "suggestion"), "selected": ("false", "true")},
    "TextField": {"style": ("filled", "outlined")},
    "Select": {"style": ("filled", "outlined")},
    "Switch": {"selected": ("false", "true")},
    "Checkbox": {"checked": ("false", "true")},
    "Radio": {"checked": ("false", "true")},
    "Slider": {},
    "ListItem": {"lines": ("1", "2", "3")},
    "Card": {"style": ("elevated", "filled", "outlined")},
    "TopAppBar": {"size": ("small", "center", "medium", "large")},
    "Tabs": {"type": ("primary", "secondary")},
    "NavigationBar": {},
    "NavigationRail": {},
    "NavigationDrawer": {"type": ("standard", "modal")},
    "Snackbar": {"action": ("false", "true")},
    "Dialog": {},
    "Divider": {},
    "Badge": {"size": ("small", "large")},
    "Progress": {"type": ("linear", "circular")},
    "Menu": {},
    "SearchBar": {},
    "SegmentedButton": {},
}

#: Sub-parts and grouping containers of Material Web that are *not* component instances of the
#: catalog (they belong to their parent component). Ground truth records them as ``meta.part``.
PARTS: dict[str, str] = {
    "md-list": "List", "md-chip-set": "ChipSet", "md-primary-tab": "Tab", "md-secondary-tab": "Tab",
    "md-menu-item": "MenuItem", "md-select-option": "SelectOption",
}


def variant(name: str, **values: Any) -> dict[str, str]:
    """Complete, validated variant dict of component ``name`` in the shared vocabulary.

    Missing keys take their default (first value); booleans become ``"true"``/``"false"``.
    Raises ``KeyError`` for an unknown component or key and ``ValueError`` for a value outside
    the vocabulary, so corpora can never drift from the matcher again.
    """
    vocab = VARIANTS[name]
    unknown = set(values) - set(vocab)
    if unknown:
        raise KeyError(f"{name}: variant keys {sorted(unknown)} not in vocabulary {sorted(vocab)}")
    out: dict[str, str] = {}
    for key, allowed in vocab.items():
        v = values.get(key, allowed[0])
        v = ("true" if v else "false") if isinstance(v, bool) else str(v).lower()
        if v not in allowed:
            raise ValueError(f"{name}.{key}={v!r} not in {allowed}")
        out[key] = v
    return out


def component_key(name: str, variant_dict: dict[str, str]) -> str:
    """Stable corpus key, e.g. ``m3.button:style=filled`` or ``m3.divider``."""
    tail = ",".join(f"{k}={variant_dict[k]}" for k in sorted(variant_dict))
    return f"m3.{name.lower()}" + (f":{tail}" if tail else "")


def component_ref(name: str, props: Optional[dict] = None, evidence: Optional[dict] = None, **values: Any) -> ComponentRef:
    """Ground-truth ``ComponentRef`` in the shared vocabulary (used by both corpora)."""
    vd = variant(name, **values)
    return ComponentRef(key=component_key(name, vd), name=name, library=LIBRARY, variant=vd,
                        props=dict(props or {}), confidence=1.0, evidence=dict(evidence or {}))


def _c(name: str) -> str:
    """Hex for a light-scheme color role (used while building signatures/tokens)."""
    return REF_PALETTE[LIGHT_SCHEME[name]]


# --------------------------------------------------------------------------- tokens
def material3_tokens() -> list[Token]:
    toks: list[Token] = []
    for k, v in REF_PALETTE.items():
        toks.append(Token(f"md.ref.palette.{k}", "color", v, {"ref": True}))
    for role, key in LIGHT_SCHEME.items():
        toks.append(Token(f"md.sys.color.{role}", "color", REF_PALETTE[key], {"palette": key, "scheme": "light"}))
    for role, (size, lh, weight, tracking, face) in TYPESCALE.items():
        toks.append(Token(f"md.sys.typescale.{role}", "typescale",
                          {"size": float(size), "line_height": float(lh), "weight": weight, "tracking": tracking, "family": TYPEFACE[face]}))
    for k, v in SHAPE.items():
        toks.append(Token(f"md.sys.shape.{k}", "shape", v))
    for k, v in ELEVATION.items():
        toks.append(Token(f"md.sys.elevation.{k}", "elevation", dict(v)))
    for s in SPACING_STEPS:
        toks.append(Token(f"md.sys.spacing.{s}", "spacing", s))
    toks.append(Token("md.ref.typeface.brand", "font", TYPEFACE["brand"]))
    toks.append(Token("md.ref.typeface.plain", "font", TYPEFACE["plain"]))
    return toks


# --------------------------------------------------------------------------- component catalog
SURFACES = ["surface", "surface-container-lowest", "surface-container-low", "surface-container",
            "surface-bright", "background"]
TRANSPARENT_OR_SURFACE = ["none"] + SURFACES
LABEL = ["label-large"]
TEXT_CHILD_ROLE_BODY = ["body-large", "body-medium", "title-medium", "title-small"]


def _cp(t: str, optional: bool = False, max: int = 1, name: str = "") -> ChildPattern:
    return ChildPattern(type=t, optional=optional, max=max, name=name)


def _spec(key: str, name: str, sig: Signature, variants: list[Variant] | None = None,
          slots: list[Slot] | None = None, meta: dict | None = None) -> ComponentSpec:
    return ComponentSpec(key=f"{LIBRARY}.{key}", name=name, library=LIBRARY, signature=sig,
                         variants=variants or [], slots=slots or [], meta=meta or {})


def _v(name: str, props: dict[str, str], **sig) -> Variant:
    return Variant(name=name, props=props, signature=Signature(**sig))


def _buttons() -> ComponentSpec:
    base = Signature(
        types=["frame", "rect"], height=(40, 40), aspect=(1.2, 12), radius="pill",
        text_roles=LABEL, text_count=(1, 1), icon_count=(0, 1), icon_size=(18, 18),
        children=[_cp("icon", optional=True), _cp("text")], order="row",
    )
    variants = [
        _v("filled", {"style": "filled"}, fills=["primary"], stroke="none", shadow="none", text_colors=["on-primary"]),
        _v("tonal", {"style": "tonal"}, fills=["secondary-container"], stroke="none", shadow="none", text_colors=["on-secondary-container"]),
        _v("elevated", {"style": "elevated"}, fills=["surface-container-low"], stroke="none", shadow="required", text_colors=["primary"]),
        _v("outlined", {"style": "outlined"}, fills=TRANSPARENT_OR_SURFACE, stroke={"tokens": ["outline"], "width": (1, 1)}, shadow="none", text_colors=["primary"]),
        _v("text", {"style": "text"}, fills=["none"], stroke="none", shadow="none", text_colors=["primary"], radius="any"),
    ]
    return _spec("button", "Button", base, variants, [Slot("label", "text"), Slot("icon", "icon")],
                 meta={"mw": "md-filled-button|md-outlined-button|md-text-button|md-elevated-button|md-filled-tonal-button"})


def _fab() -> ComponentSpec:
    base = Signature(types=["frame", "rect"], fills=["primary-container", "secondary-container", "tertiary-container", "surface-container-high"],
                     stroke="none", shadow="required", icon_count=(1, 1), text_count=(0, 0),
                     children=[_cp("icon")], order="row")
    variants = [
        _v("regular", variant("FAB", size="regular"), height=(56, 56), aspect=(0.95, 1.05), radius=16, icon_size=(24, 24)),
        _v("small", variant("FAB", size="small"), height=(40, 40), aspect=(0.95, 1.05), radius=12, icon_size=(24, 24)),
        _v("large", variant("FAB", size="large"), height=(96, 96), aspect=(0.95, 1.05), radius=28, icon_size=(36, 36)),
        # the extended FAB is told apart by its width alone (an icon OCR'd as a 1-char text must
        # not turn a square FAB into an extended one), hence the heavy aspect weight
        _v("extended", variant("FAB", size="regular", extended=True), height=(56, 56), aspect=(1.4, 10), radius=16, icon_size=(24, 24),
           weights={"aspect": 3.0},
           icon_count=(0, 1), text_count=(1, 1), text_roles=LABEL, children=[_cp("icon", optional=True), _cp("text")]),
    ]
    return _spec("fab", "FAB", base, variants, [Slot("icon", "icon"), Slot("label", "text")], meta={"mw": "md-fab"})


def _icon_buttons() -> ComponentSpec:
    base = Signature(types=["frame", "rect"], height=(40, 40), aspect=(0.95, 1.05), icon_count=(1, 1), text_count=(0, 0),
                     icon_size=(24, 24), children=[_cp("icon")], shadow="none")
    variants = [
        _v("standard", {"style": "standard"}, fills=["none"], stroke="none", radius="any"),
        _v("filled", {"style": "filled"}, fills=["primary", "surface-container-highest"], stroke="none", radius="pill"),
        _v("tonal", {"style": "tonal"}, fills=["secondary-container"], stroke="none", radius="pill"),
        _v("outlined", {"style": "outlined"}, fills=TRANSPARENT_OR_SURFACE, stroke={"tokens": ["outline"], "width": (1, 1)}, radius="pill"),
    ]
    return _spec("icon-button", "IconButton", base, variants, [Slot("icon", "icon")], meta={"mw": "md-icon-button"})


def _chips() -> ComponentSpec:
    base = Signature(types=["frame", "rect"], height=(32, 32), aspect=(1.3, 12), radius=8, text_roles=LABEL,
                     text_count=(1, 1), icon_size=(18, 18), shadow="none",
                     children=[_cp("icon", optional=True), _cp("text"), _cp("icon", optional=True)], order="row",
                     text_colors=["on-surface", "on-secondary-container"])
    outlined = {"tokens": ["outline"], "width": (1, 1)}
    variants = [
        _v("assist", variant("Chip", type="assist"), fills=TRANSPARENT_OR_SURFACE, stroke=outlined, icon_count=(1, 1),
           children=[_cp("icon"), _cp("text")]),
        _v("suggestion", variant("Chip", type="suggestion"), fills=TRANSPARENT_OR_SURFACE, stroke=outlined, icon_count=(0, 0),
           children=[_cp("text")]),
        _v("filter", variant("Chip", type="filter", selected=True), fills=["secondary-container"], stroke="none", icon_count=(0, 1)),
        _v("filter-unselected", variant("Chip", type="filter"), fills=TRANSPARENT_OR_SURFACE, stroke=outlined, icon_count=(0, 1),
           children=[_cp("icon", optional=True), _cp("text")]),
        _v("input", variant("Chip", type="input"), fills=TRANSPARENT_OR_SURFACE, stroke=outlined, icon_count=(1, 2),
           children=[_cp("icon|ellipse|image", optional=True), _cp("text"), _cp("icon")]),
        _v("elevated", variant("Chip", type="assist"), fills=["surface-container-low"], stroke="none", shadow="required"),
    ]
    return _spec("chip", "Chip", base, variants, [Slot("label", "text"), Slot("icon", "icon"), Slot("trailing_icon", "icon")],
                 meta={"mw": "md-assist-chip|md-filter-chip|md-input-chip|md-suggestion-chip"})


def _text_fields() -> ComponentSpec:
    base = Signature(types=["frame", "rect"], height=(56, 56), aspect=(1.5, 20), text_count=(1, 3),
                     text_roles=["body-large", "body-small"], icon_count=(0, 2), icon_size=(24, 24),
                     children=[_cp("line|rect|instance:Divider", optional=True, name="indicator"), _cp("icon", optional=True), _cp("text", max=2), _cp("icon", optional=True)], order="row",
                     text_colors=["on-surface", "on-surface-variant", "primary"])
    variants = [
        _v("outlined", {"style": "outlined"}, radius=4, fills=TRANSPARENT_OR_SURFACE,
           stroke={"tokens": ["outline", "primary", "on-surface"], "width": (1, 3)}, shadow="none"),
        # shadow "any": the 1px active indicator at the bottom edge reads as a soft halo to perceive
        _v("filled", {"style": "filled"}, radius=(0, 4), fills=["surface-container-highest"], stroke="any", shadow="any"),
    ]
    return _spec("text-field", "TextField", base, variants,
                 [Slot("label", "text"), Slot("value", "text"), Slot("leading_icon", "icon"), Slot("trailing_icon", "icon")],
                 meta={"mw": "md-outlined-text-field|md-filled-text-field"})


def _select() -> ComponentSpec:
    """md-*-select: a text-field-shaped anchor whose last atom is the drop-down arrow (a 24px
    ``arrow_drop_down`` icon in the M3 kit, a 10x5 svg triangle in Material Web)."""
    base = Signature(types=["frame", "rect"], height=(56, 56), aspect=(1.5, 20), text_count=(1, 2),
                     text_roles=["body-large", "body-small"], icon_count=(0, 2), icon_size=(5, 24),
                     # the arrow is the only required atom, so a field without it scores 0 here
                     children=[_cp("icon", optional=True), _cp("text", optional=True, max=2), _cp("icon|vector", name="arrow")], order="row",
                     text_colors=["on-surface", "on-surface-variant", "primary"], critical=["children"])
    variants = [
        _v("outlined", {"style": "outlined"}, radius=4, fills=TRANSPARENT_OR_SURFACE,
           stroke={"tokens": ["outline", "primary", "on-surface"], "width": (1, 3)}, shadow="none"),
        _v("filled", {"style": "filled"}, radius=(0, 4), fills=["surface-container-highest"], stroke="any", shadow="any"),
    ]
    return _spec("select", "Select", base, variants,
                 [Slot("label", "text"), Slot("value", "text"), Slot("leading_icon", "icon")],
                 meta={"mw": "md-outlined-select|md-filled-select"})


def _switch() -> ComponentSpec:
    # The unselected track's 2px outline is often absorbed by perceive (box inside the ring, the
    # ring read as a soft halo): the pill + handle structure and the track fill decide instead.
    base = Signature(types=["frame", "rect"], height=(32, 32), width=(52, 52), radius="pill", text_count=(0, 0),
                     children=[_cp("ellipse|rect|frame", name="handle"), _cp("icon", optional=True)], shadow="any")
    variants = [
        _v("selected", {"selected": "true"}, fills=["primary"], stroke="none"),
        _v("unselected", {"selected": "false"}, fills=["surface-container-highest"], stroke="any"),
    ]
    return _spec("switch", "Switch", base, variants, [], meta={"mw": "md-switch"})


def _checkbox() -> ComponentSpec:
    base = Signature(types=["frame", "rect"], height=(18, 18), aspect=(0.95, 1.05), radius=2, text_count=(0, 0),
                     children=[_cp("icon|vector", optional=True)], shadow="none")
    variants = [
        _v("checked", {"checked": "true"}, fills=["primary"], stroke="none", icon_count=(0, 1)),
        _v("unchecked", {"checked": "false"}, fills=["none"], stroke={"tokens": ["on-surface-variant"], "width": (2, 2)}, icon_count=(0, 0)),
    ]
    return _spec("checkbox", "Checkbox", base, variants, [], meta={"mw": "md-checkbox"})


def _radio() -> ComponentSpec:
    base = Signature(types=["ellipse", "frame", "rect"], height=(20, 20), aspect=(0.95, 1.05), radius="pill",
                     text_count=(0, 0), fills=["none"], children=[_cp("ellipse", optional=True)], shadow="none")
    variants = [
        _v("selected", {"checked": "true"}, stroke={"tokens": ["primary"], "width": (2, 2)}, children=[_cp("ellipse")]),
        _v("unselected", {"checked": "false"}, stroke={"tokens": ["on-surface-variant"], "width": (2, 2)}, children=[]),
    ]
    return _spec("radio", "Radio", base, variants, [], meta={"mw": "md-radio"})


def _slider() -> ComponentSpec:
    sig = Signature(types=["frame"], height=(4, 44), aspect=(3, 100), fills=["none"], stroke="none", text_count=(0, 0),
                    children=[_cp("rect|line", max=2), _cp("ellipse|rect", name="handle")], order="row", shadow="none")
    return _spec("slider", "Slider", sig, [], [], meta={"mw": "md-slider"})


def _list_item() -> ComponentSpec:
    # text roles and the children pattern (a headline text is required) are critical: a 22px
    # title-large row (top app bar) or a strip of tab cells without own text is not a list item
    base = Signature(types=["frame", "rect"], aspect=(2.5, 40), radius="none", fills=TRANSPARENT_OR_SURFACE, stroke="none",
                     shadow="none", text_roles=["body-large", "body-medium", "label-small", "title-medium"],
                     text_colors=["on-surface", "on-surface-variant"], icon_size=(24, 40), critical=["text", "children"],
                     children=[_cp("icon|image|ellipse|instance", optional=True, name="leading"), _cp("text", max=3),
                               _cp("icon|instance|text", optional=True, name="trailing")], order="row")
    variants = [
        _v("one-line", {"lines": "1"}, height=(48, 60), text_count=(1, 2)),
        _v("two-line", {"lines": "2"}, height=(64, 80), text_count=(2, 3)),
        _v("three-line", {"lines": "3"}, height=(84, 96), text_count=(3, 4)),
    ]
    return _spec("list-item", "ListItem", base, variants,
                 [Slot("headline", "text"), Slot("supporting", "text", multiple=True), Slot("leading", "icon|image|ellipse|instance"),
                  Slot("trailing", "icon|instance")], meta={"mw": "md-list-item"})


def _cards() -> ComponentSpec:
    base = Signature(types=["frame", "rect"], height=(48, 2000), width=(48, 2000), aspect=(0.3, 6), radius=12, text_count=(0, 20))
    variants = [
        _v("elevated", {"style": "elevated"}, fills=["surface-container-low"], stroke="none", shadow="required"),
        _v("filled", {"style": "filled"}, fills=["surface-container-highest"], stroke="none", shadow="none"),
        _v("outlined", {"style": "outlined"}, fills=["surface", "surface-container-lowest"], stroke={"tokens": ["outline-variant"], "width": (1, 1)}, shadow="none"),
    ]
    return _spec("card", "Card", base, variants, [Slot("content", "any", multiple=True)], meta={"mw": "md-elevated-card|md-filled-card|md-outlined-card"})


def _top_app_bar() -> ComponentSpec:
    # The headline is what makes a bar a *top app bar*: a title-large/title-medium (small, center)
    # or headline-small/medium (medium, large) text is a critical constraint; Tabs instead need
    # repeated equal-width cells plus an indicator line (see _tabs).
    base = Signature(types=["frame", "rect"], width=(280, 4000), aspect=(2.5, 60), radius="none",
                     fills=["surface", "surface-container"], stroke="none", icon_size=(24, 24), icon_count=(0, 5),
                     text_count=(1, 2), children=[_cp("icon|instance", optional=True), _cp("text"), _cp("icon|instance|text|frame", optional=True, max=5)],
                     order="row", text_colors=["on-surface"], critical=["text", "text_count"], weights={"text": 2.0})
    variants = [
        _v("small", {"size": "small"}, height=(64, 64), text_roles=["title-large", "title-medium"], title_align="start"),
        _v("center", {"size": "center"}, height=(64, 64), text_roles=["title-large", "title-medium"], title_align="center"),
        _v("medium", {"size": "medium"}, height=(112, 112), text_roles=["headline-small"], order="column"),
        _v("large", {"size": "large"}, height=(152, 152), text_roles=["headline-medium"], order="column"),
    ]
    return _spec("top-app-bar", "TopAppBar", base, variants, [Slot("title", "text"), Slot("navigation_icon", "icon"), Slot("actions", "icon", multiple=True)])


def _navigation_bar() -> ComponentSpec:
    sig = Signature(types=["frame", "rect"], height=(80, 80), width=(280, 4000), aspect=(3, 60), radius="none",
                    fills=["surface-container", "surface"], stroke="none", icon_count=(3, 5), text_count=(0, 5),
                    icon_size=(24, 24), text_roles=["label-medium"], order="row")
    return _spec("navigation-bar", "NavigationBar", sig, [], [Slot("destinations", "icon", multiple=True)])


def _navigation_rail() -> ComponentSpec:
    sig = Signature(types=["frame", "rect"], width=(80, 80), aspect=(0.02, 0.4), radius="none", fills=["surface", "surface-container"],
                    stroke="none", icon_count=(1, 8), icon_size=(24, 24), order="column")
    return _spec("navigation-rail", "NavigationRail", sig, [], [Slot("destinations", "icon", multiple=True)])


def _navigation_drawer() -> ComponentSpec:
    base = Signature(types=["frame", "rect"], width=(300, 360), aspect=(0.05, 0.9), radius=(0, 16), stroke="none",
                     text_count=(1, 30), order="column")
    variants = [
        _v("standard", {"type": "standard"}, fills=["surface", "surface-container-low"], shadow="none"),
        _v("modal", {"type": "modal"}, fills=["surface-container-low"], shadow="required"),
    ]
    return _spec("navigation-drawer", "NavigationDrawer", base, variants, [Slot("items", "any", multiple=True)])


def _tabs() -> ComponentSpec:
    # A tab strip is >= 2 equal-width cells (one label and/or icon each) plus a 2-3 px active
    # indicator; both are critical so a top app bar (title + actions, no indicator) never passes.
    # (no text/icon counts: labels sit either directly in the strip or inside painted tab cells)
    base = Signature(types=["frame", "rect"], width=(120, 4000), aspect=(2, 60), radius="none", fills=["surface", "none"],
                     stroke="none", shadow="none", text_roles=["title-small"], order="row",
                     children=[_cp("text|icon|frame|instance", max=16), _cp("line|rect", optional=True, max=2)],
                     cells="equal", indicator=(2, 3), critical=["cells", "indicator"])
    variants = [  # md-primary-tab indicator 3px (under the label), md-secondary-tab 2px (full cell)
        _v("primary", {"type": "primary"}, height=(48, 48), indicator=(3, 3)),
        _v("primary-with-icon", {"type": "primary"}, height=(64, 64), indicator=(3, 3)),
        _v("secondary", {"type": "secondary"}, height=(48, 48), indicator=(2, 2)),
    ]
    return _spec("tabs", "Tabs", base, variants, [Slot("tabs", "text", multiple=True)], meta={"mw": "md-tabs"})


def _dialog() -> ComponentSpec:
    sig = Signature(types=["frame", "rect"], width=(280, 560), height=(100, 2000), radius=28, fills=["surface-container-high"],
                    stroke="none", shadow="required", text_count=(1, 20), text_roles=["headline-small", "body-medium"],
                    order="column", children=[_cp("icon", optional=True), _cp("text", max=6), _cp("frame|instance|instance:Button", optional=True, max=4)])
    return _spec("dialog", "Dialog", sig, [], [Slot("headline", "text"), Slot("supporting", "text", multiple=True), Slot("actions", "instance", multiple=True)], meta={"mw": "md-dialog"})


def _snackbar() -> ComponentSpec:
    # Heights are discrete (48 single-line, 68 two-line) so a 56px text field is never a snackbar.
    base = Signature(types=["frame", "rect"], width=(200, 800), aspect=(3, 20), radius=4, fills=["inverse-surface"],
                     stroke="none", text_roles=["body-medium", "label-large"], text_colors=["inverse-on-surface", "inverse-primary"],
                     order="row")
    message = dict(text_count=(1, 1), children=[_cp("text"), _cp("instance|icon|frame", optional=True)])
    action = dict(text_count=(2, 2), children=[_cp("text"), _cp("text|instance"), _cp("instance|icon|frame", optional=True)])
    variants = [
        _v("message", {"action": "false"}, height=(48, 48), **message),
        _v("message-two-line", {"action": "false"}, height=(68, 68), **message),
        _v("with-action", {"action": "true"}, height=(48, 48), **action),
        _v("with-action-two-line", {"action": "true"}, height=(68, 68), **action),
    ]
    return _spec("snackbar", "Snackbar", base, variants, [Slot("message", "text"), Slot("action", "text")])


# Dividers and linear progress tracks span a container; a bar shorter than one 24px M3 icon is a
# glyph stroke (e.g. the 18x2 bars of an unidentified `menu` icon), not a component.
MIN_LINE_LEN = 24


def _divider() -> ComponentSpec:
    sig = Signature(types=["line", "rect", "frame"], height=(1, 1), width=(MIN_LINE_LEN, 100000), aspect=(12, 100000),
                    fills=["outline-variant", "outline"],
                    text_count=(0, 0), shadow="none", radius="any", stroke="any")
    return _spec("divider", "Divider", sig, [], [], meta={"mw": "md-divider"})


def _badge() -> ComponentSpec:
    base = Signature(types=["ellipse", "frame", "rect"], fills=["error"], stroke="none", radius="pill", shadow="none")
    variants = [
        _v("small", {"size": "small"}, height=(6, 6), aspect=(0.9, 1.1), text_count=(0, 0)),
        _v("large", {"size": "large"}, height=(16, 16), aspect=(0.9, 3), text_count=(0, 1), text_roles=["label-small"], text_colors=["on-error"]),
    ]
    return _spec("badge", "Badge", base, variants, [Slot("label", "text")])


def _progress() -> ComponentSpec:
    base = Signature(stroke="any", text_count=(0, 0), shadow="none")
    variants = [
        _v("linear", {"type": "linear"}, types=["rect", "frame", "line"], height=(4, 4), width=(MIN_LINE_LEN, 100000),
           aspect=(10, 100000),
           fills=["surface-container-highest", "primary", "primary-container"], radius="any"),
        _v("circular", {"type": "circular"}, types=["ellipse", "frame"], height=(48, 48), aspect=(0.95, 1.05),
           fills=["none"], stroke={"tokens": ["primary", "surface-container-highest"], "width": (4, 4)}, radius="pill"),
    ]
    return _spec("progress", "Progress", base, variants, [], meta={"mw": "md-linear-progress|md-circular-progress"})


def _menu() -> ComponentSpec:
    sig = Signature(types=["frame", "rect"], width=(112, 280), height=(48, 1000), radius=4, fills=["surface-container"],
                    stroke="none", shadow="required", text_count=(1, 20), text_roles=["body-large", "label-large"], order="column",
                    children=[_cp("text|instance|instance:ListItem|frame", max=20)])
    return _spec("menu", "Menu", sig, [], [Slot("items", "text", multiple=True)], meta={"mw": "md-menu"})


def _search_bar() -> ComponentSpec:
    sig = Signature(types=["frame", "rect"], height=(56, 56), aspect=(4, 20), radius="pill", fills=["surface-container-high"],
                    stroke="none", text_count=(1, 1), text_roles=["body-large"], icon_count=(1, 3), icon_size=(24, 24), order="row",
                    children=[_cp("icon|ellipse"), _cp("text"), _cp("icon|ellipse|image", optional=True, max=2)])
    return _spec("search-bar", "SearchBar", sig, [], [Slot("placeholder", "text"), Slot("leading_icon", "icon"), Slot("trailing_icon", "icon")])


def _segmented_button() -> ComponentSpec:
    sig = Signature(types=["frame", "rect"], height=(40, 40), aspect=(3, 20), radius="pill", fills=TRANSPARENT_OR_SURFACE + ["secondary-container"],
                    stroke={"tokens": ["outline"], "width": (1, 1)}, text_count=(2, 5), text_roles=LABEL, icon_size=(18, 18), order="row",
                    children=[_cp("icon|text|frame|line|rect|instance", max=12)])
    return _spec("segmented-button", "SegmentedButton", sig, [], [Slot("segments", "text", multiple=True)])


def material3_components() -> list[ComponentSpec]:
    return [
        _buttons(), _fab(), _icon_buttons(), _chips(), _text_fields(), _select(), _switch(), _checkbox(), _radio(), _slider(),
        _list_item(), _cards(), _top_app_bar(), _navigation_bar(), _navigation_rail(), _navigation_drawer(), _tabs(),
        _dialog(), _snackbar(), _divider(), _badge(), _progress(), _menu(), _search_bar(), _segmented_button(),
    ]


def material3() -> DesignSystem:
    """The Material 3 baseline (light) design system with its component catalog."""
    return DesignSystem(
        name=LIBRARY, tokens=material3_tokens(), components=material3_components(), fonts=["Roboto"],
        meta={"source": "@material/web 2.5 tokens v0_192", "scheme": "light", "dark_scheme": {k: REF_PALETTE[v] for k, v in DARK_SCHEME.items()},
              "spacing_grid": SPACING_GRID, "color_priority": COLOR_PRIORITY, "typescale_priority": TYPESCALE_PRIORITY,
              "variants": {name: {k: list(vals) for k, vals in keys.items()} for name, keys in VARIANTS.items()},
              "parts": sorted(set(PARTS.values()))},
    )


# --------------------------------------------------------------------------- verification against node_modules
def _parse_scss_map(text: str) -> dict[str, str]:
    """Parse `'key': if($exclude-hardcoded-values, null, VALUE)` and `'key': map.get($deps, 'ns', 'ref')` entries."""
    out: dict[str, str] = {}
    flat = re.sub(r"\s+", " ", text)
    for m in re.finditer(r"'([a-z0-9-]+)':\s*if\(\$exclude-hardcoded-values, null, ([^)]*)\)", flat):
        out[m.group(1)] = m.group(2).strip()
    for m in re.finditer(r"'([a-z0-9-]+)':\s*map\.get\(\$deps, '([a-z-]+)', '([a-z0-9-]+)'\)", flat):
        out.setdefault(m.group(1), f"{m.group(2)}.{m.group(3)}")
    return out


def parse_material_web_tokens(tokens_dir: str = MW_TOKENS_DIR) -> Optional[dict]:
    """Parse palette / light scheme / shape / elevation / component metrics from Material Web scss.
    Returns None when the package is not installed."""
    if not os.path.isdir(tokens_dir):
        return None

    def read(name: str) -> str:
        with open(os.path.join(tokens_dir, name)) as f:
            return f.read()

    palette = {k: (v if len(v) == 7 else "#" + "".join(ch * 2 for ch in v[1:])) for k, v in _parse_scss_map(read("_md-ref-palette.scss")).items()}
    color_src = read("_md-sys-color.scss")
    light_part = color_src.split("@function values-light")[1]
    light = {k: v.split(".")[-1] for k, v in _parse_scss_map(light_part).items()}
    shape = {k: v for k, v in _parse_scss_map(read("_md-sys-shape.scss")).items()}
    elevation = {k: v for k, v in _parse_scss_map(read("_md-sys-elevation.scss")).items()}
    comps: dict[str, dict[str, str]] = {}
    for fn in sorted(os.listdir(tokens_dir)):
        if fn.startswith("_md-comp-") and fn.endswith(".scss"):
            comps[fn[len("_md-comp-"):-len(".scss")]] = _parse_scss_map(read(fn))
    return {"palette": palette, "light": light, "shape": shape, "elevation": elevation, "components": comps}


def verify_against_material_web(tokens_dir: str = MW_TOKENS_DIR) -> Optional[list[str]]:
    """Compare the embedded constants with the installed Material Web token files.
    Returns a list of discrepancies (empty = in sync) or None if node_modules is absent."""
    parsed = parse_material_web_tokens(tokens_dir)
    if parsed is None:
        return None
    issues: list[str] = []
    for k, v in parsed["palette"].items():
        if REF_PALETTE.get(k) != v:
            issues.append(f"palette {k}: embedded {REF_PALETTE.get(k)} != scss {v}")
    for role, key in parsed["light"].items():
        if LIGHT_SCHEME.get(role) != key:
            issues.append(f"light {role}: embedded {LIGHT_SCHEME.get(role)} != scss {key}")
    for k, v in parsed["shape"].items():
        if k in SHAPE:
            want = "9999px" if SHAPE[k] == "full" else f"{SHAPE[k]}px"
            if v != want:
                issues.append(f"shape {k}: embedded {want} != scss {v}")
    for k, v in parsed["elevation"].items():
        if k in ELEVATION and str(ELEVATION[k]["dp"]) != v:
            issues.append(f"elevation {k}: embedded {ELEVATION[k]['dp']} != scss {v}")
    comps = parsed["components"]
    checks = {
        "filled-button": ("container-height", "40px"), "fab-primary": ("container-height", "56px"),
        "fab-primary-small": ("container-height", "40px"), "fab-primary-large": ("container-height", "96px"),
        "switch": ("track-width", "52px"), "checkbox": ("container-size", "18px"), "input-chip": ("container-height", "32px"),
        "top-app-bar-small": ("container-height", "64px"), "navigation-bar": ("container-height", "80px"),
        "list": ("list-item-two-line-container-height", "72px"), "search-bar": ("container-height", "56px"),
    }
    for comp, (key, want) in checks.items():
        got = comps.get(comp, {}).get(key)
        if got is not None and got != want:
            issues.append(f"{comp}.{key}: catalog assumes {want}, scss says {got}")
    return issues


def write_fixture(path: Optional[str] = None) -> str:
    path = path or os.path.join(FIXTURES_DIR, "material3.json")
    return material3().save(path)


if __name__ == "__main__":  # pragma: no cover
    print("wrote", write_fixture())
    issues = verify_against_material_web()
    print("verify:", "node_modules absent" if issues is None else (issues or "in sync"))
