"""Token-library ingestion: W3C DTCG design-tokens JSON and Material Theme Builder exports.

    from_dtcg(json_obj)                   -> list[Token]
    from_material_theme_builder(json_obj) -> list[Token]
    load_tokens_file(path)                -> list[Token]   (format auto-detected)
    design_system_from_tokens(name, tokens, base=material3()) -> DesignSystem

DTCG (https://tr.designtokens.org/format/): nested groups, each token has `$value` and a
`$type` (own or inherited from the group).  Supported types: color (hex string or the 2024
object form `{colorSpace, components, alpha, hex}`), dimension ("16px" or `{value, unit}`),
number, fontFamily, fontWeight, typography (composite), shadow, and `{alias.path}` references.

Material Theme Builder export (material-theme.json): `schemes.light/dark.<camelCaseRole>`,
`palettes.<name>.<tone>`, `seed`, `coreColors`.  Roles are renamed to md.sys.color.<kebab-role>
so a re-themed catalog keeps the baseline component signatures.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional

from dt.ir import Color
from dt.mapping.design_system import DesignSystem, Token, merge_tokens

_DTCG_TYPES = {"color", "dimension", "number", "fontFamily", "fontWeight", "typography", "shadow", "duration", "cubicBezier",
               "strokeStyle", "border", "transition", "gradient", "fontSize", "lineHeight", "letterSpacing"}
_CATEGORY = {"color": "color", "dimension": "dimension", "number": "dimension", "fontFamily": "font", "fontWeight": "font",
             "typography": "typescale", "shadow": "elevation"}


# --------------------------------------------------------------------------- helpers
def kebab(s: str) -> str:
    """camelCase / spaces / slashes -> kebab-case ('onPrimaryContainer' -> 'on-primary-container')."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", s)
    s = re.sub(r"[\s/_]+", "-", s.strip())
    return s.lower()


def parse_px(v: Any) -> Optional[float]:
    """'16px' | '1rem' | 16 | {'value': 16, 'unit': 'px'} -> px (rem assumed 16px)."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        val = float(v.get("value", 0))
        return val * 16 if v.get("unit") == "rem" else val
    s = str(v).strip().lower()
    m = re.match(r"^(-?[0-9.]+)\s*(px|rem|em|%)?$", s)
    if not m:
        return None
    val = float(m.group(1))
    return val * 16 if m.group(2) in ("rem", "em") else val


def parse_color(v: Any) -> Optional[str]:
    """DTCG color value -> '#rrggbb' (alpha dropped unless < 1 -> '#rrggbbaa')."""
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("#"):
            c = Color.from_hex(s)
        elif s.startswith("rgb"):
            nums = [float(x) for x in re.findall(r"[0-9.]+", s)]
            if len(nums) < 3:
                return None
            c = Color(int(nums[0]), int(nums[1]), int(nums[2]), nums[3] if len(nums) > 3 else 1.0)
        else:
            return None
    elif isinstance(v, dict):
        if v.get("hex"):
            c = Color.from_hex(v["hex"])
        else:
            comps = v.get("components") or []
            if len(comps) < 3:
                return None
            c = Color.from_rgb(comps[0] * 255, comps[1] * 255, comps[2] * 255, float(v.get("alpha", 1.0)))
    else:
        return None
    return c.hex() if c.a >= 0.999 else "%s%02x" % (c.hex(), int(round(c.a * 255)))


# --------------------------------------------------------------------------- DTCG
def _walk_dtcg(obj: dict, path: list[str], inherited: Optional[str], out: dict[str, tuple[Optional[str], Any, dict]]) -> None:
    t = obj.get("$type", inherited)
    if "$value" in obj:
        out[".".join(path)] = (t, obj["$value"], {k[1:]: v for k, v in obj.items() if k.startswith("$") and k not in ("$value", "$type")})
        return
    for k, v in obj.items():
        if k.startswith("$") or not isinstance(v, dict):
            continue
        _walk_dtcg(v, path + [k], t, out)


def _resolve_alias(v: Any, raw: dict[str, tuple[Optional[str], Any, dict]], depth: int = 0) -> Any:
    if depth > 16:
        return v
    if isinstance(v, str):
        m = re.fullmatch(r"\{([^}]+)\}", v.strip())
        if m:
            ref = m.group(1).replace("/", ".")
            if ref in raw:
                return _resolve_alias(raw[ref][1], raw, depth + 1)
            return v
        return v
    if isinstance(v, dict):
        return {k: _resolve_alias(x, raw, depth + 1) for k, x in v.items()}
    if isinstance(v, list):
        return [_resolve_alias(x, raw, depth + 1) for x in v]
    return v


def _typography(v: dict) -> dict:
    out: dict[str, Any] = {}
    if v.get("fontFamily") is not None:
        fam = v["fontFamily"]
        out["family"] = fam[0] if isinstance(fam, list) and fam else fam
    if v.get("fontSize") is not None:
        out["size"] = parse_px(v["fontSize"])
    if v.get("fontWeight") is not None:
        w = v["fontWeight"]
        out["weight"] = int(w) if isinstance(w, (int, float)) or str(w).isdigit() else {"regular": 400, "medium": 500, "bold": 700, "semibold": 600, "light": 300}.get(str(w).lower(), 400)
    if v.get("lineHeight") is not None:
        lh = v["lineHeight"]
        out["line_height"] = (float(lh) * out.get("size", 16)) if isinstance(lh, (int, float)) and float(lh) < 4 else parse_px(lh)
    if v.get("letterSpacing") is not None:
        out["tracking"] = parse_px(v["letterSpacing"])
    return out


def _shadow(v: Any) -> dict:
    s = v[0] if isinstance(v, list) and v else v
    if not isinstance(s, dict):
        return {}
    return {"color": parse_color(s.get("color")), "dx": parse_px(s.get("offsetX")) or 0.0, "dy": parse_px(s.get("offsetY")) or 0.0,
            "blur": parse_px(s.get("blur")) or 0.0, "spread": parse_px(s.get("spread")) or 0.0}


def from_dtcg(obj: dict, prefix: str = "") -> list[Token]:
    """W3C DTCG token document -> tokens. Names are dotted group paths (optionally prefixed)."""
    raw: dict[str, tuple[Optional[str], Any, dict]] = {}
    _walk_dtcg(obj, [], None, raw)
    toks: list[Token] = []
    for name, (t, value, extra) in raw.items():
        value = _resolve_alias(value, raw)
        if t is None:  # infer from the resolved value
            if isinstance(value, str) and (value.startswith("#") or value.startswith("rgb")):
                t = "color"
            elif isinstance(value, dict) and ("components" in value or "hex" in value):
                t = "color"
            elif parse_px(value) is not None:
                t = "dimension"
            elif isinstance(value, dict) and "fontSize" in value:
                t = "typography"
        cat = _CATEGORY.get(t or "", "other")
        meta = {"dtcg_type": t, **extra}
        if t == "color":
            val: Any = parse_color(value)
            if val is None:
                continue
        elif t in ("dimension", "number", "fontSize", "lineHeight", "letterSpacing"):
            val = parse_px(value)
            if val is None:
                continue
            cat = "dimension"
        elif t == "typography" and isinstance(value, dict):
            val = _typography(value)
        elif t == "shadow":
            val = _shadow(value)
        elif t == "fontWeight":
            val = value
        elif t == "fontFamily":
            val = value[0] if isinstance(value, list) and value else value
        else:
            val = value
        toks.append(Token(name=(prefix + name) if prefix else name, category=cat, value=val, meta=meta))
    return toks


# --------------------------------------------------------------------------- Material Theme Builder
def from_material_theme_builder(obj: dict, scheme: str = "light") -> list[Token]:
    """Material Theme Builder export -> md.sys.color.* (chosen scheme) + md.ref.palette.* tokens.
    Other schemes are kept under meta so nothing is lost."""
    toks: list[Token] = []
    schemes = obj.get("schemes", {})
    chosen = schemes.get(scheme) or next(iter(schemes.values()), {})
    for role, hexv in chosen.items():
        c = parse_color(hexv)
        if c:
            toks.append(Token(f"md.sys.color.{kebab(role)}", "color", c, {"scheme": scheme, "source": "material-theme-builder"}))
    for pname, tones in obj.get("palettes", {}).items():
        for tone, hexv in tones.items():
            c = parse_color(hexv)
            if c:
                toks.append(Token(f"md.ref.palette.{kebab(pname)}{tone}", "color", c, {"ref": True, "source": "material-theme-builder"}))
    if obj.get("seed"):
        toks.append(Token("md.ref.seed", "color", parse_color(obj["seed"]), {"ref": True}))
    for name, hexv in (obj.get("coreColors") or {}).items():
        c = parse_color(hexv)
        if c:
            toks.append(Token(f"md.ref.core.{kebab(name)}", "color", c, {"ref": True}))
    return toks


# --------------------------------------------------------------------------- entry points
def detect_format(obj: dict) -> str:
    if "schemes" in obj and isinstance(obj.get("schemes"), dict):
        return "material-theme-builder"
    return "dtcg"


def load_tokens_file(path: str) -> list[Token]:
    with open(path) as f:
        obj = json.load(f)
    fmt = detect_format(obj)
    return from_material_theme_builder(obj) if fmt == "material-theme-builder" else from_dtcg(obj)


def design_system_from_tokens(name: str, tokens: Iterable[Token], base: Optional[DesignSystem] = None) -> DesignSystem:
    """New DesignSystem from tokens; with `base`, the base catalog is kept and tokens override/extend it
    (e.g. re-theme Material 3 with a Theme Builder export)."""
    if base is not None:
        return merge_tokens(base, tokens, name=name)
    toks = list(tokens)
    fonts = sorted({str(t.value) for t in toks if t.category == "font" and isinstance(t.value, str)} |
                   {str(t.value.get("family")) for t in toks if t.category == "typescale" and isinstance(t.value, dict) and t.value.get("family")})
    return DesignSystem(name=name, tokens=toks, fonts=fonts)
