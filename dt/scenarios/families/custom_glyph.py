"""custom_glyph: non-Material bullets and marks (6-20 px) drawn as inline SVG.

Generalises two knowledge/failures.jsonl entries:
* R3 "4 of 6 custom 4-point sparkle bullets (~10px) are unnamed icons and render as nothing"
  (refine: the as-is raster fallback required min_side 20 px);
* R4 "non-Material glyphs named as icons: hexagon -> brightness_1, 7px ring bullets ->
  circle_notifications / report" (perceive.icons accepted on a mostly-background crop).

The family lays out real browser HTML (bullet lists, inline marks after a label, rating rows)
with stars, sparkles, dots, diamonds, rings, triangles and hexagons in any accent colour, light
or dark, and takes the ground truth from the DOM (svg -> ``vector`` nodes).

Success: every glyph is reproduced (mean ΔE2000 over its box <= ``scenarios.glyph.max_de``),
by raster or vector, and none is named as a Material Symbol.
"""
from __future__ import annotations

import numpy as np

from dt.ir import Box
from dt.params import P, register
from dt.scenarios.families._common import ACCENTS, DARK_ACCENTS, DARK_BGS, DARK_INK, LIGHT_BGS, LIGHT_INK, phrase, pick, rng_for
from dt.scenarios.sources import html_case
from dt.scenarios.spec import Criterion, ScenarioFamily

register("scenarios.glyph.max_de", 3.0,
         "custom_glyph: max mean ΔE2000 over a glyph's box (target vs render) for the glyph to count as reproduced",
         (1.0, 8.0))

NAME = "custom_glyph"

SHAPES = {
    "star": '<polygon points="12,1.5 14.9,8.6 22.5,9.2 16.7,14.1 18.5,21.5 12,17.5 5.5,21.5 7.3,14.1 1.5,9.2 9.1,8.6" fill="{c}"/>',
    "sparkle": '<path d="M12 0C12.8 7 17 11.2 24 12C17 12.8 12.8 17 12 24C11.2 17 7 12.8 0 12C7 11.2 11.2 7 12 0Z" fill="{c}"/>',
    "dot": '<circle cx="12" cy="12" r="8" fill="{c}"/>',
    "diamond": '<polygon points="12,1 23,12 12,23 1,12" fill="{c}"/>',
    "ring": '<circle cx="12" cy="12" r="9" fill="none" stroke="{c}" stroke-width="3"/>',
    "triangle": '<polygon points="12,2 23,21 1,21" fill="{c}"/>',
    "hexagon": '<polygon points="12,1 21.5,6.5 21.5,17.5 12,23 2.5,17.5 2.5,6.5" fill="{c}"/>',
}


def _svg(shape: str, size: int, color: str) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" style="flex:none;display:block">'
            + SHAPES[shape].format(c=color) + "</svg>")


def generate(p: dict, seed: int):
    rng = rng_for(seed, NAME)
    dark = p["theme"] == "dark"
    bg = pick(rng, DARK_BGS if dark else LIGHT_BGS)
    ink = pick(rng, DARK_INK if dark else LIGHT_INK)
    accents = DARK_ACCENTS if dark else ACCENTS
    W = int(p["width"])
    size, fs, gap = int(p["glyph_px"]), int(p["font_px"]), int(p["gap"])
    shapes = list(SHAPES)
    rows = []
    n = int(p["n_items"])
    for i in range(n):
        shape = p["shape"] if p["mix"] == "uniform" else pick(rng, shapes)
        color = p_color = pick(rng, accents) if p["color"] == "accent" else ink
        label = phrase(rng, 1, 4)
        if p["layout"] == "bullets":
            rows.append(f'<div class="r">{_svg(shape, size, color)}<span>{label}</span></div>')
        elif p["layout"] == "trailing":
            rows.append(f'<div class="r"><span>{label}</span>{_svg(shape, size, color)}</div>')
        else:  # rating: k marks after a number
            k = rng.randint(1, 5)
            marks = "".join(_svg(shape, size, p_color) for _ in range(k))
            rows.append(f'<div class="r"><span>{rng.randint(1, 4)}.{rng.randint(0, 9)}</span>'
                        f'<div style="display:flex;gap:{max(1, size // 6)}px">{marks}</div><span>{label}</span></div>')
    lh = max(size, round(fs * 1.5)) + int(p["row_pad"])
    H = 24 + n * lh + 24
    css = (f"body{{background:{bg};}} .wrap{{padding:24px {int(p['margin'])}px;}} "
           f".r{{display:flex;align-items:center;gap:{gap}px;height:{lh}px;font:{int(p['weight'])} {fs}px Roboto;color:{ink};"
           f"white-space:nowrap;}}")
    body = f'<div class="wrap">{"".join(rows)}</div>'
    case = html_case(NAME, seed, p, body, W, H, extra_css=css)
    glyphs = [n_ for n_ in case.gt.walk() if n_.type == "vector"]
    case.meta["glyphs"] = [g.box for g in glyphs]
    case.meta["highlight_boxes"] = [g.box.expand(1) for g in glyphs]
    if glyphs:
        u = glyphs[0].box
        for g in glyphs[1:]:
            u = u.union(g.box)
        case.roi = u.expand(4).intersect(Box(0, 0, W, H))
    return case


def metrics(case, pred, rendered) -> dict:
    from dt.validate.fidelity import conform_to_target, delta_e2000_map
    r, _ = conform_to_target(case.target_rgb, rendered)
    de = delta_e2000_map(case.target_rgb, r)
    h, w = de.shape
    max_de = float(P["scenarios.glyph.max_de"])
    des, misnamed = [], 0
    named = [n for n in pred.walk() if n.visible and n.icon_name]
    for b in case.meta["glyphs"]:
        x0, y0, x1, y1 = b.expand(1).as_int()
        cell = de[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        des.append(float(cell.mean()) if cell.size else 99.0)
        gb = b.expand(2)
        if any(gb.contains_point(n.box.cx, n.box.cy) or n.box.iou(b) > 0.3 for n in named):
            misnamed += 1
    if not des:
        return {"glyph_reproduced": None, "glyph_misnamed": misnamed, "glyph_de_max": None}
    return {"glyph_reproduced": float(np.mean([d <= max_de for d in des])), "glyph_misnamed": misnamed,
            "glyph_de_max": float(max(des)), "n_glyphs": len(des)}


FAMILY = ScenarioFamily(
    name=NAME,
    description="Non-Material bullets/marks (6-20 px SVG) must be reproduced (raster or vector) and never named as "
                "Material icons.",
    stage="translate",
    refine_iters=4,
    failure_refs=["R3 real/m3_buttons_overview_types_mobile (sparkle bullets): unnamed ~10px icons render as nothing "
                  "(raster fallback min_side 20px)",
                  "R4 test_icons::test_foreign_glyph_not_misnamed: hexagon -> brightness_1, 7px ring bullets -> "
                  "circle_notifications / report"],
    param_space={"theme": ["light", "light", "dark"], "shape": list(SHAPES), "mix": ["uniform", "mixed"],
                 "layout": ["bullets", "bullets", "trailing", "rating"], "color": ["accent", "ink"],
                 "glyph_px": (6, 20), "font_px": (12, 18), "weight": [400, 500], "gap": (4, 16), "row_pad": (4, 16),
                 "n_items": (2, 6), "margin": (12, 48), "width": (320, 520)},
    generate=generate,
    criteria=[Criterion("glyph_reproduced", ">=", 1.0, doc="every glyph re-rendered within scenarios.glyph.max_de"),
              Criterion("glyph_misnamed", "<=", 0, doc="no glyph named as a Material Symbol")],
    source="html",
    tune_prefixes=("perceive.icons.", "refine.raster.", "refine.critic."),
    metrics=metrics,
)
