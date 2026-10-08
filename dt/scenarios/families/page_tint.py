"""page_tint: a whole page (or most of it) tinted 1-4 ΔE2000 away from a white / near-white app bar.

Generalises knowledge/failures.jsonl R3 "geometry nearly exact ... but 19% of non-text pixels are
past JND: the whole page background is ~2.5 dE2000 off" (root fill estimated from border
sampling picks the white app bar; refine never recolours the root because per-node error is
below refine.critic.min_node_de although the error *mass* is huge). The family varies the tint
(1-4 ΔE, any hue, chroma), the bar (white or within 1 ΔE of it, 48-72 px, optional bottom bar),
cards painted in the bar colour (more white area that votes for the wrong background), page
size and light/dark theme (dark page vs near-black bar).

Success = the full translate (perceive -> map -> refine k) brings the non-text JND fraction of the
whole page under the visually-identical gate.
"""
from __future__ import annotations

from dt.ir import Box, Color, Document
from dt.scenarios.families._common import (ACCENTS, DARK_ACCENTS, DARK_INK, LIGHT_INK, phrase, pick, rect, rng_for,
                                           text_node)
from dt.scenarios.sources import color_at_de, de2000, ir_case
from dt.scenarios.spec import Criterion, ScenarioFamily

NAME = "page_tint"


def generate(p: dict, seed: int):
    rng = rng_for(seed, NAME)
    dark = p["theme"] == "dark"
    ref = Color.from_hex("#000000" if dark else "#ffffff")
    bar = color_at_de(ref, float(p["bar_de"]), hue_deg=float(p["hue"]) + 180.0, chroma_frac=0.2, lighter=dark) \
        if float(p["bar_de"]) > 0.05 else ref
    page = color_at_de(bar, float(p["tint_de"]), hue_deg=float(p["hue"]), chroma_frac=float(p["chroma_frac"]), lighter=dark)
    ink = pick(rng, DARK_INK if dark else LIGHT_INK)
    accent = pick(rng, DARK_ACCENTS if dark else ACCENTS)
    W, H = int(p["width"]), int(p["height"])
    doc = Document.blank(W, H, page)
    bh = int(p["bar_h"])
    top = rect("app-bar", Box(0, 0, W, bh), bar)
    top.type = "frame"
    top.children.append(text_node(phrase(rng, 1, 2), 16, (bh - 28) / 2, 22, ink, 400, 28))
    doc.root.children.append(top)
    y = bh + 16
    bottom_h = int(p["bottom_h"]) if p["bottom_bar"] else 0
    limit = H - bottom_h - 16
    for i in range(int(p["n_text"])):
        if y + 24 > limit:
            break
        doc.root.children.append(text_node(phrase(rng, 2, 6), 16, y, 14 if i else 16, ink, 500 if i == 0 else 400, 20))
        y += 28
    for i in range(int(p["n_cards"])):
        ch = rng.randint(64, 140)
        if y + ch > limit:
            break
        card = rect(f"card{i}", Box(16, y, W - 32, ch), bar, float(p["card_radius"]))
        card.type = "frame"
        card.children.append(text_node(phrase(rng, 1, 3), 32, y + 16, 16, ink, 500, 24))
        if ch >= 100:
            card.children.append(text_node(phrase(rng, 3, 6), 32, y + 44, 14, ink, 400, 20))
        doc.root.children.append(card)
        y += ch + 12
    if p["fab"] and limit - 72 > y:
        fab = rect("fab", Box(W - 72, limit - 56, 56, 56), accent, 16)
        doc.root.children.append(fab)
    if bottom_h:
        nav = rect("bottom-bar", Box(0, H - bottom_h, W, bottom_h), bar)
        doc.root.children.append(nav)
    meta = {"page": page.hex(), "bar": bar.hex(), "tint_de": de2000(page, bar), "highlight_boxes": []}
    return ir_case(NAME, seed, p, doc, roi=None, meta=meta)


def metrics(case, pred, rendered) -> dict:
    """``page_bg_de``: ΔE2000 between the rendered and target colour on pixels where the target
    shows the page tint (informational; the gate is the validator's JND fraction)."""
    import numpy as np

    from dt.validate.fidelity import conform_to_target, delta_e2000_map
    page = Color.from_hex(case.meta["page"])
    t = case.target_rgb.astype(np.int16)
    mask = (np.abs(t - np.array([page.r, page.g, page.b], dtype=np.int16)).max(axis=2) <= 1)
    if not mask.any():
        return {"page_bg_de": None, "tint_de": case.meta["tint_de"]}
    r, _ = conform_to_target(case.target_rgb, rendered)
    de = delta_e2000_map(case.target_rgb, r)
    root = pred.root.fill_color
    return {"page_bg_de": float(de[mask].mean()), "tint_de": case.meta["tint_de"],
            "root_fill_de": de2000(root, page) if root is not None else None}


FAMILY = ScenarioFamily(
    name=NAME,
    description="A page tinted 1-4 ΔE2000 away from a white/near-white app bar must be reproduced under the "
                "visually-identical JND gate after refine.",
    stage="translate",
    refine_iters=4,
    failure_refs=["R3 real/m3_buttons_overview_types_mobile: whole page background ~2.5 dE2000 off (root fill from "
                  "border sampling = white app bar; refine never recolours the root)"],
    param_space={"theme": ["light", "light", "dark"], "tint_de": (1.0, 4.0), "hue": (0.0, 360.0),
                 "chroma_frac": (0.2, 0.9), "bar_de": (0.0, 1.0), "bar_h": (48, 72), "bottom_bar": [False, True],
                 "bottom_h": (56, 80), "n_text": (0, 4), "n_cards": (0, 3), "card_radius": (0, 16), "fab": [False, True],
                 "width": (360, 600), "height": (480, 720)},
    generate=generate,
    criteria=[Criterion("jnd_frac_nontext", "<=", "param:validate.gate.ident.jnd_frac", scale=0.05,
                        doc="whole page under the visually-identical JND gate")],
    source="ir",
    tune_prefixes=("perceive.seg.", "refine.critic."),
    metrics=metrics,
)
