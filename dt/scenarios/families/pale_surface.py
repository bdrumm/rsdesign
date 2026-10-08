"""pale_surface: low-contrast cards / bands that segmentation drops.

Generalises knowledge/failures.jsonl R3 "a pale illustration card (#f3edf7 on #fdf7fe) ... loses
its card background" (stage perceive: card fill within bg_dist of the page background so
segmentation never emits it). The family spans the whole mode: 1.5-5 ΔE2000 surfaces on light
and dark pages, darker or lighter than the page, any hue, radius 0-28, with/without an M3
level-1 shadow, 1-3 surfaces, empty or with content, full-width bands or inset cards.

Success: every surface is recovered as a non-raster node (IoU >= 0.8) whose fill is within
1 ΔE2000 of the truth, and the surfaces' region re-renders under the visually-identical JND gate.
"""
from __future__ import annotations

from dt.ir import Box, Color, Document
from dt.params import P, register
from dt.scenarios.families._common import (ACCENTS, DARK_ACCENTS, DARK_BGS, DARK_INK, LIGHT_BGS, LIGHT_INK, doc_nodes,
                                           is_raster, phrase, pick, rect, rng_for, text_node)
from dt.scenarios.sources import color_at_de, de2000, ir_case
from dt.scenarios.spec import Criterion, ScenarioFamily
from dt.selftest.grammar import M3_ELEVATION

register("scenarios.pale.match_iou", 0.8, "pale_surface: IoU a predicted node needs with a surface to count as found", (0.5, 0.95))
register("scenarios.pale.fill_de", 1.0, "pale_surface: max ΔE2000 between the found node's fill and the surface fill", (0.3, 3.0))

NAME = "pale_surface"


def generate(p: dict, seed: int):
    rng = rng_for(seed, NAME)
    dark = p["theme"] == "dark"
    bg = Color.from_hex(pick(rng, DARK_BGS if dark else LIGHT_BGS))
    ink = pick(rng, DARK_INK if dark else LIGHT_INK)
    lighter = dark if p["direction"] == "auto" else not dark
    L = bg.lab()[0]
    if lighter and L > 97.5:  # nothing lighter than white: fall back to the usual direction
        lighter = False
    if not lighter and L < 3.0:
        lighter = True
    surface = color_at_de(bg, float(p["de"]), hue_deg=float(p["hue"]), chroma_frac=float(p["chroma_frac"]), lighter=lighter)
    W, H = int(p["width"]), int(p["height"])
    doc = Document.blank(W, H, bg)
    n = int(p["n_surfaces"])
    top, gap = 16, int(p["gap"])
    slot = (H - 2 * top - (n - 1) * gap) / n
    surfaces = []
    for i in range(n):
        h = max(48.0, slot * rng.uniform(0.7, 1.0))
        y = top + i * (slot + gap) + (slot - h) / 2
        if p["layout"] == "band":
            x, w, r = 0.0, float(W), 0.0
        else:
            mx = rng.randint(12, 40)
            x, w, r = float(mx), float(W - 2 * mx), float(min(p["radius"], h / 2))
        box = Box(round(x), round(y), round(w), round(h))
        node = rect(f"surface{i}", box, surface, r)
        if p["shadow"] and p["layout"] != "band":
            node.effects = list(M3_ELEVATION[1])
        doc.root.children.append(node)
        surfaces.append({"box": box, "fill": surface.hex()})
        if p["content"] != "none" and box.h >= 56:
            pad = 16
            node.children.append(text_node(phrase(rng, 1, 3), box.x + pad, box.y + pad, 16, ink, 500, 24))
            if box.h >= 88:
                node.children.append(text_node(phrase(rng, 3, 6, cap=True), box.x + pad, box.y + pad + 28, 14, ink, 400, 20))
            if p["content"] == "text_button" and box.h >= 120 and box.w >= 160:
                accent = pick(rng, DARK_ACCENTS if dark else ACCENTS)
                bw, bh = 96, 40
                btn = rect("button", Box(box.x2 - pad - bw, box.y2 - pad - bh, bw, bh), accent, 20)
                lab = text_node("Open", btn.box.x + 30, btn.box.y + 10, 14, "#ffffff" if not dark else "#381e72", 500, 20)
                btn.children.append(lab)
                node.children.append(btn)
        if node.children:
            node.type = "frame"
    u = surfaces[0]["box"]
    for s in surfaces[1:]:
        u = u.union(s["box"])
    roi = u.expand(8).intersect(Box(0, 0, W, H))
    meta = {"surfaces": surfaces, "bg": bg.hex(), "surface_de": de2000(bg, surface),
            "highlight_boxes": [s["box"] for s in surfaces]}
    return ir_case(NAME, seed, p, doc, roi=roi, meta=meta)


def metrics(case, pred, rendered) -> dict:
    nodes = [n for n in doc_nodes(pred) if n is not pred.root and n.type in ("frame", "rect", "instance", "ellipse", "image")]
    thr, max_de = float(P["scenarios.pale.match_iou"]), float(P["scenarios.pale.fill_de"])
    found, fill_des = 0, []
    for s in case.meta["surfaces"]:
        best, best_iou = None, thr
        for n in nodes:
            iou = n.box.iou(s["box"])
            if iou >= best_iou and not is_raster(n) and n.fill_color is not None:
                best, best_iou = n, iou
        if best is None:
            continue
        d = de2000(best.fill_color, Color.from_hex(s["fill"]))
        fill_des.append(d)
        found += int(d <= max_de)
    return {"surface_recall": found / len(case.meta["surfaces"]),
            "surface_fill_de": max(fill_des) if fill_des else None,
            "surface_de_from_bg": case.meta["surface_de"]}


FAMILY = ScenarioFamily(
    name=NAME,
    description="Low-contrast surfaces (1.5-5 ΔE2000 from the page) must be perceived as filled nodes and re-rendered "
                "within the JND.",
    stage="perceive",
    failure_refs=["R3 real/m3_buttons_overview_types_mobile (embedded button illustration): pale card #f3edf7 on #fdf7fe "
                  "loses its card background (perceive: fill within bg_dist of the page background)"],
    param_space={"theme": ["light", "dark"], "de": (1.5, 5.0), "hue": (0.0, 360.0), "chroma_frac": (0.0, 0.8),
                 "direction": ["auto", "auto", "invert"], "radius": (0, 28), "shadow": [False, True],
                 "n_surfaces": (1, 3), "layout": ["card", "card", "band"], "content": ["none", "text", "text_button"],
                 "width": (320, 520), "height": (280, 460), "gap": (12, 32)},
    generate=generate,
    criteria=[Criterion("surface_recall", ">=", 1.0, doc="every surface found as a filled node with the right fill"),
              Criterion("jnd_frac_nontext", "<=", "param:validate.gate.ident.jnd_frac", roi="roi", scale=0.05,
                        doc="surfaces re-render under the visually-identical JND gate")],
    source="ir",
    tune_prefixes=("perceive.seg.",),
    metrics=metrics,
)
