"""Icon identification (dt.perceive.icons): render known Material Symbols, identify them from pixels,
re-render the identification and require the re-render to match the original crop.

Accuracy is measured as *visual identity* (mean ΔE2000 of the re-rendered crop), not name equality:
Material Symbols has many aliases with identical glyphs (star/grade, check/done), and for a
pixel-faithful translation any of them is correct. Name accuracy is reported but not gated.
"""
import random

import numpy as np
import pytest

from dt.common.image import crop
from dt.ir import Box, Color, Document, Fill, Node
from dt.perceive import icons as IC
from dt.render.screenshot import render_doc
from dt.validate.fidelity import delta_e2000_map

CELL = 56
COLS = 10
SIZES = [18, 20, 24, 24, 32, 40]
SCHEMES = [("#ffffff", "#49454f"), ("#fef7ff", "#1d1b20"), ("#1d1b20", "#e6e0e9"), ("#6750a4", "#ffffff"), ("#e8def8", "#1d192b")]


def _common_sample(n: int, seed: int = 3) -> list[str]:
    names = sorted(IC.common_icons() & set(IC._read_names()))
    return random.Random(seed).sample(names, min(n, len(names)))


def _grid(names: list[str]) -> tuple[Document, list[tuple[str, Box, str, str]]]:
    rows = (len(names) + COLS - 1) // COLS
    doc = Document.blank(COLS * CELL, rows * CELL, "#ffffff")
    truth = []
    for i, nm in enumerate(names):
        sz = SIZES[i % len(SIZES)]
        bg, fg = SCHEMES[i % len(SCHEMES)]
        x, y = (i % COLS) * CELL, (i // COLS) * CELL
        doc.root.children.append(Node(type="rect", box=Box(x, y, CELL, CELL), fills=[Fill.solid(bg)]))
        b = Box(x + (CELL - sz) / 2, y + (CELL - sz) / 2, sz, sz)
        doc.root.children.append(Node(type="icon", icon_name=nm, box=b, fills=[Fill.solid(fg)]))
        truth.append((nm, b, bg, fg))
    return doc, truth


@pytest.fixture(scope="module")
def grid():
    names = _common_sample(60)
    doc, truth = _grid(names)
    return doc, truth, render_doc(doc)


def _perceived_doc(doc: Document, truth, target: np.ndarray) -> Document:
    """What perceive would hand annotate_icons: icon nodes at the tight ink box, unnamed."""
    out = Document.blank(doc.width, doc.height, "#ffffff")
    for nm, b, bg, fg in truth:
        cell = Box(b.cx - CELL / 2, b.cy - CELL / 2, CELL, CELL)
        out.root.children.append(Node(type="rect", box=cell, fills=[Fill.solid(bg)]))
        sub = crop(target, cell.expand(-4))
        _, ink = IC.ink_mask_from_crop(sub, Color.from_hex(bg))
        ink_abs = ink.translate(int(cell.x) + 4, int(cell.y) + 4)
        out.root.children.append(Node(type="icon", box=ink_abs, fills=[Fill.solid(fg)], meta={"bg": Color.from_hex(bg).to_dict()}))
    return out


def test_atlas_loads_both_fill_axes():
    atlas = IC.load_atlas((0, 1))
    n = len(IC._read_names())
    assert len(atlas.names) == 2 * n and atlas.masks.shape[1:] == (IC.CANVAS, IC.CANVAS)
    assert set(np.unique(atlas.fills)) == {0, 1}


def test_identified_icons_render_identically(grid):
    doc, truth, target = grid
    pdoc = _perceived_doc(doc, truth, target)
    summary = IC.annotate_icons(pdoc, target)
    assert summary["icons"] == len(truth)
    rendered = render_doc(pdoc)
    de = delta_e2000_map(target, rendered)
    identical, named_right, box_ok = 0, 0, 0
    icons = [n for n in pdoc.walk() if n.type == "icon"]
    fails = []
    for (nm, b, bg, fg), n in zip(truth, icons):
        x0, y0, x1, y1 = b.expand(4).as_int()
        cell_de = float(de[y0:y1, x0:x1].mean())
        if cell_de < 1.0:
            identical += 1
        else:
            fails.append((nm, n.icon_name, round(cell_de, 2)))
        named_right += int(n.icon_name == nm)
        box_ok += int(abs(n.box.x - b.x) <= 1 and abs(n.box.y - b.y) <= 1 and abs(n.box.w - b.w) <= 1.5)
    k = len(truth)
    assert identical / k >= 0.9, (identical, k, fails[:10])
    assert box_ok / k >= 0.9, box_ok
    assert named_right / k >= 0.75, named_right  # informational floor; aliases make 100% impossible


def test_unknown_glyph_stays_unnamed():
    # a filled square blob is not a Material Symbol at any good score -> candidates recorded, no name
    doc = Document.blank(80, 80, "#ffffff")
    doc.root.children.append(Node(type="rect", box=Box(30, 30, 5, 21), fills=[Fill.solid("#000000")], radius=(2, 2, 2, 2)))
    img = render_doc(doc)
    p = Document.blank(80, 80, "#ffffff")
    p.root.children.append(Node(type="icon", box=Box(30, 30, 5, 21), fills=[Fill.solid("#000000")]))
    IC.annotate_icons(p, img)
    ic = p.root.children[0]
    assert "icon_candidates" in ic.meta
    if ic.icon_name is not None:  # if something matched, the renderer must have accepted its silhouette
        # (this used to read meta["icon_score"], which annotate_icons never sets: a KeyError waiting to happen)
        assert ic.meta["icon_mis_e"] <= IC.P["perceive.icons.accept_mis"]
        assert ic.meta["icon_de"] <= IC.P["perceive.icons.accept_de"]


# ---- adversarial cases (skeptic icons-fonts)
_HEXAGON = ("<svg style='position:absolute;left:18px;top:18px' width='20' height='20' viewBox='0 0 24 24' fill='currentColor'>"
            "<polygon points='12 2 21 7 21 17 12 22 3 17 3 7'/></svg>")
_RING7 = "<div style='position:absolute;left:20px;top:20px;width:7px;height:7px;box-sizing:border-box;border:1.5px solid;border-radius:50%'></div>"


@pytest.mark.parametrize("name,markup,bg,fg", [
    ("solid hexagon (another icon set)", _HEXAGON, "#ffffff", "#49454f"),        # was brightness_1 (a circle), mean ΔE 1.32
    ("7px ring bullet (CSS border)", _RING7, "#fff8f0", "#7d5700"),               # was 'report' (octagon with '!'), mean ΔE 1.74
])
def test_foreign_glyph_not_misnamed(name, markup, bg, fg):
    """Glyphs that are not Material Symbols must stay unnamed unless a symbol reproduces their silhouette:
    the mean crop ΔE alone is diluted by background and accepted both of these."""
    from dt.render.screenshot import html_to_png
    img = html_to_png(f"<!doctype html><body style='margin:0;background:{bg};color:{fg}'>{markup}</body>", 56, 56)
    _, ink = IC.ink_mask_from_crop(img, Color.from_hex(bg))
    p = Document.blank(img.shape[1], img.shape[0], bg)
    p.root.children.append(Node(type="icon", box=ink, fills=[Fill.solid(fg)], meta={"bg": Color.from_hex(bg).to_dict()}))
    IC.annotate_icons(p, img)
    ic = p.root.children[0]
    assert ic.icon_name is None, (name, ic.icon_name, ic.meta.get("icon_candidates"))


def test_glyph_mismatch_ignores_resampling_but_not_shape():
    """The silhouette measure must be ~0 for the same glyph rendered at 2x and area-downscaled (anti-aliasing
    differs everywhere) and large for a different glyph of similar mass."""
    import cv2
    bg, fg = Color.from_hex("#ffffff"), Color.from_hex("#1d1b20")

    def render(name, scale, fill=0):
        d = Document.blank(48 * scale, 48 * scale, "#ffffff")
        d.root.children.append(Node(type="icon", icon_name=name, box=Box(12 * scale, 12 * scale, 24 * scale, 24 * scale),
                                    fills=[Fill.solid(fg)], meta={"icon_fill": fill}))
        img = render_doc(d)
        return img if scale == 1 else cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)

    for name in ("radio_button_unchecked", "settings", "star"):
        a, b = IC._coverage(render(name, 1), bg, fg), IC._coverage(render(name, 2), bg, fg)
        assert IC.glyph_mismatch(a, b)[1] <= 0.02, name
    a, b = IC._coverage(render("circle", 1, 1), bg, fg), IC._coverage(render("square", 1, 1), bg, fg)
    assert IC.glyph_mismatch(a, b)[1] > 2 * IC.P["perceive.icons.accept_mis"]


def test_fill1_coloured_2x_downscaled_icons_identified():
    """Filled glyphs, coloured on tinted surfaces, captured at 2x and downscaled: still named and identical."""
    import cv2
    names = _common_sample(20, seed=11)
    schemes = [("#d3e3fd", "#0b57d0"), ("#f9dedc", "#b3261e"), ("#c4eed0", "#146c2e"), ("#1d1b20", "#d0bcff")]
    doc, truth = _grid(names)
    big = Document.blank(doc.width * 2, doc.height * 2, "#ffffff")
    truth2 = []
    for i, (nm, b, _, _) in enumerate(truth):
        bg, fg = schemes[i % len(schemes)]
        cell = Box(b.cx - CELL / 2, b.cy - CELL / 2, CELL, CELL)
        big.root.children.append(Node(type="rect", box=Box(cell.x * 2, cell.y * 2, CELL * 2, CELL * 2), fills=[Fill.solid(bg)]))
        big.root.children.append(Node(type="icon", icon_name=nm, box=Box(b.x * 2, b.y * 2, b.w * 2, b.h * 2), fills=[Fill.solid(fg)],
                                      meta={"icon_fill": 1}))
        truth2.append((nm, b, bg, fg))
    hi = render_doc(big)
    target = cv2.resize(hi, (doc.width, doc.height), interpolation=cv2.INTER_AREA)
    pdoc = _perceived_doc(doc, truth2, target)
    IC.annotate_icons(pdoc, target)
    de = delta_e2000_map(target, render_doc(pdoc))
    icons = [n for n in pdoc.walk() if n.type == "icon"]
    identical = sum(float(de[slice(*b.expand(4).as_int()[1::2]), slice(*b.expand(4).as_int()[0::2])].mean()) < 1.0
                    for (_, b, _, _) in truth2)
    assert sum(n.icon_name is not None for n in icons) >= 0.9 * len(truth2)
    assert identical >= 0.9 * len(truth2), identical
