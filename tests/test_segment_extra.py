"""Hollow (outline-only) containers and dialog scrims in dt.perceive.segment.

Ground truth comes from the REAL renderer: each test builds an IR document, renders it with
dt.render and checks what segmentation recovers.

Run: python -m pytest tests/test_segment_extra.py -q
"""
from __future__ import annotations

import contextlib

import numpy as np
import pytest

from dt.compare import diff_map
from dt.ir import Box, Color, Document, Fill, Node, Stroke, TextStyle
from dt.params import P
from dt.perceive import perceive
from dt.perceive.hierarchy import build_tree
from dt.perceive.segment import apply_scrim, detect_scrim, estimate_background, find_regions
from dt.render.screenshot import render_doc

SURFACE = "#fef7ff"
OUTLINE = "#79747e"
SCRIM_A = 0.32


@contextlib.contextmanager
def _param(key, value):
    old = P[key]
    P.set(key, value)
    try:
        yield
    finally:
        P.set(key, old)


def _text(x, y, s, size=12.0, color="#49454f", w=120.0) -> Node:
    return Node(type="text", name=s, box=Box(x, y, w, size * 1.2 + 2), text=s,
                text_style=TextStyle(size=size, weight=400, color=Color.from_hex(color)))


def _near(a, b, tol) -> bool:
    return abs(a - b) <= tol


def _region_at(regions, box: Box, tol=2.0):
    for r in regions:
        b = r.box
        if _near(b.x, box.x, tol) and _near(b.y, box.y, tol) and _near(b.w, box.w, tol) and _near(b.h, box.h, tol):
            return r
    raise AssertionError(f"no region within ±{tol}px of {box}: " + ", ".join(f"{r.kind}{r.box}" for r in regions))


# --------------------------------------------------------------------------- hollow
def _outlined_text_field_doc() -> tuple[Document, Box]:
    """M3 outlined text field: 1px outline, radius 4, floating label whose background patch cuts
    a gap into the top edge (exactly how Material Web paints it)."""
    doc = Document.blank(480, 140, SURFACE)
    box = Box(24, 40, 400, 56)
    field = Node(type="frame", name="TextField/outlined", box=box,
                 strokes=[Stroke(color=Color.from_hex(OUTLINE), width=1, align="inside")], radius=(4, 4, 4, 4))
    label_bg = Node(type="rect", name="label-bg", box=Box(36, 30, 46, 20), fills=[Fill.solid(SURFACE)])
    field.children = [label_bg, _text(40, 32, "Notes", 12, "#49454f", 40)]
    doc.root.children.append(field)
    return doc, box


def test_outlined_text_field_with_label_gap_is_stroke_only():
    doc, box = _outlined_text_field_doc()
    rgb = render_doc(doc)
    # no OCR boxes at all: the label gap in the ring is NOT masked, the ring is open
    regs = find_regions(rgb, [])
    r = _region_at(regs, box, 2)
    assert r.fill is None, r.fill
    assert r.stroke is not None and r.stroke[0].delta_e(Color.from_hex(OUTLINE)) < 3 and r.stroke[1] == 1, r.stroke
    assert _near(r.radius, 4, 1.5), (r.radius, r.meta.get("corners"))
    assert r.meta.get("hollow") is True
    # no solid box in the stroke colour anywhere
    assert not [g for g in regs if g.fill is not None and g.fill.delta_e(Color.from_hex(OUTLINE)) < 5 and g.box.area > 200]


def test_outlined_text_field_full_perceive_renders_hollow():
    doc, box = _outlined_text_field_doc()
    rgb = render_doc(doc)
    got = perceive(rgb)
    n = next(n for n in got.walk() if n.id != "root" and n.type != "text" and _near(n.box.x, box.x, 2)
             and _near(n.box.y, box.y, 2) and _near(n.box.w, box.w, 2) and _near(n.box.h, box.h, 2))
    assert not n.fills and n.strokes and n.strokes[0].color.delta_e(Color.from_hex(OUTLINE)) < 3
    # the re-render of the empty field's interior is the page surface, not the outline colour
    re = render_doc(got)
    inner = (slice(int(box.y) + 12, int(box.y2) - 12), slice(int(box.x) + 60, int(box.x2) - 12))
    assert float(diff_map(rgb[inner], re[inner]).mean()) < 1.0


@pytest.mark.parametrize("stroke", ["#cac4d0", "#79747e"])
def test_outlined_card_radius_12_is_stroke_only(stroke):
    doc = Document.blank(400, 260, SURFACE)
    box = Box(30, 30, 300, 180)
    card = Node(type="frame", name="Card/outlined", box=box,
                strokes=[Stroke(color=Color.from_hex(stroke), width=1, align="inside")], radius=(12, 12, 12, 12))
    card.children = [_text(54, 54, "Card title", 16, "#1d1b20", 120), _text(54, 90, "Supporting text", 14, "#49454f", 140)]
    doc.root.children.append(card)
    rgb = render_doc(doc)
    regs = find_regions(rgb, [n.box for n in card.children])
    r = _region_at(regs, box, 2)
    assert r.fill is None and r.stroke is not None
    assert r.stroke[0].delta_e(Color.from_hex(stroke)) < 3 and r.stroke[1] == 1
    assert _near(r.radius, 12, 2), (r.radius, r.meta.get("corners"))


def test_open_ring_does_not_swallow_solid_shapes():
    """A filled rect, an L-shaped pair of dividers and a filled card keep their old classification."""
    doc = Document.blank(400, 200, SURFACE)
    doc.root.children += [
        Node(type="rect", box=Box(20, 20, 120, 60), fills=[Fill.solid("#e8def8")], radius=(8, 8, 8, 8)),
        Node(type="line", box=Box(200, 40, 160, 1), fills=[Fill.solid("#cac4d0")]),
        Node(type="line", box=Box(200, 40, 1, 120), fills=[Fill.solid("#cac4d0")]),
    ]
    regs = find_regions(render_doc(doc), [])
    r = _region_at(regs, Box(20, 20, 120, 60), 1)
    assert r.fill is not None and r.fill.delta_e(Color.from_hex("#e8def8")) < 3 and r.stroke is None
    assert not any(g.meta.get("hollow") for g in regs), [(g.kind, g.box) for g in regs]


# --------------------------------------------------------------------------- scrim
CARDS = [  # (box, fill, stroke, radius)
    (Box(24, 24, 260, 150), "#e6e0e9", None, 12),
    (Box(316, 24, 260, 150), "#eaddff", None, 12),
    (Box(24, 316, 260, 140), None, "#cac4d0", 12),  # outlined card, not occluded by the dialog
    (Box(316, 316, 260, 140), "#f3edf7", None, 12),
]
BUTTON = (Box(476, 120, 88, 40), "#6750a4")
DIALOG = (Box(170, 120, 280, 170), "#ece6f0", 28)


def _scrim_doc(with_scrim: bool = True) -> Document:
    doc = Document.blank(600, 480, SURFACE)
    for box, fill, stroke, rad in CARDS:
        n = Node(type="rect", box=box, radius=(rad,) * 4)
        if fill:
            n.fills = [Fill.solid(fill)]
        if stroke:
            n.strokes = [Stroke(color=Color.from_hex(stroke), width=1, align="inside")]
        doc.root.children.append(n)
    doc.root.children.append(Node(type="rect", box=BUTTON[0], fills=[Fill.solid(BUTTON[1])], radius=(20,) * 4))
    if with_scrim:
        doc.root.children.append(Node(type="rect", name="scrim", box=Box(0, 0, 600, 480), fills=[Fill.solid(Color(0, 0, 0, SCRIM_A))]))
        dlg = Node(type="frame", name="Dialog", box=DIALOG[0], fills=[Fill.solid(DIALOG[1])], radius=(DIALOG[2],) * 4)
        dlg.children = [Node(type="rect", box=Box(330, 234, 96, 40), fills=[Fill.solid("#6750a4")], radius=(20,) * 4)]
        doc.root.children.append(dlg)
    return doc


def test_scrim_detected_with_alpha_and_true_colours():
    rgb = render_doc(_scrim_doc())
    sc = detect_scrim(rgb)
    assert sc is not None
    assert _near(sc.alpha, SCRIM_A, 0.04), sc.alpha
    assert sc.page_bg.delta_e(Color.from_hex(SURFACE)) < 3
    assert estimate_background(rgb).delta_e(Color.from_hex(SURFACE)) < 3

    regs = find_regions(rgb, [])
    scrims = [r for r in regs if r.meta.get("scrim")]
    assert len(scrims) == 1
    s = scrims[0]
    assert s.box.x == 0 and s.box.y == 0 and s.box.w == 600 and s.box.h == 480
    assert s.fill is not None and _near(s.fill.a, SCRIM_A, 0.04) and s.fill.delta_e(Color(0, 0, 0)) < 3

    # dialog surface: clear (above the scrim), observed colour is its true colour
    d = _region_at(regs, DIALOG[0], 2)
    assert d.meta.get("scrim_clear") and not d.meta.get("under_scrim")
    assert d.fill.delta_e(Color.from_hex(DIALOG[1])) < 3
    assert _near(d.radius, DIALOG[2], 3)

    # cards under the scrim: true fills recovered by un-blending (ΔE < 3)
    for box, fill, stroke, _ in CARDS:
        if fill is None or box.contains(DIALOG[0]):
            continue
        r = next((r for r in regs if r.meta.get("under_scrim") and r.fill is not None and r.box.iou(box) > 0.6), None)
        if fill == "#f3edf7":  # surface-container: darkened it is within bg_dist of the darkened page
            continue
        assert r is not None, (box, [(g.kind, g.box, g.meta) for g in regs])
        assert Color.from_hex(r.meta["true_fill"]).delta_e(Color.from_hex(fill)) < 3, (fill, r.meta["true_fill"])
    oc = next(r for r in regs if r.meta.get("under_scrim") and r.stroke is not None and r.box.iou(CARDS[2][0]) > 0.6)
    assert oc.fill is None and Color.from_hex(oc.meta["true_stroke"]).delta_e(Color.from_hex("#cac4d0")) < 3
    btn = next(r for r in regs if r.meta.get("under_scrim") and r.box.iou(BUTTON[0]) > 0.8)
    assert Color.from_hex(btn.meta["true_fill"]).delta_e(Color.from_hex(BUTTON[1])) < 3


def test_scrim_unblend_mode_puts_true_fill_on_region():
    rgb = render_doc(_scrim_doc())
    with _param("perceive.seg.scrim_unblend", 1):
        regs = find_regions(rgb, [])
    r = next(r for r in regs if r.meta.get("under_scrim") and r.fill is not None and r.box.iou(CARDS[1][0]) > 0.8)
    assert r.fill.delta_e(Color.from_hex(CARDS[1][1])) < 3
    assert Color.from_hex(r.meta["blended_fill"]).delta_e(Color.from_hex(CARDS[1][1])) > 10


@pytest.mark.parametrize("unblend", [0, 1])
def test_apply_scrim_restacks_tree_and_rerenders_pixel_exact(unblend):
    """apply_scrim on the perceived tree: under-scrim subtrees with true colours below the scrim,
    the dialog above it, root = true page colour; the re-render reproduces the target."""
    target = render_doc(_scrim_doc())
    with _param("perceive.seg.scrim_unblend", unblend), _param("perceive.scrim.restack", False):
        got = perceive(target)
    assert apply_scrim(got.root)
    with _param("perceive.scrim.restack", False):
        assert not apply_scrim(perceive(render_doc(_scrim_doc(with_scrim=False))).root)
    kids = got.root.children
    si = next(i for i, c in enumerate(kids) if c.meta.get("scrim"))
    assert not kids[si].children and kids[si].type == "rect"
    assert all(c.meta.get("under_scrim") for c in kids[:si] if c.type != "text")
    assert kids[si + 1:] and all(c.meta.get("scrim_clear") for c in kids[si + 1:] if c.type != "text")
    assert got.root.fill_color.delta_e(Color.from_hex(SURFACE)) < 3
    for box, fill, _, _ in CARDS[:2]:
        n = next(n for n in got.walk() if n.meta.get("under_scrim") and n.fills and n.box.iou(box) > 0.6)
        assert n.fill_color.delta_e(Color.from_hex(fill)) < 3, (fill, n.fill_color.hex())
    re = render_doc(got)
    dm = diff_map(target, re)
    assert float(dm.mean()) < 1.0, float(dm.mean())
    assert float((dm > 5).mean()) < 0.02


def test_default_tree_reproduces_scrimmed_background():
    """Without apply_scrim (plain containment nesting) the root carries the true page colour and the
    full-page scrim frame darkens it to the observed colour; observed fills stay on the regions."""
    target = render_doc(_scrim_doc())
    got = perceive(target)
    assert got.root.fill_color.delta_e(Color.from_hex(SURFACE)) < 3
    sn = [n for n in got.walk() if n.meta.get("scrim")]
    assert len(sn) == 1 and sn[0].fills and _near(sn[0].fills[0].color.a, SCRIM_A, 0.04)
    re = render_doc(got)
    seen = Color.from_hex(SURFACE)
    bgpix = np.all(np.abs(target.astype(int) - target[2, 2].astype(int)) <= 1, axis=-1)
    assert bgpix.mean() > 0.3
    assert float(diff_map(target, re)[bgpix].mean()) < 0.5
    assert seen.delta_e(Color(*target[2, 2])) > 10  # the observed page colour is the darkened one


def test_no_scrim_on_plain_pages():
    assert detect_scrim(render_doc(_scrim_doc(with_scrim=False))) is None
    # a dark-ish grey page with white cards is not a scrim unless other colours fit the same blend
    doc = Document.blank(400, 300, "#ded8e1")
    doc.root.children.append(Node(type="rect", box=Box(20, 20, 200, 120), fills=[Fill.solid("#ffffff")], radius=(12,) * 4))
    assert detect_scrim(render_doc(doc)) is None
    with _param("perceive.seg.scrim_enable", 0):
        assert detect_scrim(render_doc(_scrim_doc())) is None


def test_new_params_registered_with_docs_and_ranges():
    keys = [k for k in P.defaults() if k.startswith("perceive.seg.") and any(
        s in k for s in ("scrim", "hollow_bg", "ring_", "split_min", "split_colors", "split_core"))]
    assert len(keys) >= 16, keys
    docs, ranges = P.docs(), P.ranges()
    for k in keys:
        assert docs.get(k), k
        lo, hi = ranges[k]
        assert lo <= P.defaults()[k] <= hi, k



def test_perceive_restacks_scrim_by_default():
    """perceive() applies apply_scrim itself: the scrim is a top-level leaf, the dialog paints above it,
    and the re-render still reproduces the target."""
    target = render_doc(_scrim_doc())
    got = perceive(target)
    assert got.meta["perceive"]["scrim"] is True
    kids = got.root.children
    si = next(i for i, c in enumerate(kids) if c.meta.get("scrim"))
    assert not kids[si].children
    assert any(c.meta.get("scrim_clear") for c in kids[si + 1:])
    assert float(diff_map(target, render_doc(got)).mean()) < 1.0


# --------------------------------------------------------------------------- occluded corners
def _occluded_card_docs():
    """A radius-16 card whose two left corners are covered: by a translucent overlapping card, and
    by two opaque badges sitting on the corners (overlapping layers are a curriculum item in
    docs/SELF_IMPROVEMENT.md). A straight cut by an abutting panel is NOT this case: the visible
    shape then really has square corners (per-corner radii, a known gap)."""
    under = Node(type="rect", box=Box(40, 40, 300, 200), fills=[Fill.solid("#6750a4")], radius=(16, 16, 16, 16))
    card = Node(type="rect", box=Box(200, 120, 300, 200), fills=[Fill.solid(Color(255, 0, 0, 0.5))], radius=(16, 16, 16, 16))
    glass = Node(type="rect", box=Box(120, 200, 200, 100), fills=[Fill.solid(Color(255, 255, 255, 0.6))], radius=(8, 8, 8, 8))
    translucent = Document.blank(560, 360, SURFACE)
    translucent.root.children = [under, card, glass]
    card2 = Node(type="rect", box=Box(200, 60, 300, 200), fills=[Fill.solid("#e8def8")], radius=(16, 16, 16, 16))
    badges = [Node(type="ellipse", box=Box(x - 20, y - 20, 40, 40), fills=[Fill.solid("#b3261e")]) for x, y in ((200, 60), (200, 260))]
    opaque = Document.blank(560, 320, SURFACE)
    opaque.root.children = [card2, *badges]
    return [(translucent, Box(200, 120, 300, 200)), (opaque, Box(200, 60, 300, 200))]


@pytest.mark.parametrize("case", [0, 1])
def test_radius_ignores_occluded_corners(case):
    doc, visible = _occluded_card_docs()[case]
    rgb = render_doc(doc)
    regions = find_regions(rgb, [], estimate_background(rgb))
    r = max((r for r in regions if r.box.intersect(visible).area > 0.8 * visible.area), key=lambda r: r.box.area, default=None)
    assert r is not None, [(r.kind, r.box) for r in regions]
    assert r.kind == "rect" and abs(r.radius - 16) <= 2, (r.kind, r.radius, r.meta.get("corners"), r.box)


def test_radius_fit_param_registered():
    assert "perceive.seg.radius_fit_rms" in P.docs() and "perceive.seg.radius_fit_rms" in P.ranges()
