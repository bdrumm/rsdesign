"""Tests for dt.perceive: ground truth is produced by the REAL renderer (dt.render), never mocked.

Run: python -m pytest tests/test_perceive.py -q

To re-derive the OCR calibration constants (perceive.ocr.size_fit.*, ascent_ratio, stroke
thresholds) render Roboto text at sizes 10..40 / weights 400,500,700 with dt.render, run
``ocr_lines`` and fit ``size = a * (baseline - ink_top) + b`` per top class; see the module
docstring of dt/perceive/ocr.py.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from dt.ir import Box, Color, Document, Fill, Layout, Node, Shadow, Stroke, TextStyle
from dt.params import P
from dt.perceive import perceive, perceive_parts, perceive_region
from dt.perceive.ocr import available_backends, get_backend, ocr_lines
from dt.perceive.segment import estimate_radius, find_regions
from dt.render.screenshot import render_doc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MWC_PNG = os.path.join(ROOT, "out", "mwc_test.png")

LH = float(P["perceive.ocr.line_height_ratio"])  # renderer 'normal' line height / size


# --------------------------------------------------------------------------- helpers
def _text(x: float, y: float, s: str, size: float, weight: int = 400, color: str = "#1d1b20", w: float = 300) -> Node:
    return Node(type="text", name=s, box=Box(x, y, w, size * LH + 2), text=s,
                text_style=TextStyle(size=size, weight=weight, color=Color.from_hex(color)))


def _perceive_doc(doc: Document) -> tuple[Document, np.ndarray]:
    rgb = render_doc(doc)
    return perceive(rgb), rgb


def _texts(doc: Document) -> dict[str, Node]:
    return {n.text: n for n in doc.walk() if n.type == "text" and n.text}


def _nontext(doc: Document) -> list[Node]:
    return [n for n in doc.walk() if n.type != "text" and n.id != "root"]


def _near(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def _find_box(doc: Document, box: Box, tol: float = 2.0, types: tuple[str, ...] | None = None) -> Node:
    for n in doc.walk():
        if n.id == "root" or (types and n.type not in types):
            continue
        b = n.box
        if _near(b.x, box.x, tol) and _near(b.y, box.y, tol) and _near(b.w, box.w, tol) and _near(b.h, box.h, tol):
            return n
    raise AssertionError(f"no node within ±{tol}px of {box}; have: " + ", ".join(f"{n.type}{n.box}" for n in doc.walk()))


def _assert_doc_valid(doc: Document) -> None:
    r = doc.root
    assert r.box.x == 0 and r.box.y == 0 and r.box.w == doc.width and r.box.h == doc.height
    assert r.fill_color is not None
    img = Box(0, 0, doc.width, doc.height)
    for n in doc.walk():
        assert img.contains(n.box, tol=0.01), f"{n.type} {n.box} outside image"
        if n.type == "text":
            assert n.text_style is not None and n.text
            assert not n.children
        if n.type == "frame" and len(n.children) >= 2:
            assert n.layout is not None


# --------------------------------------------------------------------------- backend
def test_backend_available():
    be = get_backend()
    assert be.name in ("vision", "rapidocr", "tesseract")
    if sys.platform == "darwin":
        assert "vision" in available_backends()
    with pytest.raises(ValueError):
        get_backend("nope")


def test_params_registered_with_docs():
    docs = P.docs()
    keys = [k for k in P.defaults() if k.startswith("perceive.")]
    assert len(keys) > 40
    missing = [k for k in keys if k not in docs]
    assert not missing, missing


# --------------------------------------------------------------------------- text lines
@pytest.mark.parametrize("size,weight,color,bg", [
    (12, 400, "#1d1b20", "#ffffff"),
    (14, 500, "#6750a4", "#fef7ff"),
    (16, 700, "#1d1b20", "#ffffff"),
    (22, 400, "#b3261e", "#ffffff"),
    (28, 500, "#ffffff", "#6750a4"),
    (36, 700, "#1d1b20", "#f3edf7"),
])
def test_single_text_line(size, weight, color, bg):
    doc = Document.blank(420, 120, bg)
    doc.root.children.append(_text(24, 30, "Hello Figma 42", size, weight, color))
    got, _ = _perceive_doc(doc)
    _assert_doc_valid(got)
    ts = _texts(got)
    assert "Hello Figma 42" in ts, list(ts)
    n = ts["Hello Figma 42"]
    st = n.text_style
    assert _near(st.size, size, 1.5), (st.size, size)
    assert st.weight == weight, (st.weight, weight, n.meta)
    assert st.color.delta_e(Color.from_hex(color)) < 3, (st.color.hex(), color)
    assert _near(n.box.x, 24, 2) and _near(n.box.y, 30, 2), n.box
    assert _near(n.box.h, size * LH, 2), (n.box.h, size * LH)
    assert got.parent_of(n.id) is got.root


def test_paragraph_line_height():
    doc = Document.blank(400, 160)
    n = _text(20, 20, "First line of text\nSecond line here\nThird and last", 16, 400)
    n.text_style.line_height = 24
    n.box = Box(20, 20, 340, 72)
    doc.root.children.append(n)
    got, _ = _perceive_doc(doc)
    ts = _texts(got)
    assert "First line of text\nSecond line here\nThird and last" in ts, list(ts)
    p = ts["First line of text\nSecond line here\nThird and last"]
    assert _near(p.text_style.line_height or 0, 24, 1.5)
    assert _near(p.box.y, 20, 2) and _near(p.box.h, 72, 2.5), p.box
    assert _near(p.text_style.size, 16, 1.5)


# --------------------------------------------------------------------------- shapes
def test_filled_pill_button_with_icon_and_label():
    doc = Document.blank(300, 120, "#fef7ff")
    btn = Node(type="frame", name="button", box=Box(40, 40, 128, 40), fills=[Fill.solid("#6750a4")], radius=(20, 20, 20, 20))
    icon = Node(type="icon", box=Box(56, 51, 18, 18), icon_name="add", fills=[Fill.solid("#ffffff")])
    label = _text(82, 50, "Save", 14, 500, "#ffffff", w=60)
    btn.children = [icon, label]
    doc.root.children.append(btn)
    got, _ = _perceive_doc(doc)
    _assert_doc_valid(got)
    b = _find_box(got, btn.box, 2)
    assert b.fill_color is not None and b.fill_color.delta_e(Color.from_hex("#6750a4")) < 3
    assert _near(b.uniform_radius(), 20, 2), b.radius
    # hierarchy: text inside button inside root; icon inside button
    label_n = _texts(got)["Save"]
    assert got.parent_of(label_n.id) is b and got.parent_of(b.id) is got.root
    icons = [c for c in b.children if c.type == "icon"]
    assert len(icons) == 1 and icon.box.contains(icons[0].box, tol=1.5), icons[0].box
    # 1.5px icon strokes never fully cover a pixel, so the glyph colour is a slight blend (known gap)
    assert icons[0].fill_color.delta_e(Color(255, 255, 255)) < 6
    assert label_n.text_style.color.delta_e(Color(255, 255, 255)) < 3
    assert b.layout is not None and b.layout.mode == "row" and b.layout.align_items == "center", b.layout


def test_outlined_card_with_stroke_radius_shadow_and_texts():
    doc = Document.blank(400, 240, "#f3edf7")
    card = Node(type="frame", name="card", box=Box(30, 30, 300, 160), fills=[Fill.solid("#ffffff")],
                strokes=[Stroke(color=Color.from_hex("#79747e"), width=1)], radius=(12, 12, 12, 12),
                effects=[Shadow(color=Color(0, 0, 0, 0.3), dx=0, dy=2, blur=6)])
    card.children = [_text(54, 54, "Card title", 20, 500), _text(54, 96, "Supporting body text", 14, 400, "#49454f")]
    doc.root.children.append(card)
    got, _ = _perceive_doc(doc)
    _assert_doc_valid(got)
    c = _find_box(got, card.box, 2)
    assert c.fill_color is not None and c.fill_color.delta_e(Color(255, 255, 255)) < 3
    assert c.strokes and c.strokes[0].color.delta_e(Color.from_hex("#79747e")) < 3 and _near(c.strokes[0].width, 1, 1)
    assert _near(c.uniform_radius(), 12, 2), c.radius
    assert c.effects, "shadow not detected"
    ts = _texts(got)
    assert "Card title" in ts and "Supporting body text" in ts, list(ts)
    assert got.parent_of(ts["Card title"].id) is c and got.parent_of(ts["Supporting body text"].id) is c
    assert ts["Card title"].text_style.weight == 500 and _near(ts["Card title"].text_style.size, 20, 1.5)
    assert ts["Supporting body text"].text_style.color.delta_e(Color.from_hex("#49454f")) < 3
    assert c.layout is not None and c.layout.mode == "column"


def test_divider():
    doc = Document.blank(400, 100)
    doc.root.children.append(Node(type="line", box=Box(40, 50, 320, 1), fills=[Fill.solid("#cac4d0")]))
    got, _ = _perceive_doc(doc)
    n = _find_box(got, Box(40, 50, 320, 1), 2, types=("line",))
    assert n.fill_color.delta_e(Color.from_hex("#cac4d0")) < 3


def test_column_list_of_three_rows_layout():
    doc = Document.blank(360, 240)
    cont = Node(type="frame", name="list", box=Box(24, 24, 312, 192), fills=[Fill.solid("#f3edf7")], radius=(16, 16, 16, 16))
    rows = [Node(type="rect", box=Box(40, 40 + i * 56, 280, 48), fills=[Fill.solid("#e8def8")], radius=(8, 8, 8, 8)) for i in range(3)]
    cont.children = rows
    doc.root.children.append(cont)
    got, rgb = _perceive_doc(doc)
    _assert_doc_valid(got)
    c = _find_box(got, cont.box, 2)
    assert _near(c.uniform_radius(), 16, 2)
    kids = [k for k in c.children if k.type in ("rect", "frame")]
    assert len(kids) == 3, [k.box for k in c.children]
    for r, k in zip(rows, sorted(kids, key=lambda k: k.box.y)):
        assert _near(k.box.x, r.box.x, 2) and _near(k.box.y, r.box.y, 2) and _near(k.box.w, r.box.w, 2) and _near(k.box.h, r.box.h, 2), (k.box, r.box)
        assert k.fill_color.delta_e(Color.from_hex("#e8def8")) < 3
        assert _near(k.uniform_radius(), 8, 2), k.radius
    L = c.layout
    assert L is not None and L.mode == "column", L
    assert _near(L.gap, 8, 1), L.gap
    assert all(_near(p, 16, 1.5) for p in L.padding), L.padding
    # layout consistency: flex re-render must reproduce the absolute re-render
    a = render_doc(got, mode="absolute").astype(np.int16)
    f = render_doc(got, mode="flex").astype(np.int16)
    assert np.abs(a - f).mean() < 0.5


def test_icon_blob():
    doc = Document.blank(200, 200)
    doc.root.children.append(Node(type="icon", box=Box(88, 88, 24, 24), icon_name="search", fills=[Fill.solid("#1d1b20")]))
    got, _ = _perceive_doc(doc)
    icons = [n for n in got.walk() if n.type == "icon"]
    assert len(icons) == 1, [(n.type, n.box) for n in got.walk()]
    assert Box(88, 88, 24, 24).contains(icons[0].box, tol=1.5), icons[0].box
    assert icons[0].box.w >= 12 and icons[0].box.h >= 12
    assert icons[0].fill_color.delta_e(Color.from_hex("#1d1b20")) < 3


@pytest.mark.parametrize("radius", [0, 4, 8, 12, 24, 40])
def test_radius_estimation_on_rendered_rects(radius):
    doc = Document.blank(200, 120)
    doc.root.children.append(Node(type="rect", box=Box(20, 20, 160, 80), fills=[Fill.solid("#6750a4")], radius=(radius,) * 4))
    regs = find_regions(render_doc(doc), [])
    assert len(regs) == 1, [(r.kind, r.box) for r in regs]
    r = regs[0]
    assert _near(r.box.x, 20, 1) and _near(r.box.y, 20, 1) and _near(r.box.w, 160, 1) and _near(r.box.h, 80, 1), r.box
    assert _near(r.radius, radius, 2), (radius, r.radius, r.meta.get("corners"))
    assert r.kind == ("pill" if radius == 40 else "rect")


def test_perceive_region_translates_boxes():
    doc = Document.blank(300, 200)
    doc.root.children.append(Node(type="rect", box=Box(120, 80, 60, 30), fills=[Fill.solid("#6750a4")], radius=(6, 6, 6, 6)))
    rgb = render_doc(doc)
    sub = perceive_region(rgb, Box(100, 60, 120, 80))
    assert _near(sub.box.x, 100, 0.01) and _near(sub.box.y, 60, 0.01)
    rects = [n for n in sub.walk() if n.type == "rect"]
    assert rects and _near(rects[0].box.x, 120, 2) and _near(rects[0].box.y, 80, 2), [n.box for n in sub.walk()]


# --------------------------------------------------------------------------- real Material Web page
@pytest.mark.skipif(not os.path.exists(MWC_PNG), reason="out/mwc_test.png missing")
def test_mwc_reference_page():
    doc = perceive(MWC_PNG)
    _assert_doc_valid(doc)
    non_text = _nontext(doc)
    assert len(non_text) >= 9, [(n.type, n.box) for n in non_text]
    found = " ".join(_texts(doc).keys())
    for word in ("Filled", "Outlined", "Text", "Tonal", "Assist", "Filter", "Email"):
        assert word in found, (word, found)
    # the filled button: pill 82x40 at (16,24) with primary fill and its label nested inside
    btn = _find_box(doc, Box(16, 24, 82, 40), 2)
    assert btn.fill_color.delta_e(Color.from_hex("#6750a4")) < 3 and _near(btn.uniform_radius(), 20, 2)
    assert doc.parent_of(_texts(doc)["Filled"].id) is btn
    # outlined text field: bordered box 199x56 at (16,84)
    tf = _find_box(doc, Box(16, 84, 199, 56), 2)
    assert tf.strokes and tf.strokes[0].color.delta_e(Color.from_hex("#79747e")) < 6
    # icon glyphs OCR'd into labels are split off ('+ Tonal' -> 'Tonal' + icon)
    tonal = _find_box(doc, Box(298, 24, 101, 40), 2)
    assert any(c.type == "icon" for c in tonal.children), [c.type for c in tonal.children]
