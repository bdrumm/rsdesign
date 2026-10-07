"""dt.perceive.layout must only emit layouts that a flex render reproduces.

ARCHITECTURE.md's validation rule for module A: rendering in ``mode="flex"`` must reproduce the
absolute render. These tests measure that with the REAL renderer at the DOM level (every node's
bounding rect in flex vs absolute mode), which is far stricter than the image-mean ΔE behind
``layout_consistency`` (a 100px jump of one short label barely moves a page-wide mean).

Run: python -m pytest tests/test_perceive_layout.py -q
"""
from __future__ import annotations

import pytest

from dt.common.image import downscale_dpr
from dt.ir import Box, Color, Document, Fill, Node, TextStyle
from dt.params import P
from dt.perceive import perceive
from dt.perceive.layout import infer_frame_layout, infer_layout
from dt.render.html import render_html
from dt.render.screenshot import _tmp_html, html_to_png, render_doc, render_url

LH = float(P["perceive.ocr.line_height_ratio"])
DRIFT = float(P["perceive.layout.max_drift"]) if "perceive.layout.max_drift" in P.docs() else 1.5
_RECTS = "() => [...document.querySelectorAll('[id]')].map(e => { const r = e.getBoundingClientRect(); return [e.id, r.x, r.y, r.width, r.height]; })"


def _text(x: float, y: float, s: str, size: float, color: str = "#1d1b20", w: float = 150) -> Node:
    return Node(type="text", name=s, box=Box(x, y, w, round(size * LH + 2)), text=s,
                text_style=TextStyle(size=size, color=Color.from_hex(color)))


def _card(box: Box, fill: str, r: float, kids: list[Node]) -> Node:
    return Node(type="frame", name="card", box=box, fills=[Fill.solid(fill)], radius=(r, r, r, r), children=kids)


def _dom_rects(doc: Document, mode: str) -> dict[str, tuple[float, float, float, float]]:
    _, res = render_url("file://" + _tmp_html(render_html(doc, mode=mode)), doc.width, doc.height, wait_ms=0,
                        script=_RECTS, wait_until="load")
    return {i: (x, y, w, h) for i, x, y, w, h in res}


def _max_flex_drift(doc: Document) -> tuple[float, str]:
    a, f = _dom_rects(doc, "absolute"), _dom_rects(doc, "flex")
    worst = max(((max(abs(p - q) for p, q in zip(a[k], f[k])), k) for k in a if k in f), default=(0.0, ""))
    return worst


def _nested_cards() -> Document:
    lvl3 = _card(Box(80, 120, 200, 80), "#e8def8", 8, [_text(96, 140, "Level three", 14)])
    lvl2 = _card(Box(48, 80, 264, 150), "#f3edf7", 12, [_text(64, 90, "Level two", 14), lvl3])
    lvl1 = _card(Box(16, 16, 328, 240), "#ffffff", 16, [_text(32, 30, "Level one", 16), lvl2])
    return Document(width=360, height=280, root=Node(id="root", type="frame", box=Box(0, 0, 360, 280),
                                                     fills=[Fill.solid("#fef7ff")], children=[lvl1]))


def _left_aligned_card(W: int = 360, H: int = 160) -> Document:
    """Title + body of nearly equal width, both left-aligned at x=32 (their centres agree within 3px)."""
    card = _card(Box(16, 16, 328, 120), "#f3edf7", 12,
                 [_text(32, 32, "Card title", 22), _text(32, 72, "Body text here", 14, "#49454f")])
    return Document(width=W, height=H, root=Node(id="root", type="frame", box=Box(0, 0, W, H),
                                                 fills=[Fill.solid("#fef7ff")], children=[card]))


# --------------------------------------------------------------------------- unit (no OCR)
def test_indented_children_are_not_collapsed_onto_one_edge():
    """No shared cross edge or centre -> the frame stays absolute (main fell back to align 'start',
    which moved the indented child card 16px left)."""
    frame = _card(Box(48, 80, 264, 150), "#f3edf7", 12,
                  [_text(64, 90, "Level two", 14, w=69), _card(Box(80, 120, 200, 80), "#e8def8", 8, [])])
    assert infer_frame_layout(frame) is None


def test_centre_aligned_column_keeps_true_padding_and_renders_in_place():
    """Two left-aligned labels whose centres agree within center_pref_tol: whatever alignment is
    chosen, a flex render must leave them where they are (main symmetrised the cross padding to
    min(left, right) and the labels jumped ~100px to the card's middle)."""
    doc = _left_aligned_card()
    card = doc.root.children[0]
    card.children = [_text(32, 32, "Card title", 22, w=95.5), _text(32.5, 72, "Body text here", 14, w=93.6)]
    infer_layout(doc.root)
    assert card.layout is not None and card.layout.mode == "column", card.layout
    drift, nid = _max_flex_drift(doc)
    assert drift <= DRIFT, (drift, nid, card.layout)


def test_accumulated_gap_deviation_rejects_layout():
    """Every gap is within gap_tol of the median, but the flex positions accumulate the deviations
    (6 x 1.4px): the column must stay absolute."""
    rows = [Node(type="rect", box=Box(20, y, 100, 20), fills=[Fill.solid("#e8def8")]) for y in
            (10, 38, 66, 94, 122, 150, 179.4, 208.8, 238.2, 267.6, 297, 326.4)]
    frame = Node(type="frame", box=Box(10, 0, 120, 360), fills=[Fill.solid("#f3edf7")], children=rows)
    doc = Document(width=200, height=360, root=Node(id="root", type="frame", box=Box(0, 0, 200, 360),
                                                    fills=[Fill.solid("#ffffff")], children=[frame]))
    infer_layout(doc.root)
    drift, nid = _max_flex_drift(doc)
    assert drift <= DRIFT, (drift, nid, frame.layout)


def test_regular_column_still_gets_a_layout():
    rows = [Node(type="rect", box=Box(20, 10 + i * 28, 100, 20), fills=[Fill.solid("#e8def8")]) for i in range(8)]
    frame = Node(type="frame", box=Box(10, 0, 120, 240), fills=[Fill.solid("#f3edf7")], children=rows)
    L = infer_frame_layout(frame)
    assert L is not None and L.mode == "column" and L.gap == 8 and L.align_items in ("start", "center")


# --------------------------------------------------------------------------- perceived (real renderer + OCR)
def _assert_perceived_flex_matches(doc: Document) -> None:
    drift, nid = _max_flex_drift(doc)
    assert drift <= DRIFT, (drift, nid, [(n.id, n.type, n.box, n.layout) for n in doc.walk() if n.layout])


def test_nested_cards_three_deep_flex_reproduces_absolute():
    _assert_perceived_flex_matches(perceive(render_doc(_nested_cards())))


def test_dpr2_capture_flex_reproduces_absolute():
    src = _left_aligned_card(360, 240)
    rgb2 = html_to_png(render_html(src), 720, 480, dpr=2.0)  # @2x capture of the same page
    got = perceive(rgb2, dpr=2.0)
    assert (got.width, got.height) == (360, 240)
    card = [n for n in got.walk() if n.type == "frame" and n.id != "root"]
    assert card and abs(card[0].box.x - 16) <= 2 and abs(card[0].box.w - 328) <= 2, [n.box for n in card]
    _assert_perceived_flex_matches(got)


def test_new_layout_param_registered():
    assert "perceive.layout.max_drift" in P.docs() and "perceive.layout.max_drift" in P.ranges()
