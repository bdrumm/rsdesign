"""dt.perceive.hierarchy: containment rules checked on REAL renders.

Run: python -m pytest tests/test_perceive_hierarchy.py -q
"""
from __future__ import annotations

from dt.ir import Box, Color, Document, Fill, Node, Stroke, TextStyle
from dt.params import P
from dt.perceive import perceive, perceive_parts
from dt.perceive.hierarchy import LEAF_TYPES, build_tree
from dt.perceive.ocr import ocr_lines
from dt.render.screenshot import render_doc

LH = float(P["perceive.ocr.line_height_ratio"])


def _text(x, y, s, size, w, color, weight=400) -> Node:
    return Node(type="text", name=s, box=Box(x, y, w, round(size * LH + 2)), text=s,
                text_style=TextStyle(size=size, weight=weight, color=Color.from_hex(color)))


def _calendar_glyph_doc() -> Document:
    """A Google-Calendar-like 28x30 outlined glyph with the date '31' inside, next to a title
    (gws_calendar.png has exactly this: segmentation calls the glyph an icon, OCR reads the date)."""
    glyph = Node(type="frame", box=Box(130, 150, 28, 30), radius=(4, 4, 4, 4),
                 strokes=[Stroke(color=Color.from_hex("#1a73e8"), width=2, align="inside")],
                 children=[_text(137, 158, "31", 12, 16, "#1a73e8", 700)])
    doc = Document.blank(320, 240, "#ffffff")
    doc.root.children = [glyph, _text(180, 156, "Calendar", 22, 120, "#3c4043")]
    return doc


def test_leaf_nodes_never_get_children_from_build_tree():
    rgb = render_doc(_calendar_glyph_doc())
    lines, texts, regions, bg = perceive_parts(rgb)
    assert any(r.kind == "icon" for r in regions), [(r.kind, r.box) for r in regions]  # the precondition
    root = build_tree(regions, texts, 320, 240, bg)
    bad = [(n.type, [c.text for c in n.children]) for n in root.walk() if n.type in LEAF_TYPES and n.children]
    assert not bad, bad


def test_text_inside_icon_glyph_survives_the_rerender():
    rgb = render_doc(_calendar_glyph_doc())
    doc = perceive(rgb)
    assert "31" in [n.text for n in doc.walk() if n.type == "text"]
    assert not [n for n in doc.walk() if n.type in LEAF_TYPES and n.children]
    # main dropped it with the icon's children; the unnamed glyph itself re-renders blank (icons pass), so
    # OCR may read the date and the title as one line
    assert any("31" in l.text for l in ocr_lines(render_doc(doc)))
