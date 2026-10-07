"""Extra OCR tests: per-line granularity of list text, second-chance OCR (badges, dark surfaces),
icon glyphs rejected, homoglyph normalisation. Ground truth comes from the REAL renderer.

Run: python -m pytest tests/test_ocr_extra.py -q
"""
from __future__ import annotations

import pytest

from dt.ir import Box, Color, Document, Fill, Node, TextStyle
from dt.params import P
from dt.perceive import perceive
from dt.perceive.ocr import (RawLine, _refine_line, estimate_text_style, normalize_homoglyphs, ocr_lines,
                             second_chance_candidates)
from dt.render.screenshot import render_doc

LH = float(P["perceive.ocr.line_height_ratio"])


def _text(x: float, y: float, s: str, size: float, weight: int = 400, color: str = "#1d1b20", w: float = 300) -> Node:
    return Node(type="text", name=s, box=Box(x, y, w, size * LH + 2), text=s,
                text_style=TextStyle(size=size, weight=weight, color=Color.from_hex(color)))


def _texts(doc: Document) -> dict[str, Node]:
    return {n.text: n for n in doc.walk() if n.type == "text" and n.text}


def _near(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- (a) list text per line
@pytest.mark.parametrize("bg", ["#fef7ff", "#f7f2fa"])
def test_two_line_list_item_gives_two_text_nodes(bg):
    """M3 two-line list item: 16px on-surface headline over 14px on-surface-variant supporting
    text (line boxes 24px / 20px) must stay two nodes with their own size and colour."""
    doc = Document.blank(360, 160, bg)
    for i, (head, sup) in enumerate((("Travel plan", "Modified 9:30 AM"), ("Roadmap 2025", "Modified Oct 3"))):
        y = 16 + i * 72
        doc.root.children.append(_text(56, y + 4, head, 16, 400, "#1d1b20", w=240))
        doc.root.children.append(_text(56, y + 28, sup, 14, 400, "#49454f", w=240))
    got = perceive(render_doc(doc))
    ts = _texts(got)
    for head, sup in (("Travel plan", "Modified 9:30 AM"), ("Roadmap 2025", "Modified Oct 3")):
        assert head in ts and sup in ts, list(ts)
        h, s = ts[head].text_style, ts[sup].text_style
        assert _near(h.size, 16, 1.0) and _near(s.size, 14, 1.0), (h.size, s.size)
        assert h.color.delta_e(Color.from_hex("#1d1b20")) < 3, h.color.hex()
        assert s.color.delta_e(Color.from_hex("#49454f")) < 3, s.color.hex()
        assert ts[sup].box.y > ts[head].box.y + 16


def test_line_colour_not_taken_from_neighbouring_line():
    """The crop around a light line reaches the darker line above; colour must be the line's own."""
    doc = Document.blank(320, 80)
    doc.root.children.append(_text(20, 10, "Brand assets", 16, 400, "#1d1b20"))
    doc.root.children.append(_text(20, 32, "Modified Yesterday", 14, 400, "#49454f"))
    rgb = render_doc(doc)
    lines = {l.text: l for l in ocr_lines(rgb) if l.text}
    assert "Modified Yesterday" in lines, list(lines)
    st = estimate_text_style(rgb, lines["Modified Yesterday"])
    assert st.color.delta_e(Color.from_hex("#49454f")) < 3, st.color.hex()


def test_over_tall_ocr_box_measures_only_its_own_line():
    """An OCR box stretched over two lines (e.g. by a merged leading icon) keeps one line's ink."""
    doc = Document.blank(360, 80)
    doc.root.children.append(_text(56, 10, "Travel plan", 16, 400, "#1d1b20"))
    doc.root.children.append(_text(56, 34, "Reminder: the deadline is end of day Friday.", 14, 400, "#49454f"))
    rgb = render_doc(doc)
    raw = RawLine("Reminder: the deadline is end of day Friday.", Box(50, 8, 290, 50), [], 1.0)
    line = _refine_line(rgb, raw)
    assert line is not None
    assert line.box.y > 30 and line.box.h < 16, line.box  # 'Travel plan' (y~14) is not part of it
    assert _near(estimate_text_style(rgb, line).size, 14, 1.0)


def test_same_style_paragraph_still_grouped():
    """Tightened tolerances must not split a real same-style paragraph."""
    doc = Document.blank(400, 120)
    n = _text(20, 20, "First line of text\nSecond line here", 14, 400, "#49454f")
    n.text_style.line_height = 20
    n.box = Box(20, 20, 340, 40)
    doc.root.children.append(n)
    ts = _texts(perceive(render_doc(doc)))
    assert "First line of text\nSecond line here" in ts, list(ts)


# --------------------------------------------------------------------------- (b) second chance
def _badge(x: float, y: float, label: str) -> list[Node]:
    """M3 large badge: 16px tall error pill, 11px/500 white label, 4px side padding."""
    w = max(16.0, 8 + 7 * len(label))
    pill = Node(type="rect", name="badge", box=Box(x, y, w, 16), fills=[Fill.solid("#b3261e")], radius=(8, 8, 8, 8))
    t = _text(x + 4, y + (16 - 11 * LH) / 2, label, 11, 500, "#ffffff", w=w)
    return [pill, t]


def _assert_badge_text(doc: Document, label: str, pill: Box) -> None:
    ts = _texts(doc)
    assert label in ts, list(ts)
    n = ts[label]
    assert pill.contains(Box.from_dict(n.meta["ink"]), tol=1.0), (n.meta["ink"], pill)
    assert n.text_style.color.delta_e(Color(255, 255, 255)) < 6, n.text_style.color.hex()
    assert _near(n.text_style.size, 11, 2.0), n.text_style.size


def test_badge_on_red_pill_found():
    doc = Document.blank(200, 80, "#f3edf7")
    doc.root.children.extend(_badge(90, 30, "53"))
    _assert_badge_text(perceive(render_doc(doc)), "53", Box(90, 30, 22, 16))


def test_badge_anchored_on_icon_found():
    """Badges sit on the icon's top-right corner, so the glyph-edge blob is icon+badge."""
    doc = Document.blank(200, 80, "#f3edf7")
    doc.root.children.append(Node(type="icon", box=Box(80, 32, 24, 24), icon_name="notifications", fills=[Fill.solid("#49454f")]))
    doc.root.children.extend(_badge(94, 24, "62"))
    _assert_badge_text(perceive(render_doc(doc)), "62", Box(94, 24, 22, 16))


def test_light_text_on_dark_snackbar_found():
    doc = Document.blank(400, 100, "#fef7ff")
    doc.root.children.append(Node(type="rect", name="snackbar", box=Box(20, 26, 344, 48), fills=[Fill.solid("#322f35")], radius=(4, 4, 4, 4)))
    doc.root.children.append(_text(36, 42, "Message archived", 14, 400, "#f5eff7", w=200))
    doc.root.children.append(_text(318, 42, "OK", 14, 500, "#f5eff7", w=40))
    got = perceive(render_doc(doc))
    ts = _texts(got)
    assert "OK" in ts, list(ts)
    ok = ts["OK"]
    assert Box(310, 34, 50, 32).contains(Box.from_dict(ok.meta["ink"]), tol=1.0), ok.meta["ink"]
    assert ok.text_style.color.delta_e(Color.from_hex("#f5eff7")) < 6, ok.text_style.color.hex()
    assert _near(ok.text_style.size, 14, 1.5)


def test_icon_glyphs_not_read_as_text():
    """Crop-level OCR reads some icons as letters ('Do' for person, 'OJ' for attach_file);
    the ink-width plausibility check must reject them."""
    doc = Document.blank(300, 80)
    for i, name in enumerate(("person", "attach_file", "videocam", "home", "settings")):
        doc.root.children.append(Node(type="icon", box=Box(20 + i * 56, 28, 24, 24), icon_name=name, fills=[Fill.solid("#49454f")]))
    lines = [l for l in ocr_lines(render_doc(doc)) if l.text]
    assert not lines, [(l.text, l.box, l.meta) for l in lines]


def test_second_chance_candidates_skip_covered_text():
    doc = Document.blank(300, 60)
    doc.root.children.append(_text(20, 20, "Covered label", 14))
    rgb = render_doc(doc)
    lines = [l for l in ocr_lines(rgb) if l.text]
    assert [l.text for l in lines] == ["Covered label"]
    assert not any(l.meta.get("second_chance") for l in lines)
    assert second_chance_candidates(rgb, [l.box for l in lines]) == []


def test_second_chance_can_be_disabled():
    doc = Document.blank(200, 80, "#f3edf7")
    doc.root.children.extend(_badge(90, 30, "53"))
    rgb = render_doc(doc)
    old = P["perceive.ocr.sc.enabled"]
    try:
        P.set("perceive.ocr.sc.enabled", 0)
        assert not any(l.meta.get("second_chance") for l in ocr_lines(rgb))
    finally:
        P.set("perceive.ocr.sc.enabled", old)


# --------------------------------------------------------------------------- misc
def test_homoglyph_normalisation():
    assert normalize_homoglyphs("ОK") == "OK"          # Cyrillic O + Latin K
    assert normalize_homoglyphs("Rеply") == "Reply"    # Cyrillic e
    assert normalize_homoglyphs("Привет") == "Привет"  # real Cyrillic text is left alone
    assert normalize_homoglyphs("Hello") == "Hello"


def test_new_params_registered():
    docs, ranges = P.docs(), P.ranges()
    keys = [k for k in P.defaults() if k.startswith("perceive.ocr.sc.")] + [
        "perceive.ocr.band_merge_gap", "perceive.ocr.fg_refit_de", "perceive.ocr.para_size_tol", "perceive.ocr.para_color_de"]
    assert len(keys) > 15
    for k in keys:
        assert k in docs and k in ranges, k


# --------------------------------------------------------------------------- interior punctuation is text
PUNCT_LINES = [
    "Quarterly report draft 3 - please review",  # dense Gmail-like subject line (13px)
    "Tom & Jerry",
    "Price 5 / month",
]


@pytest.mark.parametrize("size", [13, 16])
def test_interior_punctuation_is_not_split_off_as_an_icon(size):
    """Vision returns ' - ', ' & ', ' / ' as one-character words at word spacing; split_icon_words
    used to drop every single non-alphanumeric word anywhere in a line (and mark it as an icon)."""
    doc = Document.blank(520, 40 + 40 * len(PUNCT_LINES))
    doc.root.children = [_text(20, 20 + 40 * i, s, size, w=480) for i, s in enumerate(PUNCT_LINES)]
    got = [l.text for l in ocr_lines(render_doc(doc))]
    for s in PUNCT_LINES:
        assert s in got, (s, got)


def test_leading_icon_glyph_still_split_off():
    """The other side of the rule: a '+' glyph before a button label stays an icon word."""
    doc = Document.blank(240, 80, "#e8def8")
    doc.root.children = [Node(type="icon", box=Box(24, 28, 18, 18), icon_name="add", fills=[Fill.solid("#1d192b")]),
                         _text(50, 27, "Tonal", 14, 500, "#1d192b", w=60)]
    lines = [l for l in ocr_lines(render_doc(doc)) if l.text]
    assert [l.text for l in lines] == ["Tonal"], [(l.text, l.meta.get("icon_words")) for l in lines]
