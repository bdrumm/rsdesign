"""Tests for dt.perceive.group: unpainted container frames inferred from sibling geometry.

Ground truth comes from the REAL renderer: each scene is an IR document whose containers
(list items, list, app bar, icon buttons, text buttons, navigation bar destinations) are
transparent frames; it is rendered with dt.render, perceived with dt.perceive.perceive, and the
inferred frames must reproduce the generating containers within ±2 px without ever crossing a
painted boundary.

Run: python -m pytest tests/test_perceive_group.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from dt.ir import Box, Color, Document, Fill, Node, TextStyle
from dt.params import P
from dt.perceive import perceive
from dt.perceive.group import group_tree
from dt.render.screenshot import render_doc

BG = "#fef7ff"
ON_SURFACE = "#1d1b20"
ON_SURFACE_VAR = "#49454f"
PRIMARY = "#6750a4"
LH = float(P["perceive.ocr.line_height_ratio"])
TOL = 2.0


# --------------------------------------------------------------------------- helpers
def _text(x: float, y: float, s: str, size: float, weight: int = 400, color: str = ON_SURFACE, w: float = 240) -> Node:
    return Node(type="text", name=s, box=Box(x, y, w, round(size * LH + 2)), text=s,
                text_style=TextStyle(size=size, weight=weight, color=Color.from_hex(color)))


def _icon(x: float, y: float, name: str, color: str = ON_SURFACE_VAR) -> Node:
    return Node(type="icon", name=name, box=Box(x, y, 24, 24), icon_name=name, fills=[Fill.solid(color)])


def _frame(name: str, box: Box, children: list[Node]) -> Node:
    return Node(type="frame", name=name, box=box, children=children)


def _groups(doc: Document, kind: str) -> list[Node]:
    return sorted((n for n in doc.walk() if n.meta.get("group_kind") == kind), key=lambda n: (n.box.y, n.box.x))


def _close(a: Box, b: Box, tol: float = TOL) -> bool:
    return all(abs(u - v) <= tol for u, v in ((a.x, b.x), (a.y, b.y), (a.w, b.w), (a.h, b.h)))


def _assert_frames(doc: Document, kind: str, expected: list[Box]) -> list[Node]:
    got = _groups(doc, kind)
    for e in expected:
        assert any(_close(g.box, e) for g in got), f"no {kind} frame within ±{TOL}px of {e}; have {[g.box for g in got]}"
    return got


def _assert_no_crossing(doc: Document) -> None:
    """No inferred frame partially overlaps a painted node it does not contain (or that does not contain it)."""
    painted = [n for n in doc.walk() if (n.fills or n.strokes or n.type == "line") and n.id != "root"]
    for f in doc.walk():
        if f.meta.get("src") != "group":
            continue
        inside = {id(n) for n in f.walk()}
        for p in painted:
            if id(p) in inside:
                continue
            inter = f.box.intersect(p.box)
            if inter.w <= 0.5 or inter.h <= 0.5:
                continue
            assert p.box.contains(f.box, tol=0.5), f"{f.name} {f.box} crosses painted {p.type} {p.box}"


def _assert_members_inside(doc: Document) -> None:
    for f in doc.walk():
        if f.meta.get("src") == "group":
            assert f.children, f"empty inferred frame {f.name}"
            for c in f.children:
                assert f.box.contains(c.box, tol=0.5), (f.name, f.box, c.box)
            assert not f.fills and not f.strokes


def _perceive(doc: Document) -> Document:
    got = perceive(render_doc(doc))
    _assert_no_crossing(got)
    _assert_members_inside(got)
    return got


# --------------------------------------------------------------------------- scenes
W = 412


def _list_doc() -> tuple[Document, list[Box], Box]:
    """Three 2-line list items: leading icon, headline + supporting text, trailing switch."""
    doc = Document.blank(W, 360, BG)
    items, rows = [], []
    heads = [("Wi-Fi", "Connected to Home"), ("Bluetooth", "Two devices paired"), ("Hotspot", "Sharing is off")]
    h1, h2, gap = round(16 * LH + 2), round(14 * LH + 2), 4
    top = (72 - (h1 + gap + h2)) / 2.0  # M3: the text block is centred in the 72px item
    for i, (h, s) in enumerate(heads):
        y = 64 + 72 * i
        track = Node(type="frame", name="switch", box=Box(W - 16 - 52, y + 20, 52, 32), fills=[Fill.solid(PRIMARY)], radius=(16, 16, 16, 16))
        track.children = [Node(type="ellipse", name="handle", box=Box(W - 16 - 28, y + 24, 24, 24), fills=[Fill.solid("#ffffff")])]
        item = _frame("item", Box(0, y, W, 72), [
            _icon(16, y + 24, ("wifi", "bluetooth", "wifi_tethering")[i]),
            _text(56, y + top, h, 16, 400, ON_SURFACE),
            _text(56, y + top + h1 + gap, s, 14, 400, ON_SURFACE_VAR),
            track,
        ])
        items.append(item)
        rows.append(item.box)
    lst = _frame("list", Box(0, 64, W, 216), items)
    doc.root.children.append(lst)
    return doc, rows, lst.box


def _app_bar_doc() -> tuple[Document, Box, list[Box]]:
    """Small top app bar (same colour as the page): title + three trailing icon buttons."""
    doc = Document.blank(W, 200, BG)
    btns = []
    for k, name in enumerate(("add", "close", "more_vert")):
        bx = W - 4 - 40 * (3 - k)
        btns.append(_frame("icon_button", Box(bx, 12, 40, 40), [_icon(bx + 8, 20, name)]))
    bar = _frame("top_app_bar", Box(0, 0, W, 64), [_text(16, 18, "Inbox", 22, 400, ON_SURFACE, w=160)] + btns)
    doc.root.children.append(bar)
    return doc, bar.box, [b.box for b in btns]


def _label_ink_width(label: str) -> float:
    """Rendered ink width of a label-large button label (measured on the real renderer)."""
    d = Document.blank(240, 60, BG)
    d.root.children.append(_text(20, 20, label, 14, 500, PRIMARY, w=200))
    rgb = render_doc(d).astype(np.int16)
    c = Color.from_hex(BG)
    bg = np.array([c.r, c.g, c.b], dtype=np.int16)
    cols = np.where((np.abs(rgb - bg).sum(axis=2) > 40).any(axis=0))[0]
    return float(cols.max() - cols.min() + 1)


def _text_buttons_doc() -> tuple[Document, list[Box]]:
    """A row of three text buttons (primary-coloured labels, no container paint)."""
    doc = Document.blank(W, 160, BG)
    pad = float(P["perceive.group.text_button_pad"])
    mw = float(P["perceive.group.text_button_min_w"])
    x, boxes = 24.0, []
    for label in ("Archive", "Discard", "Reply all"):
        ink = _label_ink_width(label)
        w = max(round(ink + 2 * pad), mw)
        lx = x + (w - ink) / 2.0
        ty = 60 + (40 - 14 * LH) / 2.0  # M3: the label's line box is centred in the 40px button
        btn = _frame("text_button", Box(x, 60, w, 40), [_text(lx, ty, label, 14, 500, PRIMARY, w=ink + 8)])
        doc.root.children.append(btn)
        boxes.append(btn.box)
        x += w + 24  # (at an 8px gap the OCR may merge adjacent labels into one line: known gap)
    return doc, boxes


def _nav_bar_doc() -> tuple[Document, Box, list[Box]]:
    """Navigation bar (painted surface-container) with four icon-over-label destinations."""
    H = 400
    doc = Document.blank(W, H, BG)
    bar = Node(type="frame", name="navigation_bar", box=Box(0, H - 80, W, 80), fills=[Fill.solid("#f3edf7")])
    w = W / 4
    dests = []
    for i, (icon, label) in enumerate((("home", "Home"), ("search", "Search"), ("favorite", "Saved"), ("person", "Profile"))):
        cx = w * i + w / 2
        d = _frame("destination", Box(w * i, H - 80, w, 80), [
            _icon(cx - 12, H - 80 + 16, icon),
            _text(cx - 24, H - 80 + 48, label, 12, 500, ON_SURFACE_VAR, w=48),
        ])
        d.children[1].text_style.align = "center"
        dests.append(d)
    bar.children = dests
    doc.root.children.append(bar)
    return doc, bar.box, [d.box for d in dests]


# --------------------------------------------------------------------------- tests
def test_list_items_and_list_frame():
    doc, rows, lst = _list_doc()
    got = _perceive(doc)
    _assert_frames(got, "list_item", rows)
    lists = _assert_frames(got, "list", [lst])
    items = [c for c in lists[0].children if c.meta.get("group_kind") == "list_item"]
    assert len(items) == 3, [c.box for c in lists[0].children]
    for it in items:  # leading icon + texts + trailing switch end up inside the item
        assert any(c.type == "text" for c in it.children)
        assert any(c.fills and c.box.w >= 40 for c in it.children), [(c.type, c.box) for c in it.children]


def test_top_app_bar_with_icon_buttons():
    doc, bar, btns = _app_bar_doc()
    got = _perceive(doc)
    _assert_frames(got, "top_app_bar", [bar])
    ibs = _assert_frames(got, "icon_button", btns)
    for b in ibs:
        assert any(c.type in ("icon", "rect", "text") for c in b.children)
    title = next(n for n in got.walk() if n.type == "text" and "Inbox" in (n.text or ""))
    assert got.parent_of(title.id).meta.get("group_kind") == "top_app_bar"


def test_row_of_text_buttons():
    doc, boxes = _text_buttons_doc()
    got = _perceive(doc)
    tbs = _assert_frames(got, "text_button", boxes)
    assert len(tbs) == 3, [t.box for t in tbs]
    assert not _groups(got, "list_item"), "a row of text buttons is not a list item"
    for t in tbs:
        assert [c.type for c in t.children] == ["text"]


def test_navigation_bar_destinations():
    doc, bar, dests = _nav_bar_doc()
    got = _perceive(doc)
    ds = _assert_frames(got, "destination", dests)
    assert len(ds) == 4
    for d in ds:
        kinds = sorted(c.type for c in d.children)
        assert "text" in kinds and len(kinds) >= 2, kinds  # icon (glyph) over label
    painted_bar = next(n for n in got.walk() if n.fills and _close(n.box, bar))
    assert all(got.parent_of(d.id) is painted_bar for d in ds)


def test_grouping_can_be_disabled():
    doc, rows, _ = _list_doc()
    rgb = render_doc(doc)
    P.set("perceive.group.enabled", False)
    try:
        got = perceive(rgb)
    finally:
        P.reset("perceive.group.enabled")
    assert not [n for n in got.walk() if n.meta.get("src") == "group"]


def test_lone_icon_and_paragraph_are_not_grouped():
    """No frames for content that has no container semantics."""
    doc = Document.blank(W, 240, BG)
    doc.root.children += [
        _icon(194, 120, "search"),
        _text(16, 160, "Just a sentence of body text", 14, 400, ON_SURFACE, w=300),
    ]
    got = _perceive(doc)
    assert not [n for n in got.walk() if n.meta.get("src") == "group"], [(n.name, n.box) for n in got.walk()]


def test_group_tree_is_idempotent_on_grouped_tree():
    doc, rows, _ = _list_doc()
    got = perceive(render_doc(doc))
    n0 = sum(1 for _ in got.walk())
    group_tree(got.root)
    assert sum(1 for _ in got.walk()) == n0


@pytest.mark.parametrize("key", [k for k in P.defaults() if k.startswith("perceive.group.")])
def test_group_params_documented(key):
    assert P.docs().get(key), key
    if isinstance(P.defaults()[key], (int, float)) and not isinstance(P.defaults()[key], bool):
        assert key in P.ranges(), key


# --------------------------------------------------------------------------- glyph clustering (complexity)
def _reference_cluster(pieces, mg, gmax):
    """The original restart-after-every-merge agglomeration (O(n^3)); the fast one must match it."""
    from dt.perceive.group import _gap
    clusters = [(p.box, [p]) for p in pieces]
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                bi, bj = clusters[i][0], clusters[j][0]
                u = bi.union(bj)
                if _gap(bi, bj) <= mg and max(u.w, u.h) <= gmax:
                    clusters[i] = (u, clusters[i][1] + clusters[j][1])
                    clusters.pop(j)
                    merged = True
                    break
            if merged:
                break
    return clusters


def _fragments(rng, n, W=360.0, H=600.0):
    out = []
    for _ in range(n):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        out.append(Node(type="rect", box=Box(round(x, 1), round(y, 1), round(rng.uniform(1, 14), 1), round(rng.uniform(1, 14), 1))))
    return out


@pytest.mark.parametrize("seed", range(6))
def test_glyph_clustering_matches_reference(seed):
    from dt.perceive.group import _cluster_glyphs
    rng = np.random.default_rng(seed)
    pieces = _fragments(rng, 220)
    mg, gmax = float(P["perceive.group.glyph_merge_gap"]), float(P["perceive.group.glyph_max"])
    ref = _reference_cluster(pieces, mg, gmax)
    got = _cluster_glyphs(pieces, mg, gmax)
    assert [(b.as_tuple() if hasattr(b, "as_tuple") else (b.x, b.y, b.w, b.h), [id(n) for n in ns]) for b, ns in got] == \
           [(b.as_tuple() if hasattr(b, "as_tuple") else (b.x, b.y, b.w, b.h), [id(n) for n in ns]) for b, ns in ref]


def test_glyph_clustering_scales_to_long_lists():
    """120 list rows x (person icon + 3-dot more_vert) = 480 fragments: the old O(n^3) scan took
    ~4.5s on this alone (on a perceived 60-row list it was 5.9s of a 10.5s perceive)."""
    import time
    from dt.perceive.group import _cluster_glyphs
    pieces = []
    for i in range(120):
        y = i * 56.0
        pieces.append(Node(type="icon", box=Box(18, y + 18, 20, 20)))
        pieces += [Node(type="rect", box=Box(330, y + 20 + 6 * k, 4, 4)) for k in range(3)]
    t = time.perf_counter()
    got = _cluster_glyphs(pieces, float(P["perceive.group.glyph_merge_gap"]), float(P["perceive.group.glyph_max"]))
    took = time.perf_counter() - t
    assert len(got) == 240 and all(len(ns) in (1, 3) for _, ns in got)
    assert took < 1.0, took
