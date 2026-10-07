"""Tests for module E (self-test corpora): dt.selftest.mwc_corpus + dt.selftest.synth.

Everything here uses the REAL renderer (Playwright on Chrome); ground truth comes from the DOM
(mwc) or from the IR itself (synth). Three pages of each corpus are generated into a tmp dir.
"""
from __future__ import annotations

import json
import os
import random
from typing import Optional

import numpy as np
import pytest

from dt.common.image import load_rgb
from dt.ir import Box, Document, Node
from dt.selftest import mwc_corpus, synth
from dt.selftest.mwc_corpus import _Page, extract_ground_truth, page_html

N = 3
SEED = 7


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def mwc_dir(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("mwc"))
    mwc_corpus.generate(N, d, seed=SEED)
    return d


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("synth"))
    synth.generate(N, d, seed=SEED)
    return d


@pytest.fixture(scope="module")
def sink(tmp_path_factory):
    """Deterministic kitchen-sink page with one of every supported Material Web element."""
    d = str(tmp_path_factory.mktemp("sink"))
    pg = _Page(random.Random(3), 600, 1000)
    body = f"""<div class="main">
<div class="form">
<md-outlined-text-field label="Email" value="a@b.com"></md-outlined-text-field>
<md-filled-text-field label="Search" placeholder="Type here"><md-icon slot="leading-icon">search</md-icon></md-filled-text-field>
<md-outlined-select label="Repeat"><md-select-option value="0"><div slot="headline">Daily</div></md-select-option><md-select-option selected value="1"><div slot="headline">Weekly on Monday</div></md-select-option></md-outlined-select>
<md-slider value="40" labeled></md-slider>
</div>
<div class="row"><md-switch selected></md-switch><md-switch></md-switch><md-checkbox checked></md-checkbox><md-radio name="a" checked></md-radio></div>
<div class="progress-row"><md-linear-progress value="0.4" style="flex:1"></md-linear-progress><md-circular-progress value="0.6"></md-circular-progress></div>
<div class="row"><md-fab variant="primary" label="Compose"><md-icon slot="icon">edit</md-icon></md-fab><md-fab size="small"><md-icon slot="icon">add</md-icon></md-fab>
<md-filled-button>Filled</md-filled-button><md-outlined-button>Outlined</md-outlined-button><md-text-button>Text</md-text-button><md-elevated-button>Elevated</md-elevated-button><md-filled-tonal-button>Tonal</md-filled-tonal-button>
<md-icon-button><md-icon>share</md-icon></md-icon-button><md-filled-icon-button><md-icon>share</md-icon></md-filled-icon-button><md-filled-tonal-icon-button><md-icon>star</md-icon></md-filled-tonal-icon-button><md-outlined-icon-button><md-icon>star</md-icon></md-outlined-icon-button></div>
<div class="row"><md-chip-set><md-assist-chip label="Assist"></md-assist-chip><md-filter-chip label="Filter" selected></md-filter-chip><md-input-chip label="Input"></md-input-chip><md-suggestion-chip label="Suggest"></md-suggestion-chip></md-chip-set></div>
<md-tabs data-active="1"><md-primary-tab>Primary</md-primary-tab><md-primary-tab>Social</md-primary-tab></md-tabs>
{pg.menu_block()}
<md-list><md-list-item><md-icon slot="start">folder</md-icon><div slot="headline">Roadmap</div><div slot="supporting-text">Modified Mon</div></md-list-item><md-divider></md-divider><md-list-item><div slot="headline">One line</div><md-switch slot="end" selected></md-switch></md-list-item></md-list>
<div class="card elevated" data-component="Card" data-variant="elevated"><div class="headline">Card</div></div>
<div class="row"><md-checkbox></md-checkbox><md-radio name="b"></md-radio><span class="badge-wrap"><md-icon-button><md-icon>notifications</md-icon></md-icon-button><span class="badge" data-component="Badge" data-variant-size="large">3</span></span></div>
</div>
<div class="snackbar" data-component="Snackbar" data-variant-action="true" style="bottom:16px"><span class="msg">Message archived</span><span class="act">Undo</span></div>
<md-dialog open><div slot="headline">Discard draft?</div><form slot="content" method="dialog">This cannot be undone.</form><div slot="actions"><md-text-button>Cancel</md-text-button></div></md-dialog>
"""
    html_path = os.path.join(d, "sink.html")
    with open(html_path, "w") as f:
        f.write(page_html(random.Random(0), 600, 1000, body_html=body))
    doc = extract_ground_truth(html_path, 600, 1000, os.path.join(d, "sink.png"))
    with open(html_path) as f:
        return doc, f.read()


def _cases(d: str) -> list[tuple[Document, str]]:
    with open(os.path.join(d, "manifest.json")) as f:
        m = json.load(f)
    assert m["n"] == N and len(m["ids"]) == N
    return [(Document.load(os.path.join(d, cid + ".gt.json")), os.path.join(d, cid + ".png")) for cid in m["ids"]]


def test_load_case_resolves_paths(mwc_dir, synth_dir):
    for mod, d in ((mwc_corpus, mwc_dir), (synth, synth_dir)):
        with open(os.path.join(d, "manifest.json")) as f:
            cid = json.load(f)["ids"][0]
        doc, png = mod.load_case(d, cid)
        assert os.path.isabs(png) and os.path.exists(png) and doc.source_image == png
        raw = Document.load(os.path.join(d, cid + ".gt.json"))
        assert raw.source_image == cid + ".png"  # stored relative: fixtures are portable


def _ink_fraction(rgb: np.ndarray, n: Node) -> float:
    """Fraction of 'ink' pixels in a text box, or 0 if the ink is not the node's text colour.

    Ink = pixels that differ from the box's background (its most frequent colour; the top-left
    pixel is not reliable since text boxes are line boxes, whose corners can fall outside a small
    rounded surface such as a badge pill). The most saturated
    ink pixels must match the expected colour (composited under a black scrim of
    ``meta.overlay_alpha`` when flagged), which checks both box placement and colour.
    """
    x0, y0, x1, y1 = n.box.as_int()
    crop = rgb[y0:y1, x0:x1].astype(np.int32)
    if crop.size == 0:
        return 1.0
    ts = n.text_style
    col = np.array([ts.color.r, ts.color.g, ts.color.b], dtype=np.float64) * (1 - float(n.meta.get("overlay_alpha", 0.0)))
    cols, counts = np.unique(crop.reshape(-1, 3), axis=0, return_counts=True)
    bg = cols[int(np.argmax(counts))]
    dist_bg = np.abs(crop - bg).sum(axis=2)
    ink = dist_bg > 60
    if ink.mean() == 0:
        return 0.0
    strongest = crop[ink][np.argsort(-dist_bg[ink])[: max(1, int(ink.sum() * 0.1))]]
    if np.abs(strongest.mean(axis=0) - col).sum() > 90:
        return 0.0
    return float(ink.mean())


def _check_tree(doc: Document) -> None:
    root = doc.root
    assert root.box.w == doc.width and root.box.h == doc.height
    for n in doc.walk():
        assert root.box.contains(n.box, 0.75), f"{n.id} {n.name} outside page: {n.box}"
        assert n.box.w > 0 and n.box.h > 0, f"{n.id} degenerate box"
        for c in n.children:
            assert n.box.contains(c.box, 0.75), f"{c.id} {c.name} {c.box} outside parent {n.id} {n.box}"
        if n.type == "text":
            assert n.text and n.text.strip(), f"{n.id} empty text"
            assert n.text_style is not None and n.text_style.size > 0 and n.text_style.family
        if n.type == "icon":
            assert n.icon_name


# --------------------------------------------------------------------------- mwc corpus
def test_mwc_files_and_sizes(mwc_dir):
    for doc, png in _cases(mwc_dir):
        assert os.path.exists(png)
        assert os.path.exists(os.path.join(mwc_dir, doc.meta["html"]))
        rgb = load_rgb(png)
        assert rgb.shape[:2] == (doc.height, doc.width)
        assert doc.width in mwc_corpus.WIDTHS and 600 <= doc.height <= 1000


def test_mwc_tree_invariants(mwc_dir):
    for doc, _ in _cases(mwc_dir):
        _check_tree(doc)
        assert doc.root.fill_color is not None
        assert doc.root.count() > 10


def _refs(doc: Document) -> list[Node]:
    return [n for n in doc.walk() if n.component]


def _by(doc: Document, name: str, **variant: str) -> list[Node]:
    return [n for n in _refs(doc) if n.component.name == name and all(n.component.variant.get(k) == v for k, v in variant.items())]


def test_mwc_every_md_element_has_component_ref(sink):
    doc, html = sink
    import re
    from dt.mapping.material3 import PARTS
    authored = re.findall(r"<(md-[a-z-]+)", html)
    refs = _refs(doc)
    tags = {n.component.evidence.get("tag") for n in refs}
    parts = {n.meta.get("tag") for n in doc.walk() if n.meta.get("part")}
    # md-icon -> icon node (not a component); md-select-option sits in the closed select's
    # hidden menu, so it is not rendered and must not get a node; sub-parts and grouping
    # containers (md-list, md-chip-set, md-*-tab, md-menu-item) are recorded as meta.part
    for tag in set(authored) - {"md-icon", "md-select-option"}:
        if tag in PARTS:
            assert tag in parts and tag not in tags, f"<{tag}> must be a part, not a component"
        else:
            assert tag in tags, f"no component ref for authored <{tag}>"
    assert "md-select-option" not in tags | parts
    # expected (name, variant) pairs in the shared vocabulary, for a sample of elements
    have = {(n.component.name, tuple(sorted(n.component.variant.items()))) for n in refs}
    expected = [
        ("Button", {"style": s}) for s in ("filled", "outlined", "text", "elevated", "tonal")
    ] + [("IconButton", {"style": s}) for s in ("standard", "filled", "tonal", "outlined")] + [
        ("TextField", {"style": "outlined"}), ("TextField", {"style": "filled"}), ("Select", {"style": "outlined"}),
        ("Slider", {}), ("Switch", {"selected": "true"}), ("Switch", {"selected": "false"}),
        ("Checkbox", {"checked": "true"}), ("Checkbox", {"checked": "false"}), ("Radio", {"checked": "true"}),
        ("Radio", {"checked": "false"}), ("Progress", {"type": "linear"}), ("Progress", {"type": "circular"}),
        ("FAB", {"size": "regular", "extended": "true"}), ("FAB", {"size": "small", "extended": "false"}),
        ("Chip", {"type": "assist", "selected": "false"}), ("Chip", {"type": "filter", "selected": "true"}),
        ("Chip", {"type": "input", "selected": "false"}), ("Chip", {"type": "suggestion", "selected": "false"}),
        ("Tabs", {"type": "primary"}), ("Menu", {}), ("ListItem", {"lines": "2"}), ("ListItem", {"lines": "1"}),
        ("Divider", {}), ("Card", {"style": "elevated"}), ("Snackbar", {"action": "true"}), ("Dialog", {}),
        ("Badge", {"size": "large"}),
    ]
    for name, v in expected:
        assert (name, tuple(sorted(v.items()))) in have, f"missing {name}{v}; have {sorted(have)}"
    for n in refs:
        assert n.component.library == "material3" and n.component.key.startswith("m3.")
    # props: labels / icons / values are carried
    assert _by(doc, "Button", style="filled")[0].component.props["label"] == "Filled"
    assert _by(doc, "TextField", style="outlined")[0].component.props == {"label": "Email", "value": "a@b.com"}
    assert _by(doc, "FAB", extended="true")[0].component.props == {"label": "Compose", "icon": "edit"}
    assert _by(doc, "Select")[0].component.props["value"] == "Weekly on Monday"
    assert _by(doc, "ListItem", lines="2")[0].component.props["label"] == "Roadmap"


def test_mwc_variants_use_shared_vocabulary(sink, mwc_dir, synth_dir):
    """Every gt ComponentRef (both corpora) carries exactly the vocabulary keys with allowed values."""
    from dt.mapping.material3 import VARIANTS, component_key
    docs = [sink[0]] + [d for d, _ in _cases(mwc_dir)] + [d for d, _ in _cases(synth_dir)]
    seen = set()
    for doc in docs:
        for n in _refs(doc):
            vocab = VARIANTS[n.component.name]
            assert set(n.component.variant) == set(vocab), (n.component.name, n.component.variant)
            for k, v in n.component.variant.items():
                assert v in vocab[k], (n.component.name, k, v)
            assert n.component.key == component_key(n.component.name, n.component.variant)
            seen.add(n.component.name)
    assert {"Button", "Card", "ListItem", "Divider"} <= seen


def test_mwc_paint_lifted_onto_hosts(sink):
    """Unpainted md-* hosts take the paint of their covering child (checked against the screenshot)."""
    doc, _ = sink
    rgb = load_rgb(doc.source_image)
    fb = _by(doc, "Button", style="filled")[0]
    assert fb.fill_color.hex() == "#6750a4" and fb.box.h == 40 and min(fb.radius) == 20
    assert not any(c.type == "rect" for c in fb.children), "covering background must be lifted, not kept"
    assert fb.meta["lifted_from"][0]["cls"] == "background"
    # every visible lifted fill is what the screenshot shows just inside the host's left edge
    # (composited under the dialog's black scrim where flagged with meta.overlay_alpha)
    from dt.ir import Color
    dialog = _by(doc, "Dialog")[0].box  # partially covered nodes are not flagged occluded
    checked = 0
    for n in doc.walk():
        if not n.meta.get("lifted_from") or n.fill_color is None or n.meta.get("occluded") or n.box.intersect(dialog).area > 0:
            continue
        if n.fill_color.a < 1 or n.box.h < 12 or n.box.w < 16:
            continue
        k = 1.0 - float(n.meta.get("overlay_alpha", 0.0))
        want = Color(int(round(n.fill_color.r * k)), int(round(n.fill_color.g * k)), int(round(n.fill_color.b * k)))
        r, g, b = rgb[int(n.box.cy), int(n.box.x + 6)]
        assert Color(int(r), int(g), int(b)).delta_e(want) < 3, (n.name, n.box, n.fill_color.hex())
        checked += 1
    assert checked >= 3
    label = next(c for c in fb.children if c.type == "text")
    assert label.text == "Filled" and label.text_style.weight == 500 and label.text_style.color.hex() == "#ffffff"
    ob = _by(doc, "Button", style="outlined")[0]
    assert ob.fill_color is None and ob.strokes and ob.strokes[0].width == 1 and ob.strokes[0].color.hex() == "#79747e"
    assert min(ob.radius) == 20
    # FAB keeps its elevation shadows and gets its container colour + 16px corners
    fab = _by(doc, "FAB", extended="true")[0]
    assert fab.effects and fab.fill_color is not None and min(fab.radius) == 16
    # unselected switch: track fill + 2px outline on the host, the handle stays a child
    sw = _by(doc, "Switch", selected="false")[0]
    assert sw.fill_color is not None and sw.strokes and sw.strokes[0].width == 2 and min(sw.radius) == 16
    assert any(c.type == "ellipse" for c in sw.children)
    # the menu surface keeps its stacking layer when lifted onto the host
    menu = _by(doc, "Menu")[0]
    assert menu.fill_color is not None and int(menu.meta.get("z", 0)) > 1


def test_mwc_vectors_icons_badge_dialog(sink):
    doc, _ = sink
    rgb = load_rgb(doc.source_image)
    # radio svg: colour of the ring, never the white mask rect
    for rad in _by(doc, "Radio"):
        vec = next(c for c in rad.walk() if c.type == "vector")
        want = "#6750a4" if rad.component.variant["checked"] == "true" else "#49454f"
        assert vec.fill_color is not None and vec.fill_color.hex() == want, (rad.component.variant, vec.fill_color)
    # icons keep the 24px font box; meta.ink_box is the tight glyph inside it, on real ink
    icons = [n for n in doc.walk() if n.type == "icon" and not n.meta.get("occluded") and not n.meta.get("overlay_alpha")]
    assert icons and all("ink_box" in i.meta for i in icons)
    for i in icons:
        x, y, w, h = i.meta["ink_box"]
        assert i.box.contains(Box(x, y, w, h), 1.0) and 0 < w <= i.box.w and 0 < h <= i.box.h
        crop = rgb[int(y):int(y + h), int(x):int(x + w)].astype(np.int32)
        assert np.abs(crop - crop.reshape(-1, 3).max(axis=0)).sum(axis=2).max() > 60, f"no ink in {i.icon_name} ink box"
    # the badge component is the red pill itself
    badge = _by(doc, "Badge", size="large")[0]
    assert badge.fill_color.hex() == "#b3261e" and badge.box.h == 16
    # Dialog is the rounded surface; the full-viewport host is a part holding the translucent scrim
    dlg = _by(doc, "Dialog")[0]
    assert dlg.box.w < doc.width and min(dlg.radius) == 28 and any(c.type == "text" for c in dlg.walk())
    host = next(n for n in doc.walk() if n.meta.get("part") == "DialogHost")
    scrim = next(c for c in host.children if c.type == "rect" and c.box.w == doc.width)
    assert 0 < scrim.fill_color.a < 1


def test_mwc_node_types_and_paint(sink):
    doc, _ = sink
    _check_tree(doc)
    types = {n.type for n in doc.walk()}
    for t in ("frame", "rect", "text", "icon", "line", "ellipse", "vector"):
        assert t in types, f"no {t} node; have {types}"
    # elevated button carries md-elevation shadows
    assert _by(doc, "Button", style="elevated")[0].effects, "elevation shadows not attached"
    # icon node has the ligature name and a colour
    icons = [n for n in doc.walk() if n.type == "icon"]
    assert any(i.icon_name == "edit" for i in icons) and all(i.fill_color is not None for i in icons)
    # divider is a 1px line in outline-variant
    div = _by(doc, "Divider")[0]
    assert div.type == "line" and div.box.h == 1 and div.fill_color.hex() == "#cac4d0"
    # dialog scrim is a translucent full-page rect; nodes under it are flagged occluded only when fully covered
    host = next(n for n in doc.walk() if n.meta.get("part") == "DialogHost")
    scrim = next(c for c in host.children if c.type == "rect" and c.box.w == doc.width)
    assert 0 < scrim.fill_color.a < 1
    assert any(n.meta.get("occluded") for n in doc.walk())
    # ... and nodes under the translucent scrim (but not under the opaque container) are marked
    assert any(n.meta.get("overlay_alpha") == scrim.fill_color.a for n in doc.walk() if n.type == "text")
    # input value text measured from <input>
    assert any(n.text == "a@b.com" and n.meta.get("src") == "input" for n in doc.walk())


def test_mwc_text_matches_render(sink):
    """Text boxes must sit on dark glyph pixels in the real screenshot (sanity of Range rects)."""
    doc, _ = sink
    rgb = load_rgb(doc.source_image)
    checked = 0
    for n in doc.walk():
        if n.type != "text" or n.meta.get("occluded") or n.text_style.color.a < 1:
            continue
        assert _ink_fraction(rgb, n) > 0.03, f"text {n.text!r} box {n.box} has no pixels of its colour"
        checked += 1
    assert checked >= 20


def test_mwc_deterministic(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    ma = mwc_corpus.generate(2, a, seed=11)
    mb = mwc_corpus.generate(2, b, seed=11)
    assert ma == mb
    for cid in ma["ids"]:
        da = Document.load(os.path.join(a, cid + ".gt.json")).to_dict()
        db = Document.load(os.path.join(b, cid + ".gt.json")).to_dict()
        da["source_image"] = db["source_image"] = None
        da["meta"].pop("html", None)
        db["meta"].pop("html", None)
        assert da == db
    mc = mwc_corpus.generate(2, str(tmp_path / "c"), seed=12)
    assert mc["cases"][0]["html_sha1"] != ma["cases"][0]["html_sha1"]


def test_mwc_manifest(mwc_dir):
    with open(os.path.join(mwc_dir, "manifest.json")) as f:
        m = json.load(f)
    assert m["kind"] == "mwc" and m["seed"] == SEED
    assert sum(m["component_histogram"].values()) > 0
    for c in m["cases"]:
        assert c["nodes"] > 0 and c["width"] in mwc_corpus.WIDTHS


# --------------------------------------------------------------------------- synth corpus
def test_synth_files_and_sizes(synth_dir):
    for doc, png in _cases(synth_dir):
        rgb = load_rgb(png)
        assert rgb.shape[:2] == (doc.height, doc.width)
        assert doc.source_image == os.path.basename(png)


def test_synth_tree_invariants(synth_dir):
    for doc, _ in _cases(synth_dir):
        _check_tree(doc)
        types = {n.type for n in doc.walk()}
        assert {"frame", "text", "icon"} <= types
        assert any(n.component for n in doc.walk())
        assert all(n.tokens.get("fill", "").startswith("md.sys.color.") for n in doc.walk() if n.fills and n.type != "text")


def test_synth_gt_equals_render(synth_dir):
    """gt == IR: re-rendering the saved IR reproduces the stored png (near-)exactly."""
    from dt.render.screenshot import render_doc
    for doc, png in _cases(synth_dir):
        a = load_rgb(png).astype(np.int32)
        b = render_doc(doc).astype(np.int32)
        assert a.shape == b.shape
        assert np.abs(a - b).max() <= 2
        # text boxes are tight: every text box contains pixels of its own colour
        for n in doc.walk():
            if n.type == "text":
                assert _ink_fraction(a, n) > 0.03, n.text


def test_synth_deterministic(tmp_path):
    ma = synth.generate(2, str(tmp_path / "a"), seed=5)
    mb = synth.generate(2, str(tmp_path / "b"), seed=5)
    assert ma == mb
    da = Document.load(os.path.join(str(tmp_path / "a"), ma["ids"][0] + ".gt.json")).to_dict()
    db = Document.load(os.path.join(str(tmp_path / "b"), ma["ids"][0] + ".gt.json")).to_dict()
    da["source_image"] = db["source_image"] = None
    assert da == db


# --------------------------------------------------------------------------- shipped corpora
@pytest.mark.parametrize("kind", ["mwc", "synth"])
def test_shipped_corpus_present(kind):
    from dt.selftest import corpus_dir
    d = corpus_dir(kind)
    if not os.path.exists(os.path.join(d, "manifest.json")):
        pytest.skip("shipped corpus not generated")
    with open(os.path.join(d, "manifest.json")) as f:
        m = json.load(f)
    assert m["n"] == 12 and m["seed"] == 1
    for cid in m["ids"]:
        doc = Document.load(os.path.join(d, cid + ".gt.json"))
        _check_tree(doc)
        assert os.path.exists(os.path.join(d, cid + ".png"))


# --------------------------------------------------------------------------- render-only oracle (gt -> render == page)
def _ink_centroid_y(rgb: np.ndarray, box: Box, pad: int = 4) -> Optional[float]:
    """Ink-weighted vertical centroid (page px) of the glyphs in ``box`` (padded), or None."""
    x0, y0 = max(0, int(box.x) - pad), max(0, int(box.y) - pad)
    x1, y1 = min(rgb.shape[1], int(box.x2) + pad), min(rgb.shape[0], int(box.y2) + pad)
    crop = rgb[y0:y1, x0:x1].astype(np.float64)
    if crop.size == 0:
        return None
    ink = np.abs(crop - crop[0, 0]).sum(axis=2)
    ink[ink < 30] = 0
    rows = ink.sum(axis=1)
    if rows.sum() <= 0:
        return None
    return y0 + float((rows * np.arange(len(rows))).sum() / rows.sum())


def _render_gt(doc: Document) -> np.ndarray:
    from dt.render.screenshot import render_doc
    return render_doc(doc)


def test_mwc_text_boxes_are_line_boxes(sink):
    """IR contract: a text node's box is its line box (height == line_height), like synth gt."""
    doc, _ = sink
    checked = 0
    for n in doc.walk():
        if n.type != "text" or not n.text_style.line_height or n.meta.get("src") != "dom-text":
            continue
        if n.box.y <= 0.5 or n.box.y2 >= doc.height - 0.5:
            continue  # clipped by the viewport
        assert abs(n.box.h - n.text_style.line_height) < 0.02, (n.text, n.box, n.text_style.line_height)
        checked += 1
    assert checked >= 20


def test_mwc_gt_render_reproduces_text_position(sink):
    """render(gt) puts every text line where Material Web drew it (vertical ink centroid <= 0.5px)."""
    doc, _ = sink
    target = load_rgb(doc.source_image)
    rendered = _render_gt(doc)
    dys = []
    for n in doc.walk():
        if n.type != "text" or n.meta.get("occluded") or n.text_style.color.a < 1:
            continue
        a, b = _ink_centroid_y(target, n.box), _ink_centroid_y(rendered, n.box)
        if a is None or b is None:
            continue
        dys.append(b - a)
    assert len(dys) >= 15
    assert abs(float(np.median(dys))) <= 0.5, sorted(dys)
    assert float(np.percentile(np.abs(dys), 90)) <= 1.0, sorted(dys)


def test_mwc_open_menu_paints_above_later_content(sink):
    """An open md-menu (stacking layer > page content) paints after the siblings that follow it in
    the DOM, so render(gt) shows the menu, not the list under it."""
    doc, _ = sink
    order = {id(n): i for i, n in enumerate(doc.walk())}
    menu = _by(doc, "Menu")[0]
    end = order[id(menu)] + menu.count()
    later_lower = [n for n in doc.walk() if order[id(n)] >= end and int(n.meta.get("z", 0)) < int(menu.meta["z"])
                   and n.box.intersect(menu.box).area > 0]
    assert not later_lower, [(n.name, n.box) for n in later_lower[:5]]
    target = load_rgb(doc.source_image).astype(np.int32)
    rendered = _render_gt(doc).astype(np.int32)
    x0, y0, x1, y1 = menu.box.as_int()
    assert np.abs(target[y0:y1, x0:x1] - rendered[y0:y1, x0:x1]).sum(axis=2).mean() < 6


def test_mwc_menu_inside_painted_card_is_hoisted(tmp_path):
    """Adversarial: an open menu nested in a painted card overlaps the NEXT card. In CSS the menu
    (z-index 20) paints above it; gt must re-parent the menu to the root after the second card."""
    body = """<div class="main">
<div class="card filled" data-component="Card" data-variant-style="filled" style="height:120px">
<div class="headline">First</div>
<span class="menu-anchor"><md-outlined-button id="m1-a">Options</md-outlined-button><md-menu id="m1" anchor="m1-a" open quick>
<md-menu-item><div slot="headline">Rename</div></md-menu-item><md-menu-item><div slot="headline">Move to trash</div></md-menu-item>
<md-menu-item><div slot="headline">Share</div></md-menu-item><md-menu-item><div slot="headline">Download</div></md-menu-item></md-menu></span>
</div>
<div class="card elevated" data-component="Card" data-variant-style="elevated" style="height:200px"><div class="headline">Second card</div><div class="supporting">Covered by the menu</div></div>
</div>"""
    p = tmp_path / "menu.html"
    p.write_text(page_html(random.Random(0), 412, 600, body_html=body, html_dir=str(tmp_path)))
    doc = extract_ground_truth(str(p), 412, 600, str(tmp_path / "menu.png"))
    _check_tree(doc)
    menu = _by(doc, "Menu")[0]
    cards = _by(doc, "Card")
    assert len(cards) == 2 and menu.box.intersect(cards[1].box).area > 0
    assert menu in doc.root.children and menu.meta.get("reparented_from")
    order = [id(n) for n in doc.walk()]
    assert order.index(id(menu)) > order.index(id(cards[1]))
    target = load_rgb(doc.source_image).astype(np.int32)
    rendered = _render_gt(doc).astype(np.int32)
    x0, y0, x1, y1 = menu.box.intersect(cards[1].box).as_int()
    assert np.abs(target[y0:y1, x0:x1] - rendered[y0:y1, x0:x1]).sum(axis=2).mean() < 6


def test_mwc_slider_active_track_is_clipped(sink):
    """md-slider's active track is a full-width ::after clipped by clip-path: inset(); gt must
    carry the clipped geometry (value 40 -> active track ~40% of the inactive one)."""
    doc, _ = sink
    sl = _by(doc, "Slider")[0]
    tracks = sorted((c for c in sl.walk() if c.type == "rect" and c.box.h <= 6 and c.box.w > 20), key=lambda c: -c.box.w)
    assert len(tracks) >= 2, [(c.name, c.box) for c in sl.walk()]
    inactive, active = tracks[0], tracks[1]
    assert active.box.x == pytest.approx(inactive.box.x, abs=1)
    assert 0.3 < active.box.w / inactive.box.w < 0.5, (active.box, inactive.box)
    target = load_rgb(doc.source_image).astype(np.int32)
    rendered = _render_gt(doc).astype(np.int32)
    x0, y0, x1, y1 = inactive.box.as_int()
    assert np.abs(target[y0:y1, x0:x1] - rendered[y0:y1, x0:x1]).sum(axis=2).mean() < 12


def test_mwc_filled_field_active_indicator(sink):
    """md-filled-field draws its 1px bottom active indicator as a 0px-tall ::before with a
    border-bottom; the painted box is the border box, so gt must have that 1px line."""
    doc, _ = sink
    tf = _by(doc, "TextField", style="filled")[0]
    lines = [c for c in tf.walk() if c is not tf and c.box.h <= 1.01 and c.box.w >= tf.box.w - 1
             and abs(c.box.y2 - tf.box.y2) <= 1.01]
    assert lines, [(c.name, c.box) for c in tf.walk()]
    ln = lines[0]
    col = ln.fill_color or (ln.strokes[0].color if ln.strokes else None)
    assert col is not None and col.hex() == "#49454f"


def test_mwc_html_is_portable(mwc_dir):
    """Saved pages reference fonts and the Material Web bundle relative to the corpus dir, never
    by the absolute path of the checkout that generated them."""
    for doc, _ in _cases(mwc_dir):
        with open(os.path.join(mwc_dir, doc.meta["html"])) as f:
            html = f.read()
        assert "file://" not in html
        assert "mwc.bundle.js" in html and "roboto.local.css" in html


def test_mwc_gt_render_oracle(mwc_dir):
    """Render-only oracle: render(gt IR) reproduces the Material Web page (no perception involved)."""
    from dt.compare import diff_map, pixel_metrics
    for doc, png in _cases(mwc_dir):
        t, r = load_rgb(png), _render_gt(doc)
        pm = pixel_metrics(t, r, diff=diff_map(t, r))
        assert pm["mean_de"] < 0.6, (doc.meta["id"], pm["mean_de"])


def test_mwc_slider_ticks_and_hidden_label(tmp_path):
    """Adversarial: a labeled slider with ticks. The value label is scale(0) (invisible) and the
    tick marks are a repeating radial-gradient of dots; neither may become a solid gt shape."""
    body = '<div class="main"><div class="form"><md-slider value="71" labeled ticks step=10></md-slider></div></div>'
    p = tmp_path / "slider.html"
    p.write_text(page_html(random.Random(0), 412, 200, body_html=body, html_dir=str(tmp_path)))
    doc = extract_ground_truth(str(p), 412, 200, str(tmp_path / "slider.png"))
    sl = _by(doc, "Slider")[0]
    big_round = [c for c in sl.walk() if c.type == "ellipse" and c.box.w > 24]
    assert not big_round, [(c.meta.get("cls"), c.box) for c in big_round]
    assert not [c for c in sl.walk() if c.meta.get("gradient")], "tick-dot pattern emitted as a solid fill"
    target = load_rgb(doc.source_image).astype(np.int32)
    rendered = _render_gt(doc).astype(np.int32)
    x0, y0, x1, y1 = sl.box.as_int()
    assert np.abs(target[y0:y1, x0:x1] - rendered[y0:y1, x0:x1]).sum(axis=2).mean() < 6


def test_mwc_snackbar_over_outlined_field_paint_order(tmp_path):
    """Adversarial (mwc_1_009): a snackbar (positioned, layer 1) over an outlined text field whose
    outline + label paint in layer 2 while its leading icon paints in layer 1. No subtree order
    expresses that, so the outline/label must leave the field subtree; render(gt) == page there."""
    body = """<div class="main"><div class="form">
<md-outlined-text-field label="Password" style="width:1088px"><md-icon slot="leading-icon">lock</md-icon></md-outlined-text-field>
</div></div>
<div class="snackbar" data-component="Snackbar" data-variant-action="false" style="bottom:72px"><span class="msg">Copied link</span></div>"""
    p = tmp_path / "snack.html"
    p.write_text(page_html(random.Random(0), 1200, 140, body_html=body, html_dir=str(tmp_path)))
    doc = extract_ground_truth(str(p), 1200, 140, str(tmp_path / "snack.png"))
    _check_tree(doc)
    tf, sb = _by(doc, "TextField")[0], _by(doc, "Snackbar")[0]
    ov = tf.box.intersect(sb.box)
    assert ov.area > 400, (tf.box, sb.box)
    lock = next(n for n in doc.walk() if n.type == "icon" and n.icon_name == "lock")
    assert lock.meta.get("occluded"), "the layer-1 icon is under the later layer-1 snackbar"
    target = load_rgb(doc.source_image).astype(np.int32)
    rendered = _render_gt(doc).astype(np.int32)
    x0, y0, x1, y1 = ov.as_int()
    err = np.abs(target[y0:y1, x0:x1] - rendered[y0:y1, x0:x1]).sum(axis=2)
    assert err.mean() < 6, err.mean()
    # the field's bottom outline (layer 2) is drawn above the snackbar along its whole length;
    # lifting the long outline piece onto the field host would bury it under the snackbar
    assert (err > 60).sum(axis=1).max() < 0.25 * err.shape[1], ("a whole outline row differs", (err > 60).sum(axis=1).max(), err.shape)
