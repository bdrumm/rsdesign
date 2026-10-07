"""Export round-trip (docs/VALIDATION.md): IR -> plan -> code.js on fake Figma -> Figma tree -> IR -> render.

Every case renders the IR directly and through the export, and the mean ΔE2000 between the two must stay
under ``validate.roundtrip.max_mean_de``. The adversarial cases are the ones that were lossy on main:
Figma auto-layout re-positioning children (inexact inferred layouts, glyph-metric dependent rows), auto-width
text with vertical alignment, filled Material Symbols, rounded line nodes, local image files, collapsed
instances, and 0.01px coordinate rounding.
"""
from __future__ import annotations

import glob
import os
import shutil

import numpy as np
import pytest
from PIL import Image

from dt.export import to_build_plan
from dt.export.figma_layout import apply_auto_layout, demote_inexact_layouts
from dt.ir import Box, Color, ComponentRef, Document, Fill, GradientStop, Layout, Node, Stroke, TextStyle
from dt.params import P
from dt.validate.roundtrip import plan_tree_to_ir, roundtrip, run_plugin

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run the Figma plugin")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PURPLE, ON_SURFACE = Color(103, 80, 164), Color(29, 27, 32)


def _ts(**kw) -> TextStyle:
    base = dict(family="Roboto", size=14, weight=500, color=ON_SURFACE)
    base.update(kw)
    return TextStyle(**base)


# --------------------------------------------------------------------------- adversarial documents
def doc_filled_icon() -> Document:
    doc = Document.blank(160, 80, "#ffffff")
    for i, fill in enumerate((1, 0)):
        ic = Node(id=f"ic{i}", type="icon", box=Box(16 + 72 * i, 16, 48, 48), icon_name="favorite",
                  fills=[Fill.solid(Color(179, 38, 30))], meta={"icon_fill": fill})
        doc.root.children.append(ic)
    return doc


def doc_valign_text() -> Document:
    doc = Document.blank(360, 200, "#ffffff")
    doc.root.children = [
        Node(id="t_mid", type="text", box=Box(16, 8, 140, 56), text="Inbox", text_style=_ts(valign="middle")),
        Node(id="t_bot", type="text", box=Box(180, 8, 140, 56), text="Starred", text_style=_ts(valign="bottom", line_height=20)),
        Node(id="t_ml", type="text", box=Box(16, 80, 200, 100), text="Two\nlines", text_style=_ts(valign="middle")),
    ]
    return doc


def doc_rounded_line() -> Document:
    doc = Document.blank(200, 60, "#f7f2fa")
    doc.root.children = [Node(id="handle", type="line", box=Box(68, 20, 64, 12), radius=6, fills=[Fill.solid("#79747e")])]
    return doc


def doc_local_image(tmp_dir: str) -> Document:
    yy, xx = np.mgrid[0:60, 0:90]
    img = np.stack([xx * 2.8, yy * 4.0, 255 - xx * 2.8], -1).clip(0, 255).astype(np.uint8)
    path = os.path.join(tmp_dir, "photo.png")
    Image.fromarray(img).save(path)
    doc = Document.blank(140, 100, "#ffffff")
    doc.root.children = [Node(id="img", type="image", box=Box(20, 20, 90, 60), image_ref=path)]
    return doc


def doc_collapsed_instance() -> Document:
    """A list item mapped to a component and collapsed (`dt map --collapse`): its real subtree is in meta."""
    from dt.mapping.matcher import collapse_instances
    doc = Document.blank(320, 80, "#ffffff")
    item = Node(id="li", type="instance", box=Box(0, 8, 320, 56), fills=[Fill.solid("#f3edf7")],
                component=ComponentRef(key="no-such-key", name="List item", library="material3",
                                       props={"label": "Headline"}, confidence=0.9))
    item.children = [
        Node(id="li_icon", type="icon", box=Box(16, 24, 24, 24), icon_name="person", fills=[Fill.solid(PURPLE)]),
        Node(id="li_text", type="text", box=Box(56, 26, 120, 20), text="Headline", text_style=_ts(size=16, weight=400)),
    ]
    doc.root.children = [item]
    return collapse_instances(doc)


def doc_chip_row() -> Document:
    """Filter chip [check][label] centred in a fixed frame; the label box is wider than its glyphs (OCR box)."""
    doc = Document.blank(200, 60, "#ffffff")
    chip = Node(id="chip", type="frame", box=Box(20, 10, 110, 32), fills=[Fill.solid("#e8def8")], radius=8,
                layout=Layout(mode="row", gap=8, padding=(7, 6, 7, 6), align_items="center", justify="center"))
    chip.children = [Node(id="chk", type="icon", box=Box(26, 17, 18, 18), icon_name="done", fills=[Fill.solid(ON_SURFACE)]),
                     Node(id="lbl", type="text", box=Box(52, 17, 72, 18), text="Filter", text_style=_ts())]
    doc.root.children = [chip]
    return doc


def doc_inexact_column() -> Document:
    """Left-aligned lines in a wide card, but the inferred layout says align=center with symmetric padding
    (perceive prefers centre when centres agree within 3px) -- Figma would centre them in the card."""
    doc = Document.blank(260, 90, "#ffffff")
    card = Node(id="card", type="frame", box=Box(10, 10, 220, 70), fills=[Fill.solid("#b2dfdb")], radius=6,
                layout=Layout(mode="column", gap=10, padding=(13, 14, 9, 14), align_items="center", justify="start"))
    card.children = [Node(id="l1", type="text", box=Box(24, 23, 92, 15), text="Hiking options", text_style=_ts(size=12, weight=400)),
                      Node(id="l2", type="text", box=Box(23.5, 48, 90, 15), text="Multnomah Falls", text_style=_ts(size=12, weight=400))]
    doc.root.children = [card]
    return doc


def doc_exact_layouts() -> Document:
    """Layouts Figma reproduces whatever the glyph metrics: must keep auto-layout and round-trip exactly."""
    doc = Document.blank(300, 140, "#ffffff")
    col = Node(id="col", type="frame", box=Box(10, 10, 200, 60), fills=[Fill.solid("#fef7ff")],
               layout=Layout(mode="column", gap=4, padding=(8, 8, 8, 12), align_items="start", justify="start"))
    col.children = [Node(id="c1", type="text", box=Box(22, 18, 120, 20), text="Title", text_style=_ts(line_height=20)),
                    Node(id="c2", type="text", box=Box(22, 42, 160, 20), text="Supporting text", text_style=_ts(weight=400, line_height=20))]
    row = Node(id="row", type="frame", box=Box(10, 80, 200, 48), fills=[Fill.solid("#eaddff")], radius=24,
               layout=Layout(mode="row", gap=8, padding=(12, 16, 12, 16), align_items="start", justify="start"))
    row.children = [Node(id="r1", type="icon", box=Box(26, 92, 24, 24), icon_name="add", fills=[Fill.solid(PURPLE)]),
                    Node(id="r2", type="text", box=Box(58, 92, 100, 24), text="Compose", text_style=_ts(line_height=24))]
    doc.root.children = [col, row]
    return doc


def doc_subpixel_text() -> Document:
    """The outlined text-field label from the mwc_test translation: parent-relative coords rounded to 0.01px
    moved the glyphs by 0.005px, enough to change their rasterisation."""
    doc = Document.blank(240, 150, "#fffbfe")
    field = Node(id="tf", type="frame", box=Box(16, 84, 199, 56), strokes=[Stroke(Color(121, 116, 126), 1, "inside")], radius=4)
    doc.root.children = [field, Node(id="email", type="text", box=Box(32.405, 76.94847, 33.355, 13.94561), text="Email",
                                     text_style=_ts(size=11.9, weight=500, color=Color(73, 69, 79)))]
    return doc


def doc_gradients() -> Document:
    doc = Document.blank(300, 140, "#ffffff")
    stops = [GradientStop(0, Color(103, 80, 164)), GradientStop(1, Color(255, 216, 228))]
    doc.root.children = [
        Node(id="lin", type="rect", box=Box(10, 10, 200, 50), fills=[Fill(kind="linear", stops=stops, angle=45)], radius=(12, 4, 12, 4)),
        Node(id="rad", type="rect", box=Box(10, 70, 120, 60), fills=[Fill(kind="radial", stops=stops)],
             strokes=[Stroke(Color(0, 0, 0), 2, "center")]),
        Node(id="ell", type="ellipse", box=Box(150, 70, 60, 60), fills=[Fill.solid("#21005d")],
             strokes=[Stroke(Color(255, 0, 0), 3, "outside")]),
    ]
    return doc


ADVERSARIAL = {
    "filled_icon": doc_filled_icon, "valign_text": doc_valign_text, "rounded_line": doc_rounded_line,
    "collapsed_instance": doc_collapsed_instance, "chip_row": doc_chip_row, "inexact_column": doc_inexact_column,
    "exact_layouts": doc_exact_layouts, "subpixel_text": doc_subpixel_text, "gradients": doc_gradients,
}


def _assert_lossless(rep, name: str) -> None:
    assert not rep.lossy and rep.mean_de <= float(P["validate.roundtrip.max_mean_de"]), (name, rep.mean_de, rep.worst_box, rep.warnings)


# --------------------------------------------------------------------------- tests
@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_adversarial_roundtrip_is_lossless(name):
    rep = roundtrip(ADVERSARIAL[name]())
    _assert_lossless(rep, name)
    assert rep.mean_de < 0.05, (name, rep.mean_de, rep.worst_box)
    assert rep.frac_bad == 0.0, (name, rep.max_de, rep.worst_box)  # not one pixel above JND


def test_local_image_file_is_embedded(tmp_path):
    doc = doc_local_image(str(tmp_path))
    plan = to_build_plan(doc)
    paint = plan["root"]["children"][0]["fills"][-1]
    assert paint["type"] == "IMAGE" and paint["imageRef"].startswith("data:image/png;base64,")
    rep = roundtrip(doc, plan=plan)
    _assert_lossless(rep, "local_image")
    assert rep.mean_de < 0.05
    assert not any("placeholder" in w for w in rep.warnings)


def test_filled_icon_flag_survives_plugin():
    plan = to_build_plan(doc_filled_icon())
    filled, outlined = plan["root"]["children"]
    assert filled["iconFill"] == 1 and "iconFill" not in outlined
    tree = run_plugin(plan)["tree"][0]
    assert tree["children"][0]["pluginData"]["dt.iconFill"] == "1"
    assert tree["children"][0]["pluginData"]["dt.id"] == "ic0"
    back = plan_tree_to_ir(tree, 160, 80, measure=False)
    assert back.root.find("ic0").meta["icon_fill"] == 1 and "icon_fill" not in back.root.find("ic1").meta


def test_inexact_and_metric_sensitive_layouts_are_demoted():
    for build, fid in ((doc_chip_row, "chip"), (doc_inexact_column, "card")):
        plan = to_build_plan(build())
        frame = plan["root"]["children"][0]
        assert frame["id"] == fid and "layoutMode" not in frame, frame.get("layoutMode")
        assert any(fid in w and "auto-layout" in w for w in plan["warnings"])
    # with verification off the same plans are lossy: Figma moves the children
    old = P["export.figma.verify_layout"]
    P.set("export.figma.verify_layout", False)
    try:
        assert roundtrip(doc_chip_row()).mean_de > 0.02  # the label/check shift (~4px) on a small canvas
        assert roundtrip(doc_inexact_column()).lossy
    finally:
        P.set("export.figma.verify_layout", old)


def test_exact_layouts_are_kept():
    plan = to_build_plan(doc_exact_layouts())
    col, row = plan["root"]["children"]
    assert col["layoutMode"] == "VERTICAL" and row["layoutMode"] == "HORIZONTAL"
    assert not any("auto-layout" in w for w in plan["warnings"])


def test_auto_layout_model_matches_css_flex_render():
    """The Figma layout model agrees with dt.render's flex mode on an exact layout (sanity of the model)."""
    from dt.render.screenshot import render_doc
    from dt.validate.fidelity import delta_e2000_map
    doc = doc_exact_layouts()
    assert float(delta_e2000_map(render_doc(doc), render_doc(doc, mode="flex")).mean()) < 0.01
    plan = to_build_plan(doc)
    sim = apply_auto_layout(__import__("copy").deepcopy(plan["root"]))
    for a, b in zip(plan["root"]["children"], sim["children"]):
        for ca, cb in zip(a["children"], b["children"]):
            assert abs(ca["x"] - cb["x"]) < 0.01 and abs(ca["y"] - cb["y"]) < 0.01, (ca["id"], ca["x"], cb["x"])


def test_demote_is_innermost_first():
    """A parent whose only problem is a bad child frame keeps its own auto-layout once the child is fixed."""
    inner = {"id": "in", "type": "FRAME", "x": 0, "y": 0, "width": 50, "height": 20, "layoutMode": "HORIZONTAL",
             "itemSpacing": 0, "paddingTop": 0, "paddingRight": 0, "paddingBottom": 0, "paddingLeft": 0,
             "primaryAxisAlignItems": "MIN", "counterAxisAlignItems": "MIN", "primaryAxisSizingMode": "AUTO",
             "counterAxisSizingMode": "FIXED", "layoutSizingHorizontal": "HUG", "layoutSizingVertical": "FIXED",
             "children": [{"id": "a", "type": "RECTANGLE", "x": 5, "y": 0, "width": 20, "height": 20}]}
    outer = {"id": "out", "type": "FRAME", "x": 0, "y": 0, "width": 100, "height": 20, "layoutMode": "HORIZONTAL",
             "itemSpacing": 10, "paddingTop": 0, "paddingRight": 0, "paddingBottom": 0, "paddingLeft": 0,
             "primaryAxisAlignItems": "MIN", "counterAxisAlignItems": "MIN", "primaryAxisSizingMode": "FIXED",
             "counterAxisSizingMode": "FIXED",
             "children": [inner, {"id": "b", "type": "RECTANGLE", "x": 60, "y": 0, "width": 20, "height": 20,
                                  "layoutSizingHorizontal": "FIXED", "layoutSizingVertical": "FIXED"}]}
    warnings: list[str] = []
    assert demote_inexact_layouts(outer, warnings) == 1
    assert "layoutMode" not in inner and outer["layoutMode"] == "HORIZONTAL", warnings


def test_fake_corner_radius_follows_figma_semantics():
    doc = Document.blank(100, 60, "#ffffff")
    doc.root.children = [Node(id="u", type="rect", box=Box(10, 10, 30, 30), radius=8, fills=[Fill.solid("#000000")]),
                         Node(id="p", type="rect", box=Box(50, 10, 30, 30), radius=(8, 0, 8, 0), fills=[Fill.solid("#000000")])]
    tree = run_plugin(to_build_plan(doc))["tree"][0]
    u, p = tree["children"]
    assert (u["cornerRadius"], u["topLeftRadius"], u["bottomRightRadius"]) == (8, 8, 8)
    assert p["cornerRadius"] == "MIXED" and (p["topLeftRadius"], p["topRightRadius"]) == (8, 0)


def test_fake_image_hash_is_content_based(tmp_path):
    """Two different PNGs of the same byte length must not share an imageHash (they did: 'img_<len>')."""
    import base64
    import io
    uris = []
    for color in ((255, 0, 0), (0, 0, 255)):
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), color).save(buf, format="PNG", compress_level=0)  # stored: length is content-free
        uris.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    assert len(uris[0]) == len(uris[1])
    doc = Document.blank(40, 20, "#ffffff")
    doc.root.children = [Node(id=f"i{k}", type="image", box=Box(2 + 20 * k, 2, 16, 16), image_ref=u) for k, u in enumerate(uris)]
    out = run_plugin(to_build_plan(doc))
    hashes = [c["fills"][-1]["imageHash"] for c in out["tree"][0]["children"]]
    assert hashes[0] != hashes[1] and set(hashes) <= set(out["images"])
    _assert_lossless(roundtrip(doc), "two_images")


@pytest.mark.parametrize("case", ["mwc/mwc_1_002", "mwc/mwc_1_009", "synth/synth_1_005"])
def test_corpus_ground_truth_roundtrip(case):
    path = os.path.join(ROOT, "fixtures", "corpus", case + ".gt.json")
    if not os.path.exists(path):
        pytest.skip("corpus not generated")
    rep = roundtrip(Document.load(path))
    _assert_lossless(rep, case)
