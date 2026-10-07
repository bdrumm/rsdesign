"""Tests for dt.compare — ground truth comes from the real renderer (dt.render), never mocks."""
from __future__ import annotations

import copy
import json
import os

import numpy as np
import pytest

from dt.common.image import crop, load_rgb
from dt.compare import (
    LossReport,
    alignment_offset,
    diff_map,
    evaluate,
    layout_consistency,
    levenshtein,
    match_nodes,
    per_node_errors,
    pixel_metrics,
    residual_regions,
    save_diff_image,
    structural_metrics,
    text_cer,
    text_term_from_lines,
)
from dt.ir import Box, Color, ComponentRef, Document, Fill, Layout, Node, TextStyle
from dt.params import P
from dt.render.screenshot import render_doc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MWC_PNG = os.path.join(ROOT, "out", "mwc_test.png")
OUT_DIR = os.path.join(ROOT, "out", "test_compare")


def make_doc() -> Document:
    """Small screen: button, text, circle, divider, card with a title."""
    doc = Document.blank(400, 300)
    btn = Node(id="btn", type="rect", box=Box(40, 40, 120, 48), fills=[Fill.solid("#1a73e8")], radius=8)
    txt = Node(id="txt", type="text", box=Box(180, 52, 180, 24), text="Hello world",
               text_style=TextStyle(size=16, weight=500, color=Color(32, 33, 36)))
    circ = Node(id="circ", type="ellipse", box=Box(300, 200, 48, 48), fills=[Fill.solid("#d93025")])
    div = Node(id="div", type="line", box=Box(40, 150, 320, 1), fills=[Fill.solid("#dadce0")])
    card = Node(id="card", type="frame", box=Box(40, 180, 200, 90), fills=[Fill.solid("#f1f3f4")], radius=12,
                children=[Node(id="cardtxt", type="text", box=Box(56, 196, 160, 20), text="Card title",
                               text_style=TextStyle(size=14))])
    doc.root.children += [btn, txt, circ, div, card]
    return doc


@pytest.fixture(scope="module")
def base():
    doc = make_doc()
    return doc, render_doc(doc)


# --------------------------------------------------------------------------- pixel / loss
def test_identical_is_zero(base):
    doc, target = base
    rep = evaluate(doc, target, target, use_ocr=False)
    assert rep.total == pytest.approx(0.0, abs=1e-9)
    assert rep.pixel["mean_de"] == 0.0 and rep.pixel["frac_bad"] == 0.0
    assert rep.pixel["ssim"] == pytest.approx(1.0)
    assert rep.residual_regions == []
    assert all(v == 0.0 for v in rep.per_node.values())
    assert rep.diff_map.shape == (300, 400)


def test_shifted_node_residual_and_alignment(base):
    doc, target = base
    d2 = copy.deepcopy(doc)
    btn = d2.find("btn")
    btn.box = btn.box.translate(6, 0)
    rendered = render_doc(d2)
    rep = evaluate(d2, target, rendered, use_ocr=False)
    assert rep.total > 0
    # residual regions sit on the moved button (old ∪ new position), nowhere else
    moved = doc.find("btn").box.union(btn.box).expand(3)
    assert rep.residual_regions, "expected residual regions for a 6px shift"
    assert all(moved.contains(r) for r in rep.residual_regions)
    assert any(r.intersect(btn.box).area > 0 for r in rep.residual_regions)
    # the moved node has the highest per-node error
    assert rep.worst_nodes(1)[0][0] == "btn"
    # phase correlation on a crop around the button recovers the shift
    window = doc.find("btn").box.expand(20)
    dx, dy, resp = alignment_offset(crop(target, window), crop(rendered, window))
    assert abs(dx - 6) <= 1 and abs(dy) <= 1
    assert resp > 0.3


def test_recolored_node_has_highest_error(base):
    doc, target = base
    d3 = copy.deepcopy(doc)
    d3.find("btn").fills = [Fill.solid("#d93025")]
    rep = evaluate(d3, target, render_doc(d3), use_ocr=False)
    worst = rep.worst_nodes(2)
    assert worst[0][0] == "btn"
    assert worst[0][1] > 20  # blue vs red is a large ΔE
    assert rep.per_node["txt"] == 0.0 and rep.per_node["circ"] == 0.0


def test_per_node_excludes_children(base):
    doc, target = base
    d4 = copy.deepcopy(doc)
    d4.find("cardtxt").text = "Different"
    rep = evaluate(d4, target, render_doc(d4), use_ocr=False)
    # the card's own (child-excluded) error stays ~0; the child text carries the error
    assert rep.per_node["cardtxt"] > 1.0
    assert rep.per_node["card"] < 0.5
    assert rep.per_node["card"] < rep.per_node["cardtxt"]


def test_per_node_integral_matches_bruteforce(base):
    doc, target = base
    rng = np.random.default_rng(0)
    diff = rng.random((300, 400), dtype=np.float32) * 10
    errs = per_node_errors(doc.root, diff)
    card = doc.find("card")
    x0, y0, x1, y1 = card.box.as_int()
    mask = np.ones((y1 - y0, x1 - x0), bool)
    cx0, cy0, cx1, cy1 = card.children[0].box.as_int()
    mask[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = False
    expected = float(diff[y0:y1, x0:x1][mask].mean())
    assert errs["card"] == pytest.approx(expected, rel=1e-5)
    assert errs["btn"] == pytest.approx(float(diff[40:88, 40:160].mean()), rel=1e-5)


def test_pixel_metrics_size_mismatch(base):
    doc, target = base
    bigger = np.full((320, 420, 3), 255, np.uint8)
    bigger[:300, :400] = target
    m = pixel_metrics(target, bigger)
    assert m["mean_de"] == 0.0
    rep = evaluate(doc, bigger, target, use_ocr=False)
    assert rep.pixel["size_mismatch"] is True
    assert 0 < rep.pixel["size_term"] < 0.2
    assert rep.total > 0


def test_residual_regions_params():
    diff = np.zeros((100, 100), np.float32)
    diff[10:20, 10:40] = 30.0  # 300 px blob
    diff[60:62, 60:62] = 30.0  # 4 px speck
    regs = residual_regions(diff, thr=6.0, min_area=16, dilate=0)
    assert len(regs) == 1 and regs[0] == Box(10, 10, 30, 10)
    regs = residual_regions(diff, thr=6.0, min_area=1, dilate=0)
    assert len(regs) == 2
    assert residual_regions(np.zeros((10, 10), np.float32)) == []


def test_alignment_flat_crop_is_safe():
    flat = np.full((40, 40, 3), 200, np.uint8)
    assert alignment_offset(flat, flat) == (0.0, 0.0, 0.0)


@pytest.mark.skipif(not os.path.exists(MWC_PNG), reason="reference screenshot missing")
def test_mwc_reference_shift():
    mwc = load_rgb(MWC_PNG)
    shifted = np.roll(mwc, 3, axis=1)
    assert pixel_metrics(mwc, mwc)["mean_de"] == 0.0
    dx, dy, resp = alignment_offset(mwc, shifted)
    assert abs(dx - 3) <= 0.5 and abs(dy) <= 0.5 and resp > 0.5
    d = diff_map(mwc, shifted)
    regs = residual_regions(d)
    assert regs and len(regs) <= P["compare.pixel.residual_max_regions"]
    m = pixel_metrics(mwc, shifted, diff=d)
    assert 0 < m["frac_bad"] < 0.5 and m["ssim"] < 1.0


# --------------------------------------------------------------------------- structural
def test_structural_identity(base):
    doc, _ = base
    m = structural_metrics(copy.deepcopy(doc), doc)
    assert m["n_pred"] == m["n_gt"] == m["n_matched"] == 7
    assert m["node_precision"] == 1.0 and m["node_recall"] == 1.0 and m["mean_iou"] == 1.0
    assert m["color_de"] == 0.0 and m["text_cer"] == 0.0 and m["text_recall"] == 1.0
    assert m["radius_mae"] == 0.0 and m["type_acc"] == 1.0
    assert m["component_acc"] is None and m["token_acc"] is None


def test_structural_delete_node_drops_recall(base):
    doc, _ = base
    pred = copy.deepcopy(doc)
    pred.root.children = [c for c in pred.root.children if c.id != "circ"]
    m = structural_metrics(pred, doc)
    assert m["node_recall"] == pytest.approx(6 / 7)
    assert m["node_precision"] == 1.0
    assert all(gi != "circ" for _, gi, _ in m["matches"])


def test_structural_text_change_cer(base):
    doc, _ = base
    pred = copy.deepcopy(doc)
    pred.find("txt").text = "Hallo wrld"
    m = structural_metrics(pred, doc)
    assert 0 < m["text_cer"] < 0.5
    pred.find("txt").text = "completely different"
    m = structural_metrics(pred, doc)
    assert m["text_recall"] == pytest.approx(0.5)


def test_structural_perturbed_geometry_and_color(base):
    doc, _ = base
    pred = copy.deepcopy(doc)
    pred.find("btn").box = Box(44, 42, 120, 48)  # small shift keeps the match
    pred.find("btn").radius = (12, 12, 12, 12)
    pred.find("circ").fills = [Fill.solid("#ff0000")]
    m = structural_metrics(pred, doc)
    assert m["n_matched"] == 7 and m["mean_iou"] < 1.0
    assert m["radius_mae"] == pytest.approx(4 / 5)  # 5 non-text matched pairs
    assert m["color_de"] > 0
    pred.find("btn").box = Box(200, 100, 120, 48)  # far away: unmatched
    m = structural_metrics(pred, doc)
    assert m["n_matched"] == 6


def test_structural_component_and_tokens(base):
    doc, _ = base
    gt = copy.deepcopy(doc)
    gt.find("btn").component = ComponentRef(key="k", name="Button", variant={"style": "filled"})
    gt.find("btn").tokens = {"fill": "md.sys.color.primary", "radius": "md.sys.shape.corner.full"}
    pred = copy.deepcopy(gt)
    assert structural_metrics(pred, gt)["component_acc"] == 1.0
    assert structural_metrics(pred, gt)["token_acc"] == 1.0
    pred.find("btn").component.variant = {"style": "outlined"}
    pred.find("btn").tokens = {"fill": "md.sys.color.primary"}
    m = structural_metrics(pred, gt)
    assert m["component_acc"] == 0.0 and m["token_acc"] == 0.5
    pred.find("btn").component = None
    assert structural_metrics(pred, gt)["component_acc"] == 0.0


def test_match_nodes_type_gating():
    gt = [Node(id="g", type="text", box=Box(0, 0, 50, 20), text="x")]
    pred_rect = [Node(id="p", type="rect", box=Box(0, 0, 50, 20))]
    pred_text = [Node(id="p", type="text", box=Box(0, 0, 50, 20), text="x")]
    assert match_nodes(pred_rect, gt) == []
    assert match_nodes(pred_text, gt) == [("p", "g", 1.0)]
    # containers are interchangeable; icon <-> vector
    assert match_nodes([Node(id="a", type="frame", box=Box(0, 0, 10, 10))],
                       [Node(id="b", type="instance", box=Box(0, 0, 10, 10))])[0][2] == 1.0
    assert match_nodes([Node(id="a", type="vector", box=Box(0, 0, 10, 10))],
                       [Node(id="b", type="icon", box=Box(0, 0, 10, 10))])
    # hungarian prefers the better-overlapping assignment
    gt2 = [Node(id="g1", type="rect", box=Box(0, 0, 20, 20)), Node(id="g2", type="rect", box=Box(30, 0, 20, 20))]
    pr2 = [Node(id="p2", type="rect", box=Box(31, 0, 20, 20)), Node(id="p1", type="rect", box=Box(1, 0, 20, 20))]
    assert sorted((p, g) for p, g, _ in match_nodes(pr2, gt2)) == [("p1", "g1"), ("p2", "g2")]


def test_levenshtein_and_cer():
    assert levenshtein("", "") == 0 and levenshtein("abc", "") == 3
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("flaw", "lawn") == 2
    assert text_cer("Hello", "Hello") == 0.0
    assert text_cer("Hello  world", "Hello world") == 0.0  # whitespace normalised
    assert text_cer("", "abcd") == 1.0 and text_cer("x", "") == 1.0 and text_cer("", "") == 0.0


def test_text_term_from_lines(base):
    doc, _ = base
    assert text_term_from_lines([], doc) is None
    assert text_term_from_lines(["Hello world", "Card title"], doc) == 0.0
    assert text_term_from_lines(["zzzz"], doc) == 1.0
    assert text_term_from_lines(["zzzz"], Document.blank(10, 10)) == 1.0


# --------------------------------------------------------------------------- layout consistency
def _layout_doc() -> Document:
    doc = Document.blank(200, 100)
    row = Node(id="row", type="frame", box=Box(20, 20, 150, 56), fills=[Fill.solid("#eeeeee")],
               layout=Layout(mode="row", gap=10, padding=(8, 8, 8, 8)),
               children=[Node(id=f"c{i}", type="rect", box=Box(28 + 50 * i, 28, 40, 40), fills=[Fill.solid("#1a73e8")])
                         for i in range(3)])
    doc.root.children.append(row)
    return doc


def test_layout_consistency():
    assert layout_consistency(Document.blank(50, 50)) is None
    doc = _layout_doc()
    good = layout_consistency(doc)
    assert good is not None and good > 0.99
    doc.find("row").layout.gap = 30
    bad = layout_consistency(doc)
    assert bad < good - 0.1


# --------------------------------------------------------------------------- report / viz / params
def test_report_to_dict_json_and_diff_image(base):
    doc, target = base
    d2 = copy.deepcopy(doc)
    d2.find("btn").box = d2.find("btn").box.translate(6, 0)
    rendered = render_doc(d2)
    rep = evaluate(d2, target, rendered, gt=doc, use_ocr=False)
    assert isinstance(rep, LossReport)
    d = rep.to_dict()
    json.dumps(d)  # must be JSON-safe
    assert "diff_map" not in d and d["structure"]["n_matched"] == 7
    assert "diff_map" in rep.to_dict(include_diff=True)
    path = save_diff_image(target, rendered, rep.diff_map, os.path.join(OUT_DIR, "diff.png"), rep.residual_regions)
    img = load_rgb(path)
    assert img.shape[1] == 3 * 400 and img.shape[0] >= 300


def test_params_registered():
    for key in ("compare.pixel.bad_de", "compare.pixel.residual_min_area", "compare.struct.iou_thr",
                "compare.loss.w_pixel", "compare.loss.w_bad", "compare.loss.w_text", "compare.loss.de_norm",
                "compare.visualize.heat_max_de", "compare.struct.line_band_overlap"):
        assert key in P.all() and key in P.ranges() and key in P.docs()


# --------------------------------------------------------------------------- metrics-gate: fair matching
SYNTH_GT = os.path.join(ROOT, "fixtures", "corpus", "synth", "synth_1_000.gt.json")


def test_map_document_of_gt_keeps_full_node_recall():
    """Mapping must never *lower* structural recall: a node recognised as a component (e.g. a divider
    -> Divider instance) is still matched by its geometry (meta.orig_type)."""
    from dt.mapping import map_document, material3
    gt = Document.load(SYNTH_GT)
    stripped = Document.from_dict(gt.to_dict())
    for n in stripped.walk():  # what the bench does with stages=('map',)
        n.component, n.tokens = None, {}
    mapped = map_document(stripped, material3())
    assert any(n.type == "instance" and n.meta.get("orig_type") == "line" for n in mapped.walk())
    for pred in (mapped, map_document(Document.from_dict(gt.to_dict()), material3())):
        m = structural_metrics(pred, gt)
        assert m["node_recall"] == 1.0 and m["node_precision"] == 1.0
        assert m["type_acc"] == 1.0  # instance-of-line vs line is the same geometry


def test_instance_matching_uses_geometry():
    from dt.compare import geom_type
    line_gt = [Node(id="g", type="line", box=Box(0, 10, 100, 1))]
    inst_line = Node(id="p", type="instance", box=Box(0, 10, 100, 1), meta={"orig_type": "line"})
    assert geom_type(inst_line) == "line"
    assert match_nodes([inst_line], line_gt) == [("p", "g", 1.0)]
    # an instance of unknown geometry may match any shape (line, icon, ...), never text
    bare = Node(id="p", type="instance", box=Box(0, 10, 100, 1))
    assert match_nodes([bare], line_gt) == [("p", "g", 1.0)]
    assert match_nodes([Node(id="p", type="instance", box=Box(0, 0, 24, 24))],
                       [Node(id="g", type="icon", box=Box(0, 0, 24, 24))])
    assert match_nodes([bare], [Node(id="g", type="text", box=Box(0, 10, 100, 1), text="x")]) == []
    # but its recorded geometry is respected: an instance of a frame is not an icon
    inst_frame = Node(id="p", type="instance", box=Box(0, 0, 24, 24), meta={"orig_type": "frame"})
    assert match_nodes([inst_frame], [Node(id="g", type="icon", box=Box(0, 0, 24, 24))]) == []


def _two_line_gt() -> Document:
    doc = Document.blank(360, 120)
    style = TextStyle(size=16, color=Color(29, 27, 32))
    doc.root.children += [
        Node(id="l1", type="text", box=Box(20, 20, 200, 24), text="Travel plan", text_style=style),
        Node(id="l2", type="text", box=Box(20, 44, 260, 24), text="Modified 9:30 AM", text_style=style),
    ]
    return doc


def test_text_recall_is_line_fair_for_paragraphs():
    gt = _two_line_gt()
    rgb = render_doc(gt)
    # the renderer really puts ink in both line bands (the gt geometry is real)
    bg = rgb[5, 5].astype(int)
    ink = np.abs(rgb.astype(int) - bg).sum(axis=2) > 60
    assert ink[20:44, 20:220].any() and ink[44:68, 20:280].any()
    para = Node(id="p", type="text", box=Box(20, 20, 260, 48), text="Travel plan\nModified 9:30 AM")
    pred = Document.blank(360, 120)
    pred.root.children.append(para)
    m = structural_metrics(pred, gt)
    assert m["text_recall"] == 1.0  # each gt line credited by content + vertical overlap
    # a line only counts in its own band: swapped order -> neither line's band overlaps its gt line
    pred.root.children[0] = Node(id="p", type="text", box=Box(20, 20, 260, 48), text="Modified 9:30 AM\nTravel plan")
    assert structural_metrics(pred, gt)["text_recall"] == 0.0
    # content must match (CER <= compare.struct.text_match_cer)
    pred.root.children[0] = Node(id="p", type="text", box=Box(20, 20, 260, 48), text="Travel plan\nSomething else")
    assert structural_metrics(pred, gt)["text_recall"] == pytest.approx(0.5)


def test_icon_iou_uses_gt_ink_box():
    from dt.compare import pair_iou
    gt = Document.blank(120, 80)
    icon = Node(id="g", type="icon", box=Box(40, 20, 24, 24), icon_name="settings", fills=[Fill.solid("#49454f")])
    gt.root.children.append(icon)
    rgb, blank = render_doc(gt), render_doc(Document.blank(120, 80))
    ys, xs = np.nonzero(np.abs(rgb.astype(int) - blank.astype(int)).sum(axis=2) > 30)
    assert len(xs) > 20, "the icon font must render ink"
    ink = Box(float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))
    pred = Node(id="p", type="vector", box=ink)  # a perceiver that reports the tight ink box
    em_iou = pred.box.iou(icon.box)
    assert em_iou < 1.0
    icon.meta["ink_box"] = ink.to_dict()
    assert pair_iou(pred, icon) == pytest.approx(1.0)  # max(IoU vs em box, IoU vs ink box)
    assert match_nodes([pred], [icon], iou_thr=0.95) == [("p", "g", 1.0)]
    del icon.meta["ink_box"]
    assert match_nodes([pred], [icon], iou_thr=0.95) == []


def test_structure_that_paints_nothing_is_not_credited(base):
    """Metrics-gaming: gt-shaped nodes that cannot be seen (transparent text, zero-opacity or
    hidden subtrees) under a screenshot crop scored perfect structure; they must not count."""
    doc, img = base
    pred = copy.deepcopy(doc)
    for n in pred.walk():
        if n.type == "text":
            n.text_style.color = Color(0, 0, 0, 0.0)  # right text, invisible glyphs
    m = structural_metrics(pred, doc)
    assert m["text_recall"] == 0.0 and m["text_cer"] is None and m["node_recall"] == pytest.approx(5 / 7)
    pred = copy.deepcopy(doc)
    pred.find("card").opacity = 0.0  # card and its title paint nothing
    pred.find("circ").visible = False
    m = structural_metrics(pred, doc)
    assert m["n_pred"] == 4 and m["node_recall"] == pytest.approx(4 / 7)
    hidden = copy.deepcopy(doc)
    hidden.find("card").visible = False  # a hidden parent hides its (visible) children too
    assert structural_metrics(hidden, doc)["n_pred"] == 5
    # translucent-but-visible paint still counts
    pred = copy.deepcopy(doc)
    pred.find("card").opacity = 0.5
    assert structural_metrics(pred, doc)["node_recall"] == 1.0
