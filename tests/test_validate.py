"""Validation layer must (a) pass all gates on identical renders, (b) flag geometry shifts, (c) flag colour changes."""
import copy

import numpy as np
import pytest

from dt.ir import Box, Color, Document, Fill, Node, Stroke, TextStyle
from dt.params import P
from dt.render.screenshot import render_doc
from dt.validate import validate


def _doc():
    doc = Document.blank(320, 180, "#fef7ff")
    btn = Node(id="btn", type="frame", box=Box(20, 20, 120, 40), fills=[Fill.solid("#6750a4")], radius=(20,) * 4)
    btn.children = [Node(id="lbl", type="text", text="Create", box=Box(50, 30, 60, 20),
                         text_style=TextStyle(size=14, weight=500, color=Color.from_hex("#ffffff"), line_height=20))]
    card = Node(id="card", type="frame", box=Box(20, 80, 280, 80), fills=[Fill.solid("#ffffff")], radius=(12,) * 4,
                strokes=[Stroke(color=Color.from_hex("#cac4d0"), width=1)])
    card.children = [Node(id="h", type="text", text="Headline", box=Box(36, 92, 200, 24),
                          text_style=TextStyle(size=16, weight=500, color=Color.from_hex("#1d1b20"), line_height=24))]
    doc.root.children = [btn, card]
    return doc


@pytest.fixture(scope="module")
def base():
    doc = _doc()
    return doc, render_doc(doc)


def test_identical_passes_all_gates(base, tmp_path):
    doc, img = base
    rep = validate(img, img.copy(), doc=doc, out_dir=str(tmp_path), with_text=False)
    assert rep.jnd_frac == 0.0 and rep.chamfer == 0.0 and rep.edge_iou == 1.0
    assert all(rep.gates.values()), rep.gate_failures
    assert (tmp_path / "validation.json").exists() and (tmp_path / "blink.html").exists()


def test_shift_detected_by_geometry(base):
    doc, img = base
    d2 = copy.deepcopy(doc)
    d2.root.children[0].box = Box(26, 20, 120, 40)  # shift button 6px right
    d2.root.children[0].children[0].box = Box(56, 30, 60, 20)
    img2 = render_doc(d2)
    rep = validate(img, img2, doc=doc, with_text=False)
    assert rep.chamfer_tile_max > 2.0, rep.chamfer_tile_max
    assert 0 <= rep.chamfer_tile_argmax["x"] <= 160 and rep.chamfer_tile_argmax["y"] < 80  # the button's area
    assert not rep.gates["pixel-exact"]
    assert rep.edge_within1 < 0.99


def test_recolor_detected_by_regions(base):
    doc, img = base
    d2 = copy.deepcopy(doc)
    d2.root.children[1].fills = [Fill.solid("#e8def8")]  # card fill changed
    img2 = render_doc(d2)
    rep = validate(img, img2, doc=doc, with_text=False)
    assert rep.region_de_worst > 5
    worst = rep.worst_regions[0]
    # worst region should be the card area
    assert worst.box["w"] > 200 and worst.box["h"] > 50
    assert not rep.gates["visually-identical"]


def test_text_fidelity_when_ocr_available(base):
    pytest.importorskip("dt.perceive.ocr")
    doc, img = base
    rep = validate(img, img.copy(), doc=doc, with_text=True)
    if rep.text_available:
        assert rep.text_cer == 0.0
        assert rep.text_dpos_max <= 1.0


# --------------------------------------------------------------------------- metrics-gaming (skeptic lens)
def _uri(rgb):
    import base64
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb[..., :3])).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_render_of_another_size_fails_every_gate(base):
    """A 48x48 render (identical to the target's top-left corner) used to be cropped into
    agreement with the 320x180 target and passed pixel-exact."""
    doc, img = base
    for render in (img[:48, :48].copy(), np.pad(img, ((0, 40), (0, 40), (0, 0)), constant_values=255)):
        rep = validate(img, render, doc=doc, with_text=False)
        assert rep.size_match is False and (rep.render_width, rep.render_height) == render.shape[1::-1]
        assert (rep.width, rep.height) == (320, 180)
        assert not any(rep.gates.values()), rep.gates
        assert any("render size" in f for f in rep.gate_failures["structurally-faithful"])
    small = validate(img, img[:48, :48].copy(), doc=doc, with_text=False)
    assert small.jnd_frac > 0.9 and small.edge_within1 < 0.5  # the uncovered frame counts as wrong


def test_full_page_raster_fails_every_gate(base):
    """A doc that is one screenshot of the target is pixel-perfect but not a design; flagged by
    the critic or not, it must fail every tier."""
    doc, img = base
    for meta in ({"rasterised": True}, {}):
        cheat = Document.blank(320, 180, "#ffffff")
        cheat.root.children = [Node(id="shot", type="image", box=Box(0, 0, 320, 180), image_ref=_uri(img), meta=meta)]
        rend = render_doc(cheat)
        rep = validate(img, rend, doc=cheat, with_text=False)
        assert rep.jnd_frac == 0.0 and rep.raster_frac == pytest.approx(1.0)
        assert not any(rep.gates.values()), (meta, rep.gates)
    # a small crop (an illustration) stays allowed
    ok = copy.deepcopy(doc)
    ok.root.children.append(Node(id="ill", type="image", box=Box(200, 20, 60, 40), image_ref=_uri(img[20:60, 200:260]),
                                 meta={"rasterised": True}))
    rep = validate(img, render_doc(ok), doc=ok, with_text=False)
    assert rep.raster_frac == pytest.approx(60 * 40 / (320 * 180)) and rep.gates["structurally-faithful"], rep.gate_failures


def test_text_replaced_by_a_crop_fails_gates():
    """Deleting a text node and painting its pixels with a crop left no text node 'under a
    raster', so the old check passed it; the target OCR line under the crop must be counted."""
    pytest.importorskip("dt.perceive.ocr")
    doc = _doc()
    img = render_doc(doc)
    cheat = copy.deepcopy(doc)
    card = cheat.root.children[1]
    h = card.children[0]
    x0, y0, x1, y1 = h.box.as_int()
    card.children = [Node(id="h", type="image", box=h.box, image_ref=_uri(img[y0:y1, x0:x1]), meta={"rasterised": True})]
    rend = render_doc(cheat)
    rep = validate(img, rend, doc=cheat, with_text=True)
    if not rep.text_available:
        pytest.skip("no OCR backend")
    assert rep.text_cer == 0.0 and rep.text_under_raster == 0  # the old measures are fooled
    assert rep.text_rasterised == 1
    assert not rep.gates["structurally-faithful"]
    # a multi-colour wordmark rasterised by the critic (meta.logo) is an image asset: allowed
    card.children[0].meta["logo"] = True
    rep2 = validate(img, rend, doc=cheat, with_text=True)
    assert rep2.text_rasterised == 0


def test_structural_gate_ranks_a_1px_shift_above_a_mosaic_and_admits_the_gt_ir():
    """Exact-pixel edge IoU (the old structurally-faithful measure) gave the gt IR of a real
    Material Web page 0.56 (gate unreachable) and ranked a perfect doc shifted by 1 px (0.08)
    below a 6 px colour mosaic of the target (0.11). The tolerant edge F1 must order them."""
    import os

    from dt.common.image import load_rgb
    from dt.validate.fidelity import edge_f1
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gt = Document.load(os.path.join(root, "fixtures", "corpus", "mwc", "mwc_1_000.gt.json"))
    target = load_rgb(os.path.join(root, "fixtures", "corpus", "mwc", "mwc_1_000.png"))
    for n in gt.walk():
        n.component, n.tokens = None, {}
    rep = validate(target, render_doc(gt), doc=gt, with_text=False)
    # after the gt text-box fix (line boxes) exact edge IoU on this page is ~0.94; the point stands for
    # a 1px shift below, where exact IoU collapses while edge F1 @1px does not
    assert rep.edge_f1 > 0.85 and rep.edge_f1 >= rep.edge_iou
    assert rep.gates["structurally-faithful"], rep.gate_failures
    shifted = copy.deepcopy(gt)
    for n in shifted.walk():
        if n is not shifted.root:
            n.box = Box(n.box.x + 1, n.box.y + 1, n.box.w, n.box.h)
    H, W = target.shape[:2]
    mosaic = target.copy()
    for y in range(0, H, 6):
        for x in range(0, W, 6):
            mosaic[y:y + 6, x:x + 6] = target[y:y + 6, x:x + 6].reshape(-1, 3).mean(0)
    f_shift, f_mosaic = edge_f1(target, render_doc(shifted)), edge_f1(target, mosaic)
    assert f_shift > f_mosaic + 0.2 and f_mosaic < P["validate.gate.struct.edge_f1"]
    assert not validate(target, mosaic, with_text=False).gates["structurally-faithful"]
