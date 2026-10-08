"""ir-space adversary: the IR differ (typed differences), pixel support, the fixed point, and an end-to-end
run with the real perceiver + renderer on a render-verified perturbation."""
from __future__ import annotations

import copy
import random

import numpy as np
import pytest

from dt.adversary import irspace as IS
from dt.adversary.taxonomy import TYPES, Finding
from dt.ir import Box, Color, Document, Fill, Node, Shadow, Stroke, TextStyle


def _doc() -> Document:
    """A small page: a card with an icon, a title and a body line; a row of three chips; a lone button."""
    W, H = 400, 300
    root = Node(id="root", type="frame", box=Box(0, 0, W, H), fills=[Fill.solid(Color(255, 255, 255))])
    card = Node(id="card", type="rect", box=Box(16, 16, 200, 80), fills=[Fill.solid(Color(230, 224, 240))], radius=12)
    card.children = [
        Node(id="ic", type="icon", box=Box(28, 28, 24, 24), icon_name="mail", fills=[Fill.solid(Color(73, 69, 79))]),
        Node(id="t1", type="text", box=Box(64, 28, 90, 20), text="Inbox", text_style=TextStyle(size=16, color=Color(29, 27, 32))),
        Node(id="t2", type="text", box=Box(64, 56, 120, 16), text="Three new mails", text_style=TextStyle(size=14, color=Color(73, 69, 79))),
    ]
    chips = [Node(id=f"c{k}", type="rect", box=Box(16 + k * 80, 120, 64, 32), radius=8, fills=[Fill.solid(Color(232, 222, 248))],
                  strokes=[Stroke(color=Color(121, 116, 126), width=1)]) for k in range(3)]
    btn = Node(id="btn", type="rect", box=Box(260, 200, 100, 40), radius=20, fills=[Fill.solid(Color(103, 80, 164))])
    root.children = [card] + chips + [btn]
    return Document(width=W, height=H, root=root)


def _types(diffs) -> list[str]:
    return sorted(d.type for d in diffs)


def test_identical_documents_have_no_differences():
    d = _doc()
    assert IS.compare_docs(d, copy.deepcopy(d)) == []


def test_shift_of_a_subtree_is_one_geometry_shift_with_its_offset():
    a = _doc()
    b = copy.deepcopy(a)
    for n in b.find("card").walk():
        n.box = n.box.translate(0, 6)
    diffs = IS.compare_docs(a, b)
    assert _types(diffs) == ["geometry.shift"]
    d = diffs[0]
    assert d.magnitude == pytest.approx(6.0)
    assert d.box.contains(Box(16, 16, 200, 86))


def test_lone_text_move_is_text_position_and_resize_is_geometry_size():
    a = _doc()
    b = copy.deepcopy(a)
    b.find("t2").box = b.find("t2").box.translate(3, 0)
    b.find("btn").box = Box(260, 200, 110, 40)
    assert _types(IS.compare_docs(a, b)) == ["geometry.size", "text.position"]
    size = next(d for d in IS.compare_docs(a, b) if d.type == "geometry.size")
    assert size.magnitude == pytest.approx(10.0)


def test_recolour_stroke_radius_shadow_glyph_and_text_readings():
    a = _doc()
    b = copy.deepcopy(a)
    b.find("btn").fills[0].color = Color(180, 40, 60)
    b.find("c0").strokes[0].color = Color(200, 30, 30)
    b.find("c1").radius = (2.0, 2.0, 2.0, 2.0)
    b.find("c2").effects = [Shadow(Color(0, 0, 0, 0.3), 0, 2, 6, 0)]
    b.find("ic").icon_name = "star"
    b.find("t1").text = "Inbax"
    got = _types(IS.compare_docs(a, b))
    assert got == sorted(["color.fill", "color.stroke", "geometry.radius", "effect.shadow", "icon.glyph", "text.content"])
    tc = next(d for d in IS.compare_docs(a, b) if d.type == "text.content")
    assert tc.magnitude == pytest.approx(1 / 5)


def test_text_style_change_is_text_style():
    a = _doc()
    b = copy.deepcopy(a)
    b.find("t1").text_style.size = 20
    b.find("t1").box = Box(64, 27, 112, 22)
    d = IS.compare_docs(a, b)
    assert _types(d) == ["text.style"] and d[0].magnitude == pytest.approx(4.0)


def test_missing_extra_and_icon_missing():
    a = _doc()
    b = copy.deepcopy(a)
    b.root.children = [c for c in b.root.children if c.id != "btn"]
    b.find("card").children = [c for c in b.find("card").children if c.id != "ic"]
    b.root.children.append(Node(id="new", type="rect", box=Box(40, 220, 60, 30), fills=[Fill.solid(Color(0, 120, 0))]))
    got = _types(IS.compare_docs(a, b))
    assert got == ["icon.missing", "structure.extra", "structure.missing"]


def test_split_and_merge():
    a = _doc()
    b = copy.deepcopy(a)
    btn = b.find("btn")
    left, right = copy.deepcopy(btn), copy.deepcopy(btn)
    right.id = "btn2"
    left.box, right.box = Box(260, 200, 48, 40), Box(312, 200, 48, 40)
    b.root.children = [c for c in b.root.children if c.id != "btn"] + [left, right]
    d = IS.compare_docs(a, b)
    assert _types(d) == ["structure.split"] and d[0].magnitude == pytest.approx(4.0)
    d2 = IS.compare_docs(b, a)  # the reverse direction is a merge
    assert _types(d2) == ["structure.merge"]


def test_spacing_drift_is_one_layout_spacing_finding():
    a = _doc()
    b = copy.deepcopy(a)
    for k in (1, 2):
        b.find(f"c{k}").box = b.find(f"c{k}").box.translate(4 * k, 0)
    d = IS.compare_docs(a, b)
    assert _types(d) == ["layout.spacing"] and d[0].magnitude == pytest.approx(4.0)


def test_global_tint_and_opacity():
    a = _doc()
    b = copy.deepcopy(a)
    from dt.ir import rgb_to_lab
    from dt.adversary.perturb import _lab_to_rgb
    for n in b.walk():  # same Lab offset on every colour
        for f in n.fills:
            L, A, B_ = rgb_to_lab(f.color.r, f.color.g, f.color.b)
            f.color = Color(*_lab_to_rgb(L - 1.0, A + 6.0, B_ - 4.0))
        if n.text_style is not None:
            c = n.text_style.color
            L, A, B_ = rgb_to_lab(c.r, c.g, c.b)
            n.text_style.color = Color(*_lab_to_rgb(L - 1.0, A + 6.0, B_ - 4.0))
    d = IS.compare_docs(a, b)
    assert "color.tint_global" in _types(d)
    assert not [x for x in d if x.type == "color.fill"]  # recolours explained by the tint are absorbed
    # opacity: the button blended 50 % toward the white page
    c = copy.deepcopy(a)
    c.find("btn").fills[0].color = Color(179, 168, 210)
    d = IS.compare_docs(a, c)
    assert _types(d) == ["effect.opacity"] and d[0].magnitude == pytest.approx(0.5, abs=0.05)


def test_findings_need_pixel_support():
    """Bias mitigation: an IR difference over pixels that did not change is never reported."""
    a = _doc()
    b = copy.deepcopy(a)
    b.find("btn").fills[0].color = Color(180, 40, 60)
    diffs = IS.compare_docs(a, b)
    img = np.full((a.height, a.width, 3), 255, np.uint8)
    for k in range(12):  # unchanged page content (the noise floor is estimated from the page's edges)
        img[20 + 14 * k: 28 + 14 * k, 230: 380 - 5 * k] = 40
    ev = IS.pixel_evidence(img, img.copy())
    assert IS._finalise(diffs, ev, [], img, img, None) == []
    img2 = img.copy()
    img2[200:240, 260:360] = (180, 40, 60)
    ev = IS.pixel_evidence(img, img2)
    out = IS._finalise(diffs, ev, [], img, img2, None)
    assert [f.type for f in out] == ["color.fill"] and out[0].evidence["support_px"] > 100
    # an unstable (fixed-point) box lowers the confidence
    out2 = IS._finalise(diffs, ev, [Box(250, 190, 120, 60)], img, img2, None)
    assert out2[0].confidence < out[0].confidence
    for f in out:
        assert f.type in TYPES and Finding.from_dict(f.to_dict()).type == f.type


# --------------------------------------------------------------------------- real perceiver + renderer
@pytest.fixture()
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(IS, "CACHE_DIR", str(tmp_path))
    IS._MEM.clear()
    yield tmp_path
    IS._MEM.clear()


def test_end_to_end_shift_found_and_identical_pair_silent(cache):
    from dt.adversary import perturb as PT
    from dt.adversary.benchmark import localises, load_doc
    from dt.render.screenshot import render_doc
    doc = load_doc("synth:synth_1_002")
    clean = render_doc(doc)
    cand_ir, gt = PT.perturb(doc, [("geometry.shift", 0.9)], random.Random(3))
    assert gt and gt[0].type == "geometry.shift"
    cand = render_doc(cand_ir)
    assert PT.verify(clean, cand, gt)["ok"]
    fs = IS.make_finder("t", fp_enabled=False)(clean, cand, cand_ir)
    page = Box(0, 0, doc.width, doc.height)
    hit = [f for f in fs if localises(f.box, gt[0], page)]
    assert hit, [f.to_dict() for f in fs]
    assert hit[0].type in ("geometry.shift", "layout.spacing", "text.position")
    # same image on both sides: the perceiver agrees with itself, no pixel support -> silent
    assert IS.make_finder("t", fp_enabled=False)(clean, clean.copy(), None) == []
    # the perception cache was used (one reading per distinct image)
    assert len(list(cache.rglob("p_*.json"))) == 2


def test_fixed_point_reports_stability(cache):
    from dt.adversary.benchmark import load_doc
    from dt.render.screenshot import render_doc
    rgb = render_doc(load_doc("synth:synth_1_002"))
    fp = IS.fixed_point(rgb)
    assert fp["elements"] > 5 and 0.0 <= fp["stable_frac"] <= 1.0
    assert fp["stable_frac"] >= 0.5  # our perceiver mostly reproduces its own reading on a clean render
    assert all(f.type in TYPES for f in fp["findings"])


def test_opened_seam_in_a_text_line_is_a_split_not_a_style_change():
    """The perceiver's font fit explains an opened seam with a bolder weight; the ink-gap test retypes it."""
    a = _doc()
    b = copy.deepcopy(a)
    b.find("t2").text_style.weight = 500
    b.find("t2").box = Box(64, 56, 126, 16)
    diffs = IS.compare_docs(a, b)
    assert _types(diffs) == ["text.style"]
    img_t = np.full((a.height, a.width, 3), 255, np.uint8)
    img_c = img_t.copy()
    for x0, x1 in ((64, 100), (104, 184)):  # target: two words, 4 px apart
        img_t[58:70, x0:x1] = 30
    for x0, x1 in ((64, 100), (110, 190)):  # candidate: the second word 6 px further right
        img_c[58:70, x0:x1] = 30
    assert IS._max_gap(img_t, Box(63, 55, 123, 18)) == 4.0
    out = IS._retype_width_changes(diffs, img_t, img_c)
    assert [d.type for d in out] == ["structure.split"] and out[0].magnitude == pytest.approx(6.0)
    # an unchanged gap keeps the style reading
    assert [d.type for d in IS._retype_width_changes(diffs, img_t, img_t)] == ["text.style"]
