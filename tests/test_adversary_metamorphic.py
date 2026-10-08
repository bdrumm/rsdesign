"""Metamorphic testing of the translator: typed IR differ, relations, differential adversary, fuzzing."""
from __future__ import annotations

import copy
import os
import random

import numpy as np
import pytest

from dt.adversary import fuzz as F
from dt.adversary import metamorphic as M
from dt.adversary import perturb as PT
from dt.adversary.benchmark import localises
from dt.adversary.taxonomy import ERROR_TYPES, Finding
from dt.ir import Box, Color, Document, Fill, Node, TextStyle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SYNTH = os.path.join(ROOT, "fixtures", "corpus", "synth", "synth_1_002.gt.json")


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "CACHE_DIR", str(tmp_path / "cache"))


def _gt() -> Document:
    d = Document.load(SYNTH)
    d.source_image = None
    return d


# --------------------------------------------------------------------------- the typed differ (no perception)
def test_ir_diff_identity_is_silent():
    d = _gt()
    assert M.ir_diff(d, copy.deepcopy(d)) == []
    assert M.ir_diff(d, copy.deepcopy(d), painted_only=True) == []


@pytest.mark.parametrize("t", ERROR_TYPES)
def test_ir_diff_types_every_taxonomy_perturbation(t):
    """Every typed perturbation of a ground-truth IR is reported with its own type at its own place
    (at least 3 of 4 seeds: a fill change towards the backdrop is legitimately read as a fade)."""
    doc = _gt()
    clean, _ = PT.sanitize(copy.deepcopy(doc))
    page = Box(0, 0, doc.width, doc.height)
    ok = tried = 0
    for seed in range(4):
        cand, gt = PT.perturb(doc, [(t, 0.6)], random.Random(seed))
        if not gt:
            continue
        tried += 1
        fs = M.ir_diff(clean, cand, painted_only=True, aspects=("geometry", "text", "color", "radius", "effect", "icon"))
        ok += any(f.type == t and localises(f.box, gt[0], page) for f in fs)
    assert tried and ok >= min(3, tried), (t, ok, tried)


def test_ir_diff_collapses_moved_subtree_and_missing_subtree():
    d = _gt()
    card = max((n for n in d.root.children if len(list(n.walk())) >= 3), key=lambda n: n.count())
    moved = copy.deepcopy(d)
    for n in moved.walk():
        if n.id == card.id:
            for m in n.walk():
                m.box = m.box.translate(9, 0)
    fs = M.ir_diff(d, moved)
    shifts = [f for f in fs if f.type in ("geometry.shift", "text.position")]
    assert len(shifts) == 1 and shifts[0].evidence["e_id"] == card.id and abs(shifts[0].magnitude - 9) < 1e-6
    gone = copy.deepcopy(d)
    gone.root.children = [c for c in gone.root.children if c.id != card.id]
    fs = M.ir_diff(d, gone)
    assert [f.type for f in fs] == ["structure.missing"] and fs[0].evidence["e_id"] == card.id


def test_ir_diff_reports_out_of_taxonomy_component_identity():
    d = _gt()
    inst = next(n for n in d.walk() if n.component is not None)
    other = copy.deepcopy(d)
    for n in other.walk():
        if n.id == inst.id:
            n.component.name = "Definitely" + n.component.name
    fs = M.ir_diff(d, other)
    assert [f.type for f in fs] == ["component.identity"] and "component.identity" in M.OUTSIDE_TAXONOMY


# --------------------------------------------------------------------------- relation transforms (exact, no perception)
def test_translate_relation_is_exact_on_pixels_and_ir():
    from dt.render.screenshot import render_doc
    d = _gt()
    rgb = np.full((60, 80, 3), 250, np.uint8)
    rgb[10:20, 10:30] = (20, 30, 40)
    base = Document(80, 60, Node(id="root", type="frame", box=Box(0, 0, 80, 60), fills=[Fill.solid(Color(250, 250, 250))],
                                 children=[Node(id="n1", type="rect", box=Box(10, 10, 20, 10), fills=[Fill.solid(Color(20, 30, 40))])]))
    inst = M.rel_translate(rgb, base)[0]
    dx, dy = inst.params["dx"], inst.params["dy"]
    assert inst.image.shape == (60 + dy, 80 + dx, 3)
    assert (inst.image[dy:, dx:] == rgb).all() and (inst.image[:dy] == 250).all()
    n1 = next(n for n in inst.expected.walk() if n.id == "n1")
    assert n1.box == Box(10 + dx, 10 + dy, 20, 10)
    assert inst.to_source(n1.box) == Box(10, 10, 20, 10)
    assert M.check(inst, inst.expected, Box(0, 0, 80, 60)) == []
    del d, render_doc


def test_mirror_doc_is_an_involution():
    d = _gt()
    back = M.mirror_doc(M.mirror_doc(d))
    for a, b in zip(d.walk(), back.walk()):
        assert max(abs(u - v) for u, v in zip((a.box.x, a.box.y, a.box.w, a.box.h), (b.box.x, b.box.y, b.box.w, b.box.h))) < 1e-9
        assert tuple(a.radius) == tuple(b.radius) and len(a.children) == len(b.children)


def test_row_relations_add_and_remove_exactly_one_item():
    rows = [Node(id=f"r{i}", type="frame", box=Box(0, 10 + 40 * i, 200, 32), fills=[Fill.solid(Color(230, 224, 233))],
                 children=[Node(id=f"t{i}", type="text", box=Box(8, 18 + 40 * i, 60, 16), text=f"Row {i}",
                                text_style=TextStyle(size=14, color=Color(20, 20, 20)))]) for i in range(4)]
    base = Document(200, 180, Node(id="root", type="frame", box=Box(0, 0, 200, 180), fills=[Fill.solid(Color(255, 255, 255))], children=rows))
    rgb = np.full((180, 200, 3), 255, np.uint8)
    for r in rows:
        rgb[int(r.box.y):int(r.box.y2), :200] = (230, 224, 233)
    add, rem = M.rel_rows(rgb, base)
    p = add.params["pitch"]
    assert p == 40 and add.image.shape[0] == 180 + p and rem.image.shape[0] == 180 - p
    assert len(add.expected.root.children) == 5 and len(rem.expected.root.children) == 3
    assert M.check(add, add.expected, Box(0, 0, 200, 180)) == [] and M.check(rem, rem.expected, Box(0, 0, 200, 180)) == []


def test_recolor_relation_maps_ir_colours_like_pixels():
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[:2] = (103, 80, 164)
    rgb[2:] = (254, 247, 255)
    base = Document(4, 4, Node(id="root", type="frame", box=Box(0, 0, 4, 4), fills=[Fill.solid(Color(254, 247, 255))],
                               children=[Node(id="a", type="rect", box=Box(0, 0, 4, 2), fills=[Fill.solid(Color(103, 80, 164))])]))
    tint, swap = M.rel_recolor(rgb, base)
    a = next(n for n in tint.expected.walk() if n.id == "a")
    assert tuple(tint.image[0, 0]) == (a.fills[0].color.r, a.fills[0].color.g, a.fills[0].color.b)
    assert tuple(swap.image[0, 0]) == (164, 80, 103)


def test_scenario_spec_maps_instances_to_relations():
    f = Finding("geometry.size", Box(0, 0, 3, 3), 1.0)
    s = M.scenario_spec({"image": "x.png"}, "row_add", {"band": [1, 2]}, [f])
    assert s["family"] == "meta.row_add" and s["relation"] == "rows" and s["violations"] == {"geometry.size": 1}
    e = M.ledger_entry([s], {"violations": 1}, {"bench": True})
    assert e["kind"] == "scenario_baseline" and e["accepted"] is False and e["id"].startswith("scenario_baseline-")


# --------------------------------------------------------------------------- with the real renderer + perception
def test_translation_equivariance_and_differential_on_a_rendered_scene():
    """Real renderer + perception: a simple scene translates exactly, and the differential adversary
    types a moved button as geometry.shift where it moved (and is silent on identical images)."""
    from dt.render.screenshot import render_doc
    doc, parts = F.scene("filled_button")
    rgb = render_doc(doc)
    base = M.translate_ir(rgb)
    assert F.detect(base, parts)[0]
    inst = M.rel_translate(rgb, base)[0]
    assert M.check(inst, M.translate_ir(inst.image), Box(0, 0, doc.width, doc.height)) == []
    assert M.find(rgb, rgb.copy()) == []
    moved = copy.deepcopy(doc)
    for n in moved.root.children[0].walk():
        n.box = n.box.translate(-14, 0)
    fs = M.find(rgb, render_doc(moved))
    assert any(f.type == "geometry.shift" and f.box.contains(doc.root.children[0].box, tol=1) for f in fs), [f.type for f in fs]
    # the cache returns the same translation
    assert M.translate_ir(rgb).to_dict() == base.to_dict()


def test_fuzz_scene_contrast_zero_is_lost_and_nominal_found():
    from dt.render.screenshot import render_doc
    for comp in ("outlined_card", "icon"):
        doc, parts = F.scene(comp)
        assert F.detect(M.translate_ir(render_doc(doc)), parts)[0], comp
        doc0, parts0 = F.scene(comp, contrast=0.0)
        assert not F.detect(M.translate_ir(render_doc(doc0)), parts0)[0], comp
    doc2, parts2 = F.scene("filled_button", spacing=8)
    assert len(parts2) == 2 and all(Box(0, 0, doc2.width, doc2.height).contains(b) for ps in parts2 for _, b, _ in ps)


def test_same_glyph_aliases():
    from dt.perceive.icons import atlas_path
    if not os.path.exists(atlas_path(0)):
        pytest.skip("icon atlas not built")
    assert M.same_glyph("mail", "email")
    assert not M.same_glyph("add", "settings")
