"""Tests for dt.refine — ground truth is a real render (dt.render) of a known IR document.

Each case perturbs one node of a clean document, refines the perturbed document against the
clean render and checks that >= 90% of the loss gap is recovered, that the touched boxes come
back within +-1px, and that a case runs in under 60s.
"""
from __future__ import annotations

import copy
import os
import random
import time

import numpy as np
import pytest

from dt.common.image import load_rgb
from dt.compare import match_nodes
from dt.compare.structural import normalize_text
from dt.ir import Box, Color, Document, Fill, Node, Stroke, TextStyle
from dt.params import P
from dt.refine import Hypothesis, accepted_moves, critique, loss_of, refine, target_text_lines
from dt.render.screenshot import render_doc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTH_DIR = os.path.join(ROOT, "fixtures", "corpus", "synth")
OUT_DIR = os.path.join(ROOT, "out", "test_refine")
MWC_PNG = os.path.join(ROOT, "out", "mwc_test.png")
RECOVERY = 0.9
TIME_LIMIT_S = 60.0


# --------------------------------------------------------------------------- documents
def hand_built_doc() -> Document:
    """Fallback screen when the synth corpus is unavailable: card with two texts, button, divider."""
    doc = Document.blank(480, 360, "#fef7ff")
    card = Node(id="card", type="frame", name="Card/filled", box=Box(16, 40, 300, 120), fills=[Fill.solid("#e6e0e9")], radius=12,
                children=[
                    Node(id="title", type="text", box=Box(32, 56, 200, 24), text="Photos from the weekend",
                         text_style=TextStyle(size=16, weight=500, line_height=24, letter_spacing=0.15, color=Color(29, 27, 32))),
                    Node(id="body", type="text", box=Box(32, 84, 260, 20), text="The files are in the shared folder",
                         text_style=TextStyle(size=14, weight=400, line_height=20, letter_spacing=0.25, color=Color(73, 69, 79))),
                ])
    btn = Node(id="btn", type="frame", name="Button/filled", box=Box(16, 200, 120, 40), fills=[Fill.solid("#6750a4")], radius=20,
               children=[Node(id="label", type="text", box=Box(40, 210, 72, 20), text="Continue",
                              text_style=TextStyle(size=14, weight=500, line_height=20, letter_spacing=0.1, color=Color(255, 255, 255), align="center"))])
    outlined = Node(id="outlined", type="frame", name="Card/outlined", box=Box(340, 40, 120, 120), strokes=[Stroke(Color(202, 196, 208), 1)], radius=12)
    divider = Node(id="divider", type="line", box=Box(16, 270, 448, 1), fills=[Fill.solid("#cac4d0")])
    caption = Node(id="caption", type="text", box=Box(16, 290, 300, 20), text="Attached is the latest draft",
                   text_style=TextStyle(size=14, weight=400, line_height=20, letter_spacing=0.25, color=Color(29, 27, 32)))
    doc.root.children += [card, btn, outlined, divider, caption]
    return doc


def _synth_case() -> Document | None:
    """A synth corpus document that has a filled card, if the corpus (module E) is available."""
    try:
        from dt.selftest.synth import load_case, make_document
    except Exception:
        return None
    manifest = os.path.join(SYNTH_DIR, "manifest.json")
    if os.path.exists(manifest):
        import json
        ids = json.load(open(manifest)).get("ids", [])
        docs = []
        for cid in ids:
            try:
                docs.append(load_case(SYNTH_DIR, cid)[0])
            except Exception:
                continue
        for want in ("Card/filled", "Card"):  # a flat-filled card first; elevated cards are low-contrast
            for doc in docs:
                if any(n.name.startswith(want) and n.fill_color for n in doc.walk()):
                    return doc
    try:  # corpus present but without a card: build one deterministic document
        os.makedirs(OUT_DIR, exist_ok=True)
        for seed in range(6, 30):
            doc = make_document(random.Random(seed), 600, 900, OUT_DIR)
            if any(n.name.startswith("Card") and n.fill_color for n in doc.walk()):
                return doc
    except Exception:
        return None
    return None


@pytest.fixture(scope="module")
def case():
    """(clean_doc, target_rgb, target_lines, clean_loss). The target is the clean doc's own render."""
    doc = _synth_case() or hand_built_doc()
    target = render_doc(doc)
    lines = target_text_lines(target) if bool(P["refine.opt.use_ocr"]) else None
    clean, _ = loss_of(doc, target, target, lines)
    return doc, target, lines, clean


def _card(doc: Document) -> Node:
    cards = [n for n in doc.walk() if n.name.startswith("Card/filled") and n.fill_color] or [n for n in doc.walk() if n.name.startswith("Card") and n.fill_color]
    if not cards:
        cards = [n for n in doc.walk() if n.type in ("frame", "rect") and n.fill_color and n.id != doc.root.id and n.box.w > 60 and n.box.h > 40]
    return cards[0]


def _texts(doc: Document) -> list[Node]:
    return [n for n in doc.walk() if n.type == "text" and n.text and "\n" not in n.text and len(n.text) >= 5 and n.text_style and n.text_style.size >= 14]


def _run(case, perturb, label: str):
    """Perturb a copy, refine, return (refined_doc, history, recovery, perturbed_doc)."""
    clean_doc, target, lines, clean_loss = case
    doc = copy.deepcopy(clean_doc)
    perturb(doc)
    before, _ = loss_of(doc, target, None, lines)
    assert before > clean_loss + 1e-4, f"{label}: perturbation did not change the render"
    t0 = time.monotonic()
    out, hist = refine(doc, target)
    elapsed = time.monotonic() - t0
    after, _ = loss_of(out, target, None, lines)
    recovery = (before - after) / (before - clean_loss)
    assert elapsed < TIME_LIMIT_S, f"{label}: took {elapsed:.1f}s"
    assert hist and hist[-1]["kind"] == "stop"
    assert after <= before + 1e-9, f"{label}: loss increased"
    return out, hist, recovery, doc


def _assert_box(a: Box, b: Box, tol: float = 1.0) -> None:
    assert abs(a.x - b.x) <= tol and abs(a.y - b.y) <= tol and abs(a.w - b.w) <= tol and abs(a.h - b.h) <= tol, f"{a} vs {b}"


# --------------------------------------------------------------------------- recovery cases
def test_recovers_shift(case):
    clean_doc = case[0]
    nid = _card(clean_doc).id

    def shift(doc):
        for m in doc.find(nid).walk():
            m.box = m.box.translate(5, 0)

    out, hist, rec, _ = _run(case, shift, "shift")
    assert rec >= RECOVERY, rec
    _assert_box(out.find(nid).box, clean_doc.find(nid).box)
    kinds = {h["kind"] for h in accepted_moves(hist)}
    assert "shift" in kinds


def test_recovers_recolor(case):
    clean_doc = case[0]
    nid = _card(clean_doc).id
    out, hist, rec, _ = _run(case, lambda d: d.find(nid).fills.__setitem__(0, Fill.solid("#d93025")), "recolor")
    assert rec >= RECOVERY, rec
    assert out.find(nid).fill_color.delta_e(clean_doc.find(nid).fill_color) < 2.0
    assert any(h["kind"] == "recolor" and h["node_id"] == nid for h in accepted_moves(hist))


def test_recovers_resize(case):
    clean_doc = case[0]
    nid = _card(clean_doc).id

    def resize(doc):
        n = doc.find(nid)
        n.box = Box(n.box.x, n.box.y, n.box.w + 8, n.box.h)

    out, hist, rec, _ = _run(case, resize, "resize")
    assert rec >= RECOVERY, rec
    _assert_box(out.find(nid).box, clean_doc.find(nid).box)
    assert any(h["kind"] == "resize" for h in accepted_moves(hist))


def test_recovers_deleted_text(case):
    clean_doc = case[0]
    victim = _texts(clean_doc)[1]

    def delete(doc):
        p = doc.parent_of(victim.id)
        p.children = [c for c in p.children if c.id != victim.id]

    out, hist, rec, _ = _run(case, delete, "delete")
    assert rec >= RECOVERY, rec
    assert any(h["kind"] == "missing" for h in accepted_moves(hist))
    want = normalize_text(victim.text)
    found = [n for n in out.walk() if n.type == "text" and normalize_text(n.text) == want]
    assert found, [n.text for n in out.texts()]
    got = found[0]
    assert abs(got.box.x - victim.box.x) <= 2 and abs(got.box.y - victim.box.y) <= 2, (got.box, victim.box)
    assert abs(got.text_style.size - victim.text_style.size) <= 1.0
    # the recovered node matches the deleted one structurally
    assert any(g == victim.id for _, g, _ in match_nodes(out, clean_doc))


def test_recovers_text_size(case):
    clean_doc = case[0]
    victim = _texts(clean_doc)[0]

    def grow(doc):
        doc.find(victim.id).text_style.size += 2

    out, hist, rec, _ = _run(case, grow, "size")
    assert rec >= RECOVERY, rec
    n = out.find(victim.id)
    assert abs(n.text_style.size - victim.text_style.size) <= 0.5
    _assert_box(n.box, victim.box)
    assert any(h["kind"] == "text" and h["node_id"] == victim.id for h in accepted_moves(hist))


def test_recovers_combined_perturbation(case):
    """All five perturbations at once on different nodes."""
    clean_doc = case[0]
    card = _card(clean_doc)
    texts = [t for t in _texts(clean_doc) if card.find(t.id) is None]
    assert len(texts) >= 3

    def perturb(doc):
        for m in doc.find(card.id).walk():
            m.box = m.box.translate(0, 4)
        doc.find(texts[0].id).text_style.size += 2
        p = doc.parent_of(texts[1].id)
        p.children = [c for c in p.children if c.id != texts[1].id]
        others = [n for n in doc.walk() if n.type in ("frame", "rect", "line") and n.fill_color and n.id not in (card.id, doc.root.id)]
        if others:
            others[0].fills[0] = Fill.solid("#d93025")

    out, hist, rec, _ = _run(case, perturb, "combined")
    assert rec >= 0.8, rec
    _assert_box(out.find(card.id).box, card.box)
    assert abs(out.find(texts[0].id).text_style.size - texts[0].text_style.size) <= 0.5


# --------------------------------------------------------------------------- critic
def test_critique_shift_hypothesis_is_exact(case):
    clean_doc, target, lines, _ = case
    nid = _card(clean_doc).id
    doc = copy.deepcopy(clean_doc)
    for m in doc.find(nid).walk():
        m.box = m.box.translate(5, 3)
    rendered = render_doc(doc)
    _, rep = loss_of(doc, target, rendered, lines)
    hyps = critique(doc, target, rendered, rep)
    assert hyps and all(isinstance(h, Hypothesis) for h in hyps)
    # ordered by expected_gain x learned prior, deprioritised moves last (dt.refine.priors)
    keys = [(h.deprioritised, -h.expected_gain * h.prior) for h in hyps]
    assert keys == sorted(keys)
    shifts = [h for h in hyps if h.kind == "shift" and h.node_id == nid]
    assert shifts and shifts[0].params["dx"] == -5 and shifts[0].params["dy"] == -3
    # the critic expects more from the container move than from its children's individual moves (the learned
    # prior may still order a child first; tile scoring then measures every one of them exactly)
    first_shift = max((h for h in hyps if h.kind == "shift"), key=lambda h: h.expected_gain)
    assert first_shift.node_id == nid


def test_apply_never_mutates_input(case):
    clean_doc, target, lines, _ = case
    nid = _card(clean_doc).id
    doc = copy.deepcopy(clean_doc)
    doc.find(nid).fills[0] = Fill.solid("#d93025")
    rendered = render_doc(doc)
    _, rep = loss_of(doc, target, rendered, lines)
    snapshot = doc.to_dict()
    for h in critique(doc, target, rendered, rep):
        new = h.apply(doc)
        assert new is not doc
        assert doc.to_dict() == snapshot, h.describe()
    out, _ = refine(doc, target, max_iters=1)
    assert doc.to_dict() == snapshot and out is not doc


def _flat_spot(doc: Document, w: float, h: float) -> Box:
    """A box over the bare root background that touches no node (scanning a 4px grid)."""
    boxes = [n.box for n in doc.walk() if n.id != doc.root.id]
    for y in range(8, int(doc.height - h - 8), 4):
        for x in range(8, int(doc.width - w - 8), 4):
            b = Box(x, y, w, h)
            if all(b.expand(4).intersect(o).area == 0 for o in boxes):
                return b
    pytest.skip("no flat spot on this screen")


def test_extra_node_is_deleted(case):
    clean_doc = case[0]
    spot = _flat_spot(clean_doc, 60, 40)

    def add_extra(doc):
        doc.root.children.append(Node(id="ghost", type="rect", box=spot, fills=[Fill.solid("#d93025")], radius=8))

    out, hist, rec, _ = _run(case, add_extra, "extra")
    assert rec >= RECOVERY, rec
    assert out.find("ghost") is None
    assert any(h["kind"] == "extra" and h["node_id"] == "ghost" for h in accepted_moves(hist))


def test_radius_recovered(case):
    clean_doc = case[0]
    nid = _card(clean_doc).id
    want = clean_doc.find(nid).uniform_radius()
    if want < 4:
        pytest.skip("card has no radius to recover")
    out, hist, rec, _ = _run(case, lambda d: setattr(d.find(nid), "radius", (0.0, 0.0, 0.0, 0.0)), "radius")
    assert rec >= RECOVERY, rec
    assert abs(out.find(nid).uniform_radius() - want) <= 1.0
    assert any(h["kind"] == "radius" for h in accepted_moves(hist))


# --------------------------------------------------------------------------- optimizer behaviour
def test_history_schema_and_monotonic_loss(case):
    clean_doc = case[0]
    nid = _card(clean_doc).id
    out, hist, rec, _ = _run(case, lambda d: d.find(nid).fills.__setitem__(0, Fill.solid("#d93025")), "history")
    keys = {"iter", "kind", "node_id", "params", "before", "after", "accepted"}
    assert all(keys <= set(h) for h in hist)
    acc = accepted_moves(hist)
    assert acc and all(h["after"] < h["before"] for h in acc)
    stop = hist[-1]
    assert stop["kind"] == "stop" and stop["after"] <= stop["before"] and stop["params"]["renders"] >= 2
    # accepted losses never go up along the run
    seq = [h["after"] for h in acc]
    assert all(b <= a + 1e-9 for a, b in zip(seq, seq[1:]))


def test_empty_and_single_node_docs():
    empty = Document.blank(120, 80, "#ffffff")
    target = render_doc(empty)
    out, hist = refine(empty, target)
    assert out.root.count() == 1 and hist[-1]["kind"] == "stop"
    assert hist[-1]["after"] <= 1e-9
    # root only, wrong background: recolor of the root is the only possible move
    wrong = Document.blank(120, 80, "#202020")
    out, hist = refine(wrong, target)
    assert out.root.fill_color.delta_e(Color(255, 255, 255)) < 2.0
    assert any(h["kind"] == "recolor" for h in accepted_moves(hist))
    # single visible leaf, target has nothing: node must be removed or hidden by the loop, never a crash
    one = Document.blank(120, 80, "#ffffff")
    one.root.children.append(Node(id="x", type="rect", box=Box(20, 20, 40, 30), fills=[Fill.solid("#000000")]))
    out, hist = refine(one, target)
    assert hist[-1]["after"] <= hist[-1]["before"]
    # degenerate target sizes do not raise
    out, hist = refine(empty, np.zeros((1, 1, 3), np.uint8), max_iters=1)
    assert hist[-1]["kind"] == "stop"


def test_time_budget_and_log(case):
    clean_doc, target, lines, _ = case
    doc = copy.deepcopy(clean_doc)
    for m in doc.find(_card(clean_doc).id).walk():
        m.box = m.box.translate(5, 0)
    msgs: list[str] = []
    t0 = time.monotonic()
    out, hist = refine(doc, target, time_budget_s=0.01, log=msgs.append)
    assert time.monotonic() - t0 < 10.0
    assert hist[-1]["params"]["reason"] in ("time_budget", "target_loss")
    assert msgs and msgs[0].startswith("refine: start")


def test_params_registered_with_docs_and_ranges():
    docs, ranges, defaults = P.docs(), P.ranges(), P.defaults()
    keys = [k for k in defaults if k.startswith("refine.")]
    assert len(keys) >= 30
    for k in keys:
        assert k in docs and docs[k], k
        if isinstance(defaults[k], (int, float)) and not isinstance(defaults[k], bool):
            assert k in ranges, k
            lo, hi = ranges[k]
            assert lo <= defaults[k] <= hi, k


# --------------------------------------------------------------------------- real Material page
@pytest.mark.skipif(not os.path.exists(MWC_PNG), reason="out/mwc_test.png missing")
def test_refine_perceived_mwc_page():
    perceive = pytest.importorskip("dt.perceive").perceive
    target = load_rgb(MWC_PNG)
    doc = perceive(target)
    lines = target_text_lines(target)
    before, _ = loss_of(doc, target, None, lines)
    t0 = time.monotonic()
    out, hist = refine(doc, target, max_iters=4, time_budget_s=40)
    assert time.monotonic() - t0 < TIME_LIMIT_S
    after, _ = loss_of(out, target, None, lines)
    assert after <= before + 1e-9
    assert hist[-1]["kind"] == "stop"
    os.makedirs(OUT_DIR, exist_ok=True)
    out.save(os.path.join(OUT_DIR, "mwc_test.refined.json"))
    render_doc(out, os.path.join(OUT_DIR, "mwc_test.refined.png"))


# --------------------------------------------------------------------------- as-is raster fallback
def _illustration_case():
    """A target with a multi-colour 'illustration' (many tiny shapes) next to a paragraph, and a
    perceived doc that explains the illustration badly (two grey blobs) but the text correctly."""
    from dt.ir import Box, Color, Document, Fill, Node, TextStyle
    tgt = Document.blank(360, 220, "#ffffff")
    rng = random.Random(4)
    for i in range(40):
        x, y = 20 + rng.randint(0, 140), 20 + rng.randint(0, 140)
        tgt.root.children.append(Node(type="rect", box=Box(x, y, rng.randint(4, 14), rng.randint(4, 14)),
                                      fills=[Fill.solid(rng.choice(["#ea4335", "#fbbc04", "#34a853", "#4285f4"]))], radius=2))
    para = Node(id="para", type="text", text="Editable headline", box=Box(200, 90, 150, 24),
                text_style=TextStyle(size=16, weight=500, color=Color.from_hex("#1f1f1f"), line_height=24))
    tgt.root.children.append(para)
    target = render_doc(tgt)
    pred = Document.blank(360, 220, "#ffffff")
    pred.root.children += [Node(id="blob1", type="rect", box=Box(20, 20, 60, 60), fills=[Fill.solid("#cccccc")]),
                           Node(id="blob2", type="rect", box=Box(90, 90, 60, 60), fills=[Fill.solid("#bbbbbb")]),
                           copy.deepcopy(para)]
    return pred, target


def test_raster_fallback_replaces_unexplained_art_but_never_text():
    pred, target = _illustration_case()
    out, hist = refine(pred, target, max_iters=4)
    rasters = [n for n in out.walk() if n.type == "image" and n.meta.get("rasterised")]
    assert rasters, [h["kind"] for h in hist]
    para = out.find("para")
    assert para is not None and para.type == "text" and para.text == "Editable headline"
    assert not any(r.box.intersect(para.box).area > 0 for r in rasters)
    area = sum(r.box.area for r in rasters) / (out.width * out.height)
    assert area <= float(P["refine.raster.budget_frac"]) + 1e-9


# --------------------------------------------------------------------------- raster guards (adversarial)
_ART = ["#ea4335", "#fbbc04", "#34a853", "#4285f4"]


def _art(rng, x0, y0, w, h, n):
    return [Node(type="rect", box=Box(x0 + rng.randint(0, w - 14), y0 + rng.randint(0, h - 14), rng.randint(4, 14), rng.randint(4, 14)),
                 fills=[Fill.solid(rng.choice(_ART))], radius=2) for _ in range(n)]


def _rasters(doc):
    return [n for n in doc.walk() if n.type == "image" and n.meta.get("rasterised")]


def test_raster_budget_holds_when_rasters_are_batched():
    """Two disjoint illustrations, each 20% of the screen, budget 35%: each raster passes the budget check
    against the current document, but the optimizer batches independent hypotheses into one render."""
    W, H = 400, 300
    rng = random.Random(1)
    tgt = Document.blank(W, H, "#ffffff")
    tgt.root.children += _art(rng, 10, 10, 160, 150, 90) + _art(rng, 220, 140, 160, 150, 90)
    target = render_doc(tgt)
    pred = Document.blank(W, H, "#ffffff")
    pred.root.children += [Node(id="a", type="rect", box=Box(10, 10, 160, 150), fills=[Fill.solid("#cccccc")]),
                           Node(id="b", type="rect", box=Box(220, 140, 160, 150), fills=[Fill.solid("#bbbbbb")])]
    out, hist = refine(pred, target, max_iters=3)
    assert _rasters(out), "one illustration should still be rasterised"
    area = sum(r.box.area for r in _rasters(out)) / (W * H)
    assert area <= float(P["refine.raster.budget_frac"]) + 1e-9, area
    seq = [h["after"] for h in accepted_moves(hist)]
    assert all(b <= a + 1e-12 for a, b in zip(seq, seq[1:]))


def test_raster_never_painted_inside_a_component():
    """An illustration inside a mapped Card instance: the export drops instance children once the library
    component resolves, so a crop there lowers the render loss without being in the Figma output."""
    from dt.ir import ComponentRef
    W, H = 400, 300
    rng = random.Random(2)
    tgt = Document.blank(W, H, "#ffffff")
    tgt.root.children.append(Node(type="frame", box=Box(20, 20, 260, 220), fills=[Fill.solid("#f3edf7")], radius=12,
                                  children=_art(rng, 40, 40, 120, 120, 60)))
    target = render_doc(tgt)
    pred = Document.blank(W, H, "#ffffff")
    pred.root.children.append(Node(
        id="card", type="instance", name="Card", box=Box(20, 20, 260, 220), fills=[Fill.solid("#f3edf7")], radius=12,
        component=ComponentRef(key="card", name="Card", library="material3", variant={"style": "filled"}),
        children=[Node(id=f"ic{i}", type="icon", box=Box(40 + 30 * (i % 4), 40 + 30 * (i // 4), 20, 20), fills=[Fill.solid("#999999")])
                  for i in range(12)]))
    out, _ = refine(pred, target, max_iters=3)
    card = out.find("card")
    assert card is not None and card.component is not None
    assert not [n for n in card.walk() if n.meta.get("rasterised")]


def _headline_over_art(n_lines=12):
    W, H = 480, 400
    rng = random.Random(3)
    head = Node(id="head", type="text", text="Weekend photos", box=Box(40, 70, 170, 28),
                text_style=TextStyle(size=20, weight=500, color=Color.from_hex("#1f1f1f"), line_height=28))
    lines = [Node(id=f"l{i}", type="text", text=f"Message number {i} from Alex", box=Box(250, 10 + 30 * i, 220, 20),
                  text_style=TextStyle(size=14, weight=400, color=Color.from_hex("#1f1f1f"), line_height=20)) for i in range(n_lines)]
    tgt = Document.blank(W, H, "#ffffff")
    tgt.root.children += _art(rng, 20, 20, 200, 140, 120) + [copy.deepcopy(head)] + copy.deepcopy(lines)
    target = render_doc(tgt)
    pred = Document.blank(W, H, "#ffffff")
    pred.root.children += [Node(id="hero", type="rect", box=Box(20, 20, 200, 140), fills=[Fill.solid("#cccccc")]),
                           copy.deepcopy(head)] + copy.deepcopy(lines)
    return pred, target


def test_dark_text_over_colourful_art_is_not_a_logo_and_critique_is_pure():
    from dt.refine.critic import _Ctx, _glyph_hue_spread
    pred, target = _headline_over_art()
    rendered = render_doc(pred)
    _, rep = loss_of(pred, target, rendered, None)
    before = pred.to_dict()
    hyps = critique(pred, target, rendered, rep)
    assert pred.to_dict() == before, "critique must not mutate its input document"
    assert not any(h.kind == "raster" and h.node_id == "head" for h in hyps)
    assert _glyph_hue_spread(_Ctx(pred, target, rendered, rep), pred.find("head")) < float(P["refine.raster.logo_hue_spread"])


def test_raster_never_bakes_or_removes_editable_text():
    """A hero illustration with a 20px headline drawn over it (a sibling, not a child): the hero crop would
    bake the headline into the image (validator: text_under_raster) and a later region raster deleted it."""
    pred, target = _headline_over_art()
    out, _ = refine(pred, target, max_iters=3)
    head = out.find("head")
    assert head is not None and head.type == "text" and head.text == "Weekend photos"
    assert not [r for r in _rasters(out) if r.box.intersect(head.box).area > 0.3 * head.box.area]


def test_multicolour_wordmark_is_rasterised_with_its_ink():
    """A Google-style wordmark (one colour per letter) on a flat bar, perceived as one 18px-tall text line
    whose OCR box misses the first letter: the logo raster must cover all the ink and keep its text for the loss."""
    W, H = 360, 120
    tgt = Document.blank(W, H, "#ffffff")
    x = 40.0
    cols = ["#4285f4", "#ea4335", "#fbbc04", "#4285f4", "#34a853", "#ea4335"]
    adv = [15.0, 11.5, 11.5, 11.5, 6.0, 11.5]
    for ch, c, a in zip("Google", cols, adv):
        tgt.root.children.append(Node(type="text", text=ch, box=Box(x, 20, a + 4, 24),
                                      text_style=TextStyle(size=20, weight=500, color=Color.from_hex(c), line_height=24)))
        x += a
    for i in range(3):
        tgt.root.children.append(Node(type="text", text=f"Inbox item {i}", box=Box(40, 60 + 18 * i, 200, 16),
                                      text_style=TextStyle(size=12, color=Color.from_hex("#1f1f1f"), line_height=16)))
    target = render_doc(tgt)
    pred = Document.blank(W, H, "#ffffff")
    pred.root.children += [Node(id="logo", type="text", text="Google", box=Box(52, 23, 62, 18),
                                text_style=TextStyle(size=18, weight=500, color=Color.from_hex("#4285f4"), line_height=18))]
    pred.root.children += [copy.deepcopy(n) for n in tgt.root.children[6:]]
    out, hist = refine(pred, target, max_iters=3)
    logo = out.find("logo")
    assert logo is not None and logo.type == "image" and logo.meta.get("rasterised"), [h["kind"] for h in accepted_moves(hist)]
    ink = np.argwhere(np.abs(target.astype(int) - 255).sum(-1)[:50] > 60)
    y0, x0 = ink.min(0)
    y1, x1 = ink.max(0)
    assert logo.box.x <= x0 and logo.box.y <= y0 and logo.box.x2 >= x1 + 1 and logo.box.y2 >= y1 + 1, (logo.box, (x0, y0, x1, y1))
    assert logo.meta.get("alt_text") == "Google"


# --------------------------------------------------------------------------- determinism / incremental scoring
def test_incremental_scoring_is_bit_identical_and_deterministic(case):
    """Trials are scored by re-comparing only the pixels that changed; the run (every tried hypothesis, every
    before/after loss, the final document) must be identical to the full per-trial evaluate, and repeatable."""
    import json
    clean_doc, target, lines, _ = case
    doc = copy.deepcopy(clean_doc)
    card = _card(doc)
    for m in card.walk():
        m.box = m.box.translate(4, -3)
    card.fills[0] = Fill.solid("#d93025")
    pred, art_target = _illustration_case()
    runs = []
    old = P["refine.opt.incremental"]
    try:
        for inc in (True, True, False):
            P.set("refine.opt.incremental", inc)
            o1, h1 = refine(doc, target, max_iters=3)
            o2, h2 = refine(pred, art_target, max_iters=3)
            runs.append(json.dumps([h1, h2, o1.to_dict(), o2.to_dict()], sort_keys=True, default=str))
    finally:
        P.set("refine.opt.incremental", old)
    assert runs[0] == runs[1], "refine is not deterministic"
    assert runs[0] == runs[2], "incremental scoring changed the run"
    final, _ = loss_of(o1, target, None, lines)
    assert final == json.loads(runs[0])[0][-1]["after"]
