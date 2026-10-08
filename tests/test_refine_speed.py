"""Refine speed: tile-batched trials (dt.refine.tiles) and learned move priors (dt.refine.priors).

The tile prediction is checked against the real renderer: for every hypothesis the critic proposes on
the refine test cases, the loss predicted from its tile must equal the loss of a full render of the
edited document (or the tile must flag itself as leaky and go through the full-render path).
"""
from __future__ import annotations

import copy
import json
import sys
import types

import numpy as np
import pytest

from dt.ir import Box, Color, Document, Fill, Node, Shadow
from dt.params import P
from dt.refine import accepted_moves, critique, refine, target_text_lines
from dt.refine import priors as PR
from dt.refine import tiles as T
from dt.refine.critic import Hypothesis
from dt.refine.optimizer import _State
from dt.render.screenshot import render_doc
from dt.selftest import move_stats
from tests import test_refine as TR

EPS = 1e-9  # predicted vs full-render loss (they are bit-identical when the tile is exact)


# --------------------------------------------------------------------------- property: tile == full render
def _cases():
    clean = TR._synth_case() or TR.hand_built_doc()
    card = TR._card(clean)
    texts = TR._texts(clean)
    out = []

    def shifted(d):
        for m in d.find(card.id).walk():
            m.box = m.box.translate(5, 3)
        d.find(card.id).fills[0] = Fill.solid("#d93025")
        d.find(texts[0].id).text_style.size += 2
        v = texts[1]
        p = d.parent_of(v.id)
        p.children = [c for c in p.children if c.id != v.id]

    d = copy.deepcopy(clean)
    shifted(d)
    out.append(("combined", d, render_doc(clean)))
    out.append(("illustration", *TR._illustration_case()))
    out.append(("headline_over_art", *TR._headline_over_art(6)))
    return out


@pytest.mark.parametrize("which", [0, 1, 2])
def test_tile_prediction_equals_full_render_loss(which):
    name, doc, target = _cases()[which]
    lines = target_text_lines(target)
    st = _State(copy.deepcopy(doc), target, lines, None, None)
    hyps = critique(st.doc, target, st.rendered, st.report)
    assert hyps, name
    index = T._index(st.doc)
    tiles, opts = [], []
    W, H = st.doc.width, st.doc.height
    for h in hyps:
        for opt in [h] + list(h.alternatives):
            cand = opt.apply(st.doc)
            ext = T.changed_extent(st.doc, cand, index)
            if ext is None:
                continue
            tr = T.tile_rect(ext, W, H, cand)
            if tr is None:
                continue
            tiles.append(T.Tile(len(tiles), cand, tr[0], tr[1]))
            opts.append(opt)
    assert len(tiles) >= 3, f"{name}: too few local hypotheses to test ({len(tiles)})"
    pages, layers = T.render_tiles(tiles, st.rendered)
    assert pages == 1 and 1 <= layers <= len(tiles)
    checked = 0
    for t, opt in zip(tiles, opts):
        full = render_doc(t.doc)
        true, _ = st.score(t.doc, full)
        x0, y0, x1, y1 = t.rect
        outside = full.copy()
        outside[y0:y1, x0:x1] = st.rendered[y0:y1, x0:x1]
        assert np.array_equal(outside, st.rendered), f"{name}: {opt.describe()} changed pixels outside its tile"
        if t.leaky:
            continue  # flagged: the optimizer scores it with a full render instead
        pred, _ = st.score(t.doc, T.paste(st.rendered, t))
        assert abs(pred - true) <= EPS, f"{name}: {opt.describe()} predicted {pred} vs full render {true}"
        assert np.array_equal(t.pixels, full[y0:y1, x0:x1]), f"{name}: {opt.describe()} tile pixels differ"
        checked += 1
    assert checked >= 0.75 * len(tiles), f"{name}: only {checked}/{len(tiles)} tiles were exact"


def test_unchanged_document_tiles_reproduce_the_full_render():
    """Many overlapping tiles of the same document, several layers, one page: every tile is bit-identical
    (tiles grow to hold any shadow they touch whole: Skia blurs a clipped shadow slightly differently)."""
    doc = TR.hand_built_doc()
    doc.root.children.append(Node(id="sh", type="rect", box=Box(360, 200, 80, 50), fills=[Fill.solid("#ffffff")], radius=10,
                                  effects=[Shadow(Color(0, 0, 0, 0.3), 0, 2, 6, 1)]))
    full = render_doc(doc)
    rng = np.random.default_rng(3)
    tiles = []
    for i in range(24):
        w, h = int(rng.integers(8, 200)), int(rng.integers(8, 160))
        x, y = int(rng.integers(0, doc.width - w)), int(rng.integers(0, doc.height - h))
        rect, core = T.tile_rect(Box(x, y, w, h), doc.width, doc.height, doc)
        tiles.append(T.Tile(i, doc, rect, core))
    assert any(t.rect[2] - t.rect[0] > 200 or t.rect[3] - t.rect[1] > 160 for t in tiles) or \
        not any(T._overlaps(t.rect, (350, 190, 450, 260)) for t in tiles)
    pages, layers = T.render_tiles(tiles, full)
    assert pages == 1 and layers >= 2
    for t in tiles:
        x0, y0, x1, y1 = t.rect
        assert np.array_equal(t.pixels, full[y0:y1, x0:x1]), t.rect
        assert not t.leaky


# --------------------------------------------------------------------------- changed extents
def test_changed_extent_locality():
    doc = TR.hand_built_doc()
    idx = T._index(doc)
    same = copy.deepcopy(doc)
    assert T.changed_extent(doc, same, idx).area == 0
    rec = copy.deepcopy(doc)
    rec.find("btn").fills[0] = Fill.solid("#000000")
    ext = T.changed_extent(doc, rec, idx)
    assert ext is not None and ext.contains(doc.find("btn").box) and ext.area < 4 * doc.find("btn").box.area
    bg = copy.deepcopy(doc)
    bg.root.fills[0] = Fill.solid("#000000")
    assert T.changed_extent(doc, bg, idx) is None  # page background: not local
    ins = copy.deepcopy(doc)
    ins.root.children.append(Node(id="new", type="rect", box=Box(400, 300, 20, 20), fills=[Fill.solid("#ff0000")]))
    ext = T.changed_extent(doc, ins, idx)
    assert ext.contains(Box(400, 300, 20, 20)) and ext.area < 40 * 40
    reorder = copy.deepcopy(doc)
    reorder.root.children.reverse()
    assert T.changed_extent(doc, reorder, idx) is None  # z-order of kept top-level nodes
    grown = copy.deepcopy(doc)
    grown.find("caption").text = grown.find("caption").text * 4  # overflows its box to the right
    ext = T.changed_extent(doc, grown, idx)
    assert ext.x2 > doc.find("caption").box.x2 + 100


# --------------------------------------------------------------------------- the optimizer
def _perturbed():
    clean = TR._synth_case() or TR.hand_built_doc()
    target = render_doc(clean)
    doc = copy.deepcopy(clean)
    card = TR._card(clean)
    texts = [t for t in TR._texts(clean) if card.find(t.id) is None]
    for m in doc.find(card.id).walk():
        m.box = m.box.translate(0, 4)
    doc.find(texts[0].id).text_style.size += 2
    others = [n for n in doc.walk() if n.type in ("frame", "rect", "line") and n.fill_color and n.id not in (card.id, doc.root.id)]
    if others:
        others[0].fills[0] = Fill.solid("#d93025")
    return doc, target


def _with(params: dict, fn):
    old = {k: P[k] for k in params}
    try:
        for k, v in params.items():
            P.set(k, v)
        return fn()
    finally:
        for k, v in old.items():
            P.set(k, v)


def test_tiles_cut_renders_without_losing_quality():
    doc, target = _perturbed()
    out0, h0 = _with({"refine.tile.enabled": False, "refine.prior.enabled": False}, lambda: refine(doc, target, max_iters=4))
    out1, h1 = _with({"refine.tile.enabled": True}, lambda: refine(doc, target, max_iters=4))
    s0, s1 = h0[-1], h1[-1]
    assert s1["params"]["tile_pages"] >= 1 and s1["params"]["tile_evals"] >= 5
    assert s1["after"] <= s0["after"] * 1.01 + 1e-9, (s0["after"], s1["after"])
    assert s1["params"]["full_renders"] * 2 <= s0["params"]["renders"], (s0["params"], s1["params"])
    assert s1["params"]["renders"] < s0["params"]["renders"]
    # loss never increases and every accepted record is a real drop
    seq = [h["after"] for h in accepted_moves(h1)]
    assert all(b <= a + 1e-12 for a, b in zip(seq, seq[1:])) and all(h["after"] < h["before"] for h in accepted_moves(h1))
    assert any(h.get("eval") == "tile" for h in accepted_moves(h1))
    # the final document's real loss is what the history says
    from dt.refine import loss_of
    real, _ = loss_of(out1, target, None, target_text_lines(target))
    assert real == pytest.approx(s1["after"], abs=1e-12)


def test_confirm_bisects_when_the_prediction_is_wrong(monkeypatch):
    """Sabotage the tile pixels of every hypothesis (predict a perfect match): the confirming full render
    misses the prediction, the set is bisected, and the loss still never increases."""
    doc, target = _perturbed()
    real_render = T.render_tiles

    def lying(tiles, current):
        out = real_render(tiles, current)
        for t in tiles:
            x0, y0, x1, y1 = t.rect
            t.pixels = target[y0:y1, x0:x1].copy()
            t.leaky = False
        return out

    monkeypatch.setattr(T, "render_tiles", lying)
    msgs: list[str] = []
    out, hist = refine(doc, target, max_iters=2, log=msgs.append)
    assert any("bisecting" in m for m in msgs)
    assert hist[-1]["after"] <= hist[-1]["before"]
    acc = accepted_moves(hist)
    assert all(h["after"] < h["before"] for h in acc)


def test_refine_with_tiles_is_deterministic():
    doc, target = _perturbed()
    a = refine(doc, target, max_iters=2)
    b = refine(doc, target, max_iters=2)
    assert json.dumps([a[1], a[0].to_dict()], sort_keys=True, default=str) == json.dumps([b[1], b[0].to_dict()], sort_keys=True, default=str)


# --------------------------------------------------------------------------- priors
def _hyp(kind, gain, node_id=None, region=Box(0, 0, 10, 10)):
    return Hypothesis(kind, node_id, {}, gain, region, lambda d: None)


def test_priors_order_and_deprioritise_never_remove(tmp_path, monkeypatch):
    table = {"kinds": {"text": {"prior": 1.0, "deprioritised": False}, "shift": {"prior": 1.0, "deprioritised": False}},
             "buckets": {"text|-|xs": {"prior": 0.1, "deprioritised": True}, "shift|-|xs": {"prior": 3.0, "deprioritised": False},
                         "recolor|-|xs": {"prior": 1.0, "deprioritised": False}}}
    p = tmp_path / "priors.json"
    p.write_text(json.dumps(table))
    monkeypatch.setattr(PR, "PRIORS_PATH", str(p))
    doc = Document.blank(100, 100)
    hyps = [_hyp("text", 0.5), _hyp("recolor", 0.2), _hyp("shift", 0.1), _hyp("opacity", 0.05)]
    out = PR.order(hyps, doc)
    assert [h.kind for h in out] == ["shift", "recolor", "opacity", "text"]  # 0.3, 0.2, 0.05 (no prior), then deprioritised
    assert out[-1].deprioritised and len(out) == len(hyps)
    monkeypatch.setattr(PR, "PRIORS_PATH", str(tmp_path / "missing.json"))
    assert [h.kind for h in PR.order(hyps, doc)] == ["text", "recolor", "shift", "opacity"]


def test_adversary_policy_hook_with_fallback(monkeypatch):
    doc = Document.blank(100, 100)
    hyps = [_hyp("text", 0.5), _hyp("recolor", 0.2), _hyp("shift", 0.1)]
    old = P["refine.policy"]
    P.set("refine.policy", "adversary")
    try:
        monkeypatch.setitem(sys.modules, "dt.adversary.policy", None)  # module missing -> priors order
        assert len(PR.order(hyps, doc)) == 3
        fake = types.ModuleType("dt.adversary.policy")
        fake.suggest = lambda findings, d: list(reversed(findings))[:2]  # drops one: it must be kept
        monkeypatch.setitem(sys.modules, "dt.adversary.policy", fake)
        assert [h.kind for h in PR.order(hyps, doc)] == ["shift", "recolor", "text"]
        bad = types.ModuleType("dt.adversary.policy")
        bad.suggest = lambda findings, d: 1 / 0
        monkeypatch.setitem(sys.modules, "dt.adversary.policy", bad)
        assert sorted(h.kind for h in PR.order(hyps, doc)) == ["recolor", "shift", "text"]
    finally:
        P.set("refine.policy", old)


def test_move_stats_exports_priors(tmp_path):
    run = tmp_path / "run1"
    run.mkdir()
    hist = []
    for i in range(20):  # text on tiny nodes: tried a lot, never pays
        hist.append({"iter": 1, "kind": "text", "node_id": f"t{i}", "accepted": False, "before": 1.0, "after": 1.01, "batch": 1,
                     "expected_gain": 0.01, "node_type": "text", "node_area": 100.0})
    for i in range(10):  # shifts deliver more than the critic promised
        hist.append({"iter": 1, "kind": "shift", "node_id": f"s{i}", "accepted": True, "before": 1.0, "after": 0.98, "batch": 1,
                     "expected_gain": 0.01, "node_type": "frame", "node_area": 10000.0, "gain_est": 0.02})
    hist.append({"iter": 1, "kind": "stop", "accepted": True, "before": 1.0, "after": 0.8})
    (run / "refine_history.json").write_text(json.dumps(hist))
    rows = move_stats.collect([str(tmp_path)])
    pri = move_stats.priors(rows, smooth=2.0, min_tries=10, dead_gpt=0.02)
    assert pri["buckets"]["text|text|xs"]["deprioritised"] is True
    assert pri["buckets"]["shift|frame|m"]["deprioritised"] is False
    assert pri["buckets"]["shift|frame|m"]["prior"] > pri["buckets"]["text|text|xs"]["prior"]
    assert move_stats.summarise(rows)["kinds"]["shift"]["gain"] == pytest.approx(0.2)


def test_shipped_priors_load():
    table = PR.load()
    assert table.get("kinds") and table.get("buckets"), "knowledge/move_priors.json missing or empty"
    for v in table["buckets"].values():
        assert v["prior"] >= 0 and isinstance(v["deprioritised"], bool)


@pytest.mark.parametrize("edit", ["shift_child", "recolor_child"])
def test_tile_inside_an_opacity_group_is_exact(edit):
    """A group with opacity < 1 is composited as one layer: a tile that cuts it rasterises its edges +-1 level
    differently (no ring change, so the leak check cannot see it). Tiles must hold such groups whole."""
    doc = Document.blank(500, 300, "#ffffff")
    inner = Node(id="in", type="frame", box=Box(120, 80, 80, 60), fills=[Fill.solid("#3366cc")], radius=6)
    doc.root.children.append(Node(id="grp", type="frame", box=Box(100, 60, 140, 100), fills=[Fill.solid("#eeeeee")], opacity=0.7,
                                  children=[inner]))
    cand = copy.deepcopy(doc)
    if edit == "shift_child":
        cand.find("in").box = cand.find("in").box.translate(50, 30)
    else:
        cand.find("in").fills = [Fill.solid("#cc3333")]
    current = render_doc(doc)
    rect, core = T.tile_rect(T.changed_extent(doc, cand), doc.width, doc.height, cand)
    t = T.Tile(0, cand, rect, core)
    T.render_tiles([t], current)
    full = render_doc(cand)
    x0, y0, x1, y1 = t.rect
    assert t.leaky or np.array_equal(t.pixels, full[y0:y1, x0:x1]), f"tile {t.rect} differs from the full render"
