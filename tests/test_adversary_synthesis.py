"""Tests for the adversary synthesis: fusion ensemble, operator policy, novelty and the protocol.

Unit tests build synthetic findings / records; the guided-refine and feature tests use the real
renderer (``dt.render``) for ground truth.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pytest

from dt.adversary import ensemble as E
from dt.adversary import novelty as NV
from dt.adversary import policy as PO
from dt.adversary import protocol as PR
from dt.adversary.taxonomy import CRITIC_KINDS, ERROR_TYPES, Finding
from dt.ir import Box, Color, Document, Fill, Node, TextStyle
from dt.params import P, register


# --------------------------------------------------------------------------- fixtures
@dataclass
class FakeCase:
    id: str
    gt: list
    family: str = "synth"
    kind: str = "perturbed"
    noise: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)
    bench_key: str = "fake"
    doc: str = "synth:x"
    W: int = 400
    H: int = 300

    @property
    def page(self) -> Box:
        return Box(0, 0, self.W, self.H)


def F(t, x, y, w=40, h=20, conf=0.9, mag=None, node=None):
    return Finding(t, Box(x, y, w, h), mag, conf, {}, node)


def _synthetic_world(n_cases=40, seed=0):
    """Approach A: reliable for geometry.shift and color.fill; B: reliable for color.fill, mistypes shift
    as text.position, fires spuriously; C: noisy and often wrong."""
    rng = np.random.default_rng(seed)
    cases, preds = [], {"A": {}, "B": {}, "C": {}}
    for i in range(n_cases):
        gts = []
        a, b, c = [], [], []
        for j, t in enumerate(("geometry.shift", "color.fill")):
            x, y = 20 + 150 * j, 30 + 7 * (i % 10)
            gts.append(F(t, x, y, conf=1.0))
            if rng.random() < 0.9:
                a.append(F(t, x + 1, y, conf=0.8 + 0.1 * rng.random()))
            if rng.random() < 0.8:
                b.append(F("text.position" if t == "geometry.shift" else t, x, y + 1, conf=0.7))
            if rng.random() < 0.4:
                c.append(F(rng.choice(["structure.extra", t]), x, y, conf=0.5))
        if rng.random() < 0.5:
            b.append(F("structure.missing", 300, 250, conf=0.6))
        if rng.random() < 0.6:
            c.append(F("color.stroke", 330, 200, conf=0.4))
        cs = FakeCase(f"c{i}", gts)
        cases.append(cs)
        for k, v in (("A", a), ("B", b), ("C", c)):
            preds[k][E.uid(cs)] = v
    return cases, preds


# --------------------------------------------------------------------------- ensemble
def test_cluster_gives_each_approach_one_vote():
    page = Box(0, 0, 400, 300)
    by = {"A": [F("geometry.shift", 10, 10), F("geometry.shift", 12, 11, conf=0.5)],
          "B": [F("text.position", 11, 10)], "C": [F("color.fill", 200, 200)]}
    cls = E.cluster(by, page, ("A", "B", "C"))
    assert len(cls) == 2
    big = max(cls, key=lambda c: len(c.votes))
    assert set(big.votes) == {"A", "B"} and len(big.fragments) == 1


def test_fusion_learns_reliability_and_confusion():
    cases, preds = _synthetic_world()
    model = E.fit(preds, cases, ("A", "B", "C"), emit_min=0.5)
    page = Box(0, 0, 400, 300)
    # B's "text.position" agreeing with A's shift is a shift (B's confusion row is learned)
    out = model.fuse({"A": [F("geometry.shift", 20, 30)], "B": [F("text.position", 20, 31)], "C": []}, page)
    assert [f.type for f in out] == ["geometry.shift"]
    both = out[0].confidence
    # only the spurious-prone approach: not emitted (or much less confident)
    lone = model.fuse({"A": [], "B": [F("structure.missing", 300, 250, conf=0.6)], "C": []}, page)
    assert not lone or lone[0].confidence < both
    # silence of the reliable approach lowers the posterior
    no_a = model.posterior(E.Cluster({"B": F("text.position", 20, 31)}))
    with_a = model.posterior(E.Cluster({"A": F("geometry.shift", 20, 30), "B": F("text.position", 20, 31)}))
    assert 1 - no_a["none"] < 1 - with_a["none"]
    # JSON round trip
    m2 = E.FusionModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert m2.posterior(E.Cluster({"A": F("geometry.shift", 20, 30)})) == pytest.approx(model.posterior(E.Cluster({"A": F("geometry.shift", 20, 30)})))


def test_fused_beats_each_annotator_on_synthetic_world():
    cases, preds = _synthetic_world(60, seed=1)
    train, test = cases[:30], cases[30:]
    model = E.fit(preds, train, ("A", "B", "C"))
    fused = E.fuse_preds(model, preds, test)
    h_f = E.score_preds(fused, test, "fuse")["headline"]
    singles = [E.score_preds(preds[a], test, a)["headline"] for a in ("A", "B", "C")]
    assert h_f >= max(singles) - 1e-9


def test_router_picks_best_approach_per_type():
    cases, preds = _synthetic_world()
    r = E.fit_router(preds, cases, ("A", "B", "C"))
    assert r.route["geometry.shift"] == "A"
    assert r.route["color.fill"] in ("A", "B")
    out = r.apply({"A": [F("geometry.shift", 20, 30)], "B": [F("text.position", 20, 30)], "C": []}, Box(0, 0, 400, 300))
    assert [f.type for f in out] == ["geometry.shift"]


def test_uid_distinguishes_seeds_of_the_same_case_id():
    a, b = FakeCase("cal_x_0", []), FakeCase("cal_x_0", [])
    b.bench_key = "other"
    assert E.uid(a) != E.uid(b)


# --------------------------------------------------------------------------- policy
def _records():
    recs = []
    for i in range(12):
        for k in CRITIC_KINDS:
            fixed = (k == "shift")
            recs.append({"case": f"c{i}", "gt": 0, "type": "geometry.shift", "magnitude": float(i), "bin": "small", "kind": k,
                         "proposed": k in ("shift", "resize", "recolor"), "tries": 1 if k in ("shift", "resize", "recolor") else 0,
                         "fixed": fixed, "fix_frac": 1.0 if fixed else 0.0, "gain": 0.05 if fixed else 0.0, "rank": 0})
            fixed2 = (k == "recolor") and i % 2 == 0
            recs.append({"case": f"c{i}", "gt": 1, "type": "color.fill", "magnitude": 5.0 + i, "bin": "small", "kind": k,
                         "proposed": k in ("recolor", "opacity"), "tries": 1 if k in ("recolor", "opacity") else 0,
                         "fixed": fixed2, "fix_frac": 0.9 if fixed2 else 0.0, "gain": 0.02 if fixed2 else 0.0, "rank": 1})
    return recs


def test_policy_fit_orders_operators_and_backs_off_to_taxonomy_prior():
    pol = PO.Policy.fit(_records())
    assert pol.ranking("geometry.shift")[0][0] == "shift"
    assert pol.ranking("color.fill")[0][0] == "recolor"
    assert pol.p_fix("geometry.shift", "shift") > 0.7 > pol.p_fix("geometry.shift", "resize")
    # an unseen type falls back to the taxonomy's refine_kinds prior
    assert pol.p_fix("effect.shadow", "shadow") == pytest.approx(float(P["adversary.policy.prior_listed"]))
    assert pol.p_fix("effect.shadow", "text") == pytest.approx(float(P["adversary.policy.prior_other"]))
    rt = PO.Policy.from_dict(json.loads(json.dumps(pol.to_dict())))
    assert rt.ranking("color.fill") == pol.ranking("color.fill")


def test_policy_suggest_targets_nodes():
    pol = PO.Policy.fit(_records())
    doc = Document.blank(200, 100, "#ffffff")
    doc.root.children.append(Node(id="b", type="rect", box=Box(10, 10, 40, 20), fills=[Fill.solid("#6750a4")]))
    s = pol.suggest([F("geometry.shift", 10, 10, 40, 20), F("noise.antialias", 0, 0, 200, 100)], doc)
    assert s and s[0]["kind"] == "shift" and s[0]["node_id"] == "b"
    assert all(x["finding_type"] == "geometry.shift" for x in s)


def test_reorder_puts_matched_first_and_prunes_useless_operators():
    from dt.refine.critic import Hypothesis
    pol = PO.Policy.fit(_records())
    doc = Document.blank(200, 100, "#ffffff")
    doc.root.children.append(Node(id="b", type="rect", box=Box(10, 10, 40, 20), fills=[Fill.solid("#6750a4")]))
    noop = lambda d: None  # noqa: E731
    hyps = [Hypothesis("text", "b", {}, 0.9, Box(10, 10, 40, 20), noop),
            Hypothesis("recolor", "x", {}, 0.5, Box(150, 60, 20, 20), noop),
            Hypothesis("shift", "b", {}, 0.1, Box(10, 10, 40, 20), noop)]
    out, st = PO.reorder(hyps, [F("geometry.shift", 10, 10, 40, 20)], pol, doc, prune=True)
    assert out[0].kind == "shift"
    assert "text" not in [h.kind for h in out]          # useless for a shift finding: pruned
    assert any(h.kind == "recolor" for h in out)        # overlaps no finding: kept as background
    assert st["pruned"] >= 1


def _card_doc() -> Document:
    doc = Document.blank(360, 200, "#fef7ff")
    card = Node(id="card", type="frame", box=Box(20, 20, 200, 80), fills=[Fill.solid("#e6e0e9")], radius=12,
                children=[Node(id="t", type="text", box=Box(36, 44, 150, 24), text="Weekend photos",
                               text_style=TextStyle(size=16, weight=500, line_height=24, color=Color(29, 27, 32)))])
    btn = Node(id="btn", type="frame", box=Box(20, 130, 120, 40), fills=[Fill.solid("#6750a4")], radius=20)
    doc.root.children += [card, btn]
    return doc


def test_guided_refine_recovers_with_the_real_renderer():
    from dt.refine.optimizer import refine
    from dt.render.screenshot import render_doc
    clean = _card_doc()
    target = render_doc(clean)
    bad = copy.deepcopy(clean)
    for n in bad.walk():
        if n.id == "btn":
            n.box = Box(n.box.x + 6, n.box.y, n.box.w, n.box.h)
    pol = PO.Policy.fit(_records())
    runs = {}
    for mode in ("default", "guided"):
        trace: list = []
        with PO._trace(trace):
            if mode == "default":
                doc, hist = refine(bad, target, max_iters=3)
            else:
                with PO.guided(pol, prune=True):
                    doc, hist = refine(bad, target, max_iters=3)
        runs[mode] = hist[-1]
        assert trace and trace[0][1] >= trace[-1][1]
    d, g = runs["default"], runs["guided"]
    assert g["after"] <= 0.1 * g["before"]                  # >= 90 % of the loss recovered
    assert g["after"] <= d["after"] + 1e-6 or g["params"]["renders"] <= d["params"]["renders"]


# --------------------------------------------------------------------------- novelty
def _items(rng, n, page, centre):
    out = []
    for i in range(n):
        f = {k: float(rng.normal(centre.get(k, 0.0), 0.3)) for k in NV.FEATURES}
        out.append(NV.Item(page if isinstance(page, str) else page[i % len(page)], Box(0, 0, 10, 10), "residual", f, label="color.fill"))
    return out


def test_novelty_flags_outliers_and_clusters_only_multi_page_modes():
    rng = np.random.default_rng(0)
    known = _items(rng, 200, "k", {})
    model = NV.NoveltyModel.fit(known)
    normal = _items(rng, 30, "p1", {})
    odd = _items(rng, 12, ["p1", "p2", "p3"], {"tex_t": 8.0, "log_area": 6.0})
    lonely = _items(rng, 6, "p9", {"edge_c": -8.0, "icon_frac": 8.0})
    model.score(normal + odd + lonely)
    assert np.mean([NV.is_candidate(i) for i in normal]) < 0.2
    assert all(NV.is_candidate(i) for i in odd)
    cl = NV.cluster_candidates([i for i in normal + odd + lonely if NV.is_candidate(i)], model)
    multi = [c for c in cl if c["promotable"]]
    assert multi and set(multi[0]["pages"]) == {"p1", "p2", "p3"}
    assert all(not c["promotable"] for c in cl if c["pages"] == ["p9"])
    stub = multi[0]["stub"]
    for k in ("name", "description", "stage", "failure_refs", "param_space", "criteria", "generator_sketch", "python"):
        assert k in stub
    assert stub["stage"] in ("perceive", "map", "translate")
    assert all(set(c) >= {"metric", "op", "threshold"} for c in stub["criteria"])
    fam = NV.to_family(stub)
    assert fam is None or fam.name == stub["name"]


def test_novelty_features_on_real_renders():
    from dt.render.screenshot import render_doc
    clean = _card_doc()
    moved = copy.deepcopy(clean)
    for n in moved.walk():
        if n.id == "btn":
            n.box = Box(n.box.x + 2, n.box.y, n.box.w, n.box.h)
    t, c = render_doc(clean), render_doc(moved)
    f = NV.features(NV.PageCtx(t, c, moved), Box(14, 124, 136, 52))
    assert f["shift_gain"] > 0.8
    recol = copy.deepcopy(clean)
    for n in recol.walk():
        if n.id == "btn":
            n.fills = [Fill.solid("#7d5260")]
    f2 = NV.features(NV.PageCtx(t, render_doc(recol), recol), Box(20, 130, 120, 40))
    assert f2["colour_gain"] > 0.8 and f2["shift_gain"] < 0.3


# --------------------------------------------------------------------------- protocol
register("adversary.toy.thr", 20.0, "toy adversary: ΔE threshold (tests only)", (1.0, 80.0))
register("adversary.toy.always", 0, "toy adversary: 1 = also report a finding at the page corner (tests only)", (0, 1))


def toy_adversary(t, c, ir=None):
    d = np.abs(t.astype(int) - c.astype(int)).sum(-1)
    out = []
    ys, xs = np.nonzero(d > float(P["adversary.toy.thr"]))
    if len(xs):
        out.append(Finding("structure.missing", Box(int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)), None, 0.9))
    if int(P["adversary.toy.always"]):
        out.append(Finding("structure.extra", Box(0, 0, 8, 8), None, 0.1))
    return out


def _toy_cases():
    from dt.adversary.benchmark import Case
    cases = []
    rng = np.random.default_rng(3)
    for i in range(8):
        fam = "synth" if i % 2 else "mwc"
        t = np.full((60, 80, 3), 250, np.uint8)
        c = t.copy()
        x, y = 10 + 3 * i, 12 + 2 * i
        de = 4 if i % 2 == 0 else 60       # half the errors are faint (3 x 4 = 12 < 20)
        t[y:y + 14, x:x + 20] = 250 - de
        g = [Finding("structure.missing", Box(x, y, 20, 14), None)]
        cases.append(Case(f"t{i}", "calibration", fam, f"{fam}:{i}", "perturbed", g, {}, {"width": 80, "height": 60}, t, c, None))
    for i in range(2):
        t = np.full((60, 80, 3), 250, np.uint8)
        t[rng.integers(0, 60, 20), rng.integers(0, 80, 20)] = 249
        cases.append(Case(f"n{i}", "calibration", "synth" if i else "mwc", f"x:{i}", "noise", [], {"jpeg_quality": 90},
                          {"width": 80, "height": 60}, t, np.full((60, 80, 3), 250, np.uint8), None))
    return cases


def test_bounded_priors_rejects_out_of_range_yardstick_and_big_changes():
    assert PR.bounded_priors(PR.as_change({"adversary.toy.thr": 15.0}))["passed"]
    assert not PR.bounded_priors(PR.as_change({"adversary.toy.thr": 500.0}))["passed"]
    assert not PR.bounded_priors(PR.as_change({"adversary.toy.thr": 79.0}))["passed"]       # step > max_step
    assert not PR.bounded_priors(PR.as_change({"validate.jnd_de": 3.0}))["passed"]          # the yardstick
    assert not PR.bounded_priors(PR.as_change({"no.such.key": 1}))["passed"]
    many = {k: P[k] for k in list(P.ranges())[:6]}
    assert not PR.bounded_priors(PR.as_change(many))["passed"]


def test_protocol_accepts_a_generalising_change_and_rejects_a_metamorphic_cheat():
    cases = _toy_cases()
    good = PR.check({"adversary.toy.thr": 5.0}, adversary=toy_adversary, calibration=cases, fresh=[], name="lower thr")
    assert good["accepted"], good["reasons"]
    assert good["checks"]["calibration"]["gain"] > 0
    assert set(good["checks"]["leave_one_family_out"]["families"]) == {"mwc", "synth"}
    led = good["ledger"]
    for k in ("id", "ts", "kind", "scope", "source", "change", "evidence", "accepted", "reverts"):
        assert k in led
    assert led["change"]["params"]["adversary.toy.thr"] == [20.0, 5.0]
    assert set(led["evidence"]) >= {"before", "after", "gates"}
    # a change that does not win, plus one that adds an always-on finding (identity relation broken)
    cheat = PR.check({"adversary.toy.thr": 5.0, "adversary.toy.always": 1}, adversary=toy_adversary, calibration=cases, fresh=[])
    assert not cheat["accepted"]
    assert not cheat["checks"]["metamorphic"]["passed"]
    assert cheat["checks"]["metamorphic"]["after"]["by_relation"]["identity"] > 0


def test_protocol_change_fn_only_sees_training_cases():
    cases = _toy_cases()
    seen = []

    def change_fn(train):
        seen.append({c.id for c in train})
        return PR.override({"adversary.toy.thr": 5.0})
    v = PR.check(change_fn, adversary=toy_adversary, calibration=cases, fresh=[], metamorphic=False)
    ids = {c.id for c in cases}
    assert all(s <= ids for s in seen)
    fams = {c.id: c.family for c in cases}
    assert any({fams[i] for i in s} == {"synth"} for s in seen) and any({fams[i] for i in s} == {"mwc"} for s in seen)
    assert "evaluation" not in v["checks"]
