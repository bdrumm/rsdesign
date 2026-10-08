"""Adversary benchmark: taxonomy, render-verified perturbations, noise, determinism, scorer math."""
from __future__ import annotations

import copy
import hashlib
import math
import os
import random
import re

import numpy as np
import pytest

from dt.adversary import benchmark as BM
from dt.adversary import perturb as PT
from dt.adversary import taxonomy as TX
from dt.adversary.taxonomy import Finding
from dt.compare.structural import levenshtein
from dt.ir import Box, Color, Document
from dt.params import P

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SYNTH = os.path.join(ROOT, "fixtures", "corpus", "synth", "synth_1_002.gt.json")
MWC = os.path.join(ROOT, "fixtures", "corpus", "mwc", "mwc_1_001.gt.json")


def _h(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


# --------------------------------------------------------------------------- taxonomy
def test_taxonomy_covers_the_required_types_and_maps_to_critic_kinds():
    required = {"geometry.shift", "geometry.size", "geometry.radius", "color.fill", "color.stroke", "color.tint_global",
                "structure.missing", "structure.extra", "structure.split", "structure.merge", "text.content", "text.style",
                "text.position", "icon.glyph", "icon.missing", "effect.shadow", "effect.opacity", "layout.spacing",
                "noise.antialias", "noise.subpixel"}
    assert required <= set(TX.TYPES)
    src = open(os.path.join(ROOT, "dt", "refine", "critic.py")).read()
    critic_kinds = set(re.findall(r'Hypothesis\(\s*"([a-z_]+)"', src))
    assert critic_kinds == set(TX.CRITIC_KINDS)
    for k, t in TX.TYPES.items():
        assert t.parent in TX.PARENTS and t.definition and t.ir_property
        assert set(t.refine_kinds) <= critic_kinds
        if t.is_error:
            assert t.refine_kinds, k
        else:
            assert t.parent == "noise" and not t.refine_kinds
        if t.magnitude_scored:
            assert t.magnitude_unit
    # every error type has a perturbation operator, and nothing else does
    assert set(PT.OPERATORS) == set(TX.ERROR_TYPES)
    assert TX.is_noise("noise.subpixel") and not TX.is_noise("geometry.shift")
    f = Finding("geometry.shift", Box(1, 2, 3, 4), 2.5, 0.7, {"a": 1}, "n3")
    assert Finding.from_dict(f.to_dict()) == f
    assert TX.describe()["version"] == TX.TAXONOMY_VERSION


# --------------------------------------------------------------------------- perturbations (real renderer)
@pytest.fixture(scope="module")
def renders():
    from dt.render.screenshot import render_doc
    out = {}
    for p in (SYNTH, MWC):
        d = Document.load(p)
        out[p] = (d, render_doc(d))
    return out


def _doc_for(t: str) -> str:
    return SYNTH if t in ("color.stroke", "structure.merge") else (MWC if t in ("geometry.size", "text.style") else SYNTH)


@pytest.mark.parametrize("type_", list(TX.ERROR_TYPES))
def test_operator_yields_render_verified_ground_truth(type_, renders):
    from dt.render.screenshot import render_doc
    path = _doc_for(type_)
    doc, clean = renders[path]
    before = copy.deepcopy(doc.to_dict())
    got = None
    for seed in range(6):
        u = (seed % 4) / 3
        cand, gt = PT.perturb(doc, [(type_, u)], random.Random(seed))
        assert doc.to_dict() == before, "perturb must not mutate its input"
        if not gt:
            continue
        rgb = render_doc(cand)
        v = PT.verify(clean, rgb, gt)
        if v["ok"]:
            got = (cand, gt, v, u)
            break
    assert got is not None, f"{type_}: no render-verified perturbation in 6 seeds"
    cand, gt, v, u = got
    assert len(gt) == 1
    f = gt[0]
    assert f.type == type_
    page = Box(0, 0, doc.width, doc.height)
    assert page.contains(f.box) and f.box.area > 0
    assert v["visible"][0] >= int(P["adversary.perturb.min_visible_px"])
    assert v["unexplained_px"] <= int(P["adversary.perturb.max_unexplained_px"])
    assert f.evidence["bin"] == PT.magnitude_bin(u)
    spec = TX.TYPES[type_]
    if spec.magnitude_unit is None:
        assert f.magnitude is None
    else:
        assert f.magnitude is not None and f.magnitude > 0
    node = cand.find(f.node_id) if f.node_id else None
    if type_ not in ("color.tint_global", "structure.missing", "icon.missing"):
        assert node is not None, "error types that edit an existing node name it in the candidate IR"
    # type-specific exactness
    if type_ in ("geometry.shift", "text.position"):
        assert f.magnitude == pytest.approx(math.hypot(f.evidence["dx"], f.evidence["dy"]))
        assert f.box.expand(1.01).contains(node.box)
        assert f.magnitude >= 1.0
    elif type_ == "color.fill":
        c = node.text_style.color if node.type == "text" else node.fill_color
        assert c.hex() == f.evidence["to"]
        assert Color.from_hex(f.evidence["from"]).delta_e(c) == pytest.approx(f.magnitude, abs=1e-6)
    elif type_ == "color.stroke":
        assert node.strokes[0].color.hex() == f.evidence["to"]
    elif type_ == "text.content":
        assert node.text == f.evidence["to"] != f.evidence["from"]
        assert f.magnitude == pytest.approx(levenshtein(f.evidence["from"], f.evidence["to"]) / len(f.evidence["from"]))
        assert node.name == f"text:{node.text[:24]}"
    elif type_ == "icon.glyph":
        assert node.icon_name == f.evidence["to"] != f.evidence["from"]
    elif type_ in ("structure.missing", "icon.missing"):
        assert f.node_id is None and cand.root.count() < doc.root.count()
        assert f.magnitude == pytest.approx(f.box.area, rel=0.6)
    elif type_ == "structure.extra":
        assert cand.root.count() > doc.root.count() and f.box.contains(node.box)
    elif type_ == "structure.split":
        a, b = (cand.find(i) for i in f.evidence["parts"])
        assert a is not None and b is not None and cand.root.count() == doc.root.count() + 1
    elif type_ == "structure.merge":
        assert cand.root.count() < doc.root.count()
    elif type_ == "effect.opacity":
        assert node.opacity == pytest.approx(1.0 - f.magnitude)
    elif type_ == "color.tint_global":
        assert f.box.area == page.area


def test_perturbations_in_one_case_are_disjoint_and_exclusive_tint(renders):
    doc, _ = renders[SYNTH]
    plan = [(t, 0.5) for t in ("geometry.shift", "color.fill", "text.content", "icon.missing")]
    cand, gt = PT.perturb(doc, plan, random.Random(3))
    assert len(gt) >= 3
    pad = float(P["adversary.perturb.disjoint_pad"])
    for i, a in enumerate(gt):
        for b in gt[i + 1:]:
            assert a.box.expand(pad).intersect(b.box).area == 0
    _, gt2 = PT.perturb(doc, [("color.tint_global", 0.5), ("geometry.shift", 0.5)], random.Random(3))
    assert [f.type for f in gt2] == ["color.tint_global"]


def test_sanitize_strips_leaks_without_changing_pixels(renders):
    from dt.render.screenshot import render_doc
    doc, clean = renders[MWC]
    cand, _ = PT.sanitize(copy.deepcopy(doc))
    assert _h(render_doc(cand)) == _h(clean)
    ids = [n.id for n in cand.walk()]
    assert ids[0] == "root" and ids[1:] == [f"n{i}" for i in range(1, len(ids))]
    for n in cand.walk():
        assert not n.tokens and set(n.meta) <= {"icon_fill"}
        assert n.component is None or not n.component.evidence
        if n.type == "text" and n.text:
            assert n.name == f"text:{n.text[:24]}"
    assert not cand.meta and cand.source_image is None


def test_text_measurement_matches_the_renderer(renders):
    doc, _ = renders[SYNTH]
    n = next(t for t in doc.walk() if t.type == "text" and len(t.text or "") > 4)
    w, w_long = PT.measure_texts([(n.text, n), (n.text + n.text, n)])
    assert abs(w - n.box.w) <= 1.0  # synth boxes are measured glyph runs (rounded up)
    assert w_long == pytest.approx(2 * w, abs=4.0)


# --------------------------------------------------------------------------- noise
def test_noise_changes_the_target_but_never_creates_ground_truth(renders):
    doc, clean = renders[SYNTH]
    sub = PT.render_target(doc, PT.NoiseRecipe(translate=(0.3, -0.4)), clean)
    assert sub.shape == clean.shape and _h(sub) != _h(clean)
    assert _h(PT.render_target(doc, PT.NoiseRecipe(translate=(0.3, -0.4)), clean)) == _h(sub)  # deterministic
    jp = PT.render_target(doc, PT.NoiseRecipe(jpeg_quality=80), clean)
    assert _h(jp) != _h(clean)
    bl = PT.render_target(doc, PT.NoiseRecipe(blur_sigma=0.5), clean)
    assert _h(bl) != _h(clean)
    assert _h(PT.render_target(doc, PT.NoiseRecipe(), clean)) == _h(clean)
    rec = PT.sample_noise(random.Random(1), ["subpixel", "jpeg"], 2)
    assert set(rec.types()) == {"noise.subpixel", "noise.compression"}
    lo, hi = P["adversary.noise.subpixel"]
    assert all(lo <= abs(v) <= hi for v in rec.translate)
    assert all(TX.is_noise(t) for t in rec.types())


def test_alternate_browser_noise_if_installed(renders):
    alt = PT.alt_browser()
    if not alt.available:
        pytest.skip(f"no alternate browser build: {alt.error}")
    doc, clean = renders[SYNTH]
    rgb = PT.render_target(doc, PT.NoiseRecipe(browser="alt"), clean)
    assert rgb.shape == clean.shape


# --------------------------------------------------------------------------- benchmark build
MINI = BM.BenchConfig(seed=5, cases_per_doc=3, max_perturbations=3, calibration_docs=("synth:synth_1_001",),
                      evaluation_docs=("synth:synth_1_007",), include_browser_noise=False)


@pytest.fixture(scope="module")
def mini():
    P.set("adversary.bench.noise_case_frac", 0.34)
    try:
        return BM.build(MINI), BM.build(MINI)
    finally:
        P.reset("adversary.bench.noise_case_frac")


def test_build_is_deterministic(mini):
    a, b = mini
    assert [c.record() for c in a.cases] == [c.record() for c in b.cases]
    for x, y in zip(a.cases, b.cases):
        assert _h(x.target_rgb) == _h(y.target_rgb) and _h(x.candidate_rgb) == _h(y.candidate_rgb)
        assert x.candidate_ir.to_dict() == y.candidate_ir.to_dict()
    assert {c.split for c in a.cases} == {"calibration", "evaluation"}
    assert any(c.kind == "noise" for c in a.cases) and any(c.kind == "perturbed" for c in a.cases)


def test_benchmark_cases_are_consistent(mini):
    from dt.render.screenshot import render_doc
    bench, _ = mini
    for c in bench.cases:
        assert c.target_rgb.shape == c.candidate_rgb.shape == (c.meta["height"], c.meta["width"], 3)
        if c.kind == "noise":
            assert c.gt == [] and c.noise
        else:
            assert 1 <= len(c.gt) <= MINI.max_perturbations
            assert all(f.type in TX.ERROR_TYPES and f.evidence["visible_px"] >= 1 for f in c.gt)
            # the stored candidate image is the render of the stored candidate IR
            assert _h(render_doc(c.candidate_ir)) == _h(c.candidate_rgb)


def test_save_load_roundtrip(mini, tmp_path):
    bench, _ = mini
    bench2 = copy.deepcopy(bench)
    bench2.save(str(tmp_path))
    back = BM.Benchmark.load(str(tmp_path))
    assert [c.record() for c in back.cases] == [c.record() for c in bench.cases]
    assert _h(back.cases[0].target_rgb) == _h(bench.cases[0].target_rgb)


def test_seed_changes_the_benchmark():
    cfg = BM.BenchConfig(seed=6, cases_per_doc=2, calibration_docs=("synth:synth_1_001",), evaluation_docs=(), include_browser_noise=False)
    cfg5 = BM.BenchConfig(seed=5, cases_per_doc=2, calibration_docs=("synth:synth_1_001",), evaluation_docs=(), include_browser_noise=False)
    assert [c.record() for c in BM.build(cfg).cases] != [c.record() for c in BM.build(cfg5).cases]


def test_default_split_is_disjoint_by_document_and_seed():
    cal, ev = set(BM.DEFAULT_CALIBRATION), set(BM.DEFAULT_EVALUATION)
    assert not cal & ev
    fam_cal = {BM.family_of(s) for s in cal}
    fam_ev = {BM.family_of(s) for s in ev}
    assert "synth_gen" in fam_ev - fam_cal  # a held-out generator seed family
    assert BM._sub_seed(0, "calibration", "synth:x", 0) != BM._sub_seed(0, "evaluation", "synth:x", 0)


# --------------------------------------------------------------------------- scorer
def _case(cid: str, kind: str, gt: list[Finding], noise: dict | None = None) -> BM.Case:
    z = np.zeros((100, 100, 3), np.uint8)
    return BM.Case(cid, "evaluation", "synth", "synth:x", kind, gt, noise or {}, {"width": 100, "height": 100}, z, z, None)


def _scripted(responses: list[list[Finding]]):
    it = iter(responses)

    def fn(target, candidate, ir):
        return next(it)
    return fn


def test_scorer_math_on_hand_made_findings():
    g1 = Finding("geometry.shift", Box(10, 10, 20, 20), 5.0, 1, {"bin": "medium"})
    g2 = Finding("color.fill", Box(60, 60, 20, 20), 10.0, 1, {"bin": "small"})
    cases = [_case("c1", "perturbed", [g1, g2]), _case("c2", "noise", [], {"jpeg_quality": 80})]
    preds_c1 = [Finding("geometry.shift", Box(12, 12, 20, 20), 7.0, 0.9),     # IoU 0.68, right leaf
                Finding("color.stroke", Box(65, 65, 4, 4), 3.0, 0.8),        # centre in box, right parent, wrong leaf
                Finding("structure.extra", Box(0, 80, 5, 5), None, 0.95),    # spurious
                Finding("noise.antialias", Box(0, 0, 100, 100), None, 0.5)]  # abstention, never counted
    preds_c2 = [Finding("structure.missing", Box(40, 40, 10, 10), None, 0.6)]
    r = BM.score(_scripted([preds_c1, preds_c2]), cases, name="scripted")
    d = r["detection"]
    assert (d["tp"], d["pred"], d["gt"]) == (2, 4, 2)
    assert d["precision"] == pytest.approx(0.5) and d["recall"] == pytest.approx(1.0) and d["f1"] == pytest.approx(2 / 3)
    assert r["type_accuracy"] == {"leaf": 0.5, "parent": 1.0}
    pt = r["per_type"]
    assert pt["geometry.shift"]["f1"] == 1.0 and pt["geometry.shift"]["mag_mae"] == pytest.approx(2.0)
    assert pt["geometry.shift"]["mag_rel_median"] == pytest.approx(0.4)
    assert pt["color.fill"]["recall"] == 0.0 and pt["color.fill"]["loc_recall"] == 1.0
    assert pt["color.stroke"]["precision"] == 0.0 and pt["color.stroke"]["gt"] == 0
    assert r["macro_leaf"]["f1"] == pytest.approx(0.5)  # mean over the types present in ground truth
    assert r["per_parent"]["color"]["f1"] == 1.0 and r["macro_parent"]["f1"] == pytest.approx(1.0)
    # confidence ranking: 0.95 miss, 0.9 hit (1/2), 0.8 hit (2/3), 0.6 miss
    assert r["ap"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert r["noise"]["cases"] == 1 and r["noise"]["false_case_rate"] == 1.0
    assert r["noise"]["by_type"] == {"noise.compression": {"false_case_rate": 1.0, "cases": 1}}
    assert r["abstentions"] == 1
    assert r["confusion"]["geometry.shift"] == {"geometry.shift": 1}
    assert r["confusion"]["color.fill"] == {"color.stroke": 1}
    assert r["confusion"]["(spurious)"] == {"structure.extra": 1, "structure.missing": 1}
    assert r["recall_by_bin"] == {"medium": 1.0, "small": 1.0}
    assert r["headline"] == pytest.approx(0.4 * (2 / 3) + 0.3 * 0.5 + 0.3 * 0.0 * 1.0)
    assert r["runtime"]["mean_s"] >= 0


def test_localisation_rules():
    page = Box(0, 0, 100, 100)
    g = Finding("geometry.shift", Box(10, 10, 20, 20))
    assert BM.localises(Box(10, 10, 20, 20), g, page)
    assert BM.localises(Box(18, 18, 4, 4), g, page)          # tiny, centred inside: centre-in-box
    assert not BM.localises(Box(0, 0, 100, 100), g, page)    # page-sized prediction: low IoU, centre outside
    assert not BM.localises(Box(29, 29, 20, 20), g, page)    # IoU ~0.002, centre outside
    tint = Finding("color.tint_global", page)
    assert not BM.localises(Box(40, 40, 10, 10), tint, page)  # page-sized truth needs IoU
    assert BM.localises(Box(0, 0, 100, 90), tint, page)
    # one prediction can only claim one truth
    two = [Finding("text.content", Box(0, 0, 10, 10)), Finding("text.content", Box(2, 2, 10, 10))]
    assert len(BM.match([Finding("text.content", Box(1, 1, 10, 10))], two, page)) == 1


def test_floor_adversaries_and_oracle(mini):
    bench, _ = mini
    cases = bench.split("all")
    oracle = _scripted([[copy.deepcopy(f) for f in c.gt] for c in cases])
    r = BM.score(oracle, bench, "all", name="oracle")
    assert r["detection"]["f1"] == 1.0 and r["macro_leaf"]["f1"] == 1.0 and r["type_accuracy"]["leaf"] == 1.0
    assert r["noise"]["false_case_rate"] == 0.0 and r["headline"] == pytest.approx(1.0)
    null = BM.score(BM.null_adversary, bench, "all")
    assert null["detection"]["recall"] == 0.0 and null["noise"]["false_case_rate"] == 0.0 and null["headline"] == 0.0
    base = BM.score(BM.baseline_residual, bench, "all")
    assert 0.0 < base["detection"]["recall"] <= 1.0
    assert set(base["per_type"]) - set(r["per_type"]) <= {"structure.missing"}
    assert all(k == "structure.missing" or v["pred"] == 0 for k, v in base["per_type"].items())
    md = BM.format_report([base, null], bench)
    assert "baseline_residual" in md and "Confusion" in md
