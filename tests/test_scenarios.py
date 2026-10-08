"""Failure-scenario harness: deterministic families, criteria, runner, promote/gate, training
loop guards, mining. Uses the real renderer for every case."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

import dt.params as dtp
from dt.ir import Box, Color, Document, Fill, Node
from dt.learn import ledger, state
from dt.params import P
from dt.scenarios import registry, runner, suite
from dt.scenarios.sources import color_at_de, de2000, ir_case
from dt.scenarios.spec import Criterion, ScenarioFamily, sample_params

GENERATED = ("pale_surface", "page_tint", "custom_glyph", "desktop_list")


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    """Isolated learned state: params layers, DT_HOME, ledger, suite, real-crop manifest."""
    monkeypatch.setattr(dtp, "PARAMS_PATH", str(tmp_path / "params.json"))
    monkeypatch.setenv("DT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("DT_PARAMS_LOCAL", raising=False)
    monkeypatch.setenv("DT_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("DT_SCENARIO_SUITE", str(tmp_path / "scenarios.json"))
    monkeypatch.setenv("DT_REAL_MANIFEST", str(tmp_path / "real_crops.json"))
    P.reload()
    yield tmp_path
    monkeypatch.undo()
    P.reload()


# --------------------------------------------------------------------------- tiny families (fast, real renderer)
def _tiny_generate(p: dict, seed: int):
    doc = Document.blank(160, 120, "#ffffff")
    card = Node(type="rect", name="card", box=Box(20, 20, 100 + p["w"], 60), fills=[Fill.solid(p["fill"])], radius=(8,) * 4)
    doc.root.children.append(card)
    return ir_case("tiny", seed, p, doc, roi=Box(10, 10, 140, 80), meta={"card": card.box}, fit_text=False)


def _card_found(case, pred, rendered) -> dict:
    hit = any(n.box.iou(case.meta["card"]) >= 0.8 for n in pred.walk() if n is not pred.root)
    return {"card_found": 1.0 if hit else 0.0}


def tiny_family(name: str = "tiny", criteria=None, metrics=_card_found, prefixes=("perceive.seg.",)) -> ScenarioFamily:
    fam = ScenarioFamily(name=name, description="one strong-contrast card", stage="perceive", failure_refs=["test"],
                         param_space={"w": (0, 20), "fill": ["#6750a4", "#b3261e"]}, generate=_tiny_generate,
                         criteria=criteria or [Criterion("card_found", ">=", 1.0),
                                               Criterion("jnd_frac_nontext", "<=", "param:validate.gate.ident.jnd_frac",
                                                         roi="roi", scale=0.05)],
                         tune_prefixes=tuple(prefixes), metrics=metrics)
    registry.register(fam)
    return fam


# --------------------------------------------------------------------------- spec
def test_criterion_ops_margins_and_param_thresholds():
    c = Criterion("x", ">=", 0.5, scale=0.5)
    r = c.evaluate(0.75)
    assert r["passed"] and r["margin"] == pytest.approx(0.5)
    assert not c.evaluate(0.25)["passed"] and c.evaluate(0.25)["margin"] == pytest.approx(-0.5)
    assert c.evaluate(None) == {**c.evaluate(None), "passed": False, "margin": -1.0}
    lt = Criterion("y", "<", 2.0)
    assert not lt.evaluate(2.0)["passed"] and lt.evaluate(2.0)["margin"] == 0.0
    import dt.validate.fidelity  # noqa: F401
    g = Criterion("jnd_frac_nontext", "<=", "param:validate.gate.ident.jnd_frac", roi="roi")
    assert g.resolved_threshold() == P["validate.gate.ident.jnd_frac"] and g.key() == "roi.jnd_frac_nontext"
    assert Criterion.from_dict(g.to_dict()) == g
    with pytest.raises(ValueError):
        Criterion("x", "~", 1)


def test_sample_params_is_seeded_and_within_priors():
    space = {"a": (0, 10), "b": (0.5, 1.5), "c": ["x", "y", "z"], "d": [True, False]}
    p1, p2 = sample_params(space, 3), sample_params(space, 3)
    assert p1 == p2
    assert any(sample_params(space, s) != p1 for s in range(4, 10))
    for s in range(20):
        p = sample_params(space, s)
        assert isinstance(p["a"], int) and 0 <= p["a"] <= 10 and 0.5 <= p["b"] <= 1.5 and p["c"] in space["c"]


def test_color_at_de_hits_the_requested_difference():
    for base in ("#ffffff", "#fdf7fe", "#1f1f1f"):
        for de in (1.5, 3.0, 5.0):
            c = color_at_de(Color.from_hex(base), de, hue_deg=300, chroma_frac=0.4)
            assert abs(de2000(Color.from_hex(base), c) - de) < 0.35


# --------------------------------------------------------------------------- families
def test_registry_discovers_the_failure_families():
    fams = registry.families(refresh=True)
    assert set(GENERATED) | {"real_crops"} <= set(fams), registry.errors()
    for name in GENERATED:
        f = fams[name]
        assert f.failure_refs and f.criteria and f.tune_prefixes and f.param_space
        assert len(f.param_space) >= 5  # wide priors, not one page


@pytest.mark.parametrize("name", GENERATED)
def test_family_generates_deterministic_cases(name):
    fam = registry.get(name)
    a, b = fam.case(11), fam.case(11)
    assert a.params == b.params and a.fingerprint() == b.fingerprint()
    assert a.target_rgb.shape[:2] == (a.gt.height, a.gt.width)
    c = fam.case(12)
    assert c.params != a.params or c.fingerprint() != a.fingerprint()
    if a.roi is not None:
        assert 0 <= a.roi.x and a.roi.x2 <= a.width + 1e-6 and a.roi.y2 <= a.height + 1e-6


def test_family_specific_facts():
    ps = registry.get("pale_surface").case(5)
    for s in ps.meta["surfaces"]:
        assert 1.3 <= ps.meta["surface_de"] <= 5.2 and s["box"].area > 0
    cg = registry.get("custom_glyph").case(2)
    assert cg.meta["glyphs"] and all(6 <= g.w <= 21 for g in cg.meta["glyphs"])   # svg glyphs from the DOM gt
    dl = registry.get("desktop_list").case(1)
    assert dl.width >= 1152 and len(dl.meta["rows"]) >= 6
    texts = {s["text"] for s in dl.meta["segments"]}
    assert len(dl.meta["segments"]) == 3 * len(dl.meta["rows"]) and all(t for t in texts)


def test_runner_pass_rate_margins_and_images(tmp_path):
    fam = tiny_family()
    rep = runner.run_family(fam, [0, 1], out_root=str(tmp_path), run_id="r")
    assert rep["n"] == 2 and rep["n_errors"] == 0, [c.get("traceback") for c in rep["cases"]]
    assert rep["pass_rate"] == 1.0 and rep["objective"] >= 0
    assert all(os.path.exists(c["diff_image"]) for c in rep["cases"])
    assert os.path.exists(os.path.join(rep["out_dir"], "report.json")) and os.path.exists(os.path.join(rep["out_dir"], "report.md"))
    row = rep["cases"][0]
    assert {"full", "roi"} <= set(row["metrics"]) and row["metrics"]["full"]["card_found"] == 1.0
    assert "node_recall" in row["metrics"]["roi"] and "jnd_frac_nontext" in row["metrics"]["roi"]
    impossible = tiny_family("tiny_impossible", criteria=[Criterion("card_found", ">", 1.0, scale=1.0)])
    rep2 = runner.run_family(impossible, [0], out_root=None)
    assert rep2["pass_rate"] == 0.0 and rep2["cases"][0]["margin_min"] == 0.0  # fails at margin 0 (strict op)


def test_runner_survives_a_broken_generator():
    def boom(p, s):
        raise RuntimeError("nope")
    fam = ScenarioFamily(name="broken", description="", stage="perceive", failure_refs=[], param_space={"a": (0, 1)},
                         generate=boom, criteria=[Criterion("x", ">=", 1)])
    rep = runner.run_family(fam, [0], out_root=None)
    assert rep["n_errors"] == 1 and rep["pass_rate"] == 0.0 and rep["cases"][0]["error_stage"] == "generate"


def test_promote_and_gate(iso):
    fam = tiny_family()
    ent = suite.promote(fam.name, n_train=1, n_holdout=1, out_root=None)
    s = suite.load()
    assert s["families"]["tiny"]["baseline"]["holdout_pass_rate"] == 1.0
    assert s["families"]["tiny"]["holdout_seeds"] == [suite.HOLDOUT_OFFSET]
    es = ledger.read()
    assert es[-1]["kind"] == "scenario_baseline" and es[-1]["id"] == ent["ledger_id"]
    g = suite.gate()
    assert g["pass"] and g["families"]["tiny"]["current"] == 1.0
    # a family whose pass rate falls below its baseline fails the gate
    s["families"]["tiny_impossible"] = dict(s["families"]["tiny"])
    tiny_family("tiny_impossible", criteria=[Criterion("card_found", ">", 1.0)])
    suite.save(s)
    g = suite.gate()
    assert not g["pass"] and g["families"]["tiny_impossible"]["status"] == "regressed"
    # a generator change invalidates the baseline
    s["families"]["tiny"]["family_version"] = 99
    suite.save(s)
    assert suite.gate(["tiny"])["families"]["tiny"]["status"] == "version"
    # bench integration
    from dt.selftest import bench
    s["families"] = {"tiny": {**s["families"]["tiny"], "family_version": 1}}
    suite.save(s)
    sg = bench.scenario_gate()
    assert sg["pass"] and "scenario gate: PASS" in sg["text"]


def test_train_writes_nothing_unless_gates_pass(iso):
    fam = tiny_family()          # already passes everything: the holdout cannot improve
    suite.promote(fam.name, n_train=1, n_holdout=1, out_root=None, record=False)
    from dt.scenarios import train
    res = train.train(fam.name, iters=2, seed=1, bench_gate=lambda ch: (True, {}), history_path=None, verbose=False)
    assert not res["written"] and res["ledger_id"] is None
    assert not os.path.exists(dtp.PARAMS_PATH) and not os.path.exists(state.layer_path("local"))
    assert not os.path.exists(state.ledger_path("global"))
    if res["changed"]:
        assert res["guards"]["holdout"]["pass"] is False and "bench" not in res["guards"]  # short-circuit


def test_train_accepts_generalising_change_then_revert_restores(iso):
    from dt.scenarios import train
    from dt.selftest import tune
    key = sorted(tune.search_space(include=["perceive.seg."]))[0]
    base = P[key]
    spec = tune.search_space(include=[key])[key]

    def knob(case, pred, rendered):  # any move of the knob "fixes" this synthetic failure
        return {"knob": abs(float(P[key]) - float(base)) / (spec["hi"] - spec["lo"])}
    fam = tiny_family("tiny_knob", criteria=[Criterion("knob", ">=", 0.01, scale=0.1)], metrics=knob, prefixes=(key,))
    suite.promote(fam.name, n_train=1, n_holdout=1, out_root=None, record=False)
    assert suite.load()["families"]["tiny_knob"]["baseline"]["holdout_pass_rate"] == 0.0
    calls = []
    res = train.train(fam.name, prefixes=[key], iters=2, local=True, seed=0, history_path=None, verbose=False,
                      bench_gate=lambda ch: (calls.append(ch) or True, {"stub": True}))
    assert res["changed"] and res["accepted"] and res["written"], res
    assert calls == [res["changed"]]                       # bench gate ran on exactly the change
    assert res["guards"]["holdout"]["after"]["pass_rate"] == 1.0
    local = json.loads(open(state.layer_path("local")).read())
    assert local == res["changed"] and P.layers()[key] == "local"
    assert not os.path.exists(dtp.PARAMS_PATH)               # --local never touches the global layer
    e = ledger.get(res["ledger_id"])
    assert e["kind"] == "params" and e["scope"] == "local" and e["evidence"]["gates"] == {"holdout": True, "scenarios": True, "bench": True}
    ledger.revert(e["id"])
    assert json.loads(open(state.layer_path("local")).read()) == {} and P[key] == base


def test_mine_real_crops(iso, tmp_path):
    from dt.common.image import save_rgb
    from dt.scenarios import mine
    from dt.render.screenshot import render_doc
    doc = Document.blank(300, 240, "#ffffff")
    doc.root.children.append(Node(type="rect", box=Box(100, 100, 60, 20), fills=[Fill.solid("#3f51b5")]))
    rgb = render_doc(doc)
    run = tmp_path / "run1"
    os.makedirs(run / "validation")
    save_rgb(rgb, str(run / "validation" / "target.png"))
    (run / "metrics.json").write_text(json.dumps({"source": str(run / "validation" / "target.png")}))
    (run / "validation" / "validation.json").write_text(json.dumps({
        "gates": {"visually-identical": False}, "chamfer_tile_max": 0.5, "chamfer_tile_argmax": {},
        "worst_regions": [{"box": {"x": 100, "y": 100, "w": 60, "h": 20}, "delta_e": 54.9, "target_color": "#3f51b5",
                           "render_color": "#ffffff"}, {"box": {"x": 0, "y": 0, "w": 9, "h": 9}, "delta_e": 1.0}]}))
    res = mine.mine([str(run)])
    assert len(res["added"]) == 1
    man = mine.load_manifest()
    e = man["crops"][0]
    assert e["kind"] == "region" and e["measure"]["value"] == pytest.approx(54.9) and e["criteria"]
    b = Box.from_dict(e["box"])
    assert b.contains(Box(100, 100, 60, 20)) and b.w >= P["scenarios.mine.min_side"]
    px = mine.load_crop(e)
    assert px.shape[:2] == (int(b.h), int(b.w))
    assert os.path.exists(os.path.join(state.real_crops_dir(), e["id"], "meta.json"))
    fam = registry.get("real_crops")
    assert fam.available() and fam.n_cases() == 1
    case = fam.case(0)
    assert case.gt is None and case.meta["criteria"][0]["metric"] == "region_de_worst"
    assert np.array_equal(case.target_rgb, px)


def test_cli_scenario_list_and_bench_flag(capsys):
    from dt.cli import build_parser, main
    assert main(["scenario", "list", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert set(GENERATED) <= set(out["families"])
    a = build_parser().parse_args(["bench", "--gate", "--scenarios"])
    assert a.scenarios is True
    a = build_parser().parse_args(["train", "--family", "pale_surface", "--params", "perceive.seg.", "--iters", "2", "--local"])
    assert a.local and a.iters == 2
