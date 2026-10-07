"""Tests for dt.selftest.bench / metrics / tune (module E, phase 2).

The bench is exercised on real corpus cases through the real perceive -> map -> render
pipeline (no image mocks); the tuner runs a 2-iteration search against a scratch params file.
"""
from __future__ import annotations

import json
import os

import pytest

from dt.params import P
from dt.selftest import bench, tune
from dt.selftest.metrics import METRICS, aggregate, case_composite, goodness, suspect_stage, worst_entries

CORPORA = list(bench.DEFAULT_CORPORA)


# --------------------------------------------------------------------------- metrics (pure)
def test_goodness_directions_and_bounds():
    assert goodness("node_recall", 0.3) == pytest.approx(0.3)
    assert goodness("text_cer", 0.25) == pytest.approx(0.75)
    assert goodness("color_de", 0.0) == 1.0
    assert goodness("color_de", 1e9) == 0.0
    assert goodness("mean_de", P["bench.norm.mean_de"] / 2) == pytest.approx(0.5)
    assert goodness("node_recall", None) is None
    assert goodness("not_a_metric", 1.0) is None
    for m in METRICS:
        assert 0.0 <= goodness(m.name, 0.5) <= 1.0


def test_case_composite_skips_undefined_and_is_weighted():
    perfect = {m.name: (1.0 if m.higher_better else 0.0) for m in METRICS}
    assert case_composite(perfect) == pytest.approx(1.0)
    only_recall = {"node_recall": 0.5}
    assert case_composite(only_recall) == pytest.approx(0.5)
    assert case_composite({}) is None
    two = {"node_recall": 1.0, "component_acc": 0.0}  # equal weights 1.5 / 1.5
    assert case_composite(two) == pytest.approx(0.5)


def test_suspect_stage_blames_perceive_when_structure_is_missing():
    assert suspect_stage("component_acc", {}) == "map"
    assert suspect_stage("mean_de", {"node_recall": 0.95}) == "render"
    assert suspect_stage("mean_de", {"node_recall": 0.2}) == "perceive"
    assert suspect_stage("node_recall", {}) == "perceive"


def test_aggregate_and_worst_entries_pure():
    rows = [
        {"id": "a", "corpus": "x", "metrics": {"node_recall": 0.2, "color_de": 10.0}, "composite": 0.3},
        {"id": "b", "corpus": "x", "metrics": {"node_recall": 0.8, "color_de": None}, "composite": 0.8},
        {"id": "c", "corpus": "x", "metrics": {}, "composite": 0.0, "error": "boom", "error_stage": "perceive"},
    ]
    agg = aggregate(rows)
    assert agg["node_recall"]["mean"] == pytest.approx(0.5)
    assert agg["node_recall"]["worst"] == pytest.approx(0.2)
    assert agg["color_de"]["n"] == 1 and agg["color_de"]["worst"] == 10.0
    worst = worst_entries(rows, k=3)
    assert worst[0]["metric"] == "error" and worst[0]["case"] == "c"
    assert (worst[1]["case"], worst[1]["metric"], worst[1]["stage"]) == ("a", "node_recall", "perceive")
    assert len(worst) == 3


def test_param_overrides_restore_state():
    key = "bench.worst_k"  # registered by dt.selftest.metrics itself (perceive may not be imported yet)
    before = P[key]
    with bench.param_overrides({key: before + 1}):
        assert P[key] == before + 1
    assert P[key] == before


# --------------------------------------------------------------------------- bench on real cases
@pytest.fixture(scope="module")
def report(tmp_path_factory):
    out_root = str(tmp_path_factory.mktemp("bench"))
    return bench.run(CORPORA, limit=2, run_id="t", out_root=out_root)


def test_bench_runs_two_cases_per_corpus(report):
    assert report["n_cases"] == 4 and report["n_errors"] == 0
    assert {r["corpus"] for r in report["cases"]} == {"synth", "mwc"}
    assert 0.0 <= report["composite"] <= 1.0
    for r in report["cases"]:
        m = r["metrics"]
        assert 0.0 <= r["composite"] <= 1.0
        assert m["n_gt"] > 0 and m["n_pred"] > 0
        for k in ("node_recall", "node_precision", "mean_iou", "mean_de", "frac_bad", "ssim"):
            assert m[k] is not None
        assert m["text_recall"] is not None  # every corpus page has text
        assert m["text_line_recall"] >= m["text_recall"] and 0.0 <= m["component_name_acc"] <= 1.0
        assert r["timing"]["perceive_s"] > 0 and r["timing"]["render_s"] > 0 and r["timing"]["total_s"] > 0
        # the pipeline must be doing real work on these screens
        assert m["node_recall"] > 0.1 and m["mean_de"] < 15
    for name in ("node_recall", "component_acc", "mean_de"):
        a = report["metrics"][name]
        assert set(a) >= {"mean", "p50", "worst", "best", "n", "goodness"} and a["n"] == 4
    assert report["metrics"]["component_acc"]["mean"] >= 0.0  # defined: gt has components, map ran
    assert len(report["worst"]) == min(P["bench.worst_k"], 4 * len(METRICS))
    assert report["worst"][0]["deficit"] >= report["worst"][-1]["deficit"]
    assert all(e["stage"] in ("perceive", "map", "render") for e in report["worst"])


def test_bench_writes_report_and_diff_images(report):
    out = report["out_dir"]
    jp, mp = os.path.join(out, "report.json"), os.path.join(out, "report.md")
    assert os.path.exists(jp) and os.path.exists(mp)
    data = json.load(open(jp))
    assert data["composite"] == pytest.approx(report["composite"]) and data["n_cases"] == 4
    md = open(mp).read()
    assert "## Worst" in md and "| suspected stage |" in md and "## Cases" in md
    for r in report["cases"]:
        assert r["diff_image"] and os.path.exists(os.path.join(out, r["diff_image"]))
        assert f"{r['corpus']}/{r['id']}" in md


def test_stage_subsets_isolate_modules(tmp_path):
    cases = bench.find_cases([CORPORA[0]], limit=1)
    # no stages: the gt IR is the prediction -> structure is perfect, nothing else is scored
    rep0 = bench.run(cases=cases, stages=(), out_root=None)
    m0 = rep0["cases"][0]["metrics"]
    assert m0["node_recall"] == pytest.approx(1.0) and m0["mean_iou"] == pytest.approx(1.0)
    assert m0["component_acc"] is None and "mean_de" not in m0
    # map only: mapping is scored on perfect geometry (components stripped first, then re-mapped)
    rep = bench.run(cases=cases, stages=("map",), out_root=None)
    m = rep["cases"][0]["metrics"]
    assert m["component_acc"] is not None and m["node_recall"] > 0.5 and "mean_de" not in m
    # perceive only: no mapping / pixel metrics
    rep2 = bench.run(cases=cases, stages=("perceive",), out_root=None)
    m2 = rep2["cases"][0]["metrics"]
    assert m2["component_acc"] is None and m2["token_acc"] is None and "mean_de" not in m2
    assert rep2["out_dir"] is None


def test_baseline_gate(report, tmp_path):
    path = bench.save_baseline(report, str(tmp_path / "baseline.json"))
    base = bench.load_baseline(path)
    assert base["composite"] == pytest.approx(report["composite"])
    assert bench.check_regression(report, base) == []
    worse = json.loads(json.dumps(report))
    worse["composite"] -= 0.2
    worse["metrics"]["node_recall"]["goodness"] -= 0.2
    fails = bench.check_regression(worse, base)
    assert any(f.startswith("composite") for f in fails) and any(f.startswith("node_recall") for f in fails)
    drafts = bench.draft_failures(report, k=3)
    assert len(drafts) == 3 and {"case", "stage", "symptom", "metric", "value"} <= set(drafts[0])


def test_run_case_survives_broken_input(tmp_path):
    row = bench.run_case(str(tmp_path), "missing_case")
    assert row["error"] and row["error_stage"] == "load" and row["composite"] == 0.0


# --------------------------------------------------------------------------- tune
def test_tune_dry_lists_search_space(capsys):
    assert tune.main(["--dry"]) == 0
    out = capsys.readouterr().out
    space = tune.search_space()
    assert len(space) >= 10
    assert all(k in out for k in list(space)[:10])
    assert not any(k.startswith(tune.OBJECTIVE_PREFIXES) for k in space)
    assert all(s["kind"] in ("int", "float") and s["lo"] < s["hi"] for s in space.values())
    sub = tune.search_space(include=["perceive.ocr."])
    assert sub and all(k.startswith("perceive.ocr.") for k in sub)


def test_tune_two_iters_writes_params_only_on_holdout_gain(tmp_path):
    params_path = str(tmp_path / "params.json")
    history = str(tmp_path / "history.jsonl")
    before = P.all()
    res = tune.search(iters=2, corpus_dirs=CORPORA, holdout_frac=0.5, seed=1, limit=1,
                      params_path=params_path, history_path=history, verbose=False)
    assert res["n_train"] == 1 and res["n_holdout"] == 1 and res["evaluations"] >= 2
    assert P.all() == before  # live params restored
    lines = [json.loads(l) for l in open(history)]
    assert lines[0]["phase"] == "baseline" and lines[-1]["phase"] == "summary"
    assert len(lines) == res["evaluations"] + 1
    if res["written"]:
        gain = res["best"]["holdout"] - res["baseline"]["holdout"]
        assert gain > P["tune.min_improve"] and res["best"]["params"]
        saved = json.load(open(params_path))
        assert saved == res["best"]["params"]
    else:
        assert not os.path.exists(params_path)


# --------------------------------------------------------------------------- review additions
def test_run_rejects_unknown_stage_names():
    # a typo must not silently drop the stage and score the gt IR as the prediction
    with pytest.raises(ValueError, match="percieve"):
        bench.run(cases=[], stages=("percieve", "map"), out_root=None)


def test_suspect_stage_never_blames_a_stage_that_did_not_run():
    assert suspect_stage("type_acc", {}, stages=("map",)) == "map"
    assert suspect_stage("type_acc", {}, stages=("map", "render")) == "map"
    assert suspect_stage("mean_de", {"node_recall": 0.2}, stages=("map", "render")) == "map"
    assert suspect_stage("node_recall", {}, stages=()) == "gt"
    assert suspect_stage("type_acc", {}) == "perceive"  # no stage info: static table
    rows = [{"id": "a", "corpus": "x", "stages": ["map"], "metrics": {"type_acc": 0.5}, "composite": 0.5}]
    assert worst_entries(rows, k=1)[0]["stage"] == "map"


def test_regression_gate_rejects_incomparable_runs():
    base = {"composite": 0.5, "stages": ["perceive", "map", "render"], "n_cases": 24, "metrics": {}}
    same = {"composite": 0.6, "stages": ["perceive", "map", "render"], "n_cases": 24, "metrics": {}}
    assert bench.check_regression(same, base) == []
    sub = {"composite": 0.9, "stages": ["perceive"], "n_cases": 2, "metrics": {}}
    fails = bench.check_regression(sub, base)
    assert any(f.startswith("stages") for f in fails) and any(f.startswith("n_cases") for f in fails)


def test_discount_gt_translucency_threshold_is_a_param():
    from dt.ir import Box, Color, Document, Fill, Node
    doc = Document.blank(10, 10)
    doc.root.children.append(Node(type="frame", box=Box(0, 0, 5, 5), fills=[Fill.solid(Color.from_hex("#000000", 0.7))]))
    assert bench.discount_gt(doc).root.children[0].fills == []
    with bench.param_overrides({"bench.gt.opaque_alpha": 0.6}):
        assert len(bench.discount_gt(doc).root.children[0].fills) == 1


# --------------------------------------------------------------------------- metrics-gate
FIDELITY = ("jnd_frac_nontext", "chamfer_tile_max", "edge_within1")


def test_new_params_registered_with_doc_and_range():
    for key in ("bench.w.jnd_frac_nontext", "bench.w.chamfer_tile_max", "bench.w.edge_within1",
                "bench.norm.jnd_frac_nontext", "bench.norm.chamfer_tile_max", "bench.gate.eps", "bench.gate.metric_eps"):
        assert key in P.all() and key in P.ranges() and key in P.docs()
    assert P["bench.gate.eps"] == pytest.approx(0.003) and P["bench.gate.metric_eps"] == pytest.approx(0.01)
    from dt.selftest import metrics as M
    for spec in METRICS:  # every goodness normaliser is documented in the module docstring
        assert spec.name in M.__doc__
        if spec.norm:
            assert spec.norm in M.__doc__


def test_fidelity_metrics_on_real_renders():
    from dt.ir import Box, Document, Fill, Node, TextStyle
    from dt.render.screenshot import render_doc
    doc = Document.blank(240, 160)
    doc.root.children += [Node(id="card", type="rect", box=Box(20, 20, 120, 60), fills=[Fill.solid("#e8def8")], radius=12),
                          Node(id="t", type="text", box=Box(20, 100, 160, 24), text="Hello", text_style=TextStyle(size=16))]
    target = render_doc(doc)
    same = bench.fidelity_metrics(target, target, doc)
    assert same == {"jnd_frac_nontext": 0.0, "chamfer_tile_max": 0.0, "edge_within1": 1.0}
    moved = Document.from_dict(doc.to_dict())
    moved.find("card").box = Box(26, 24, 120, 60)
    worse = bench.fidelity_metrics(target, render_doc(moved), doc)
    assert worse["jnd_frac_nontext"] > 0.01 and worse["chamfer_tile_max"] > 1.0 and worse["edge_within1"] < 0.9
    # text pixels are masked out of the JND fraction (fonts are tracked separately)
    retext = Document.from_dict(doc.to_dict())
    retext.find("t").text = "Hellp"
    assert bench.fidelity_metrics(target, render_doc(retext), doc)["jnd_frac_nontext"] == 0.0


def test_bench_report_has_fidelity_and_metrics_version(report):
    assert report["metrics_version"] == bench.METRICS_VERSION >= 2
    for r in report["cases"]:
        for k in FIDELITY:
            assert r["metrics"][k] is not None
        assert r["timing"]["fidelity_s"] > 0
    for k in FIDELITY:
        assert report["metrics"][k]["n"] == 4
    assert set(report["norms"]) == {m.name for m in METRICS if m.norm}


def test_bench_never_silently_caps(tmp_path, report):
    # limit: the cases beyond it are reported, not just skipped
    assert report["n_dropped"] == len(report["dropped"]) > 0
    assert all(d["reason"].startswith("over limit") for d in report["dropped"])
    saved = json.load(open(os.path.join(report["out_dir"], "report.json")))
    assert saved["n_dropped"] == report["n_dropped"] and saved["errors"] == []
    # broken corpus: listed id without png, gt file missing from the manifest, missing dir
    src = os.path.join(CORPORA[0], "synth_1_000")
    d = tmp_path / "corp"
    d.mkdir()
    for ext in (".png", ".gt.json"):
        (d / f"ok{ext}").write_bytes(open(src + ext, "rb").read())
    (d / "nopng.gt.json").write_bytes(open(src + ".gt.json", "rb").read())
    (d / "unlisted.gt.json").write_bytes(open(src + ".gt.json", "rb").read())
    (d / "manifest.json").write_text(json.dumps({"cases": [{"id": "ok"}, {"id": "nopng"}]}))
    cases, dropped = bench.scan_cases([str(d), str(tmp_path / "nope")])
    assert [c[1] for c in cases] == ["ok"]
    reasons = {x["id"]: x["reason"] for x in dropped}
    assert "missing nopng.png" in reasons["nopng"] and "not listed" in reasons["unlisted"]
    assert any(x["id"] is None and "not found" in x["reason"] for x in dropped)
    # errored cases are listed in the report (and score 0)
    rep = bench.run(cases=[(str(d), "ok"), (str(d), "ghost")], stages=(), out_root=str(tmp_path / "o"), run_id="e")
    assert rep["n_errors"] == 1 and rep["errors"][0]["id"] == "ghost" and rep["errors"][0]["stage"] == "load"
    assert json.load(open(os.path.join(rep["out_dir"], "report.json")))["errors"] == rep["errors"]


def test_gate_thresholds_and_delta_table(report, tmp_path):
    path = str(tmp_path / "baseline.json")
    gate, text = bench.gate_and_baseline(report, None, path)
    assert gate is None and "baseline written" in text
    base = bench.load_baseline(path)
    assert {"metrics_version", "stages", "n_cases", "composite", "goodness", "metrics", "cases", "weights"} <= set(base)
    assert base["goodness"]["node_recall"] == pytest.approx(report["metrics"]["node_recall"]["goodness"])
    gate, text = bench.gate_and_baseline(report, path)
    assert gate["pass"] and "regression gate: PASS" in text and "node_recall" in text and "delta" in text

    def variant(d_comp=0.0, metric=None, d_metric=0.0):
        r = json.loads(json.dumps(report))
        r["composite"] += d_comp
        if metric:
            r["metrics"][metric]["goodness"] += d_metric
        return r

    assert bench.check_regression(variant(-0.002), base) == []           # within bench.gate.eps
    assert bench.check_regression(variant(-0.004), base)[0].startswith("composite")
    assert bench.check_regression(variant(0, "mean_iou", -0.009), base) == []   # within metric_eps
    fails = bench.check_regression(variant(0, "mean_iou", -0.011), base)
    assert len(fails) == 1 and fails[0].startswith("mean_iou")
    table = bench.format_gate_table(bench.gate_table(variant(0, "mean_iou", -0.011), base))
    assert "REGRESSED" in table and "-0.0110" in table
    old = dict(base, metrics_version=1)
    assert any("metrics_version" in f for f in bench.check_regression(report, old))
    gate, text = bench.gate_and_baseline(report, str(tmp_path / "missing.json"))
    assert not gate["pass"] and "no baseline" in text


def test_cli_bench_gate_exit_codes(tmp_path, capsys):
    from dt.cli import main as dt_main
    path = str(tmp_path / "baseline.json")
    args = ["bench", "--corpus", CORPORA[0], "--limit", "1", "--stages", "map", "--out", "", "--json"]
    assert dt_main(args + ["--set-baseline", path]) == 0
    assert os.path.exists(path)
    capsys.readouterr()
    assert dt_main(args + ["--gate", path]) == 0
    out = capsys.readouterr()
    assert json.loads(out.out)["gate"]["pass"] is True and "regression gate: PASS" in out.err
    base = json.load(open(path))
    base["composite"] += 0.01
    json.dump(base, open(path, "w"))
    assert dt_main(args + ["--gate", path]) == 1
    out = capsys.readouterr()
    assert "composite" in out.err and "REGRESSED" in out.err and "FAIL" in out.err


def test_tracked_baseline_matches_current_definitions():
    base = bench.load_baseline()
    assert base is not None, "knowledge/baseline.json must ship with the repo (dt bench --set-baseline)"
    assert bench.BASELINE_PATH.endswith(os.path.join("knowledge", "baseline.json"))
    assert base["metrics_version"] == bench.METRICS_VERSION
    assert base["stages"] == list(bench.ALL_STAGES) and base["n_cases"] == len(bench.find_cases(CORPORA))
    assert set(base["goodness"]) == {m.name for m in METRICS}
    assert base["weights"] == {m.name: float(P[m.weight_key]) for m in METRICS}
    from dt.selftest.metrics import definition_params
    assert base["definitions"] == definition_params()


# --------------------------------------------------------------------------- metrics-gaming (skeptic lens)
def _uri(rgb):
    import base64
    import io

    import numpy as np
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb[..., :3])).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_adversarial_outputs_never_beat_the_gt_ir():
    """Outputs that are visually or structurally wrong must score below the gt IR itself:
    a screenshot crop over invisible gt-shaped nodes scored 0.999 (> gt IR 0.976) and a 48x48
    canvas scored 1.0 before the render was conformed to the target frame."""
    from dt.common.image import load_rgb
    from dt.ir import Box, Color, Document, Node
    d = bench.DEFAULT_CORPORA[0]
    gt, png = bench.load_case(d, "synth_1_001")
    target = load_rgb(png)
    W, H = gt.width, gt.height

    def oracle():
        o = Document.from_dict(gt.to_dict())
        for n in o.walk():
            n.component, n.tokens = None, {}
        return o

    def score(doc):
        r = bench.run_case(d, "synth_1_001", stages=("render",), prediction=doc)
        assert r["error"] is None, r["error"]
        return r["composite"], r["metrics"]

    c_gt, m_gt = score(oracle())
    assert m_gt["raster_frac"] == 0.0 and m_gt["size_match"] is True and c_gt > 0.95

    shot = Node(id="shot", type="image", box=Box(0, 0, W, H), image_ref=_uri(target))
    over = oracle()  # gt structure that paints nothing + a full screenshot on top
    for n in over.walk():
        n.fills, n.strokes, n.effects = [], [], []
        if n.type == "text" and n.text_style is not None:
            n.text_style.color = Color(0, 0, 0, 0.0)
    over.root.children.append(shot)
    c_over, m_over = score(over)
    assert m_over["mean_de"] == pytest.approx(0.0, abs=0.05) and m_over["raster_frac"] == pytest.approx(1.0)
    assert m_over["text_recall"] == 0.0 and c_over == 0.0

    full = Document.blank(W, H, "#ffffff")
    full.root.children = [Node.from_dict(shot.to_dict())]
    assert score(full)[0] == 0.0

    small = oracle()
    small.width, small.height = 48, 48
    c_small, m_small = score(small)
    assert m_small["size_match"] is False and m_small["edge_within1"] < 0.2
    assert c_small < c_gt - 0.2
    big = oracle()  # right pixels on a canvas 1.5x too large: pixels alone cannot see the extra area
    big.width, big.height = int(W * 1.5), int(H * 1.5)
    c_big, m_big = score(big)
    assert m_big["size_match"] is False and m_big["frame_iou"] == pytest.approx(1 / 2.25)
    assert c_big < c_gt - 0.2

    shifted = oracle()  # 2px everywhere: worse than the gt, still a recognisable translation
    for n in shifted.walk():
        if n is not shifted.root:
            n.box = Box(n.box.x + 2, n.box.y + 2, n.box.w, n.box.h)
    c_shift, m_shift = score(shifted)
    assert c_small < c_shift < c_gt and m_shift["edge_within1"] < m_gt["edge_within1"]


def test_gate_rejects_changed_definitions_and_one_case_collapsing():
    from dt.selftest.metrics import definition_params
    rows = [{"id": f"c{i}", "corpus": "x", "composite": 0.8, "metrics": {}} for i in range(4)]
    base = bench.baseline_dict(bench.build_report(rows, run_id="b", stages=bench.ALL_STAGES))
    assert base["definitions"] == definition_params()
    same = bench.build_report(rows, run_id="s", stages=bench.ALL_STAGES)
    assert bench.check_regression(same, base) == []
    # a looser IoU threshold / normaliser inflates goodness without any change in the output
    with bench.param_overrides({"compare.struct.iou_thr": 0.2, "bench.norm.chamfer_tile_max": 256.0}):
        loose = bench.build_report(rows, run_id="l", stages=bench.ALL_STAGES)
    fails = bench.check_regression(loose, base)
    assert any("metric-definition params" in f and "compare.struct.iou_thr" in f for f in fails)
    assert any("metric normalisers" in f for f in fails)
    # one case collapses while the others improve: the run composite still passes the eps gate
    moved = [dict(r, composite=c) for r, c in zip(rows, (0.70, 0.84, 0.84, 0.84))]
    rep = bench.build_report(moved, run_id="m", stages=bench.ALL_STAGES)
    assert rep["composite"] >= base["composite"] - P["bench.gate.eps"]
    fails = bench.check_regression(rep, base)
    assert len(fails) == 1 and fails[0].startswith("case x/c0"), fails


def test_gate_catches_a_deliberate_perceive_regression():
    """End to end: a bad perceive param (segmentation drops every region under 5000 px) must
    fail the regression gate against a baseline of the same case."""
    cases = [(bench.DEFAULT_CORPORA[0], "synth_1_000")]
    good = bench.run(cases=cases, out_root=None)
    base = bench.baseline_dict(good)
    assert bench.check_regression(bench.run(cases=cases, out_root=None), base) == []  # deterministic
    bad = bench.run(cases=cases, out_root=None, overrides={"perceive.seg.min_area": 5000})
    fails = bench.check_regression(bad, base)
    assert bad["composite"] < good["composite"] - 0.05
    assert any(f.startswith("composite") for f in fails) and any(f.startswith("node_recall") for f in fails)
    assert any(f.startswith("case synth/synth_1_000") for f in fails)
