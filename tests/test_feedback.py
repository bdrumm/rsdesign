"""User-informed tuning (dt/feedback): every capture channel, learning, evaluation, privacy and promotion.

Runs are REAL translate runs (perceive -> map -> render, real renderer) of corpus screenshots; the Figma channel
runs the real plugin code on fake_figma.js; review.html is driven in the real browser with Playwright.
Every test gets its own DT_HOME, so nothing touches the user's ~/.rsdesign.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import socket
import zipfile
from contextlib import redirect_stderr, redirect_stdout

import pytest

from dt import cli
from dt.feedback import ledger
from dt.feedback.capture import add_corrections, apply_corrections, log_decisions
from dt.feedback.store import list_bundles, load_bundle, run_info, validate_bundle
from dt.ir import Document
from dt.params import P

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SYNTH = os.path.join(ROOT, "fixtures", "corpus", "synth")
MWC = os.path.join(ROOT, "fixtures", "corpus", "mwc")
# three screens with outlined label-only chips (same learned signature), one with nothing but other content
SCREENS = {"a": os.path.join(SYNTH, "synth_1_001.png"), "b": os.path.join(SYNTH, "synth_1_005.png"),
           "c": os.path.join(MWC, "mwc_1_010.png")}


def _cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Translate runs (no refine, no validation: fast) of the three screens, made with an empty DT_HOME."""
    from dt.pipeline import translate
    base = tmp_path_factory.mktemp("fb_runs")
    old = os.environ.get("DT_HOME")
    os.environ["DT_HOME"] = str(base / "home0")
    try:
        out = {}
        for k, png in SCREENS.items():
            d = str(base / k)
            m = translate(png, ds="material3", out_dir=d, refine_iters=0, validate=False)
            assert m["ok"] and not m["errors"], m["errors"]
            out[k] = d
        return out
    finally:
        if old is None:
            os.environ.pop("DT_HOME", None)
        else:
            os.environ["DT_HOME"] = old


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "dt_home"
    monkeypatch.setenv("DT_HOME", str(h))
    return str(h)


@pytest.fixture
def run_copy(runs, tmp_path):
    def make(k: str) -> str:
        d = str(tmp_path / f"run_{k}")
        shutil.copytree(runs[k], d)
        return d
    return make


def _doc(run_dir: str) -> Document:
    return Document.load(os.path.join(run_dir, "ir.mapped.json"))


def _chips(doc: Document) -> list:
    """Label-only chips (the matcher calls them suggestion chips); chips with a leading icon are another signature."""
    return [n for n in doc.walk() if n.component is not None and n.component.name == "Chip"
            and n.component.variant.get("type") == "suggestion"]


def _text_node(doc: Document):
    return next(n for n in doc.walk() if n.type == "text" and n.text)


# --------------------------------------------------------------------------- schema / ledger
def test_ledger_entries_conform_to_the_shared_schema(tmp_path):
    e = ledger.make_entry("rule", "local", "feedback", ["fb-0123456789ab"], {"rules": {"added": []}},
                          {"item_acc": 0.5}, {"item_acc": 1.0}, {"user_corpus": True, "bench": True}, True)
    assert ledger.validate_entry(e) == [] and e["id"].startswith("rule-") and e["reverts"] is None
    p = ledger.append(e, str(tmp_path / "ledger.jsonl"))
    assert ledger.read(p) == [json.loads(json.dumps(e, sort_keys=True))]
    bad = dict(e, kind="nope", scope="team")
    assert len(ledger.validate_entry(bad)) >= 2
    assert ledger.validate_entry(dict(e, evidence={"before": {}, "after": {}, "gates": {"bench": "yes"}}))
    with pytest.raises(ValueError):
        ledger.make_entry("rule", "local", "user", [], {}, {}, {}, {}, True)


# --------------------------------------------------------------------------- channel (a): CLI corrections file
def test_add_corrections_file_via_cli(runs, run_copy, home, tmp_path):
    rd = run_copy("a")
    doc = _doc(rd)
    chip, text = _chips(doc)[0], _text_node(doc)
    corr = {"items": [
        {"kind": "component", "node_id": chip.id, "value": {"name": "Button", "variant": {"style": "outlined"}}},
        {"kind": "text", "node_id": text.id, "value": "Undo it", "note": "typo"},
        {"kind": "missing", "box": [370, 168, 24, 24], "value": {"type": "icon", "icon_name": "more_vert"}},
        {"kind": "ok", "node_id": _chips(doc)[1].id},
    ], "run_rating": 4}
    cp = tmp_path / "corrections.json"
    cp.write_text(json.dumps(corr))
    rc, out, err = _cli(["feedback", "add", rd, str(cp), "--json"])
    assert rc == 0, err
    b = json.loads(out)
    assert validate_bundle(b) == [] and b["channel"] == "cli" and b["run_rating"] == 4
    assert b["run"]["screenshot_sha256"] == run_info(rd)["screenshot_sha256"]
    assert b["consent"] == {"store_screenshot": True, "share_screenshot": False, "share_text": False}  # private by default
    d = os.path.join(home, "feedback", b["id"])
    assert sorted(os.listdir(d)) == ["bundle.json", "ir.base.json", "ir.json", "target.png"]
    comp = b["items"][0]
    assert comp["context"]["signature_hash"] and comp["context"]["component"]["name"] == "Chip"
    assert comp["box"] == comp["context"]["box"]
    gt = Document.load(os.path.join(d, "ir.json"))
    assert gt.find(chip.id).component.name == "Button" and gt.find(text.id).text == "Undo it"
    assert any(n.type == "icon" and n.icon_name == "more_vert" for n in gt.walk())
    assert not any(n.type == "instance" for n in gt.walk())  # gt convention of the corpora
    # corrections recorded on another screenshot are refused; malformed items are a clean exit 1
    other = dict(corr, run={"screenshot_sha256": "0" * 64})
    cp.write_text(json.dumps(other))
    rc, _, err = _cli(["feedback", "add", rd, str(cp)])
    assert rc == 1 and "different screenshot" in err
    cp.write_text(json.dumps({"items": [{"kind": "colour", "node_id": chip.id}]}))
    assert _cli(["feedback", "add", rd, str(cp)])[0] == 1
    # without consent to store the screenshot, no copy is kept
    cp.write_text(json.dumps({"items": [{"kind": "extra", "node_id": text.id}]}))
    rc, out, _ = _cli(["feedback", "add", rd, str(cp), "--no-store-screenshot", "--json"])
    b2 = json.loads(out)
    assert rc == 0 and b2["files"]["target"] is None
    assert not os.path.exists(os.path.join(home, "feedback", b2["id"], "target.png"))
    assert len(list_bundles()) == 2


def test_apply_corrections_every_kind(runs):
    doc = _doc(runs["a"])
    chip, text = _chips(doc)[0], _text_node(doc)
    frame = next(n for n in doc.walk() if n.type != "text" and n is not doc.root and n.fill_color is not None)
    items = [{"id": "i1", "kind": "geometry", "node_id": chip.id, "value": {"box": [20, 70, 89, 32]}},
             {"id": "i2", "kind": "color", "node_id": frame.id, "value": {"color": "#ff0000"}},
             {"id": "i3", "kind": "should_be_image", "node_id": _chips(doc)[1].id, "value": None},
             {"id": "i4", "kind": "extra", "node_id": text.id, "value": None},
             {"id": "i5", "kind": "icon", "node_id": "nope", "value": {"icon_name": "x"}}]
    out, rep = apply_corrections(doc, items)
    assert rep["applied"] == ["i1", "i2", "i3", "i4"] and rep["skipped"][0]["id"] == "i5"
    moved = out.find(chip.id)
    assert (moved.box.x, moved.box.y) == (20, 70)
    kid0, kid1 = doc.find(chip.id).children[0], moved.children[0]
    assert (kid1.box.x - kid0.box.x, kid1.box.y - kid0.box.y) == pytest.approx((20 - chip.box.x, 70 - chip.box.y))
    assert out.find(frame.id).fill_color.hex() == "#ff0000"
    assert out.find(_chips(doc)[1].id).type == "image" and out.find(text.id) is None


# --------------------------------------------------------------------------- channel (b): decisions
def test_apply_decisions_logs_feedback(run_copy, home):
    from dt.pipeline import apply_decisions
    rd = run_copy("a")
    decisions = json.load(open(os.path.join(rd, "decisions.json")))
    comp = [d for d in decisions if d["kind"] == "component"]
    assert len(comp) >= 2
    res = apply_decisions(rd, {comp[0]["id"]: {"name": "Card", "variant": {"style": "filled"}}})
    assert res["applied"] == [comp[0]["id"]] and res["feedback_bundle"], res
    b = load_bundle(res["feedback_bundle"])
    assert b["channel"] == "decisions" and len(b["items"]) == 1
    it = b["items"][0]
    assert it["value"] == {"name": "Card", "variant": {"style": "filled"}} and it["note"] == f"decision {comp[0]['id']}"
    # the context is what the model saw *before* the answer (not the answered state)
    assert it["context"]["component"] is None or it["context"]["component"]["name"] != "Card" or comp[0]["current"]
    # a second answer extends the same bundle; re-answering a node replaces its item
    res2 = apply_decisions(rd, {comp[1]["id"]: {"name": None}, comp[0]["id"]: {"name": "ListItem"}})
    assert res2["feedback_bundle"] == res["feedback_bundle"]
    b = load_bundle(res["feedback_bundle"])
    assert sorted(i["value"]["name"] or "-" for i in b["items"]) == ["-", "ListItem"]
    # switched off -> nothing logged
    P.set("feedback.log_decisions", False)
    try:
        res3 = apply_decisions(rd, {comp[0]["id"]: {"name": "Chip"}})
        assert "feedback_bundle" not in res3
        assert len(load_bundle(res["feedback_bundle"])["items"]) == 2
    finally:
        P.reset("feedback.log_decisions")


# --------------------------------------------------------------------------- channel (c): review.html
def test_review_html_click_correct_and_download(run_copy, home, tmp_path):
    from dt.render.screenshot import _brw, _ensure_browser
    rd = run_copy("a")
    page_path = os.path.join(rd, "review.html")
    assert os.path.exists(page_path)  # written by translate
    html = open(page_path).read()
    assert "http://" not in html.replace("http://www.w3.org", "") and "https://" not in html  # self-contained, offline
    doc = _doc(rd)
    chip, text = _chips(doc)[0], next(n for n in doc.walk() if n.type == "text" and n.text and n.box.w > 30)
    _ensure_browser()
    ctx = _brw().new_context(accept_downloads=True, viewport={"width": 1280, "height": 900})
    try:
        page = ctx.new_page()
        page.goto("file://" + page_path)
        page.click(f'#target-wrap [data-node-id="{text.id}"]')
        assert page.input_value("#kind") == "text"
        page.fill("#v-text", "Corrected label")
        page.click("#add")
        page.click(f'#target-wrap [data-node-id="{chip.id}"]', position={"x": 2, "y": 2})
        page.select_option("#kind", "component")
        page.select_option("#v-component", "Button")
        page.fill("#v-variant", "style=outlined")
        page.click("#add")
        # draw a box around something we missed (image coords 300..340 x 120..140)
        page.click("#mode-missing")
        bb = page.locator("#target-wrap").bounding_box()
        s = bb["width"] / doc.width
        page.mouse.move(bb["x"] + 300 * s, bb["y"] + 120 * s)
        page.mouse.down()
        page.mouse.move(bb["x"] + 320 * s, bb["y"] + 130 * s)
        page.mouse.move(bb["x"] + 340 * s, bb["y"] + 140 * s)
        page.mouse.up()
        assert page.input_value("#kind") == "missing"
        page.select_option("#v-type", "text")
        page.fill("#v-text", "Archive")
        page.click("#add")
        page.click("#thumbs-down")
        page.check("#c-share-text")
        assert page.inner_text("#count") == "3 corrections"
        with page.expect_download() as dl:
            page.click("#download")
        assert dl.value.suggested_filename == "corrections.json"
        dest = str(tmp_path / "corrections.json")
        dl.value.save_as(dest)
    finally:
        ctx.close()
    data = json.load(open(dest))
    assert data["schema"] == "rsdesign.corrections/1" and data["run_rating"] == 1
    assert data["consent"] == {"store_screenshot": True, "share_screenshot": False, "share_text": True}
    kinds = {i["kind"]: i for i in data["items"]}
    assert kinds["text"]["node_id"] == text.id and kinds["text"]["value"] == {"text": "Corrected label"}
    assert kinds["component"]["value"] == {"name": "Button", "variant": {"style": "outlined"}}
    mb = kinds["missing"]["box"]
    assert mb == pytest.approx([300, 120, 40, 20], abs=2.0)
    b = add_corrections(rd, dest)
    assert b["channel"] == "review" and len(b["items"]) == 3 and b["consent"]["share_text"] is True
    assert b["apply_report"]["skipped"] == []


# --------------------------------------------------------------------------- channel (d): Figma
def test_figma_export_corrections_round_trip(run_copy, home):
    from dt.feedback.figma import diff_export, from_figma, run_session
    rd = run_copy("a")
    doc = _doc(rd)
    plan = json.load(open(os.path.join(rd, "figma_plan.json")))
    # untouched frame: nothing to report (auto-layout / fallback paint are not edits)
    out = run_session(plan, [])
    assert out["export"]["schema"] == "rsdesign.figma-corrections/1" and diff_export(doc, out["export"], plan) == []
    chip = _chips(doc)[0]
    button = next(n for n in doc.walk() if n.component is not None and n.component.name == "Button")
    btn_text = next(c for c in button.walk() if c.type == "text")
    list_text = next(n for n in doc.walk() if n.type == "text" and n.text and doc.parent_of(n.id) is not doc.root
                     and doc.parent_of(n.id).component is not None and doc.parent_of(n.id).component.name == "ListItem")
    removable = next(n for n in doc.walk() if n.type == "rect" and doc.parent_of(n.id) is not doc.root)
    keys = {"material3.button": "Button", "material3.chip": "Chip"}
    edits = [
        {"op": "swap", "dtId": button.id, "componentKey": "material3.chip"},     # "this is a chip, not a button"
        {"op": "set", "dtId": list_text.id, "key": "characters", "value": "Renamed in Figma"},
        {"op": "remove", "dtId": removable.id},
        {"op": "set", "dtId": chip.id, "key": "x", "value": chip.box.x + 10},
        {"op": "add", "type": "RECTANGLE", "x": 300, "y": 600,  # on the screen frame itself
         "width": 50, "height": 20, "fills": [{"type": "SOLID", "color": {"r": 1, "g": 0, "b": 0}, "opacity": 1}]},
    ]
    out = run_session(plan, edits, components=list(keys), component_names=keys)
    exp = out["export"]
    assert exp["frames"][0]["planIds"] and out["summary"]["frames"] == 1
    items = diff_export(doc, exp, plan)
    by = {(i["kind"], i.get("node_id")): i for i in items}
    assert by[("component", button.id)]["value"]["name"] == "Chip"
    assert by[("text", list_text.id)]["value"] == {"text": "Renamed in Figma"}
    assert ("extra", removable.id) in by
    assert by[("geometry", chip.id)]["value"]["box"][0] == pytest.approx(chip.box.x + 10)
    miss = [i for i in items if i["kind"] == "missing"]
    assert len(miss) == 1 and miss[0]["box"] == [300.0, 600.0, 50.0, 20.0] and miss[0]["value"]["fill"] == "#ff0000"
    # the button's own label lives inside a library instance: it is not reported as deleted
    assert ("extra", btn_text.id) not in by
    assert not any(k == "geometry" and nid != chip.id for k, nid in by)  # children of a moved node did not "move"
    ep = os.path.join(rd, "figma_corrections.json")
    json.dump(exp, open(ep, "w"))
    rc, o, err = _cli(["feedback", "from-figma", rd, ep, "--json"])
    assert rc == 0, err
    b = json.loads(o)
    assert b["channel"] == "figma" and len(b["items"]) == len(items) and b["apply_report"]["skipped"] == []
    with pytest.raises(ValueError):
        from_figma(rd, {"schema": "something-else"})


# --------------------------------------------------------------------------- learning
def _correct_chips_to_filter(run_dir: str, n: int = 1, consent=None) -> dict:
    """An unselected filter chip and a suggestion chip are pixel-identical (the matcher scores both 1.0 and picks
    suggestion); a consumer whose app uses filter chips says so. Exactly the knowledge only users have."""
    doc = _doc(run_dir)
    items = [{"kind": "component", "node_id": c.id, "value": {"name": "Chip", "variant": {"type": "filter"}}} for c in _chips(doc)[:n]]
    return add_corrections(run_dir, {"items": items}, consent=consent)


def _is_filter_chip(n) -> bool:
    return n.component is not None and n.component.name == "Chip" and n.component.variant.get("type") == "filter"


def test_learn_rule_generalises_and_eval_history(run_copy, home):
    from dt.feedback.learn import eval_history, learn, read_rules
    from dt.mapping import map_document
    from dt.mapping.matcher import load_learned_rules
    from dt.perceive import perceive
    from dt.pipeline import resolve_design_system
    ra, rb = run_copy("a"), run_copy("b")
    # one screen is not enough evidence (support counts distinct screenshots, not items)
    _correct_chips_to_filter(ra, n=2)
    res = learn(gate=False)
    assert res["changed"] and res["accepted"] and res["active"] == 0 and res["behaviour_changed"] is False
    assert res["ledger"] is None  # evidence bookkeeping only: nothing the matcher does changed
    assert read_rules(os.path.join(home, "learned_rules.json"))[0]["support"] == 1
    # a second screenshot agrees -> the rule activates, measured on the user's corpus before / after
    _correct_chips_to_filter(rb, n=1)
    res = learn(gate=False)
    assert res["accepted"] is True and res["active"] == 1, res
    assert res["gates"] == {"user_corpus": True} and "bench (--no-gate)" in res["skipped"]
    assert res["after"]["item_acc"] > res["before"]["item_acc"]
    rules = read_rules(os.path.join(home, "learned_rules.json"))
    rule = next(r for r in rules if r["active"])
    assert rule["component"] == "Chip" and rule["variant"] == {"type": "filter", "selected": "false"}
    assert rule["support"] == 2 and rule["confidence"] == 1.0 and rule["scope"] == "local"
    entries = ledger.read(os.path.join(home, "ledger.jsonl"))
    assert [e["kind"] for e in entries] == ["rule"] and all(ledger.validate_entry(e) == [] for e in entries)
    assert entries[-1]["source"]["type"] == "feedback" and set(entries[-1]["source"]["ids"]) >= {b["id"] for b in list_bundles()}
    assert entries[-1]["change"]["rules"]["changed"][rule["id"]]["active"] == [False, True]
    assert entries[-1]["change"]["rules"]["changed"][rule["id"]]["support"] == [1, 2]
    # a NEW run of a third screen with the same signature: every outlined label chip now maps to a filter chip,
    # including chips nobody corrected (other labels, other widths) -- the rule generalises
    ds = resolve_design_system("material3")
    new = map_document(perceive(SCREENS["c"]), ds)
    hits = [n for n in new.walk() if (n.meta.get("learned_rule") or {}).get("rule") == rule["id"]]
    assert len(hits) >= 2
    assert all(_is_filter_chip(n) and n.component.evidence["learned_rule"]["mode"] == "override" for n in hits)
    assert all(n.component.confidence >= P["pipeline.decisions.component_conf"] for n in hits)  # no longer asked
    assert len({round(n.box.w) for n in hits}) >= 2
    # with learned rules switched off, mapping is exactly the shipped behaviour
    plain = map_document(perceive(SCREENS["c"]), ds, learned={})
    assert not any(n.meta.get("learned_rule") for n in plain.walk())
    assert not any(_is_filter_chip(n) for n in plain.walk())
    assert sum(1 for n in plain.walk() if n.component is not None and n.component.name == "Chip") >= len(hits)
    # accuracy on the user's own cases over time
    h1 = eval_history()
    h2 = eval_history()
    assert h1["n_cases"] == 2 and h1["item_acc"] == 1.0 and h2["previous"]["ts"] == h1["ts"]
    lines = open(os.path.join(home, "feedback", "history.jsonl")).read().strip().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["model"]["id"]
    # revert restores the rules the accepted change replaced
    from dt.feedback.learn import revert
    rv = revert(entries[-1]["id"])
    assert rv["kind"] == "revert" and rv["reverts"] == entries[-1]["id"] and ledger.validate_entry(rv) == []
    assert not any(r["active"] for r in read_rules(os.path.join(home, "learned_rules.json")))
    assert load_learned_rules() == {}


def test_conflicting_corrections_lower_confidence_and_gate_rejects(run_copy, home):
    from dt.feedback.learn import learn, read_rules
    ra, rb, rc_ = run_copy("a"), run_copy("b"), run_copy("c")
    _correct_chips_to_filter(ra)
    _correct_chips_to_filter(rb)
    # a failing global bench gate keeps the old (empty) rules but still records the attempt
    res = learn(gate=True, bench_fn=lambda ov, **k: {"pass": False, "failures": ["composite dropped"], "composite": 0.8,
                                                      "baseline_composite": 0.89, "n_cases": 24})
    assert res["accepted"] is False and res["gates"]["bench"] is False
    assert read_rules(os.path.join(home, "learned_rules.json")) == []
    e = ledger.read(os.path.join(home, "ledger.jsonl"))[-1]
    assert e["accepted"] is False and e["evidence"]["gates"]["bench"] is False and ledger.validate_entry(e) == []
    # the rejected rule stays off while its evidence is unchanged (no bench re-run, nothing new to try)
    res = learn(gate=True, bench_fn=lambda ov, **k: pytest.fail("an unchanged rejected rule must not be re-gated"))
    assert res["ledger"] is None and res["active"] == 0 and res["behaviour_changed"] is False
    # a third screen confirms its chips are suggestion chips: 2 for vs 1 against -> 0.667 < feedback.rule_min_conf
    doc = _doc(rc_)
    add_corrections(rc_, {"items": [{"kind": "ok", "node_id": _chips(doc)[0].id}]})
    res = learn(gate=False)
    rule = next(r for r in read_rules(os.path.join(home, "learned_rules.json")) if r["variant"].get("type") == "filter")
    assert rule["support"] == 2 and rule["against"] == 1 and rule["confidence"] == pytest.approx(0.667, abs=1e-3)
    assert rule["active"] is False


def test_rules_never_override_strong_evidence_against(runs):
    from dt.mapping import map_document
    from dt.mapping.matcher import learned_signature, match_node
    from dt.pipeline import resolve_design_system, unmap_document
    ds = resolve_design_system("material3")
    doc = unmap_document(_doc(runs["a"]))
    mapped = map_document(Document.from_dict(doc.to_dict()), ds, learned={})
    li = next(n for n in mapped.walk() if n.component is not None and n.component.name == "ListItem" and n.component.evidence["score"] > 0.9)
    node = doc.find(li.id)
    h, feat = learned_signature(node, ds)

    def rule(component, conf=1.0):
        return {h: [{"id": "rtest", "signature_hash": h, "active": True, "component": component, "variant": {},
                     "confidence": conf, "support": 5, "scope": "local",
                     "features": {**feat, "h_range": [feat["h"], feat["h"]], "aspect_range": [feat["aspect"], feat["aspect"]]}}]}
    # the matcher is very sure this is a list item and a Switch can never be this node: the rule is vetoed
    for comp in ("Switch", "Snackbar"):
        out = map_document(Document.from_dict(doc.to_dict()), ds, learned=rule(comp))
        n = out.find(li.id)
        assert n.meta["learned_rule"]["mode"] == "vetoed" and n.component.name == "ListItem", comp
    # 'not a component': vetoed against a confident match, applied where the engine itself is unsure
    sure = max((n for n in mapped.walk() if n.component is not None), key=lambda n: n.component.confidence)
    assert sure.component.confidence >= P["map.learned.veto_score"]
    hs, fs = learned_signature(doc.find(sure.id), ds)
    reject = {hs: [{"id": "rno", "signature_hash": hs, "active": True, "component": None, "variant": {}, "confidence": 1.0,
                    "support": 3, "scope": "local",
                    "features": {**fs, "h_range": [fs["h"], fs["h"]], "aspect_range": [fs["aspect"], fs["aspect"]]}}]}
    out = map_document(Document.from_dict(doc.to_dict()), ds, learned=reject)
    assert out.find(sure.id).meta["learned_rule"]["mode"] == "vetoed" and out.find(sure.id).component is not None
    assert li.component.confidence < P["map.learned.veto_score"]  # the list item match is a 0.9 score but unsure
    out = map_document(Document.from_dict(doc.to_dict()), ds, learned=rule(None))
    n = out.find(li.id)
    assert n.meta["learned_rule"]["mode"] == "reject" and n.component is None and "component_candidates" not in n.meta
    cands = match_node(node, ds)
    assert cands[0].spec.name == "ListItem"
    # a rule below map.learned.override_conf only boosts its candidate; here that is enough to pick the variant
    chip = _chips(_doc(runs["a"]))[0]
    hc, fc = learned_signature(doc.find(chip.id), ds)
    weak = {hc: [{"id": "rweak", "signature_hash": hc, "active": True, "component": "Chip", "variant": {"type": "filter"},
                  "confidence": 0.7, "support": 2, "scope": "local",
                  "features": {**fc, "h_range": [fc["h"], fc["h"]], "aspect_range": [fc["aspect"], fc["aspect"]]}}]}
    out = map_document(Document.from_dict(doc.to_dict()), ds, learned=weak)
    n = out.find(chip.id)
    assert n.meta["learned_rule"]["mode"] == "boost" and n.component.variant["type"] == "filter"
    assert n.component.evidence["candidates"][0]["variant"]["type"] == "filter"
    # unknown components are ignored, not invented
    out = map_document(Document.from_dict(doc.to_dict()), ds, learned=rule("NoSuchWidget"))
    assert out.find(li.id).meta["learned_rule"]["mode"] == "unknown_component"
    # a rule whose size range does not cover the node does not match at all
    far = rule("Card")
    far[h][0]["features"]["h_range"] = [feat["h"] + 40, feat["h"] + 60]
    out = map_document(Document.from_dict(doc.to_dict()), ds, learned=far)
    assert "learned_rule" not in out.find(li.id).meta


# --------------------------------------------------------------------------- promotion: export / import
def _no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("feedback code must not touch the network")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_export_redaction_and_consent(run_copy, home, tmp_path, monkeypatch):
    from dt.feedback.share import export_bundles, redact_text
    assert redact_text("Inbox 12, Q3 — Ünïcode") == "Xxxxx 00, X0 — Xxxxxxx"
    ra, rb = run_copy("a"), run_copy("b")
    doc = _doc(ra)
    text = _text_node(doc)
    private = add_corrections(ra, {"items": [{"kind": "text", "node_id": text.id, "value": "Secret plan", "note": "my note"}]})
    shared = add_corrections(rb, {"items": [{"kind": "ok", "node_id": _chips(_doc(rb))[0].id, "note": "fine"}]},
                             consent={"share_screenshot": True, "share_text": True})
    _no_network(monkeypatch)
    res = export_bundles(str(tmp_path / "all.zip"))
    rows = {r["id"]: r for r in res["bundles"]}
    assert rows[private["id"]] == {"id": private["id"], "items": 1, "text": "redacted", "screenshot": False,
                                   "screenshot_excluded": "no share_screenshot consent"}
    assert rows[shared["id"]]["text"] == "shared" and rows[shared["id"]]["screenshot"] is True
    z = zipfile.ZipFile(tmp_path / "all.zip")
    names = set(z.namelist())
    assert f"bundles/{private['id']}/target.png" not in names and f"bundles/{shared['id']}/target.png" in names
    pb = json.loads(z.read(f"bundles/{private['id']}/bundle.json"))
    blob = z.read(f"bundles/{private['id']}/bundle.json").decode() + z.read(f"bundles/{private['id']}/ir.json").decode()
    assert "Secret plan" not in blob and "my note" not in blob and text.text not in blob
    assert "source_path" not in pb["run"] and "run_dir" not in pb["run"] and ra not in blob
    assert pb["items"][0]["value"]["text"] == "Xxxxxx xxxx" and pb["export"]["text"] == "redacted"
    # geometry and text metrics are kept
    rir = Document.from_dict(json.loads(z.read(f"bundles/{private['id']}/ir.json")))
    orig = Document.load(os.path.join(home, "feedback", private["id"], "ir.json"))
    for n in orig.walk():
        r = rir.find(n.id)
        assert r is not None and r.box == n.box
        if n.type == "text":
            assert len(r.text) == len(n.text) and (r.text_style.size, r.text_style.weight) == (n.text_style.size, n.text_style.weight)
    # flags only ever make an export more private
    res = export_bundles(str(tmp_path / "red.zip"), redact=True)
    assert all(r["text"] == "redacted" and not r["screenshot"] for r in res["bundles"])
    res = export_bundles(str(tmp_path / "noshot.zip"), no_screenshots=True)
    assert {r["id"]: r["screenshot"] for r in res["bundles"]} == {private["id"]: False, shared["id"]: False}
    rc, out, err = _cli(["feedback", "export", "-o", str(tmp_path / "cli.zip"), "--redact-text", "--json"])
    assert rc == 0 and len(json.loads(out)["bundles"]) == 2, err


def test_import_consent_gating_and_promotion(run_copy, home, tmp_path, monkeypatch):
    from dt.feedback.share import export_bundles, import_bundle, promote
    ra, rb, rc_ = run_copy("a"), run_copy("b"), run_copy("c")
    yes = {"share_screenshot": True, "share_text": True}
    b1 = _correct_chips_to_filter(ra, consent=yes)
    b2 = _correct_chips_to_filter(rb)                       # private: corrections only, no pixels, no text
    b3 = _correct_chips_to_filter(rc_, consent={"share_screenshot": True})  # screenshot but not text: no seed
    zp = str(tmp_path / "contrib.zip")
    export_bundles(zp)
    with zipfile.ZipFile(zp, "a") as z:  # a hostile member is never extracted
        z.writestr("../../evil.py", "print('x')")
    sdir, kdir = str(tmp_path / "scenarios" / "user_reported"), str(tmp_path / "knowledge")
    _no_network(monkeypatch)
    res = import_bundle(zp, scenarios_dir=sdir, knowledge_dir=kdir)
    assert sorted(res["imported"]) == sorted([b1["id"], b2["id"], b3["id"]]) and res["rejected"] == []
    assert res["seeds"] == [b1["id"]]
    skipped = {s["id"]: s["reason"] for s in res["seed_skipped"]}
    assert set(skipped) == {b2["id"], b3["id"]}
    assert sorted(os.listdir(sdir)) == sorted([b1["id"] + ".png", b1["id"] + ".gt.json", b1["id"] + ".feedback.json", "manifest.json"])
    assert not os.path.exists(str(tmp_path / "evil.py"))
    man = json.load(open(os.path.join(sdir, "manifest.json")))
    assert man["ids"] == [b1["id"]] and man["family"] == "user_reported"
    gt = Document.load(os.path.join(sdir, b1["id"] + ".gt.json"))
    assert any(_is_filter_chip(n) for n in gt.walk())
    cands = json.load(open(os.path.join(kdir, "feedback_candidates.json")))
    assert len(cands["observations"]) == 3 and res["eligible"] == 1  # 3 screens >= feedback.global_min_support
    assert not any("text" in json.dumps(o["features"]).lower() and o["features"].get("text") for o in cands["observations"])
    # re-import is idempotent
    assert import_bundle(zp, scenarios_dir=sdir, knowledge_dir=kdir)["observations_added"] == 0
    # promotion is always gated; a failing bench keeps the shared rules unchanged
    assert promote(sdir, kdir, gate=False)["promoted"] is False
    fail = lambda ov, **k: {"pass": False, "failures": ["x"], "composite": 0.85, "baseline_composite": 0.8931, "n_cases": 24}
    r = promote(sdir, kdir, bench_fn=fail)
    assert r["promoted"] is False and not os.path.exists(os.path.join(kdir, "learned_rules.json"))
    seen = {}

    def ok(ov, **k):
        seen.update(ov)
        return {"pass": True, "failures": [], "composite": 0.8931, "baseline_composite": 0.8931, "n_cases": 24}
    r = promote(sdir, kdir, bench_fn=ok)
    assert r["promoted"] is True and r["gates"]["bench"] is True and r["gates"]["scenarios"] is True
    assert seen["map.learned_rules"].endswith("learned_rules.json.candidate.json")  # gate ran on the candidate set only
    rules = json.load(open(os.path.join(kdir, "learned_rules.json")))["rules"]
    assert len(rules) == 1 and rules[0]["scope"] == "global" and rules[0]["component"] == "Chip" and rules[0]["support"] == 3
    entries = ledger.read(os.path.join(kdir, "ledger.jsonl"))
    assert [(e["kind"], e["scope"], e["accepted"]) for e in entries] == [("feedback_promotion", "global", False),
                                                                          ("feedback_promotion", "global", True)]
    assert all(ledger.validate_entry(e) == [] for e in entries)
    assert set(entries[-1]["source"]["ids"]) == {b1["id"], b2["id"], b3["id"]}
    # nothing new to promote afterwards
    assert promote(sdir, kdir, bench_fn=ok)["promoted"] is False


# --------------------------------------------------------------------------- MCP + status
def test_mcp_feedback_tools(run_copy, home):
    pytest.importorskip("mcp")
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from dt import mcp_server as S

    def call(name, args):
        with ThreadPoolExecutor(1) as ex:
            res = ex.submit(asyncio.run, S.server.call_tool(name, args)).result()
        if isinstance(res, tuple):
            res = res[-1] if isinstance(res[-1], dict) else res[0]
        if isinstance(res, dict):
            return res.get("result", res)
        if isinstance(res, list):
            return json.loads(res[0].text)
        return getattr(res, "structured_content", None) or json.loads(res.content[0].text)
    rd = run_copy("a")
    chip = _chips(_doc(rd))[0]
    out = call("submit_feedback", {"run_dir": rd, "items": [{"kind": "component", "node_id": chip.id, "value": {"name": "Button"}}],
                                   "rating": 3})
    assert out["items"] == 1 and out["applied"] == 1 and out["consent"]["share_text"] is False
    st = call("feedback_status", {})
    assert st["bundles"] == 1 and st["by_kind"] == {"component": 1} and st["by_channel"] == {"mcp": 1} and st["ratings"] == [3]
    rc, o, _ = _cli(["feedback", "status", "--json"])
    assert rc == 0 and json.loads(o)["bundles"] == 1
