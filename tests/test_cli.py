"""Tests for dt.cli and dt.pipeline (module G).

The end-to-end test runs the REAL pipeline (perceive -> map -> render -> refine -> export) on one
synth corpus screenshot and checks the output contract; the CLI tests exercise argument parsing,
JSON output and exit codes through ``dt.cli.main`` in-process (no subprocess, same interpreter).
"""
from __future__ import annotations

import io
import json
import os
import shutil
from contextlib import redirect_stderr, redirect_stdout

import pytest

from dt import cli
from dt.ir import Document
from dt.params import P
from dt.pipeline import OUTPUT_FILES, build_decisions, components_found, resolve_design_system, unmap_document
from dt.selftest import corpus_dir

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out", "test_cli")
SYNTH = corpus_dir("synth")

SUBCOMMANDS = ("perceive", "map", "render", "compare", "refine", "export", "export-figma", "validate", "translate",
               "apply-decisions", "corpus", "bench", "tune", "improve", "ds", "params")


def _run(argv: list[str]) -> tuple[int, str, str]:
    """Invoke the CLI in-process; returns (exit_code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


def _first_synth_png() -> str:
    ids = json.load(open(os.path.join(SYNTH, "manifest.json")))["ids"]
    return os.path.join(SYNTH, ids[0] + ".png")


# --------------------------------------------------------------------------- help / parsing
@pytest.mark.parametrize("cmd", SUBCOMMANDS)
def test_help_for_every_subcommand(cmd):
    rc, out, _ = _run([cmd, "--help"])
    assert rc == 0
    assert "usage: dt " + cmd.split("-figma")[0] in out or "usage: dt" in out
    assert "--json" in out or cmd == "corpus"  # corpus has --json on the group parser


def test_top_level_help_lists_all_subcommands():
    rc, out, _ = _run(["--help"])
    assert rc == 0
    for cmd in SUBCOMMANDS:
        assert cmd in out


def test_usage_errors_exit_2_and_runtime_errors_exit_1():
    rc, _, _ = _run(["not-a-command"])
    assert rc == 2
    rc, _, _ = _run(["translate"])  # missing positional
    assert rc == 2
    rc, out, err = _run(["perceive", os.path.join(OUT, "does-not-exist.png"), "--json"])
    assert rc == 1
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["ok"] is False and "FileNotFoundError" in payload["error"]


def test_params_lists_pipeline_params_as_json():
    rc, out, _ = _run(["params", "--prefix", "pipeline.", "--json"])
    assert rc == 0
    d = json.loads(out)
    assert "pipeline.refine_iters" in d
    for k, v in d.items():
        assert v["doc"], k
        if v["range"] is not None:
            assert v["range"][0] <= v["default"] <= v["range"][1], k


def test_resolve_design_system_specs(tmp_path):
    assert resolve_design_system(None) is None
    assert resolve_design_system("none") is None
    ds = resolve_design_system("material3")
    assert ds.name == "material3" and len(ds.components) >= 20
    p = tmp_path / "copy.json"
    ds.save(str(p))
    assert resolve_design_system(str(p)).name == "material3"
    with pytest.raises(FileNotFoundError):
        resolve_design_system("screens:" + str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError):
        resolve_design_system("no-such-design-system")


# --------------------------------------------------------------------------- end-to-end translate
@pytest.fixture(scope="module")
def translate_run():
    png = _first_synth_png()
    out_dir = os.path.join(OUT, "translate_synth")
    shutil.rmtree(out_dir, ignore_errors=True)
    rc, stdout, stderr = _run(["translate", png, "--ds", "material3", "-o", out_dir, "--refine-iters", "3",
                               "--time-budget", "60", "--json"])
    return rc, stdout, stderr, out_dir, png


def test_translate_writes_every_output_and_improves_loss(translate_run):
    rc, stdout, stderr, out_dir, png = translate_run
    brief = json.loads(stdout)
    assert brief["errors"] == [], brief["errors"]
    assert rc == 0, stderr
    for name in OUTPUT_FILES:
        assert os.path.exists(os.path.join(out_dir, name)), name
    assert os.path.exists(os.path.join(out_dir, "validation", "validation.json"))
    m = json.load(open(os.path.join(out_dir, "metrics.json")))
    assert m["ok"] is True
    assert m["loss_before"] is not None and m["loss_after"] is not None
    assert m["loss_after"] <= m["loss_before"] + 1e-9
    assert m["refine"]["max_iters"] == 3 and m["refine"]["iterations"] <= 3
    assert m["timings"]["total"] > 0 and set(m["stages_completed"]) >= {"perceive", "map", "evaluate_before", "refine", "evaluate_after", "export", "decisions", "report"}
    assert m["components"]["n_after"] == len(m["components"]["found"]) >= 1
    assert m["export"]["errors"] == []
    # the perceived IR and the final mapped IR are both loadable; the mapped one carries components
    ir = Document.load(os.path.join(out_dir, "ir.json"))
    mapped = Document.load(os.path.join(out_dir, "ir.mapped.json"))
    assert (ir.width, ir.height) == (mapped.width, mapped.height)
    assert mapped.design_system == "material3"
    assert any(n.component is not None for n in mapped.walk())
    assert not any(n.component is not None for n in ir.walk())
    # the plan is valid and the decisions / report are well-formed
    plan = json.load(open(os.path.join(out_dir, "figma_plan.json")))
    assert plan["width"] == mapped.width and plan["root"]["children"]
    decisions = json.load(open(os.path.join(out_dir, "decisions.json")))
    assert all({"id", "kind", "node_id", "question", "candidates", "evidence", "answer"} <= d.keys() for d in decisions)
    assert m["decisions"]["n"] == len(decisions)
    report = open(os.path.join(out_dir, "report.md")).read()
    assert "loss before / after" in report and "## Components" in report and "## Validation gates" in report


def test_translate_human_output_and_partial_exit_codes(translate_run, tmp_path):
    _, _, _, out_dir, png = translate_run
    # resume from the finished run with refine off: fast, and the human output names the report
    out2 = str(tmp_path / "resumed")
    rc, out, _ = _run(["translate", png, "--ds", "material3", "-o", out2, "--refine-iters", "0", "--resume", out_dir, "--no-validate", "--quiet"])
    assert rc == 0, out
    assert "report:" in out and "loss before -> after" in out
    m = json.load(open(os.path.join(out2, "metrics.json")))
    assert m["refine"]["iterations"] == 0 and m["validation"] is None
    assert m["loss_after"] == pytest.approx(m["loss_before"])
    # an unresolvable design system is a stage failure, not a crash: outputs still written, exit 3
    out3 = str(tmp_path / "bad_ds")
    rc, out, _ = _run(["translate", png, "--ds", "no-such-ds", "-o", out3, "--refine-iters", "0", "--resume", out_dir, "--no-validate", "--quiet", "--json"])
    assert rc == 3
    brief = json.loads(out)
    assert brief["errors"] and brief["errors"][0]["stage"] == "map"
    for name in OUTPUT_FILES:
        assert os.path.exists(os.path.join(out3, name)), name


def test_apply_decisions_round_trip(translate_run, tmp_path):
    _, _, _, out_dir, _ = translate_run
    run_dir = str(tmp_path / "run")
    shutil.copytree(out_dir, run_dir)
    doc = Document.load(os.path.join(run_dir, "ir.mapped.json"))
    # synthesise one decision of each kind on real nodes so the merge path is exercised fully
    text = next(n for n in doc.walk() if n.type == "text" and n.text)
    frame = next(n for n in doc.walk() if n.component is not None)
    decisions = [
        {"id": "d1", "kind": "text", "node_id": text.id, "question": "?", "candidates": [], "evidence": {}, "answer": None},
        {"id": "d2", "kind": "component", "node_id": frame.id, "question": "?", "candidates": [], "evidence": {}, "answer": None},
        {"id": "d3", "kind": "icon", "node_id": "missing-node", "question": "?", "candidates": [], "evidence": {}, "answer": None},
    ]
    json.dump(decisions, open(os.path.join(run_dir, "decisions.json"), "w"))
    answers = {"d1": {"text": "Inbox"}, "d2": {"name": "Card", "variant": {"style": "elevated"}}, "d3": {"icon_name": "search"}, "d9": {}}
    ans_path = tmp_path / "answers.json"
    ans_path.write_text(json.dumps(answers))
    before = os.path.getmtime(os.path.join(run_dir, "render.png"))
    rc, out, _ = _run(["apply-decisions", run_dir, "--answers", str(ans_path), "--json"])
    assert rc == 0, out
    res = json.loads(out)
    assert res["applied"] == ["d1", "d2"] and res["skipped"] == ["d3"] and res["unknown"] == ["d9"] and res["errors"] == []
    after = Document.load(os.path.join(run_dir, "ir.mapped.json"))
    assert after.find(text.id).text == "Inbox"
    c = after.find(frame.id).component
    assert c.name == "Card" and c.variant == {"style": "elevated"} and c.confidence == 1.0 and c.key
    dec = json.load(open(os.path.join(run_dir, "decisions.json")))
    assert dec[0]["answer"] == {"text": "Inbox"} and dec[2]["answer"] is None
    assert os.path.getmtime(os.path.join(run_dir, "render.png")) >= before
    plan = json.load(open(os.path.join(run_dir, "figma_plan.json")))
    assert any(ch.get("characters") == "Inbox" for ch in _walk_plan(plan["root"]))


def _walk_plan(node: dict):
    """Every plan node, including the fallback subtree of instances (where mapped components keep their content)."""
    yield node
    for ch in node.get("children", []) or []:
        yield from _walk_plan(ch)
    fb = node.get("fallback")
    if isinstance(fb, dict):
        yield from _walk_plan(fb)


# --------------------------------------------------------------------------- stage subcommands chained
def test_stage_commands_chain(translate_run, tmp_path):
    _, _, _, out_dir, png = translate_run
    ir = os.path.join(out_dir, "ir.json")
    mapped = str(tmp_path / "m.json")
    rc, out, _ = _run(["map", ir, "--ds", "material3", "-o", mapped, "--json"])
    assert rc == 0 and json.loads(out)["components"] >= 1
    render = str(tmp_path / "r.png")
    rc, out, _ = _run(["render", mapped, "-o", render, "--json"])
    assert rc == 0 and os.path.exists(render)
    diff = str(tmp_path / "d.png")
    rc, out, _ = _run(["compare", mapped, png, "--render", render, "--diff", diff, "--no-ocr", "--json"])
    assert rc == 0
    rep = json.loads(out)
    assert rep["total"] >= 0 and os.path.exists(diff) and "per_node" not in rep
    gt = os.path.join(SYNTH, os.path.basename(png).replace(".png", ".gt.json"))
    rc, out, _ = _run(["compare", mapped, png, "--render", render, "--gt", gt, "--no-ocr", "--json"])
    assert rc == 0 and json.loads(out)["structure"]["node_recall"] is not None
    plan = str(tmp_path / "plan.json")
    rc, out, _ = _run(["export-figma", mapped, "-o", plan, "--json"])
    assert rc == 0 and json.loads(out)["errors"] == [] and os.path.exists(plan)
    rc, out, _ = _run(["validate", png, render, "--ir", mapped, "--json"])
    assert rc == 0 and set(json.loads(out)["gates"]) == {"pixel-exact", "visually-identical", "structurally-faithful"}


# --------------------------------------------------------------------------- pipeline helpers (pure)
def test_unmap_and_components_found_and_decisions(translate_run):
    _, _, _, out_dir, _ = translate_run
    doc = Document.load(os.path.join(out_dir, "ir.mapped.json"))
    found = components_found(doc)
    assert found and all({"node_id", "name", "variant", "confidence", "box"} <= f.keys() for f in found)
    n_inst = sum(1 for n in doc.walk() if n.type == "instance")
    assert n_inst == len(found)
    unmap_document(doc)
    assert not any(n.component or n.tokens or n.type == "instance" for n in doc.walk())
    assert doc.design_system is None
    # decisions respect the cap and the OCR-confidence threshold
    doc2 = Document.load(os.path.join(out_dir, "ir.mapped.json"))
    for n in doc2.walk():
        if n.type == "text":
            n.meta["conf"] = 0.1
    P.set("pipeline.decisions.max_items", 3)
    try:
        d = build_decisions(doc2)
    finally:
        P.reset("pipeline.decisions.max_items")
    assert len(d) == 3 and [x["id"] for x in d] == ["d1", "d2", "d3"]
    assert any(x["kind"] == "text" for x in build_decisions(doc2))


# --------------------------------------------------------------------------- improve / bench glue
def test_improve_brief_from_existing_bench_report(tmp_path):
    from dt.selftest import bench
    report = {"run_id": "t", "composite": 0.5, "n_cases": 2, "n_errors": 1, "stages": ["perceive", "map", "render"], "out_dir": str(tmp_path),
              "metrics": {"node_recall": {"mean": 0.5, "p50": 0.5, "worst": 0.2, "best": 0.8, "n": 1, "goodness": 0.5}},
              "worst": [], "cases": [
                  {"id": "synth_1_000", "corpus": "synth", "png": _first_synth_png(), "width": 600, "height": 891,
                   "metrics": {"node_recall": 0.2, "n_pred": 5, "n_gt": 10, "n_matched": 2}, "composite": 0.2, "diff_image": "diffs/synth_1_000.png"},
                  {"id": "mwc_1_000", "corpus": "mwc", "png": "x.png", "metrics": {}, "composite": 0.0, "error": "boom", "error_stage": "perceive"}],
              "timing": {"wall_s": 1.0}}
    rpath = tmp_path / "report.json"
    rpath.write_text(json.dumps(report))
    brief = tmp_path / "brief.md"
    rc, out, _ = _run(["improve", "--report", str(rpath), "--top", "5", "-o", str(brief), "--knowledge", os.path.join(ROOT, "knowledge", "failures.jsonl"), "--json"])
    assert rc == 0
    res = json.loads(out)
    assert res["brief"] == str(brief) and len(res["drafts"]) >= 2
    md = brief.read_text()
    assert "synth_1_000" in md and "node_recall" in md and "suspected stage" in md and "```jsonl" in md
    assert "error at `perceive`" in md
    # drafts follow the knowledge/failures.jsonl schema
    assert {"case", "stage", "metric", "value", "symptom", "root_cause_hypothesis", "suggested_fix", "run_id"} <= res["drafts"][0].keys()
    assert bench.draft_failures(report, 5)  # same source as the CLI


def test_tune_dry_lists_search_space():
    rc, out, _ = _run(["tune", "--dry", "--include", "map.", "--json"])
    assert rc == 0
    space = json.loads(out)
    assert space and all(k.startswith("map.") for k in space)
