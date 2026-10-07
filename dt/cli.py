"""``dt`` command line: every pipeline stage as a subcommand with JSON in/out and stable exit codes.

    dt translate shot.png --ds material3 -o out/run1 [--json]
    dt perceive shot.png -o ir.json          dt map ir.json --ds material3 -o ir.mapped.json
    dt render ir.json -o render.png          dt compare ir.json shot.png --diff diff.png
    dt refine ir.json shot.png -o ir.refined.json --iters 10
    dt export ir.json -o figma_plan.json     dt validate shot.png render.png -o out/validation
    dt corpus build --kind mwc --n 12        dt bench [--gate] [--set-baseline]   (knowledge/baseline.json)
    dt tune --iters 50                       dt improve --top 10 -o brief.md
    dt apply-decisions out/run1 --answers answers.json
    dt ds from-tokens tokens.json -o fixtures/design_systems/mine.json
    dt params [--prefix perceive.]

Exit codes: 0 success · 1 failure (exception, regression gate failed, or no document produced) ·
2 usage error · 3 partial success (``translate`` wrote its outputs but one or more stages failed).
With ``--json`` the only thing on stdout is one JSON object; progress goes to stderr.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Optional, Sequence

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_PARTIAL = 0, 1, 2, 3


# --------------------------------------------------------------------------- output helpers
def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _emit(result: Any, as_json: bool, human: Optional[Callable[[Any], str]] = None) -> None:
    if as_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(human(result) if human else json.dumps(result, indent=2, default=str))


def _kv_table(d: dict, keys: Optional[Sequence[str]] = None) -> str:
    keys = list(keys or d.keys())
    width = max((len(k) for k in keys), default=1)
    return "\n".join(f"{k:{width}s}  {d.get(k)}" for k in keys)


def _dpr_arg(v: str):
    """--dpr accepts a number or 'auto' (detect from the image)."""
    return "auto" if str(v).lower() == "auto" else float(v)


def _out_dir_for(path: str) -> str:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------- stage commands
def cmd_perceive(a: argparse.Namespace) -> int:
    from dt.ir import summarize
    from dt.perceive import perceive
    dpr = a.dpr
    if dpr == "auto":
        from dt.common.image import load_rgb as _load
        from dt.perceive.dpr import detect_dpr
        dpr, _ = detect_dpr(_load(a.image))
    doc = perceive(a.image, dpr=dpr)
    doc.source_image = os.path.abspath(a.image)
    out = a.out or os.path.splitext(a.image)[0] + ".ir.json"
    _out_dir_for(out)
    doc.save(out)
    res = {"out": out, "width": doc.width, "height": doc.height, "nodes": doc.root.count(), "texts": len(doc.texts()),
           "ocr": doc.meta.get("perceive", {}).get("ocr")}
    _emit(res, a.json, lambda r: f"{summarize(doc, 40)}\n-> {out}")
    return EXIT_OK


def cmd_map(a: argparse.Namespace) -> int:
    from dt.ir import Document
    from dt.mapping import map_document
    from dt.pipeline import components_found, resolve_design_system, unmap_document
    doc = Document.load(a.ir)
    ds = resolve_design_system(a.ds, _log)
    if ds is None:
        _log("no design system given (--ds); nothing to map")
        return EXIT_USAGE
    if a.remap:
        unmap_document(doc)
    doc = map_document(doc, ds, collapse=a.collapse)
    out = a.out or a.ir.replace(".json", ".mapped.json")
    _out_dir_for(out)
    doc.save(out)
    found = components_found(doc)
    res = {"out": out, "design_system": ds.name, "components": len(found), "found": found}
    _emit(res, a.json, lambda r: "\n".join(f"{c['node_id']:12s} {c['name']:14s} {c['variant']} conf={c['confidence']:.2f}" for c in found) + f"\n{len(found)} components -> {out}")
    return EXIT_OK


def cmd_render(a: argparse.Namespace) -> int:
    from dt.ir import Document
    from dt.render.screenshot import render_doc
    doc = Document.load(a.ir)
    out = a.out or a.ir.replace(".json", ".png")
    _out_dir_for(out)
    rgb = render_doc(doc, out, mode=a.mode)
    res = {"out": out, "width": int(rgb.shape[1]), "height": int(rgb.shape[0]), "mode": a.mode}
    _emit(res, a.json, lambda r: f"rendered {r['width']}x{r['height']} ({a.mode}) -> {out}")
    return EXIT_OK


def cmd_compare(a: argparse.Namespace) -> int:
    from dt.common.image import load_rgb
    from dt.compare import evaluate, save_diff_image
    from dt.ir import Document
    doc = Document.load(a.ir)
    target = load_rgb(a.target)
    rendered = load_rgb(a.render) if a.render else None
    gt = Document.load(a.gt) if a.gt else None
    rep = evaluate(doc, target, rendered, gt=gt, use_ocr=not a.no_ocr)
    res = rep.to_dict()
    res["worst_nodes"] = rep.worst_nodes(8)
    if a.diff:
        from dt.render.screenshot import render_doc
        rendered = rendered if rendered is not None else render_doc(doc)
        _out_dir_for(a.diff)
        res["diff_image"] = save_diff_image(target, rendered, rep.diff_map, a.diff, rep.residual_regions)
    if not a.per_node:
        res.pop("per_node", None)

    def human(r: dict) -> str:
        px = r["pixel"]
        lines = [f"loss total {r['total']:.5f}", _kv_table({k: round(v, 4) if isinstance(v, float) else v for k, v in px.items()}),
                 f"alignment dx={r['alignment']['dx']:.2f} dy={r['alignment']['dy']:.2f}", f"residual regions {len(r['residual_regions'])}",
                 "worst nodes: " + ", ".join(f"{k}={v:.2f}" for k, v in r["worst_nodes"])]
        if gt is not None:
            lines.append(_kv_table({k: v for k, v in r["structure"].items() if k != "matches"}))
        if a.diff:
            lines.append(f"diff -> {a.diff}")
        return "\n".join(lines)

    _emit(res, a.json, human)
    return EXIT_OK


def cmd_refine(a: argparse.Namespace) -> int:
    from dt.common.image import load_rgb
    from dt.ir import Document
    from dt.refine.optimizer import accepted_moves, refine
    doc = Document.load(a.ir)
    target = load_rgb(a.target)
    new_doc, hist = refine(doc, target, max_iters=a.iters, patience=a.patience, time_budget_s=a.time_budget, log=_log)
    out = a.out or a.ir.replace(".json", ".refined.json")
    _out_dir_for(out)
    new_doc.save(out)
    stop = next((h for h in reversed(hist) if h.get("kind") == "stop"), {})
    acc = accepted_moves(hist)
    res = {"out": out, "loss_before": stop.get("before"), "loss_after": stop.get("after"), "accepted": len(acc),
           "tried": len(hist) - 1, "iterations": stop.get("iter"), "renders": stop.get("params", {}).get("renders"),
           "reason": stop.get("params", {}).get("reason"), "moves": acc}
    if a.history:
        _out_dir_for(a.history)
        with open(a.history, "w") as f:
            json.dump(hist, f, indent=1, default=str)
        res["history"] = a.history
    _emit(res, a.json, lambda r: f"loss {r['loss_before']:.5f} -> {r['loss_after']:.5f}; {r['accepted']} accepted / {r['tried']} tried; stop={r['reason']} -> {out}")
    return EXIT_OK


def cmd_export(a: argparse.Namespace) -> int:
    from dt.export import save_plan, to_build_plan, validate_plan
    from dt.ir import Document
    doc = Document.load(a.ir)
    plan = to_build_plan(doc)
    errors = validate_plan(plan)
    out = a.out or a.ir.replace(".json", ".figma_plan.json")
    _out_dir_for(out)
    save_plan(plan, out)
    res = {"out": out, "nodes": plan["meta"].get("nodeCount"), "warnings": plan.get("warnings", []), "errors": errors,
           "fonts": plan.get("fonts", [])}
    _emit(res, a.json, lambda r: f"build plan {r['nodes']} nodes, {len(r['warnings'])} warnings, {len(errors)} errors -> {out}")
    return EXIT_FAIL if errors else EXIT_OK


def cmd_validate(a: argparse.Namespace) -> int:
    from dt.common.image import load_rgb
    from dt.ir import Document
    from dt.validate import validate
    doc = Document.load(a.ir) if a.ir else None
    rep = validate(load_rgb(a.target), load_rgb(a.render), doc=doc, out_dir=a.out)
    res = rep.to_dict()
    res.pop("text_matches", None)
    res["worst_regions"] = res["worst_regions"][:8]
    _emit(res, a.json, lambda r: rep.markdown())
    return EXIT_OK


# --------------------------------------------------------------------------- translate / decisions
def cmd_translate(a: argparse.Namespace) -> int:
    from dt.pipeline import translate
    ds = None if (a.ds or "").lower() in ("", "none") else a.ds
    m = translate(a.image, ds=ds, out_dir=a.out, refine_iters=a.refine_iters, dpr=a.dpr, time_budget_s=a.time_budget,
                  validate=not a.no_validate, resume_from=a.resume, log=_log if not a.quiet else None)
    brief = {k: m.get(k) for k in ("ok", "out_dir", "design_system", "loss_before", "loss_after", "loss_gain_frac", "refine",
                                   "decisions", "export", "validation", "timings", "files")}
    brief["components"] = {k: v for k, v in m["components"].items() if k != "found"}
    brief["errors"] = [{"stage": e["stage"], "error": e["error"]} for e in m["errors"]]

    def human(r: dict) -> str:
        rows = {"out_dir": r["out_dir"], "design_system": r["design_system"],
                "loss before -> after": f"{r['loss_before']} -> {r['loss_after']}",
                "refine": f"{r['refine'].get('accepted')} accepted, {r['refine'].get('iterations')} iters, stop={r['refine'].get('reason')}",
                "components": f"{r['components'].get('n_after')} {r['components'].get('by_name')}",
                "decisions": f"{r['decisions'].get('n')} {r['decisions'].get('by_kind')}",
                "gates": (r.get("validation") or {}).get("gates"), "errors": len(r["errors"]), "wall_s": r["timings"].get("total")}
        return _kv_table(rows) + "\nreport: " + os.path.join(r["out_dir"], "report.md")

    _emit(brief, a.json, human)
    if not m["ok"]:
        return EXIT_FAIL
    return EXIT_PARTIAL if m["errors"] else EXIT_OK


def cmd_apply_decisions(a: argparse.Namespace) -> int:
    from dt.pipeline import apply_decisions
    with open(a.answers) as f:
        answers = json.load(f)
    if not isinstance(answers, dict):
        _log("answers must be a JSON object {decision_id: choice}")
        return EXIT_USAGE
    res = apply_decisions(a.run_dir, answers, ds_spec=a.ds, log=_log)
    _emit(res, a.json, lambda r: f"applied {len(r['applied'])}, unknown {r['unknown']}, skipped {r['skipped']}, errors {r['errors']}")
    return EXIT_FAIL if res["errors"] else EXIT_OK


# --------------------------------------------------------------------------- corpus / bench / tune / improve
def cmd_corpus(a: argparse.Namespace) -> int:
    if a.corpus_cmd != "build":
        return EXIT_USAGE
    from dt.selftest import corpus_dir
    if a.kind == "screens":
        from dt.selftest.reference_screens import DEFAULT_DIR, capture
        out = a.out or DEFAULT_DIR
        m = capture(out, names=a.only or None, verbose=not a.json)
        res = {"kind": "screens", "out": out, "n_ok": m["n_ok"], "n_failed": m["n_failed"]}
        _emit(res, a.json, lambda r: f"{r['n_ok']} ok, {r['n_failed']} failed -> {out}")
        return EXIT_OK if m["n_ok"] else EXIT_FAIL
    gen = __import__(f"dt.selftest.{'mwc_corpus' if a.kind == 'mwc' else 'synth'}", fromlist=["generate"]).generate
    out = a.out or corpus_dir(a.kind)
    manifest = gen(a.n, out, seed=a.seed)
    res = {"kind": a.kind, "out": out, "n": manifest["n"], "seed": manifest["seed"], "ids": manifest["ids"],
           "component_histogram": manifest.get("component_histogram", {})}
    _emit(res, a.json, lambda r: f"{r['n']} {a.kind} cases (seed {r['seed']}) -> {out}\n" + _kv_table(r["component_histogram"]))
    return EXIT_OK


def _bench_human(report: dict) -> str:
    lines = [f"composite {report['composite']:.4f} over {report['n_cases']} cases ({report['n_errors']} errors, "
             f"{report.get('n_dropped', 0)} dropped) in {report['timing']['wall_s']:.1f}s -> {report.get('out_dir')}"]
    for name, agg in report["metrics"].items():
        lines.append(f"  {name:20s} mean={agg['mean']:.4f} p50={agg['p50']:.4f} worst={agg['worst']:.4f} goodness={agg['goodness']:.3f}")
    for e in report.get("errors", []):
        lines.append(f"  ERROR {e['corpus']}/{e['id']} [{e['stage']}]: {e['error']}")
    for d in report.get("dropped", []):
        lines.append(f"  dropped {d['corpus']}/{d['id']}: {d['reason']}")
    return "\n".join(lines)


def cmd_bench(a: argparse.Namespace) -> int:
    from dt.selftest import bench
    stages = tuple(s.strip() for s in a.stages.split(",") if s.strip())
    report = bench.run(a.corpus or list(bench.DEFAULT_CORPORA), stages=stages, limit=a.limit, workers=a.workers,
                       run_id=a.run_id, out_root=(a.out or None), verbose=not a.json and not a.quiet)
    gate, gate_text = bench.gate_and_baseline(report, a.gate, a.set_baseline)
    rc = EXIT_FAIL if (gate is not None and not gate["pass"]) else EXIT_OK
    if gate_text and a.json:  # the delta table always reaches the terminal, even in --json mode
        _log(gate_text)
    res = {k: v for k, v in report.items() if k != "cases"}
    res["gate"] = gate
    res["baseline_written"] = a.set_baseline
    if a.cases:
        res["cases"] = report["cases"]
    _emit(res, a.json, lambda r: _bench_human(report) + ("\n" + gate_text if gate_text else ""))
    return rc


def cmd_tune(a: argparse.Namespace) -> int:
    from dt.selftest import bench, tune
    if a.dry:
        space = tune.search_space(a.include)
        _emit({k: v for k, v in space.items()}, a.json, lambda r: tune.format_space(space))
        return EXIT_OK
    stages = tuple(s.strip() for s in a.stages.split(",") if s.strip())
    res = tune.search(a.iters, a.corpus or list(bench.DEFAULT_CORPORA), a.holdout, a.seed, limit=a.limit, stages=stages,
                      workers=a.workers, include=a.include, params_path=a.params, history_path=a.history,
                      uniform_first=a.uniform_first, verbose=not a.json)
    _emit({k: v for k, v in res.items() if k != "ts"}, a.json,
          lambda r: _kv_table({"baseline": r["baseline"], "best": r["best"], "evaluations": r["evaluations"], "written": r["written"], "params_path": r["params_path"]}))
    return EXIT_OK


def improve_brief(report: dict, top: int, knowledge_path: Optional[str] = None) -> tuple[str, list[dict]]:
    """Markdown brief for an AI driver: worst cases, suspected stage, evidence, prior knowledge, and
    ``knowledge/failures.jsonl`` templates to fill in. Returns ``(markdown, drafts)``."""
    from dt.selftest import bench
    from dt.selftest.metrics import SPEC, worst_entries
    worst = worst_entries(report["cases"], top)
    drafts = bench.draft_failures(report, top)
    rows = {r["id"]: r for r in report["cases"]}
    known: list[dict] = []
    if knowledge_path and os.path.exists(knowledge_path):
        with open(knowledge_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        known.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    L = [f"# dt improve brief — run `{report['run_id']}`", "",
         f"composite **{report['composite']:.4f}** over {report['n_cases']} cases ({report['n_errors']} errors, "
         f"{report.get('n_dropped', 0)} dropped); stages {', '.join(report['stages'])}; metrics v{report.get('metrics_version', 1)}; "
         f"report dir `{report.get('out_dir')}`", "",
         *[f"- dropped `{d['corpus']}/{d['id']}`: {d['reason']}" for d in report.get("dropped", [])],
         "## Per-metric summary", "", "| metric | mean | p50 | worst | goodness | stage |", "|---|---|---|---|---|---|"]
    for name, agg in report["metrics"].items():
        spec = SPEC.get(name)
        L.append(f"| {name} | {agg['mean']:.4f} | {agg['p50']:.4f} | {agg['worst']:.4f} | {agg['goodness']:.3f} | {spec.stage if spec else '?'} |")
    by_stage: dict[str, float] = {}
    for e in worst:
        by_stage[e["stage"]] = by_stage.get(e["stage"], 0.0) + float(e["deficit"])
    L += ["", "## Where the loss is (weighted deficit of the worst entries, by suspected stage)", ""]
    L += [f"- **{s}**: {d:.3f}" for s, d in sorted(by_stage.items(), key=lambda kv: -kv[1])]
    L += ["", f"## Worst {len(worst)} (case, metric) pairs — fix the top one first", ""]
    for i, e in enumerate(worst, 1):
        row = rows.get(e["case"], {})
        met = row.get("metrics", {})
        L += [f"### {i}. `{e['corpus']}/{e['case']}` — {e['metric']} = {e['value']} (goodness {e['goodness']:.2f}, deficit {e['deficit']:.3f}) → suspected stage **{e['stage']}**", ""]
        if row.get("error"):
            L += [f"- error at `{row.get('error_stage')}`: `{row['error']}`", ""]
        L += [f"- screenshot: `{row.get('png')}` ({row.get('width')}x{row.get('height')})",
              f"- ground truth: `{os.path.join(os.path.dirname(row.get('png') or ''), e['case'] + '.gt.json')}`"]
        if row.get("diff_image"):
            L.append(f"- diff image (target | render | ΔE): `{os.path.join(report.get('out_dir') or '', row['diff_image'])}`")
        L.append("- other metrics: " + ", ".join(f"{k}={round(v, 3) if isinstance(v, float) else v}" for k, v in met.items() if v is not None and k not in ("n_pred", "n_gt", "n_matched")))
        if {"n_pred", "n_gt", "n_matched"} <= met.keys():
            L.append(f"- nodes: predicted {met['n_pred']}, ground truth {met['n_gt']}, matched {met['n_matched']}")
        prior = [k for k in known if k.get("metric") == e["metric"] and k.get("stage") == e["stage"]]
        if prior:
            L.append("- prior knowledge (knowledge/failures.jsonl):")
            for k in prior[:3]:
                L.append(f"  - *{k.get('case')}*: {k.get('root_cause_hypothesis') or k.get('symptom')} → fix: {k.get('suggested_fix') or '?'}")
        L += [f"- reproduce: `dt translate {row.get('png')} --ds material3 -o out/improve/{e['case']} --refine-iters 0` then compare `ir.mapped.json` with the gt",
              ""]
    L += ["## Protocol", "",
          "1. Pick the top entry; open its diff image and compare the predicted IR with the gt (`dt compare <ir> <png> --gt <gt.json>`).",
          "2. Edit ONE stage (`dt/perceive`, `dt/mapping`, `dt/refine`, `dt/export`); thresholds go through `dt.params`.",
          "3. `python -m pytest -q` and `dt bench --gate` (gates against the tracked `knowledge/baseline.json`) — exits 1 if the composite drops more than "
          "`bench.gate.eps` or any metric's goodness drops more than `bench.gate.metric_eps`.",
          "4. Fill in and append the matching template below to `knowledge/failures.jsonl`; `dt bench --gate --set-baseline` to move the gate (commit `knowledge/baseline.json`).", "",
          "## knowledge/failures.jsonl templates (fill root_cause_hypothesis / suggested_fix)", "", "```jsonl"]
    L += [json.dumps(d) for d in drafts]
    L += ["```", ""]
    return "\n".join(L), drafts


def cmd_improve(a: argparse.Namespace) -> int:
    from dt.selftest import bench
    if a.report:
        with open(a.report) as f:
            report = json.load(f)
    else:
        stages = tuple(s.strip() for s in a.stages.split(",") if s.strip())
        report = bench.run(a.corpus or list(bench.DEFAULT_CORPORA), stages=stages, limit=a.limit, workers=a.workers,
                           out_root=(a.bench_out or None), verbose=not a.json)
    md, drafts = improve_brief(report, a.top, a.knowledge)
    if a.out:
        _out_dir_for(a.out)
        with open(a.out, "w") as f:
            f.write(md)
    res = {"run_id": report["run_id"], "composite": report["composite"], "n_cases": report["n_cases"], "worst": report["worst"][: a.top],
           "drafts": drafts, "brief": a.out, "report_dir": report.get("out_dir")}
    _emit(res, a.json, lambda r: md if not a.out else f"brief -> {a.out} (composite {r['composite']:.4f}, {len(r['worst'])} worst entries)")
    return EXIT_OK


# --------------------------------------------------------------------------- design systems / params
def cmd_ds(a: argparse.Namespace) -> int:
    from dt.pipeline import resolve_design_system
    spec = {"from-figma": f"figma:{a.src}", "from-screens": f"screens:{a.src}", "from-tokens": f"tokens:{a.src}"}.get(a.ds_cmd)
    if spec is None:
        if a.ds_cmd == "show":
            ds = resolve_design_system(a.src, _log)
            res = {"name": ds.name, "tokens": len(ds.tokens), "components": [c.name for c in ds.components], "fonts": list(ds.fonts)}
            _emit(res, a.json, lambda r: _kv_table(r))
            return EXIT_OK
        return EXIT_USAGE
    ds = resolve_design_system(spec, _log)
    if a.name:
        ds.name = a.name
    out = a.out or os.path.join("fixtures", "design_systems", f"{ds.name}.json")
    _out_dir_for(out)
    ds.save(out)
    res = {"out": out, "name": ds.name, "tokens": len(ds.tokens), "components": len(ds.components)}
    _emit(res, a.json, lambda r: f"design system {r['name']}: {r['tokens']} tokens, {r['components']} components -> {out}")
    return EXIT_OK


def cmd_params(a: argparse.Namespace) -> int:
    import dt.pipeline  # noqa: F401 - registers pipeline.* params
    from dt.params import P
    for mod in ("dt.perceive", "dt.compare", "dt.mapping", "dt.refine", "dt.export", "dt.selftest.bench", "dt.selftest.tune", "dt.validate"):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001 - optional stages may be unavailable on this platform
            pass
    docs, ranges, allv, defaults = P.docs(), P.ranges(), P.all(), P.defaults()
    keys = sorted(k for k in allv if not a.prefix or k.startswith(a.prefix))
    res = {k: {"value": allv[k], "default": defaults.get(k), "range": list(ranges[k]) if k in ranges else None, "doc": docs.get(k, "")} for k in keys}
    _emit(res, a.json, lambda r: "\n".join(f"{k:44s} {str(v['value']):>10s}  {('[%g, %g]' % tuple(v['range'])) if v['range'] else '':>14s}  {v['doc']}" for k, v in r.items()) + f"\n{len(r)} params")
    return EXIT_OK


def cmd_doctor(a: argparse.Namespace) -> int:
    from dt import doctor
    res = doctor.run()
    _emit(doctor.summary(res), a.json, lambda _r: doctor.format_human(res))
    return EXIT_OK if not any(r.status == "fail" for r in res) else EXIT_FAIL


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dt", description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, fn: Callable, help_: str, aliases: Sequence[str] = ()) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_, description=help_, aliases=list(aliases))
        p.add_argument("--json", action="store_true", help="emit one JSON object on stdout (logs go to stderr)")
        p.set_defaults(fn=fn)
        return p

    p = add("perceive", cmd_perceive, "screenshot -> IR (ir.json)")
    p.add_argument("image"); p.add_argument("-o", "--out"); p.add_argument("--dpr", type=_dpr_arg, default=1.0, help="device pixel ratio or 'auto'")

    p = add("map", cmd_map, "IR -> IR with component refs + tokens for a design system")
    p.add_argument("ir"); p.add_argument("--ds", default="material3", help="material3 | figma:<key> | screens:<dir> | tokens:<json> | <file.json>")
    p.add_argument("-o", "--out"); p.add_argument("--collapse", action="store_true", help="move instance children into props")
    p.add_argument("--remap", action="store_true", help="drop existing component refs first")

    p = add("render", cmd_render, "IR -> PNG via the real renderer")
    p.add_argument("ir"); p.add_argument("-o", "--out"); p.add_argument("--mode", choices=("absolute", "flex"), default="absolute")

    p = add("compare", cmd_compare, "IR vs target screenshot -> loss report (+ optional diff image / gt metrics)")
    p.add_argument("ir"); p.add_argument("target"); p.add_argument("--render", help="existing render PNG (else renders the IR)")
    p.add_argument("--diff", help="write side-by-side diff PNG here"); p.add_argument("--gt", help="ground-truth IR for structural metrics")
    p.add_argument("--no-ocr", action="store_true"); p.add_argument("--per-node", action="store_true", help="keep per_node errors in JSON")

    p = add("refine", cmd_refine, "adversarial render-verified refinement of an IR against the screenshot")
    p.add_argument("ir"); p.add_argument("target"); p.add_argument("-o", "--out"); p.add_argument("--iters", type=int, default=None)
    p.add_argument("--patience", type=int, default=None); p.add_argument("--time-budget", type=float, default=None, help="seconds")
    p.add_argument("--history", help="write the full hypothesis history JSON here")

    p = add("export", cmd_export, "IR -> Figma build plan JSON (import with dt/export/figma_plugin)", aliases=("export-figma",))
    p.add_argument("ir"); p.add_argument("-o", "--out")

    p = add("validate", cmd_validate, "independent fidelity gates: target PNG vs render PNG")
    p.add_argument("target"); p.add_argument("render"); p.add_argument("--ir"); p.add_argument("-o", "--out", help="artifact dir (validation.json/md, heatmap, blink)")

    p = add("translate", cmd_translate, "full pipeline: perceive -> map -> refine -> export -> report")
    p.add_argument("image"); p.add_argument("--ds", default="material3", help="design system spec, or 'none'")
    p.add_argument("-o", "--out", default="out/run"); p.add_argument("--refine-iters", type=int, default=None)
    p.add_argument("--dpr", type=_dpr_arg, default=1.0, help="device pixel ratio or 'auto'"); p.add_argument("--time-budget", type=float, default=None, help="refine wall-clock cap (s)")
    p.add_argument("--no-validate", action="store_true"); p.add_argument("--resume", help="previous run dir: reuse its ir.mapped.json")
    p.add_argument("--quiet", action="store_true")

    p = add("apply-decisions", cmd_apply_decisions, "merge decision answers into a run (ir.mapped.json, plan, render)")
    p.add_argument("run_dir"); p.add_argument("--answers", required=True, help="JSON {decision_id: choice}"); p.add_argument("--ds")

    p = add("corpus", cmd_corpus, "ground-truth corpora")
    cs = p.add_subparsers(dest="corpus_cmd", required=True)
    b = cs.add_parser("build", help="generate a corpus (mwc | synth) or capture reference screens")
    b.add_argument("--kind", choices=("mwc", "synth", "screens"), required=True); b.add_argument("--n", type=int, default=12)
    b.add_argument("--seed", type=int, default=1); b.add_argument("--out"); b.add_argument("--only", nargs="*", help="screens: source names")
    b.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="emit one JSON object on stdout")

    _baseline = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge", "baseline.json")
    p = add("bench", cmd_bench, "run the ground-truth benchmark (perceive -> map -> render) and score it")
    p.add_argument("--corpus", nargs="*"); p.add_argument("--stages", default="perceive,map,render"); p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=1); p.add_argument("--run-id"); p.add_argument("--out", default="out/bench", help="report root ('' for none)")
    p.add_argument("--gate", "--baseline", dest="gate", nargs="?", const=_baseline, default=None,
                   help="gate against a baseline (default knowledge/baseline.json): exit 1 if the composite drops > bench.gate.eps "
                        "or any metric's goodness drops > bench.gate.metric_eps; prints a per-metric delta table")
    p.add_argument("--set-baseline", "--write-baseline", dest="set_baseline", nargs="?", const=_baseline,
                   default=None, help="write this run as the regression baseline (default knowledge/baseline.json, tracked)")
    p.add_argument("--cases", action="store_true", help="include per-case rows in JSON"); p.add_argument("--quiet", action="store_true")

    p = add("tune", cmd_tune, "parameter search over dt.params ranges (writes dt/params.json only if holdout improves)")
    p.add_argument("--iters", type=int, default=20); p.add_argument("--corpus", nargs="*"); p.add_argument("--holdout", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--limit", type=int); p.add_argument("--stages", default="perceive,map,render")
    p.add_argument("--workers", type=int, default=1); p.add_argument("--include", nargs="*"); p.add_argument("--params", default=None)
    p.add_argument("--history", default=None); p.add_argument("--uniform-first", action="store_true"); p.add_argument("--dry", action="store_true")

    p = add("improve", cmd_improve, "bench -> worst cases -> markdown brief an AI can act on")
    p.add_argument("--top", type=int, default=10); p.add_argument("--report", help="reuse an existing bench report.json")
    p.add_argument("--corpus", nargs="*"); p.add_argument("--stages", default="perceive,map,render"); p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=1); p.add_argument("--bench-out", default="out/bench"); p.add_argument("-o", "--out", help="write the brief here (.md)")
    p.add_argument("--knowledge", default=os.path.join("knowledge", "failures.jsonl"))

    p = add("ds", cmd_ds, "design systems: from-figma <key> | from-screens <dir> | from-tokens <json> | show <spec>")
    p.add_argument("ds_cmd", choices=("from-figma", "from-screens", "from-tokens", "show")); p.add_argument("src")
    p.add_argument("-o", "--out"); p.add_argument("--name")

    add("doctor", cmd_doctor, "check this install (browser, fonts, OCR backend, icon atlas, Material Web bundle) and print fixes")

    p = add("params", cmd_params, "list registered tunable parameters (value, default, range, doc)")
    p.add_argument("--prefix", default="")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:  # argparse exits 2 on usage errors, 0 on --help
        return int(e.code or 0)
    try:
        return int(a.fn(a))
    except KeyboardInterrupt:
        _log("interrupted")
        return 130
    except Exception as e:  # noqa: BLE001 - the CLI must always exit with a code, never a traceback on stdout
        import traceback
        _log(f"error: {type(e).__name__}: {e}")
        if os.environ.get("DT_DEBUG"):
            traceback.print_exc()
        if getattr(a, "json", False):
            print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}", "command": a.cmd}))
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
