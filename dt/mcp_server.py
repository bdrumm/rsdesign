"""MCP server: the harness as tools any MCP client (Claude, Cursor, an agent framework) can call.

    python -m dt.mcp_server            # stdio transport (what MCP clients launch)

Every tool returns compact JSON (paths to large artefacts, never the artefacts themselves) so an
AI driver can run the loop in AGENTS.md without reading pixels: translate -> read gates and
decisions -> answer decisions -> re-render -> validate; and, to improve the harness itself:
bench -> improve brief -> edit one stage -> bench --gate.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from mcp.server.mcpserver import MCPServer

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

# Every tool body runs on this one worker thread: the renderer's Playwright browser is bound to the
# thread that started it, so one thread == one browser for the life of the server.
_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rsdesign-worker")


def _on_worker(fn):
    """Turn a blocking tool body into an async tool that runs on the dedicated worker thread."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_WORKER, functools.partial(fn, *args, **kwargs))
    return wrapper

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

INSTRUCTIONS = """rsdesign turns a UI screenshot into a structured design (IR), maps it onto a design system
(Material 3 by default), renders it back, refines it against the screenshot, validates the result
with independent pixel/geometry/text gates and exports a Figma build plan.

Loop for one screenshot:
1. translate(image_path, out_dir) -> read `gates`, `fidelity` and `open_decisions`.
2. list_decisions(out_dir) -> answer only what the evidence supports; apply_decisions(out_dir, answers).
3. validate(target, render, ir) to re-check; gates are pixel-exact > visually-identical > structurally-faithful.
4. The Figma plan is <out_dir>/figma_plan.json (import with the plugin in dt/export/figma_plugin).

Loop for improving the harness:
1. bench(gate=True) -> composite and per-metric deltas vs knowledge/baseline.json.
2. improve_brief(top) -> worst cases with suspected stage; attribution() -> which stage costs most.
3. Edit ONE stage, run tests, bench(gate=True) must pass, record a knowledge/failures.jsonl entry.

Loop for learning from the user (user-informed tuning, local and private by default):
1. When the user corrects a run (or answers decisions), submit_feedback(run_dir, items, rating).
2. feedback_status() -> bundles, learned rules, accuracy on the user's own cases; the user runs
   `dt feedback learn` / `dt feedback eval` (or exports with consent for the shared model).
Never judge output by eye: every question has a measured answer."""

server = MCPServer(name="rsdesign", title="rsdesign screenshot-to-design harness", instructions=INSTRUCTIONS,
                   version="0.1.0")


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def _fidelity_summary(v: dict) -> dict:
    keys = ("mean_de", "jnd_frac_nontext", "chamfer", "chamfer_tile_max", "edge_within1", "edge_iou", "region_de_worst",
            "text_cer", "text_dpos_max", "text_lines_matched", "text_lines_target", "raster_frac", "text_under_raster")
    return {k: v.get(k) for k in keys if k in v}


@server.tool(description="Translate a screenshot end to end (perceive, map, render, refine, validate, export). "
                         "Returns gates, fidelity measures, loss before/after, open decisions and output paths.")
@_on_worker
def translate(image_path: str, out_dir: str, design_system: str = "material3", refine_iters: int = 4,
              dpr: float = 1.0, time_budget_s: Optional[float] = None) -> dict:
    from dt.pipeline import translate as run
    out_dir = _abs(out_dir)
    m = run(_abs(image_path), ds=design_system or None, out_dir=out_dir, refine_iters=refine_iters, dpr=dpr,
            time_budget_s=time_budget_s)
    v = {}
    vp = os.path.join(out_dir, "validation", "validation.json")
    if os.path.exists(vp):
        v = json.load(open(vp))
    decisions = []
    dp = os.path.join(out_dir, "decisions.json")
    if os.path.exists(dp):
        d = json.load(open(dp))
        decisions = d.get("decisions", d) if isinstance(d, dict) else d
    return {
        "out_dir": out_dir,
        "loss_before": m.get("loss_before"), "loss_after": m.get("loss_after"),
        "gates": v.get("gates"), "gate_failures": v.get("gate_failures"), "fidelity": _fidelity_summary(v),
        "open_decisions": sum(1 for x in decisions if isinstance(x, dict) and not x.get("answered")),
        "errors": m.get("errors", []), "timings": m.get("timings"),
        "files": {k: os.path.join(out_dir, f) for k, f in (("ir", "ir.json"), ("ir_mapped", "ir.mapped.json"),
                                                             ("render", "render.png"), ("figma_plan", "figma_plan.json"),
                                                             ("report", "report.md"), ("decisions", "decisions.json"))},
    }


@server.tool(description="Perceive a screenshot into IR JSON (no mapping/refine). Returns node counts and the "
                         "icon/font identification summary.")
@_on_worker
def perceive(image_path: str, out_path: str, dpr: float = 1.0) -> dict:
    from collections import Counter
    from dt.perceive import perceive as run
    doc = run(_abs(image_path), dpr=dpr)
    doc.save(_abs(out_path))
    return {"out_path": _abs(out_path), "size": [doc.width, doc.height],
            "node_types": dict(Counter(n.type for n in doc.walk())), "perceive": doc.meta.get("perceive", {})}


@server.tool(description="Render an IR JSON file to PNG with the pipeline renderer.")
@_on_worker
def render(ir_path: str, out_path: str) -> dict:
    from dt.ir import Document
    from dt.render.screenshot import render_doc
    doc = Document.load(_abs(ir_path))
    render_doc(doc, _abs(out_path))
    return {"out_path": _abs(out_path), "size": [doc.width, doc.height]}


@server.tool(description="Independent fidelity validation of a render against the target screenshot: gates "
                         "(pixel-exact / visually-identical / structurally-faithful) plus the measures behind them.")
@_on_worker
def validate(target_path: str, render_path: str, ir_path: Optional[str] = None, out_dir: Optional[str] = None) -> dict:
    from dt.common.image import load_rgb
    from dt.ir import Document
    from dt.validate import validate as run
    doc = Document.load(_abs(ir_path)) if ir_path else None
    rep = run(load_rgb(_abs(target_path)), load_rgb(_abs(render_path)), doc=doc, out_dir=_abs(out_dir) if out_dir else None)
    d = rep.to_dict()
    return {"gates": d["gates"], "gate_failures": d["gate_failures"], "fidelity": _fidelity_summary(d),
            "worst_regions": d.get("worst_regions", [])[:5], "artifacts": d.get("artifacts", {})}


@server.tool(description="Export an IR JSON file as a Figma build plan (for the bundled plugin). Returns plan "
                         "path, node count and validation errors.")
@_on_worker
def export_figma(ir_path: str, out_path: str) -> dict:
    from dt.export import save_plan, to_build_plan, validate_plan
    from dt.ir import Document
    plan = to_build_plan(Document.load(_abs(ir_path)))
    save_plan(plan, _abs(out_path))
    return {"out_path": _abs(out_path), "errors": validate_plan(plan), "meta": plan.get("meta", {})}


@server.tool(description="List the open decisions of a translate run (components, text, icons the engine could "
                         "not settle), each with candidates and evidence.")
@_on_worker
def list_decisions(run_dir: str, include_answered: bool = False) -> dict:
    p = os.path.join(_abs(run_dir), "decisions.json")
    if not os.path.exists(p):
        return {"decisions": [], "note": "no decisions.json in run_dir"}
    d = json.load(open(p))
    items = d.get("decisions", d) if isinstance(d, dict) else d
    if not include_answered:
        items = [x for x in items if isinstance(x, dict) and not x.get("answered")]
    return {"count": len(items), "decisions": items}


@server.tool(description="Apply answers to decisions of a translate run and refresh its outputs. answers maps "
                         "decision id to a choice: {name, variant} | {name: null} | {text} | {icon_name}.")
@_on_worker
def apply_decisions(run_dir: str, answers: dict) -> dict:
    from dt.pipeline import apply_decisions as run
    return run(_abs(run_dir), answers)


@server.tool(description="Run the ground-truth benchmark. With gate=True, compare against knowledge/baseline.json "
                         "and report whether the regression gate passes.")
@_on_worker
def bench(limit: Optional[int] = None, workers: int = 2, gate: bool = True) -> dict:
    from dt.selftest import bench as B
    rep = B.run(list(B.DEFAULT_CORPORA), limit=limit, workers=workers)
    out: dict[str, Any] = {"composite": rep["composite"], "n_cases": rep["n_cases"], "n_errors": rep.get("n_errors"),
                           "metrics": {k: v.get("mean") for k, v in rep["metrics"].items()}, "report_dir": rep.get("out_dir")}
    if gate:
        bp = os.path.join(ROOT, "knowledge", "baseline.json")
        if os.path.exists(bp):
            fails = B.check_regression(rep, json.load(open(bp)))
            out["gate"] = {"passed": not fails, "failures": fails}
        else:
            out["gate"] = {"passed": False, "failures": ["no knowledge/baseline.json"]}
    return out


@server.tool(description="Markdown brief of the worst benchmark cases with suspected stages, for deciding what "
                         "to fix next. Runs a bench first unless report_dir points at an existing run.")
@_on_worker
def improve_brief(top: int = 10, report_dir: Optional[str] = None, limit: Optional[int] = None) -> dict:
    from dt.cli import improve_brief as brief
    from dt.selftest import bench as B
    if report_dir and os.path.exists(os.path.join(_abs(report_dir), "report.json")):
        rep = json.load(open(os.path.join(_abs(report_dir), "report.json")))
    else:
        rep = B.run(list(B.DEFAULT_CORPORA), limit=limit, workers=2)
    md, rows = brief(rep, top, os.path.join(ROOT, "knowledge", "failures.jsonl"))
    return {"markdown": md, "rows": rows}


@server.tool(description="Stage attribution: substitute ground truth at each stage boundary and report how much "
                         "composite each stage (render, map, perceive) costs.")
@_on_worker
def attribution(limit: Optional[int] = None, workers: int = 2) -> dict:
    from dt.selftest import attribution as A
    res = A.run(limit=limit, workers=workers)
    return {"composite": res["composite"], "stage_cost": res["stage_cost"], "largest": res["largest"],
            "markdown": A.markdown(res)}


@server.tool(description="Refine-move statistics across translate runs under the given folders: which edit kinds "
                         "get accepted and how much loss each removes per attempt.")
@_on_worker
def move_stats(roots: Optional[list[str]] = None) -> dict:
    from dt.selftest import move_stats as M
    s = M.summarise(M.collect([_abs(r) for r in (roots or ["out"])]))
    return {"summary": s, "markdown": M.markdown(s)}


@server.tool(description="Record corrections to a translate run as local feedback (user-informed tuning; nothing "
                         "leaves the machine). items: [{kind, node_id, box, value, note}] with kind in component | text | "
                         "icon | geometry | color | missing | extra | should_be_image | should_be_editable | ok (see "
                         "docs/FEEDBACK.md); rating 1-5; consent {store_screenshot, share_screenshot, share_text}.")
@_on_worker
def submit_feedback(run_dir: str, items: list[dict], rating: Optional[int] = None, consent: Optional[dict] = None) -> dict:
    from dt.feedback.capture import add_corrections
    b = add_corrections(_abs(run_dir), {"items": items}, rating=rating, consent=consent or {}, channel="mcp")
    return {"bundle": b["id"], "items": len(b["items"]), "applied": len((b.get("apply_report") or {}).get("applied", [])),
            "skipped": (b.get("apply_report") or {}).get("skipped", []), "consent": b["consent"],
            "next": "dt feedback learn (local rules, gated) / dt feedback eval (accuracy on your cases)"}


@server.tool(description="Local feedback state: bundles and items by kind, learned rules (active), user-corpus size, "
                         "the last accuracy evaluation on the user's own cases, recent ledger entries.")
@_on_worker
def feedback_status() -> dict:
    from dt.feedback import status
    return status()


def main() -> None:  # pragma: no cover - transport entry point
    server.run("stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
