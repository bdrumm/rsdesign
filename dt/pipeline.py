"""End-to-end translation pipeline (module G): screenshot -> IR -> design system -> Figma plan.

    from dt.pipeline import translate
    metrics = translate("shot.png", ds="material3", out_dir="out/run1", refine_iters=8)

Stage order (every stage is wrapped; a failure is logged under ``metrics["errors"]`` and the
pipeline continues with the best document it has):

    load -> perceive -> map -> render/evaluate (before) -> refine -> re-map -> render/evaluate
    (after) -> export (Figma build plan) -> decisions -> validate -> report

Outputs written to ``out_dir`` (all best-effort, see :data:`OUTPUT_FILES`):

* ``ir.json``            perceived IR (before mapping / refinement)
* ``ir.mapped.json``     final IR: refined geometry + component refs + tokens
* ``render.png``         render of the final IR; ``render.before.png`` the first render
* ``diff.png``           target | render | ΔE heatmap with residual boxes (``diff.before.png`` too)
* ``figma_plan.json``    Figma build plan (``dt.export``), import with the plugin
* ``decisions.json``     judgment calls for a human/AI driver (see :func:`build_decisions`)
* ``metrics.json``       loss before/after, refine stats, components, timings, errors
* ``report.md``          human-readable summary linking everything above
* ``validation/``        independent fidelity gates (``dt.validate``), when enabled

Design-system specs accepted by :func:`resolve_design_system`: ``"material3"`` (or any name in
``fixtures/design_systems``), ``"figma:<file_key>"`` (needs ``FIGMA_TOKEN``), ``"screens:<dir>"``
(learn from reference screenshots), ``"tokens:<file.json>"`` (DTCG / Material Theme Builder tokens
over the Material 3 catalog), a path to a ``*.json`` design system, or ``None`` to skip mapping.
"""
from __future__ import annotations

import glob
import json
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from dt.common.image import downscale_dpr, load_rgb, save_rgb
from dt.ir import Box, ComponentRef, Document, Node, summarize
from dt.params import P, register

Logger = Optional[Callable[[str], None]]

# --------------------------------------------------------------------------- params
register("pipeline.refine_iters", 12, "Default refine iterations for `dt translate` (0 disables refine).", (0, 60))
register("pipeline.refine_patience", 3, "Refine iterations without an accepted move before stopping.", (1, 10))
register("pipeline.time_budget_s", 120.0, "Default wall-clock budget (s) for the refine stage of one translate run.", (5.0, 900.0))
register("pipeline.decisions.component_conf", 0.8,
         "Matched components below this confidence are also queued as decisions (with the match as the first candidate).", (0.5, 1.0))
register("pipeline.decisions.ocr_conf", 0.6, "Text nodes whose OCR confidence is below this are queued as decisions.", (0.0, 1.0))
register("pipeline.decisions.max_items", 200, "Cap on the number of entries written to decisions.json.", (10, 2000))
register("pipeline.report.max_components", 40, "Max component rows listed in report.md.", (5, 500))

OUTPUT_FILES = ("ir.json", "ir.mapped.json", "render.png", "diff.png", "figma_plan.json",
                "decisions.json", "metrics.json", "report.md")
"""Files a successful run writes to ``out_dir`` (``validation/`` is a directory, written when enabled)."""

STAGES = ("load", "perceive", "map", "evaluate_before", "refine", "remap", "evaluate_after",
          "export", "decisions", "validate", "report")


# --------------------------------------------------------------------------- design systems
def resolve_design_system(spec: Optional[str], log: Logger = None):
    """Turn a ``--ds`` spec into a :class:`dt.mapping.DesignSystem` (``None`` -> ``None``).

    See the module docstring for the accepted forms. Raises ``ValueError`` / ``FileNotFoundError``
    for unknown specs so the caller can record the failure.
    """
    if spec is None or spec == "" or spec.lower() == "none":
        return None
    from dt.mapping.design_system import DesignSystem

    say = log or (lambda _m: None)
    kind, _, arg = spec.partition(":")
    if kind == "figma" and arg:
        from dt.mapping.ingest_figma import from_figma_file
        say(f"design system: fetching Figma file {arg}")
        return from_figma_file(arg)
    if kind == "screens" and arg:
        from dt.mapping.ingest_screenshots import from_reference_screenshots
        paths = sorted(p for ext in ("png", "jpg", "jpeg") for p in glob.glob(os.path.join(arg, f"*.{ext}")))
        if not paths:
            raise FileNotFoundError(f"no screenshots under {arg}")
        say(f"design system: learning from {len(paths)} screenshots in {arg}")
        return from_reference_screenshots(paths, name=os.path.basename(os.path.normpath(arg)) or "learned")
    if kind == "tokens" and arg:
        from dt.mapping.ingest_library import design_system_from_tokens, load_tokens_file
        from dt.mapping.material3 import material3
        name = os.path.splitext(os.path.basename(arg))[0]
        say(f"design system: tokens from {arg} over material3")
        return design_system_from_tokens(name, load_tokens_file(arg), base=material3())
    if spec.endswith(".json") and os.path.exists(spec):
        return DesignSystem.load(spec)
    return DesignSystem.load_named(spec)


# --------------------------------------------------------------------------- IR helpers
def unmap_document(doc: Document) -> Document:
    """Remove component refs / tokens so :func:`dt.mapping.map_document` can run afresh (in place)."""
    for n in doc.walk():
        if n.component is not None or "orig_type" in n.meta:
            n.type = n.meta.pop("orig_type", "frame" if n.type == "instance" else n.type)
        n.component = None
        n.tokens = {}
        n.meta.pop("component_candidates", None)
        n.meta.pop("tokens_evidence", None)
    doc.design_system = None
    doc.meta.pop("mapping", None)
    return doc


def components_found(doc: Document) -> list[dict]:
    """Flat list of matched components: ``{node_id, name, variant, confidence, box, props}``."""
    out = []
    for n in doc.walk():
        c = n.component
        if c is None:
            continue
        out.append({"node_id": n.id, "name": c.name, "key": c.key, "variant": dict(c.variant),
                    "confidence": round(float(c.confidence), 3), "box": n.box.to_dict(),
                    "props": {k: v for k, v in c.props.items() if isinstance(v, (str, int, float, bool)) or v is None}})
    return out


def _node_evidence(n: Node) -> dict:
    fill = n.fill_color
    return {
        "type": n.meta.get("orig_type", n.type), "box": n.box.to_dict(), "fill": fill.hex() if fill else None,
        "radius": round(n.uniform_radius(), 1),
        "stroke": {"color": n.strokes[0].color.hex(), "width": n.strokes[0].width} if n.strokes else None,
        "shadow": bool(n.effects), "children": [c.meta.get("orig_type", c.type) for c in n.children],
        "texts": [c.text for c in n.children if c.type == "text" and c.text][:6],
        "icons": [c.icon_name for c in n.children if c.type == "icon"][:6],
    }


def build_decisions(doc: Document) -> list[dict]:
    """Questions the engine could not settle, for ``decisions.json``.

    Kinds:

    * ``component`` -- a frame with component candidates but no confident match, or a match
      below ``pipeline.decisions.component_conf``; ``candidates`` are the top-3 with scores.
    * ``text`` -- OCR line with confidence below ``pipeline.decisions.ocr_conf``.
    * ``icon`` -- icon glyph whose Material Symbol name is unknown (renders empty).

    Each entry: ``{id, kind, node_id, question, candidates, evidence, answer: null}``. Answers
    are merged back by :func:`apply_decisions`.
    """
    comp_conf = float(P["pipeline.decisions.component_conf"])
    ocr_conf = float(P["pipeline.decisions.ocr_conf"])
    items: list[dict] = []
    for n in doc.walk():
        cands = n.meta.get("component_candidates")
        if n.component is not None and n.component.confidence < comp_conf:
            cands = n.component.evidence.get("candidates") or [
                {"name": n.component.name, "key": n.component.key, "variant": dict(n.component.variant), "score": n.component.confidence}]
            items.append({"kind": "component", "node_id": n.id,
                          "question": f"Low-confidence match {n.component.name}{n.component.variant} ({n.component.confidence:.2f}); confirm or pick another",
                          "candidates": cands[:3], "evidence": _node_evidence(n), "current": n.component.to_dict()})
        elif n.component is None and cands:
            items.append({"kind": "component", "node_id": n.id,
                          "question": "Which component (if any) is this frame?",
                          "candidates": list(cands)[:3], "evidence": _node_evidence(n), "current": None})
        if n.type == "text" and n.text is not None:
            conf = n.meta.get("conf")
            if conf is not None and float(conf) < ocr_conf:
                items.append({"kind": "text", "node_id": n.id, "question": f"OCR read {n.text!r} with confidence {float(conf):.2f}; correct text?",
                              "candidates": [{"text": n.text, "score": float(conf)}],
                              "evidence": {"box": n.box.to_dict(), "size": n.text_style.size if n.text_style else None}, "current": n.text})
        if n.type == "icon" and not n.icon_name:
            col = n.fill_color or (n.text_style.color if n.text_style else None)
            items.append({"kind": "icon", "node_id": n.id, "question": "Unknown icon glyph: which Material Symbol name?",
                          "candidates": [], "evidence": {"box": n.box.to_dict(), "color": col.hex() if col else None,
                                                         "ocr_word": n.meta.get("ocr_icon_word", False)}, "current": None})
    order = {"component": 0, "text": 1, "icon": 2}
    items.sort(key=lambda it: (order.get(it["kind"], 9), -Box.from_dict(it["evidence"]["box"]).area if it["evidence"].get("box") else 0))
    items = items[: int(P["pipeline.decisions.max_items"])]
    for i, it in enumerate(items):
        it["id"] = f"d{i + 1}"
        it["answer"] = None
    return items


def apply_decisions(run_dir: str, answers: dict[str, Any], ds_spec: Optional[str] = None, log: Logger = None) -> dict:
    """Merge ``answers`` (decision id -> choice) into ``<run_dir>/ir.mapped.json`` and refresh outputs.

    Choice shapes: ``{"name": "Card", "variant": {"style": "elevated"}}`` (component; ``{"name": null}``
    rejects the match), ``{"text": "Inbox"}``, ``{"icon_name": "search"}``. Re-exports
    ``figma_plan.json`` and re-renders ``render.png``; marks the entries answered in
    ``decisions.json``. Returns ``{applied, unknown, skipped, errors}``.
    """
    say = log or (lambda _m: None)
    dec_path = os.path.join(run_dir, "decisions.json")
    ir_path = os.path.join(run_dir, "ir.mapped.json")
    decisions = json.load(open(dec_path)) if os.path.exists(dec_path) else []
    doc = Document.load(ir_path)
    by_id = {d["id"]: d for d in decisions}
    res: dict[str, Any] = {"applied": [], "unknown": [], "skipped": [], "errors": []}
    ds = None
    if ds_spec or doc.design_system:
        try:
            ds = resolve_design_system(ds_spec or doc.design_system, say)
        except Exception as e:  # noqa: BLE001
            res["errors"].append(f"design system: {e}")
    for did, choice in answers.items():
        d = by_id.get(did)
        if d is None:
            res["unknown"].append(did)
            continue
        node = doc.find(d["node_id"])
        if node is None:
            res["skipped"].append(did)
            continue
        try:
            _apply_choice(node, d["kind"], choice or {}, ds)
            d["answer"] = choice
            res["applied"].append(did)
        except Exception as e:  # noqa: BLE001
            res["errors"].append(f"{did}: {e}")
    doc.save(ir_path)
    with open(dec_path, "w") as f:
        json.dump(decisions, f, indent=2)
    try:
        from dt.export import save_plan, to_build_plan
        save_plan(to_build_plan(doc), os.path.join(run_dir, "figma_plan.json"))
        from dt.render.screenshot import render_doc
        render_doc(doc, os.path.join(run_dir, "render.png"))
    except Exception as e:  # noqa: BLE001
        res["errors"].append(f"refresh outputs: {e}")
    return res


def _apply_choice(node: Node, kind: str, choice: dict, ds) -> None:
    if kind == "component":
        name = choice.get("name")
        if not name:
            if node.component is not None:
                node.type = node.meta.pop("orig_type", "frame")
                node.component = None
            return
        variant = dict(choice.get("variant") or {})
        key, library = choice.get("key", ""), choice.get("library", "")
        if ds is not None:
            spec = next((s for s in ds.components if s.name.lower() == str(name).lower()), None)
            if spec is not None:
                key, library, name = key or spec.key, library or spec.library or ds.name, spec.name
        props = dict(node.component.props) if node.component else {}
        node.component = ComponentRef(key=key or f"decision:{name}", name=str(name), library=library, variant=variant,
                                      props=props, confidence=1.0, evidence={"source": "decision"})
        if node.type != "instance":
            node.meta["orig_type"] = node.type
            node.type = "instance"
    elif kind == "text":
        if "text" in choice:
            node.text = str(choice["text"])
            node.meta["conf"] = 1.0
            node.meta["decision"] = True
    elif kind == "icon":
        if choice.get("icon_name"):
            node.icon_name = str(choice["icon_name"])
            node.meta["decision"] = True
    else:
        raise ValueError(f"unknown decision kind {kind!r}")


# --------------------------------------------------------------------------- run state
@dataclass
class _Run:
    """Mutable state of one translate run (so stage functions stay small)."""

    image_path: str
    out_dir: str
    ds_spec: Optional[str]
    log: Callable[[str], None]
    target: Optional[np.ndarray] = None
    doc: Optional[Document] = None
    ds: Any = None
    rendered: Optional[np.ndarray] = None
    report_before: Any = None
    report_after: Any = None
    history: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    plan: Optional[dict] = None
    validation: Optional[dict] = None
    metrics: dict = field(default_factory=dict)

    def path(self, name: str) -> str:
        return os.path.join(self.out_dir, name)

    def stage(self, name: str, fn: Callable[[], None], critical: bool = False) -> bool:
        """Run ``fn`` under a timer; log failures into ``metrics['errors']``. Returns success."""
        t0 = time.perf_counter()
        try:
            fn()
            ok = True
        except Exception as e:  # noqa: BLE001 - every stage must be survivable
            ok = False
            self.metrics["errors"].append({"stage": name, "error": f"{type(e).__name__}: {e}",
                                           "traceback": traceback.format_exc()})
            self.log(f"[{name}] FAILED: {type(e).__name__}: {e}")
            if critical:
                raise
        self.metrics["timings"][name] = round(time.perf_counter() - t0, 3)
        if ok:
            self.metrics["stages_completed"].append(name)
        return ok


def _loss_summary(report) -> dict:
    d = report.to_dict()
    return {"total": d["total"], "pixel": d["pixel"], "text_term": d["structure"].get("text_term"),
            "alignment": d["alignment"], "n_residual_regions": len(d["residual_regions"])}


# --------------------------------------------------------------------------- the pipeline
def translate(image_path: str, ds: Optional[str] = "material3", out_dir: str = "out/run",
              refine_iters: Optional[int] = None, dpr: "float | str" = 1.0, time_budget_s: Optional[float] = None,
              validate: bool = True, resume_from: Optional[str] = None, log: Logger = None) -> dict:
    """Run the whole pipeline on ``image_path``; returns the ``metrics.json`` dict.

    * ``ds`` -- design-system spec (see :func:`resolve_design_system`); ``None`` skips mapping.
    * ``refine_iters`` -- max refine iterations (default ``P['pipeline.refine_iters']``; 0 = off).
    * ``dpr`` -- device pixel ratio of the screenshot (a @2x capture is downscaled to DPR 1); ``"auto"``
      detects it from the image width and the OCR'd text sizes (dt.perceive.dpr).
    * ``time_budget_s`` -- wall-clock cap for refine (default ``P['pipeline.time_budget_s']``).
    * ``validate`` -- also run the independent ``dt.validate`` gates into ``out_dir/validation``.
    * ``resume_from`` -- a previous run dir: its ``ir.mapped.json`` replaces perceive + map (used
      after ``apply_decisions``).

    Never raises for stage failures: ``metrics['errors']`` lists them and ``metrics['ok']`` is
    False only when no document could be produced at all.
    """
    os.makedirs(out_dir, exist_ok=True)
    say = log or (lambda _m: None)
    run = _Run(image_path=image_path, out_dir=out_dir, ds_spec=ds, log=say)
    run.metrics = {
        "source": os.path.abspath(image_path), "out_dir": os.path.abspath(out_dir), "dpr": dpr, "design_system": ds,
        "ok": False, "errors": [], "stages_completed": [], "timings": {}, "loss_before": None, "loss_after": None,
        "refine": {"iterations": 0, "accepted": 0, "tried": 0, "renders": 0, "reason": None},
        "components": {"n_before": 0, "n_after": 0, "found": []}, "decisions": {"n": 0, "by_kind": {}},
        "export": None, "validation": None, "files": {},
    }
    t_start = time.perf_counter()
    iters = int(P["pipeline.refine_iters"] if refine_iters is None else refine_iters)
    budget = float(P["pipeline.time_budget_s"] if time_budget_s is None else time_budget_s)

    # ---- load
    def load() -> None:
        nonlocal dpr
        rgb = load_rgb(image_path)
        if isinstance(dpr, str) or dpr is None or float(dpr) <= 0:
            from dt.perceive.dpr import detect_dpr
            dpr, ev = detect_dpr(rgb)
            run.metrics["dpr_detected"] = {"dpr": dpr, **ev}
            run.metrics["dpr"] = dpr
        if float(dpr) != 1.0:
            rgb = downscale_dpr(rgb, float(dpr))
        run.target = rgb
        run.metrics["size"] = {"w": int(rgb.shape[1]), "h": int(rgb.shape[0])}

    try:
        run.stage("load", load, critical=True)
    except Exception:
        _write_metrics(run, t_start)
        return run.metrics

    # ---- perceive (or resume)
    def perceive_stage() -> None:
        if resume_from:
            src = os.path.join(resume_from, "ir.mapped.json")
            if not os.path.exists(src):
                src = os.path.join(resume_from, "ir.json")
            run.doc = Document.load(src)
            run.doc.source_image = os.path.abspath(image_path)
            say(f"[perceive] resumed from {src}")
        else:
            from dt.perceive import perceive
            run.doc = perceive(run.target)
            run.doc.source_image = os.path.abspath(image_path)
            say(f"[perceive] {run.doc.root.count()} nodes, {len(run.doc.texts())} texts")
        run.doc.save(run.path("ir.json"))
        run.metrics["files"]["ir"] = "ir.json"
        run.metrics["n_nodes_perceived"] = run.doc.root.count()

    if not run.stage("perceive", perceive_stage):
        run.doc = _fallback_document(run.target, image_path)
        run.doc.save(run.path("ir.json"))

    # ---- design system + map
    def map_stage() -> None:
        if ds is None:
            say("[map] skipped (no design system)")
            return
        from dt.mapping import map_document
        run.ds = resolve_design_system(ds, say)
        if run.ds is None:
            return
        if any(n.component is not None for n in run.doc.walk()):
            unmap_document(run.doc)
        run.doc = map_document(run.doc, run.ds)
        run.metrics["design_system"] = run.ds.name
        run.metrics["components"]["n_before"] = sum(1 for n in run.doc.walk() if n.component is not None)
        say(f"[map] {run.metrics['components']['n_before']} components matched against {run.ds.name}")

    run.stage("map", map_stage)
    run.doc.save(run.path("ir.mapped.json"))
    run.metrics["files"]["ir_mapped"] = "ir.mapped.json"

    # ---- evaluate before
    def evaluate_before() -> None:
        from dt.compare import evaluate, save_diff_image
        from dt.render.screenshot import render_doc
        run.rendered = render_doc(run.doc, run.path("render.before.png"))
        run.report_before = evaluate(run.doc, run.target, run.rendered)
        run.metrics["loss_before"] = float(run.report_before.total)
        run.metrics["loss_before_detail"] = _loss_summary(run.report_before)
        save_diff_image(run.target, run.rendered, run.report_before.diff_map, run.path("diff.before.png"),
                        run.report_before.residual_regions)
        say(f"[evaluate] loss before refine {run.report_before.total:.5f} (mean ΔE {run.report_before.pixel.get('mean_de', 0):.2f})")

    run.stage("evaluate_before", evaluate_before)

    # ---- refine
    def refine_stage() -> None:
        if iters <= 0:
            say("[refine] skipped (0 iterations)")
            return
        from dt.refine.optimizer import accepted_moves, refine
        patience = int(P["pipeline.refine_patience"])
        new_doc, hist = refine(run.doc, run.target, max_iters=iters, patience=patience, time_budget_s=budget, log=say)
        run.doc, run.history = new_doc, hist
        stop = next((h for h in reversed(hist) if h.get("kind") == "stop"), {})
        acc = accepted_moves(hist)
        run.metrics["refine"] = {
            "iterations": int(stop.get("iter", 0)), "accepted": len(acc), "tried": sum(1 for h in hist if h.get("kind") != "stop"),
            "renders": int(stop.get("params", {}).get("renders", 0)), "reason": stop.get("params", {}).get("reason"),
            "loss_start": stop.get("before"), "loss_end": stop.get("after"),
            "accepted_by_kind": _count_by(acc, "kind"), "max_iters": iters, "time_budget_s": budget,
        }
        with open(run.path("refine_history.json"), "w") as f:
            json.dump(hist, f, indent=1, default=str)
        run.metrics["files"]["refine_history"] = "refine_history.json"

    run.stage("refine", refine_stage)

    # ---- re-map after refine (geometry/colours may have changed)
    def remap_stage() -> None:
        if run.ds is None:
            return
        from dt.mapping import map_document
        unmap_document(run.doc)
        run.doc = map_document(run.doc, run.ds)
        say(f"[remap] {sum(1 for n in run.doc.walk() if n.component is not None)} components after refine")

    run.stage("remap", remap_stage)
    run.metrics["components"]["found"] = components_found(run.doc)
    run.metrics["components"]["n_after"] = len(run.metrics["components"]["found"])
    run.metrics["components"]["by_name"] = _count_by(run.metrics["components"]["found"], "name")
    run.doc.save(run.path("ir.mapped.json"))

    # ---- evaluate after
    def evaluate_after() -> None:
        from dt.compare import evaluate, save_diff_image
        from dt.render.screenshot import render_doc
        run.rendered = render_doc(run.doc, run.path("render.png"))
        run.report_after = evaluate(run.doc, run.target, run.rendered)
        run.metrics["loss_after"] = float(run.report_after.total)
        run.metrics["loss_after_detail"] = _loss_summary(run.report_after)
        run.metrics["worst_nodes"] = [{"node_id": k, "mean_de": round(v, 3)} for k, v in run.report_after.worst_nodes(8)]
        save_diff_image(run.target, run.rendered, run.report_after.diff_map, run.path("diff.png"), run.report_after.residual_regions)
        run.metrics["files"].update(render="render.png", diff="diff.png")
        say(f"[evaluate] loss after refine {run.report_after.total:.5f} (mean ΔE {run.report_after.pixel.get('mean_de', 0):.2f})")

    if not run.stage("evaluate_after", evaluate_after) and run.rendered is not None:
        # keep the best available render/diff so the output set is complete
        save_rgb(run.rendered, run.path("render.png"))
        if run.report_before is not None and run.report_before.diff_map is not None:
            from dt.compare import save_diff_image
            save_diff_image(run.target, run.rendered, run.report_before.diff_map, run.path("diff.png"), run.report_before.residual_regions)
        run.metrics["loss_after"] = run.metrics["loss_before"]

    # ---- export
    def export_stage() -> None:
        from dt.export import save_plan, to_build_plan, validate_plan
        run.plan = to_build_plan(run.doc)
        errors = validate_plan(run.plan)
        save_plan(run.plan, run.path("figma_plan.json"))
        run.metrics["export"] = {"nodes": run.plan["meta"].get("nodeCount"), "warnings": len(run.plan.get("warnings", [])),
                                 "errors": errors, "fonts": run.plan.get("fonts", [])}
        run.metrics["files"]["figma_plan"] = "figma_plan.json"
        say(f"[export] build plan: {run.metrics['export']['nodes']} nodes, {run.metrics['export']['warnings']} warnings, {len(errors)} errors")

    run.stage("export", export_stage)

    # ---- decisions
    def decisions_stage() -> None:
        run.decisions = build_decisions(run.doc)
        with open(run.path("decisions.json"), "w") as f:
            json.dump(run.decisions, f, indent=2)
        run.metrics["decisions"] = {"n": len(run.decisions), "by_kind": _count_by(run.decisions, "kind")}
        run.metrics["files"]["decisions"] = "decisions.json"
        say(f"[decisions] {len(run.decisions)} open ({run.metrics['decisions']['by_kind']})")

    run.stage("decisions", decisions_stage)

    # ---- independent validation
    def validate_stage() -> None:
        if not validate or run.rendered is None:
            return
        from dt.validate import validate as validate_fn
        rep = validate_fn(run.target, run.rendered, doc=run.doc, out_dir=run.path("validation"))
        run.validation = rep.to_dict()
        run.metrics["validation"] = {
            "gates": rep.gates, "gate_failures": rep.gate_failures, "jnd_frac_nontext": rep.jnd_frac_nontext,
            "chamfer": rep.chamfer, "edge_iou": rep.edge_iou, "text_cer": rep.text_cer, "region_de_worst": rep.region_de_worst,
        }
        run.metrics["files"]["validation"] = "validation/validation.json"
        say(f"[validate] gates {rep.gates}")

    run.stage("validate", validate_stage)

    # ---- report + metrics
    run.metrics["ok"] = run.doc is not None and "perceive" in run.metrics["stages_completed"]
    run.metrics["timings"]["total"] = round(time.perf_counter() - t_start, 3)
    run.stage("report", lambda: _write_report(run))
    _write_metrics(run, t_start)
    say(f"[done] {out_dir} in {run.metrics['timings']['total']:.1f}s, {len(run.metrics['errors'])} errors")
    return run.metrics


# --------------------------------------------------------------------------- helpers
def _fallback_document(target: np.ndarray, image_path: str) -> Document:
    """A root-only document in the screenshot's dominant colour (used when perceive fails)."""
    from dt.common.image import dominant_color
    h, w = target.shape[:2]
    bg = dominant_color(target) or None
    doc = Document.blank(w, h, bg or "#ffffff")
    doc.source_image = os.path.abspath(image_path)
    doc.meta["fallback"] = "perceive failed; root-only document"
    return doc


def _count_by(items: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        k = str(it.get(key))
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _write_metrics(run: _Run, t_start: float) -> None:
    run.metrics["timings"]["total"] = round(time.perf_counter() - t_start, 3)
    if run.metrics.get("loss_before") is not None and run.metrics.get("loss_after") is not None:
        lb, la = run.metrics["loss_before"], run.metrics["loss_after"]
        run.metrics["loss_gain"] = round(lb - la, 6)
        run.metrics["loss_gain_frac"] = round((lb - la) / lb, 4) if lb > 0 else 0.0
    run.metrics["files"]["metrics"] = "metrics.json"
    run.metrics["files"]["report"] = "report.md"
    with open(run.path("metrics.json"), "w") as f:
        json.dump(run.metrics, f, indent=2, default=str)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.5f}" if abs(v) < 1 else f"{v:.3f}"
    return str(v)


def _write_report(run: _Run) -> None:
    m = run.metrics
    L = [f"# designtranslator run: {os.path.basename(run.image_path)}", ""]
    size = m.get("size", {})
    L += ["## Summary", "", "| field | value |", "|---|---|",
          f"| source | `{m['source']}` |", f"| size | {size.get('w')}x{size.get('h')} (dpr {m['dpr']}) |",
          f"| design system | {m.get('design_system')} |", f"| nodes (perceived / final) | {m.get('n_nodes_perceived')} / {run.doc.root.count() if run.doc else '-'} |",
          f"| loss before / after | {_fmt(m.get('loss_before'))} / {_fmt(m.get('loss_after'))} |",
          f"| refine | {m['refine'].get('accepted')} accepted of {m['refine'].get('tried')} tried, {m['refine'].get('iterations')} iters, {m['refine'].get('renders')} renders, stop={m['refine'].get('reason')} |",
          f"| components matched (before / after refine) | {m['components'].get('n_before')} / {m['components'].get('n_after')} |",
          f"| open decisions | {m['decisions'].get('n')} {m['decisions'].get('by_kind')} |",
          f"| errors | {len(m['errors'])} |", f"| wall time | {m['timings'].get('total', '-')} s |", ""]
    if m.get("validation"):
        v = m["validation"]
        L += ["## Validation gates (independent, dt.validate)", "", "| gate | result |", "|---|---|"]
        for g, ok in v["gates"].items():
            L.append(f"| {g} | {'PASS' if ok else 'FAIL: ' + '; '.join(v['gate_failures'].get(g, []))} |")
        L += ["", f"non-text JND fraction {v['jnd_frac_nontext']:.4f} · chamfer {v['chamfer']:.2f}px · edge IoU {v['edge_iou']:.3f} · text CER {v['text_cer']:.3f} · worst region ΔE {v['region_de_worst']:.1f}",
              "", "Artifacts: `validation/side_by_side.png`, `validation/heatmap.png`, `validation/blink.html`, `validation/worst_regions.png`", ""]
    for label, key in (("before refine", "loss_before_detail"), ("after refine", "loss_after_detail")):
        d = m.get(key)
        if d:
            px = d["pixel"]
            L += [f"## Loss {label}", "", "| metric | value |", "|---|---|", f"| total | {_fmt(d['total'])} |",
                  *[f"| {k} | {_fmt(px[k])} |" for k in ("mean_de", "frac_bad", "ssim", "psnr", "max_de") if k in px],
                  f"| text term (OCR) | {_fmt(d.get('text_term'))} |", f"| residual regions | {d['n_residual_regions']} |",
                  f"| global alignment dx,dy | {d['alignment']['dx']:.2f}, {d['alignment']['dy']:.2f} |", ""]
    comps = m["components"].get("found", [])
    L += [f"## Components ({len(comps)})", ""]
    if comps:
        L += ["| node | component | variant | conf | box |", "|---|---|---|---|---|"]
        for c in comps[: int(P["pipeline.report.max_components"])]:
            b = c["box"]
            L.append(f"| {c['node_id']} | {c['name']} | {c['variant']} | {c['confidence']:.2f} | {b['x']:.0f},{b['y']:.0f} {b['w']:.0f}x{b['h']:.0f} |")
        if len(comps) > int(P["pipeline.report.max_components"]):
            L.append(f"| ... | {len(comps) - int(P['pipeline.report.max_components'])} more | | | |")
    else:
        L.append("_none_")
    L.append("")
    if run.history:
        from dt.refine.optimizer import accepted_moves
        acc = accepted_moves(run.history)
        L += [f"## Accepted refine moves ({len(acc)})", ""]
        if acc:
            L += ["| iter | kind | node | params | loss before -> after |", "|---|---|---|---|---|"]
            for h in acc[:40]:
                params = {k: v for k, v in (h.get("params") or {}).items() if not str(k).startswith("_")}
                L.append(f"| {h.get('iter')} | {h.get('kind')} | {h.get('node_id')} | {json.dumps(params, default=str)[:80]} | {_fmt(h.get('before'))} -> {_fmt(h.get('after'))} |")
        else:
            L.append("_none accepted_")
        L.append("")
    if run.decisions:
        L += [f"## Decisions ({len(run.decisions)}, see decisions.json)", "", "| id | kind | node | question | top candidate |", "|---|---|---|---|---|"]
        for d in run.decisions[:30]:
            top = d["candidates"][0] if d["candidates"] else {}
            L.append(f"| {d['id']} | {d['kind']} | {d['node_id']} | {d['question'][:70]} | {top.get('name') or top.get('text') or ''} {top.get('variant', '') if top.get('name') else ''} |")
        L.append("")
    if m.get("export"):
        e = m["export"]
        L += ["## Export", "", f"- `figma_plan.json`: {e['nodes']} nodes, fonts {', '.join(f"{fo['family']} {fo['style']}" for fo in e.get('fonts', [])) or '-'}",
              f"- plan warnings: {e['warnings']}", f"- plan validation errors: {len(e['errors'])}" + (": " + "; ".join(e["errors"][:5]) if e["errors"] else ""),
              "- import: Figma -> Plugins -> Development -> Import plugin from manifest -> `dt/export/figma_plugin/manifest.json` -> paste the plan", ""]
    if m["errors"]:
        L += ["## Errors", ""] + [f"- **{e['stage']}**: {e['error']}" for e in m["errors"]] + [""]
    L += ["## Images", "", "| target | render | diff |", "|---|---|---|",
          f"| ![target]({os.path.relpath(m['source'], run.out_dir)}) | ![render](render.png) | ![diff](diff.png) |", ""]
    if run.doc is not None:
        L += ["## Final IR tree", "", "```", summarize(run.doc, max_nodes=80), "```", ""]
    L += ["## Files", ""] + [f"- `{v}`: {k}" for k, v in m["files"].items()] + [""]
    with open(run.path("report.md"), "w") as f:
        f.write("\n".join(L))


__all__ = ["translate", "resolve_design_system", "build_decisions", "apply_decisions", "unmap_document",
           "components_found", "OUTPUT_FILES", "STAGES"]
