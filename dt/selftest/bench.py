"""Benchmark the pipeline against the ground-truth corpora (module E, phase 2).

For every corpus case (``<id>.png`` + ``<id>.gt.json``) the bench runs

    perceive(png) -> map_document(doc, material3) -> render_doc(doc)

and scores the result: :func:`dt.compare.structural_metrics` (pred IR vs gt IR), pixel metrics
of the render against the gt screenshot, the independent validator's fidelity measures
(:func:`fidelity_metrics`: non-text JND fraction, worst-tile chamfer, edges within 1 px) and
``layout_consistency`` of the inferred auto-layout.
Metrics are aggregated (mean / p50 / worst) and folded into one composite in [0, 1]
(:mod:`dt.selftest.metrics`, weights under ``bench.w.*``).

    from dt.selftest import bench
    report = bench.run(["fixtures/corpus/synth", "fixtures/corpus/mwc"], run_id="r1")
    report["composite"], report["metrics"]["node_recall"]["mean"], report["worst"]

Outputs (when ``out_root`` is not None): ``out/bench/<run_id>/report.json``, ``report.md`` (with a
worst-k table: case, metric, value, suspected stage) and ``diffs/<case>.png`` side-by-side
target | rendered | ΔE heatmap images from :mod:`dt.compare.visualize`.

Stages (``stages`` argument) can be switched off to isolate a module:

* without ``"perceive"`` the ground-truth IR is used as the prediction (so ``map`` is scored on
  perfect geometry and ``render`` measures pure render fidelity);
* without ``"map"`` no component/token metrics are produced;
* without ``"render"`` no pixel/layout metrics (and no diff images) are produced.

Ground truth discounting: gt nodes flagged ``meta.occluded`` (hidden under a later opaque
box) are ignored, and nodes under a translucent overlay (``meta.overlay_alpha``) do not take
part in the colour metric, because a perceiver cannot see their true paint.

Regression gate (ships with the repo): ``save_baseline(report)`` writes the tracked
``knowledge/baseline.json`` (metrics version, stages, n_cases, case ids, weights, normalisers,
composite, per-metric aggregates incl. goodness, per-case composites) and
``check_regression(report, baseline)`` lists the failures: composite drop >
``bench.gate.eps``, any metric's goodness drop > ``bench.gate.metric_eps``, or an incomparable
run (different metrics version / stages / cases / weights). ``gate_table`` gives the per-metric
delta table. CLI: ``dt bench --set-baseline`` / ``dt bench --gate`` (exit 1 on regression), or
``python -m dt.selftest.bench --help``.

No silent caps: cases that are listed but cannot run (missing png / gt json), gt files absent
from the manifest and cases skipped by ``limit`` are reported under ``report["dropped"]``; cases
that raised (including a crashed worker process) are scored 0 and listed under
``report["errors"]``. Both are logged to stderr even with ``verbose=False``.

Thread/process model: ``workers=1`` runs in-process; ``workers>1`` uses a spawn process pool
(each worker owns its own Chrome). Parameter overrides (used by the tuner) are forwarded to
workers explicitly, so results do not depend on inherited process state.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from typing import Any, Iterable, Iterator, Optional, Sequence

from dt.ir import Document
from dt.params import P
from dt.selftest.metrics import (EDITABILITY_KEY, FRAME_KEY, METRICS, METRICS_VERSION, aggregate, case_composite,
                                 definition_params, worst_entries)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CORPORA: tuple[str, ...] = (
    os.path.join(ROOT, "fixtures", "corpus", "synth"),
    os.path.join(ROOT, "fixtures", "corpus", "mwc"),
)
DEFAULT_OUT_ROOT = os.path.join(ROOT, "out", "bench")
BASELINE_PATH = os.path.join(ROOT, "knowledge", "baseline.json")
ALL_STAGES: tuple[str, ...] = ("perceive", "map", "render")

Case = tuple[str, str]
"""``(corpus_dir, case_id)``."""


# ----------------------------------------------------------------------------- params context
@contextlib.contextmanager
def param_overrides(overrides: Optional[dict[str, Any]]) -> Iterator[None]:
    """Temporarily apply ``overrides`` to :data:`dt.params.P`; restores previous values on exit."""
    if not overrides:
        yield
        return
    before_all, defaults = P.all(), P.defaults()
    was_overridden = {k: (k in before_all and before_all[k] != defaults.get(k)) for k in overrides}
    for k, v in overrides.items():
        P.set(k, v)
    try:
        yield
    finally:
        for k in overrides:
            if was_overridden[k]:
                P.set(k, before_all[k])
            else:
                P.reset(k)


# ----------------------------------------------------------------------------- cases
def scan_cases(corpus_dirs: Iterable[str], limit: Optional[int] = None) -> tuple[list[Case], list[dict]]:
    """``(cases, dropped)``: runnable ``(corpus_dir, id)`` pairs from each corpus
    ``manifest.json`` (falls back to globbing ``*.gt.json``), plus one ``{corpus, id, reason}``
    entry for everything that will *not* run — listed ids missing their png / gt json, gt files
    absent from the manifest, a missing corpus dir, and ids beyond ``limit`` (per corpus)."""
    out: list[Case] = []
    dropped: list[dict] = []
    for d in corpus_dirs:
        d = os.path.abspath(d)
        name = os.path.basename(os.path.normpath(d))
        if not os.path.isdir(d):
            dropped.append({"corpus": name, "id": None, "reason": f"corpus dir not found: {d}"})
            continue
        on_disk = sorted(fn[: -len(".gt.json")] for fn in os.listdir(d) if fn.endswith(".gt.json"))
        man = os.path.join(d, "manifest.json")
        if os.path.exists(man):
            with open(man) as f:
                ids = [c["id"] for c in json.load(f).get("cases", [])]
            listed = set(ids)
            dropped.extend({"corpus": name, "id": i, "reason": "gt.json not listed in manifest.json"}
                           for i in on_disk if i not in listed)
        else:
            ids = on_disk
        ok: list[str] = []
        for i in ids:
            missing = [ext for ext in (".png", ".gt.json") if not os.path.exists(os.path.join(d, i + ext))]
            if missing:
                dropped.append({"corpus": name, "id": i, "reason": "missing " + ", ".join(i + m for m in missing)})
            else:
                ok.append(i)
        if limit is not None and len(ok) > int(limit):
            dropped.extend({"corpus": name, "id": i, "reason": f"over limit={int(limit)}"} for i in ok[int(limit):])
            ok = ok[: int(limit)]
        out.extend((d, i) for i in ok)
    return out, dropped


def find_cases(corpus_dirs: Iterable[str], limit: Optional[int] = None) -> list[Case]:
    """Runnable ``(corpus_dir, id)`` pairs (see :func:`scan_cases`, which also reports what is
    dropped). ``limit`` caps the number of cases taken *per corpus*."""
    return scan_cases(corpus_dirs, limit)[0]


def load_case(corpus_dir: str, cid: str) -> tuple[Document, str]:
    """``(gt_document, png_path)`` of a corpus case."""
    doc = Document.load(os.path.join(corpus_dir, cid + ".gt.json"))
    png = os.path.join(corpus_dir, cid + ".png")
    doc.source_image = png
    return doc, png


def discount_gt(gt: Document) -> Document:
    """Copy of ``gt`` with occluded nodes hidden and nodes whose true paint a perceiver cannot
    observe (under a translucent overlay, or themselves translucent such as a dialog scrim)
    stripped of fills so they do not enter the colour metric."""
    g = Document.from_dict(gt.to_dict())
    opaque = float(P["bench.gt.opaque_alpha"])
    for n in g.walk():
        if n.meta.get("occluded"):
            n.visible = False
        elif n.meta.get("overlay_alpha") or any(f.color is not None and f.color.a * f.opacity < opaque for f in n.fills):
            n.fills = []
    return g


def extra_metrics(pred: Document, gt: Document, matches: list[tuple[str, str, float]]) -> dict[str, Optional[float]]:
    """Diagnostic metrics (information only, not in the composite):

    * ``component_name_acc``: like ``component_acc`` but on the component *name* alone, so a
      variant-vocabulary mismatch between corpus and matcher does not hide correct names;
    * ``text_line_recall``: like ``text_recall`` but a predicted paragraph counts for each of
      its lines (gt corpora hold one node per rendered line, perceive groups paragraphs).
    """
    from dt.compare import text_cer
    from dt.compare.structural import normalize_text
    pm = {n.id: n for n in pred.walk()}
    mp = {g: pm[p] for p, g, _ in matches if p in pm}
    names: list[float] = []
    for g in gt.walk():
        if g.component is None or not g.visible:
            continue
        p = mp.get(g.id)
        ok = p is not None and p.component is not None and p.component.name.strip().lower() == g.component.name.strip().lower()
        names.append(1.0 if ok else 0.0)
    thr = float(P["compare.struct.text_match_cer"])
    pred_texts = [n for n in pred.walk() if n.type == "text" and n.visible and normalize_text(n.text)]
    hits: list[float] = []
    for g in gt.walk():
        if g.type != "text" or not g.visible or not normalize_text(g.text):
            continue
        cands = [n for n in pred_texts if n.box.iou(g.box) > 0]
        parts = [normalize_text(x) for n in cands for x in [n.text or ""] + (n.text or "").split("\n")]
        hits.append(1.0 if any(text_cer(x, g.text) <= thr for x in parts if x) else 0.0)
    return {"component_name_acc": (sum(names) / len(names)) if names else None,
            "text_line_recall": (sum(hits) / len(hits)) if hits else None}


def gt_text_mask(gt: Document, shape: tuple[int, int], pad: int = 2) -> "Any":
    """Boolean ``shape`` mask of the visible gt text boxes (expanded by ``pad`` px)."""
    import numpy as np
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    for n in gt.walk():
        if n.type == "text" and n.visible and (n.text or "").strip():
            x0, y0, x1, y1 = n.box.expand(pad).as_int()
            m[max(0, y0):max(0, min(h, y1)), max(0, x0):max(0, min(w, x1))] = True
    return m


def fidelity_metrics(target: "Any", rendered: "Any", gt: Document) -> dict[str, float]:
    """The independent validator's (:mod:`dt.validate.fidelity`) measures of ``rendered`` vs the
    gt screenshot ``target``, without OCR (text regions come from the gt text boxes):

    * ``jnd_frac_nontext``: fraction of non-text pixels with ΔE2000 > ``validate.jnd_de``;
    * ``chamfer_tile_max``: worst-tile symmetric Canny-edge chamfer (px);
    * ``edge_within1``: fraction of target edge pixels within 1 px of a render edge.
    """
    from dt.validate.fidelity import conform_to_target, delta_e2000_map, edge_metrics
    t, (r, _ok) = target[..., :3], conform_to_target(target, rendered)  # whole target frame, never a crop
    de = delta_e2000_map(t, r)
    jnd = float(P["validate.jnd_de"])
    nontext = ~gt_text_mask(gt, de.shape)
    jnd_frac = float((de[nontext] > jnd).mean()) if nontext.any() else float((de > jnd).mean())
    _iou, _chamfer, within1, _nt, _nr, tile_max, _arg = edge_metrics(t, r)
    return {"jnd_frac_nontext": jnd_frac, "chamfer_tile_max": float(tile_max), "edge_within1": float(within1)}


def raster_frac(pred: Document, gt: Document) -> float:
    """Share of the gt screen painted by raster image nodes of ``pred`` (``image_ref`` crops,
    :func:`dt.validate.fidelity.raster_nodes`) outside the gt's own image nodes."""
    from dt.validate.fidelity import raster_mask, raster_nodes
    rasters = raster_nodes(pred)
    if not rasters:
        return 0.0
    shape = (int(gt.height), int(gt.width))
    m = raster_mask(rasters, shape)
    gt_imgs = [n for n in gt.walk() if n.visible and n.type == "image"]
    if gt_imgs:
        m &= ~raster_mask(gt_imgs, shape)
    return float(m.mean())


# ----------------------------------------------------------------------------- one case
_DS = None


def _design_system():
    global _DS
    if _DS is None:
        from dt.mapping import material3
        _DS = material3()
    return _DS


def _timed(fn, timing: dict, key: str):
    t = time.perf_counter()
    try:
        return fn()
    finally:
        timing[key] = round(time.perf_counter() - t, 4)


def run_case(corpus_dir: str, cid: str, stages: Sequence[str] = ALL_STAGES, out_dir: Optional[str] = None,
             overrides: Optional[dict[str, Any]] = None, prediction: Optional[Document] = None) -> dict:
    """Run the pipeline on one corpus case and score it. Never raises: failures are reported
    in ``row["error"]`` (and the case scores 0).

    ``prediction``: score this document instead of running perceive/map (external or adversarial
    outputs; ``stages`` then only decides whether ``map`` and ``render`` run on it).

    Returns ``{id, corpus, png, width, height, stages, metrics, timing, composite, error,
    error_stage, diff_image}``.
    """
    stages = tuple(stages)
    row: dict[str, Any] = {"id": cid, "corpus": os.path.basename(os.path.normpath(corpus_dir)), "png": None,
                           "width": None, "height": None, "stages": list(stages), "metrics": {}, "timing": {},
                           "composite": None, "error": None, "error_stage": None, "diff_image": None}
    stage = "load"
    try:
        with param_overrides(overrides):
            gt, png = load_case(corpus_dir, cid)
            row.update(png=png, width=gt.width, height=gt.height)
            gt_eval = discount_gt(gt)
            timing = row["timing"]

            stage = "perceive"
            if prediction is not None:
                doc = Document.from_dict(prediction.to_dict())
            elif "perceive" in stages:
                from dt.perceive import perceive
                doc = _timed(lambda: perceive(png), timing, "perceive_s")
            else:
                doc = Document.from_dict(gt.to_dict())
                for n in doc.walk():  # strip the answers the mapper would otherwise copy
                    n.component, n.tokens = None, {}

            stage = "map"
            if "map" in stages:
                from dt.mapping import map_document
                ds = _design_system()
                doc = _timed(lambda: map_document(doc, ds), timing, "map_s")

            stage = "compare"
            from dt.compare import structural_metrics
            sm = _timed(lambda: structural_metrics(doc, gt_eval), timing, "struct_s")
            extra = extra_metrics(doc, gt_eval, sm.pop("matches", []))
            if "map" not in stages:
                sm["component_acc"] = sm["token_acc"] = extra["component_name_acc"] = None
            metrics: dict[str, Any] = {k: sm.get(k) for k in ("n_pred", "n_gt", "n_matched")}
            metrics.update({m.name: sm.get(m.name) for m in METRICS if m.source == "structure"})
            metrics.update(extra)
            metrics[EDITABILITY_KEY] = raster_frac(doc, gt_eval)
            iw, ih = min(doc.width, gt.width), min(doc.height, gt.height)
            inter = float(max(0, iw) * max(0, ih))
            metrics[FRAME_KEY] = inter / max(1.0, float(doc.width * doc.height + gt.width * gt.height) - inter)

            if "render" in stages:
                stage = "render"
                from dt.render.screenshot import render_doc
                rendered = _timed(lambda: render_doc(doc), timing, "render_s")
                stage = "compare"
                from dt.common.image import load_rgb
                from dt.compare import diff_map, layout_consistency, pixel_metrics, residual_regions, save_diff_image
                from dt.validate.fidelity import conform_to_target
                target = load_rgb(png)
                # score on the gt frame: a render of another size must not be cropped into agreement
                rendered, metrics["size_match"] = conform_to_target(target, rendered)
                t0 = time.perf_counter()
                diff = diff_map(target, rendered)
                pm = pixel_metrics(target, rendered, diff=diff)
                metrics.update({k: pm.get(k) for k in ("mean_de", "frac_bad", "ssim", "max_de", "mse", "psnr")})
                metrics["layout_consistency"] = layout_consistency(doc)
                timing["pixel_s"] = round(time.perf_counter() - t0, 4)
                if P["bench.fidelity"]:
                    stage = "fidelity"
                    metrics.update(_timed(lambda: fidelity_metrics(target, rendered, gt_eval), timing, "fidelity_s"))
                    stage = "compare"
                if out_dir:
                    path = os.path.join(out_dir, "diffs", f"{cid}.png")
                    save_diff_image(target, rendered, diff, path, residual_regions(diff))
                    row["diff_image"] = os.path.relpath(path, out_dir)
            row["metrics"] = metrics
            row["composite"] = case_composite(metrics)
    except Exception as e:  # noqa: BLE001 - a bench must survive any single-case failure
        row["error"] = f"{type(e).__name__}: {e}"
        row["error_stage"] = stage
        row["traceback"] = traceback.format_exc()
        row["composite"] = 0.0
    row["timing"]["total_s"] = round(sum(v for k, v in row["timing"].items() if k != "total_s"), 4)
    return row


def _run_case_star(args: tuple) -> dict:
    return run_case(*args)


# ----------------------------------------------------------------------------- the run
def run(corpus_dirs: Iterable[str] = DEFAULT_CORPORA, stages: Sequence[str] = ALL_STAGES, limit: Optional[int] = None,
        workers: int = 1, run_id: Optional[str] = None, out_root: Optional[str] = DEFAULT_OUT_ROOT,
        cases: Optional[Sequence[Case]] = None, overrides: Optional[dict[str, Any]] = None,
        verbose: bool = False) -> dict:
    """Bench the pipeline on the corpora and return the report dict (also written to disk).

    * ``corpus_dirs`` / ``limit``: which cases (``limit`` per corpus); or pass explicit ``cases``.
    * ``stages``: subset of ``("perceive", "map", "render")`` (see module docstring).
    * ``workers``: process pool size (1 = in-process).
    * ``run_id``: directory name under ``out_root`` (default: local time stamp). ``out_root=None``
      writes nothing (used by the tuner).
    * ``overrides``: ``dt.params`` values applied for this run only.

    Report keys: ``run_id, created, stages, params_overrides, n_cases, n_errors, composite,
    metrics (aggregate per metric), worst, cases (rows), timing (per stage totals + wall),
    out_dir``.
    """
    unknown = sorted(set(stages) - set(ALL_STAGES))
    if unknown:  # a typo would silently score the gt IR as the prediction (inflated composite)
        raise ValueError(f"unknown stage(s) {unknown}; valid: {list(ALL_STAGES)}")
    stages = tuple(s for s in ALL_STAGES if s in set(stages))
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(out_root, run_id) if out_root else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if cases is not None:
        case_list, dropped = list(cases), []
    else:
        case_list, dropped = scan_cases(corpus_dirs, limit)
    for dr in dropped:
        _log(f"  [bench] dropped {dr['corpus']}/{dr['id']}: {dr['reason']}")
    jobs = [(d, cid, stages, out_dir, overrides) for d, cid in case_list]

    t0 = time.perf_counter()
    rows: list[dict] = []
    if workers > 1 and len(jobs) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=int(workers), mp_context=ctx) as ex:
            futs = [ex.submit(_run_case_star, job) for job in jobs]
            for job, fut in zip(jobs, futs):  # keep case order; one crashed worker must not lose the run
                try:
                    row = fut.result()
                except Exception as e:  # noqa: BLE001 - e.g. BrokenProcessPool
                    row = _error_row(job[0], job[1], stages, "worker", e)
                rows.append(row)
                if verbose:
                    _print_row(row)
    else:
        with param_overrides(overrides):
            for job in jobs:
                row = run_case(*job)
                rows.append(row)
                if verbose:
                    _print_row(row)
    wall = time.perf_counter() - t0
    if not verbose:  # errors are never silent
        for row in rows:
            if row.get("error"):
                _print_row(row)

    report = build_report(rows, run_id=run_id, stages=stages, overrides=overrides, wall_s=wall, out_dir=out_dir,
                          dropped=dropped)
    if out_dir:
        write_report(report, out_dir)
    return report


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _error_row(corpus_dir: str, cid: str, stages: Sequence[str], stage: str, e: BaseException) -> dict:
    """Row for a case whose worker died before returning (scored 0, like any errored case)."""
    return {"id": cid, "corpus": os.path.basename(os.path.normpath(corpus_dir)), "png": None, "width": None,
            "height": None, "stages": list(stages), "metrics": {}, "timing": {"total_s": 0.0}, "composite": 0.0,
            "error": f"{type(e).__name__}: {e}", "error_stage": stage, "diff_image": None,
            "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__))}


def _print_row(row: dict) -> None:
    comp = row["composite"]
    msg = f"  {row['corpus']}/{row['id']}: composite={comp:.3f} t={row['timing'].get('total_s', 0):.2f}s"
    if row.get("error"):
        msg += f"  ERROR[{row['error_stage']}] {row['error']}"
    print(msg, file=sys.stderr, flush=True)


def build_report(rows: list[dict], run_id: str, stages: Sequence[str], overrides: Optional[dict] = None,
                 wall_s: float = 0.0, out_dir: Optional[str] = None, dropped: Optional[list[dict]] = None) -> dict:
    """Aggregate case rows into the report dict (pure; no I/O). ``dropped``: cases that were not
    run (:func:`scan_cases`); errored cases are listed under ``errors``."""
    comps = [r["composite"] for r in rows if r.get("composite") is not None]
    timing: dict[str, float] = {}
    for r in rows:
        for k, v in r.get("timing", {}).items():
            timing[k] = round(timing.get(k, 0.0) + float(v), 3)
    timing["wall_s"] = round(wall_s, 3)
    weights = {m.name: float(P[m.weight_key]) for m in METRICS}
    return {
        "run_id": run_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics_version": METRICS_VERSION,
        "stages": list(stages),
        "params_overrides": dict(overrides or {}),
        "weights": weights,
        "norms": {m.name: float(P[m.norm]) for m in METRICS if m.norm},
        "definitions": definition_params(),
        "n_cases": len(rows),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "errors": [{"corpus": r.get("corpus"), "id": r.get("id"), "stage": r.get("error_stage"), "error": r.get("error")}
                   for r in rows if r.get("error")],
        "n_dropped": len(dropped or []),
        "dropped": list(dropped or []),
        "composite": float(sum(comps) / len(comps)) if comps else 0.0,
        "metrics": aggregate(rows),
        "worst": worst_entries(rows),
        "cases": rows,
        "timing": timing,
        "out_dir": out_dir,
    }


# ----------------------------------------------------------------------------- report files
def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def report_markdown(report: dict) -> str:
    """Render a report dict as Markdown (summary, metric table, worst-k table, per-case table)."""
    L: list[str] = []
    L.append(f"# Bench report `{report['run_id']}`")
    L.append("")
    L.append(f"* created: {report['created']}  ·  stages: {', '.join(report['stages'])}  ·  "
             f"cases: {report['n_cases']} (errors: {report['n_errors']}, dropped: {report.get('n_dropped', 0)})"
             f"  ·  metrics v{report.get('metrics_version', 1)}")
    L.append(f"* **composite: {report['composite']:.4f}**  (weighted mean goodness, 1 = perfect)")
    t = report.get("timing", {})
    L.append("* timing (s, summed over cases): " + ", ".join(f"{k.replace('_s', '')}={v:.2f}" for k, v in t.items()))
    if report.get("params_overrides"):
        L.append("* params overrides: `" + json.dumps(report["params_overrides"], sort_keys=True) + "`")
    L.append("")
    L.append("## Metrics")
    L.append("")
    L.append("| metric | stage | weight | mean | p50 | worst | best | n | goodness |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in METRICS:
        a = report["metrics"].get(m.name)
        if not a:
            continue
        L.append(f"| {m.name} | {m.stage} | {report['weights'][m.name]:.2f} | {_fmt(a['mean'])} | {_fmt(a['p50'])} | "
                 f"{_fmt(a['worst'])} | {_fmt(a['best'])} | {a['n']} | {_fmt(a['goodness'])} |")
    L.append("")
    L.append(f"## Worst {len(report['worst'])} (by weighted deficit)")
    L.append("")
    L.append("| # | case | metric | value | goodness | deficit | suspected stage |")
    L.append("|---:|---|---|---:|---:|---:|---|")
    for i, e in enumerate(report["worst"], 1):
        L.append(f"| {i} | {e['corpus']}/{e['case']} | {e['metric']} | {_fmt(e['value'])} | {_fmt(e['goodness'])} | "
                 f"{_fmt(e['deficit'])} | {e['stage']} |")
    L.append("")
    L.append("## Cases")
    L.append("")
    keys = ("node_recall", "node_precision", "mean_iou", "text_recall", "text_line_recall", "component_acc",
            "component_name_acc", "mean_de", "frac_bad", "layout_consistency")
    L.append("| case | size | composite | " + " | ".join(keys) + " | time s | diff |")
    L.append("|---|---|---:|" + "---:|" * len(keys) + "---:|---|")
    for r in report["cases"]:
        m = r.get("metrics") or {}
        diff = f"[png]({r['diff_image']})" if r.get("diff_image") else "-"
        size = f"{r.get('width')}x{r.get('height')}" if r.get("width") else "-"
        vals = " | ".join(_fmt(m.get(k)) for k in keys)
        L.append(f"| {r['corpus']}/{r['id']} | {size} | {_fmt(r['composite'])} | {vals} | "
                 f"{_fmt(r['timing'].get('total_s'), 2)} | {diff} |")
    errs = [r for r in report["cases"] if r.get("error")]
    if errs:
        L.append("")
        L.append("## Errors")
        L.append("")
        for r in errs:
            L.append(f"* `{r['corpus']}/{r['id']}` [{r['error_stage']}]: {r['error']}")
    if report.get("dropped"):
        L.append("")
        L.append("## Dropped (not run)")
        L.append("")
        for d in report["dropped"]:
            L.append(f"* `{d['corpus']}/{d['id']}`: {d['reason']}")
    L.append("")
    return "\n".join(L)


def write_report(report: dict, out_dir: str) -> tuple[str, str]:
    """Write ``report.json`` and ``report.md`` into ``out_dir``; returns both paths."""
    os.makedirs(out_dir, exist_ok=True)
    jp, mp = os.path.join(out_dir, "report.json"), os.path.join(out_dir, "report.md")
    slim = dict(report)
    slim["cases"] = [{k: v for k, v in r.items() if k != "traceback"} for r in report["cases"]]
    with open(jp, "w") as f:
        json.dump(slim, f, indent=2, sort_keys=True)
    with open(mp, "w") as f:
        f.write(report_markdown(report))
    return jp, mp


# ----------------------------------------------------------------------------- baseline gate
def _case_key(r: dict) -> str:
    return f"{r.get('corpus')}/{r.get('id')}"


def baseline_dict(report: dict) -> dict:
    """The regression-baseline record of a bench report (what ``knowledge/baseline.json`` holds)."""
    return {
        "metrics_version": report.get("metrics_version", METRICS_VERSION),
        "run_id": report["run_id"],
        "created": report["created"],
        "stages": list(report["stages"]),
        "n_cases": report["n_cases"],
        "n_errors": report.get("n_errors", 0),
        "n_dropped": report.get("n_dropped", 0),
        "weights": dict(report.get("weights", {})),
        "norms": dict(report.get("norms", {})),
        "definitions": dict(report.get("definitions") or definition_params()),
        "gate": {"eps": float(P["bench.gate.eps"]), "metric_eps": float(P["bench.gate.metric_eps"]),
                 "case_eps": float(P["bench.gate.case_eps"])},
        "composite": report["composite"],
        "goodness": {k: v["goodness"] for k, v in report["metrics"].items()},
        "metrics": {k: dict(v) for k, v in report["metrics"].items()},
        "cases": {_case_key(r): r["composite"] for r in report["cases"]},
    }


def save_baseline(report: dict, path: str = BASELINE_PATH) -> str:
    """Persist the regression baseline (:func:`baseline_dict`) at ``path`` (default: the tracked
    ``knowledge/baseline.json``)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(baseline_dict(report), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def load_baseline(path: str = BASELINE_PATH) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def gate_table(report: dict, baseline: dict, eps: Optional[float] = None,
               metric_eps: Optional[float] = None) -> list[dict]:
    """Per-metric gate rows ``{metric, baseline, current, delta, tol, ok}`` (goodness units;
    first row is the composite). A metric present in the baseline but missing now fails."""
    eps = float(P["bench.gate.eps"] if eps is None else eps)
    metric_eps = float(P["bench.gate.metric_eps"] if metric_eps is None else metric_eps)
    rows = []
    bc, cc = float(baseline["composite"]), float(report["composite"])
    rows.append({"metric": "composite", "baseline": bc, "current": cc, "delta": cc - bc, "tol": eps,
                 "ok": cc >= bc - eps})
    names = list(baseline.get("metrics", {}))
    names += [m for m in report.get("metrics", {}) if m not in names]
    for name in names:
        b = baseline.get("metrics", {}).get(name)
        c = report.get("metrics", {}).get(name)
        bg = None if b is None else float(b["goodness"])
        cg = None if c is None else float(c["goodness"])
        delta = None if (bg is None or cg is None) else cg - bg
        ok = True if bg is None else (cg is not None and cg >= bg - metric_eps)
        rows.append({"metric": name, "baseline": bg, "current": cg, "delta": delta, "tol": metric_eps, "ok": ok})
    return rows


def format_gate_table(rows: list[dict]) -> str:
    """Plain-text per-metric delta table for :func:`gate_table` rows."""
    def f(v: Optional[float], sign: bool = False) -> str:
        return "-" if v is None else (f"{v:+.4f}" if sign else f"{v:.4f}")
    out = [f"{'metric':20s} {'baseline':>9s} {'current':>9s} {'delta':>9s} {'tol':>7s}  status"]
    for r in rows:
        status = "ok" if r["ok"] else "REGRESSED"
        if r["baseline"] is None:
            status = "new"
        out.append(f"{r['metric']:20s} {f(r['baseline']):>9s} {f(r['current']):>9s} {f(r['delta'], True):>9s} "
                   f"{r['tol']:7.4f}  {status}")
    return "\n".join(out)


def check_regression(report: dict, baseline: dict, tol: Optional[float] = None,
                     metric_tol: Optional[float] = None) -> list[str]:
    """Gate failures of ``report`` versus ``baseline`` (empty list = gate passes):

    * incomparable runs: different metrics version, stage set, case count / case ids, or
      composite weights (composites are only comparable on the same cases and definitions);
    * composite drop > ``tol`` (default ``bench.gate.eps``);
    * any metric's goodness drop > ``metric_tol`` (default ``bench.gate.metric_eps``), or a
      baseline metric that is no longer reported.
    """
    fails: list[str] = []
    bv, cv = baseline.get("metrics_version", 1), report.get("metrics_version", 1)
    if bv != cv:
        fails.append(f"metrics_version {cv} != baseline {bv} (metric definitions changed: re-baseline)")
    if "stages" in baseline and list(report.get("stages", [])) != list(baseline["stages"]):
        fails.append(f"stages {list(report.get('stages', []))} differ from baseline {list(baseline['stages'])}")
    if "n_cases" in baseline and report.get("n_cases") != baseline["n_cases"]:
        fails.append(f"n_cases {report.get('n_cases')} != baseline {baseline['n_cases']}")
    if isinstance(baseline.get("cases"), dict) and isinstance(report.get("cases"), list):
        cur = {_case_key(r) for r in report["cases"]}
        missing = sorted(set(baseline["cases"]) - cur)
        extra = sorted(cur - set(baseline["cases"]))
        if missing or extra:
            fails.append(f"case set differs from baseline (missing {missing[:5]}{'...' if len(missing) > 5 else ''}, "
                         f"new {extra[:5]}{'...' if len(extra) > 5 else ''})")
    if baseline.get("weights") and report.get("weights"):
        diff = sorted(k for k in set(baseline["weights"]) | set(report["weights"])
                      if abs(float(baseline["weights"].get(k, 0.0)) - float(report["weights"].get(k, 0.0))) > 1e-12)
        if diff:
            fails.append(f"composite weights differ from baseline: {diff}")
    # normalisers / thresholds that define a metric's value: a looser norm or IoU threshold inflates
    # goodness without any change in the output (params.json is human-editable)
    for key, label in (("norms", "metric normalisers"), ("definitions", "metric-definition params")):
        b_def, c_def = baseline.get(key), report.get(key)
        if isinstance(b_def, dict) and isinstance(c_def, dict):
            diff = sorted(k for k in b_def if k not in c_def or c_def[k] != b_def[k])
            if diff:
                fails.append(f"{label} differ from baseline: {diff[:8]}{'...' if len(diff) > 8 else ''}")
    # one case collapsing must not be hidden by small gains elsewhere
    case_eps = float(P["bench.gate.case_eps"])
    if isinstance(baseline.get("cases"), dict) and isinstance(report.get("cases"), list):
        for r in report["cases"]:
            b = baseline["cases"].get(_case_key(r))
            c = r.get("composite")
            if b is not None and c is not None and float(c) < float(b) - case_eps:
                fails.append(f"case {_case_key(r)} composite {float(c):.4f} < baseline {float(b):.4f} - {case_eps}")
    for r in gate_table(report, baseline, tol, metric_tol):
        if r["ok"]:
            continue
        if r["current"] is None:
            fails.append(f"{r['metric']} missing (baseline goodness {r['baseline']:.4f})")
        elif r["metric"] == "composite":
            fails.append(f"composite {r['current']:.4f} < baseline {r['baseline']:.4f} - {r['tol']}")
        else:
            fails.append(f"{r['metric']} goodness {r['current']:.4f} < baseline {r['baseline']:.4f} - {r['tol']}")
    return fails


# ----------------------------------------------------------------------------- failure drafts
_SYMPTOM = {
    "node_recall": "gt nodes not recovered by any predicted node (IoU >= thr)",
    "node_precision": "predicted nodes that match no gt node (spurious regions / split text)",
    "mean_iou": "matched boxes are loose (edges off by several px)",
    "color_de": "fill colours of matched nodes differ",
    "text_cer": "OCR text differs from gt text",
    "text_recall": "gt text lines not found by OCR",
    "radius_mae": "corner radius estimate off",
    "type_acc": "node type differs on matched pairs (rect vs ellipse/line/icon)",
    "component_acc": "component name/variant not resolved on gt component nodes",
    "token_acc": "token bindings of gt nodes not reproduced",
    "mean_de": "re-render differs from target over large areas",
    "frac_bad": "many pixels differ by more than the bad-pixel ΔE",
    "ssim": "structural similarity of the re-render is low",
    "layout_consistency": "flex re-render of inferred layout differs from absolute render",
    "jnd_frac_nontext": "many non-text pixels differ visibly (ΔE2000 above the JND) from the target",
    "chamfer_tile_max": "edges in one tile are displaced: an element is missing, misplaced or missized",
    "edge_within1": "target edges have no render edge within 1 px (geometry off by >= 2 px)",
}


def draft_failures(report: dict, k: Optional[int] = None) -> list[dict]:
    """Template failure entries (``knowledge/failures.jsonl`` schema) from the worst-k table.

    The root-cause hypothesis and fix fields are left for a human/AI driver to fill in from
    the diff images; ``symptom`` is derived from the metric programmatically.
    """
    out = []
    for e in worst_entries(report["cases"], k):
        out.append({"case": f"{e['corpus']}/{e['case']}", "stage": e["stage"], "metric": e["metric"],
                    "value": e["value"], "symptom": _SYMPTOM.get(e["metric"], e["metric"]),
                    "root_cause_hypothesis": "", "suggested_fix": "", "run_id": report["run_id"]})
    return out


# ----------------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="dt.selftest.bench", description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", nargs="*", default=list(DEFAULT_CORPORA), help="corpus dirs (manifest.json)")
    ap.add_argument("--stages", default=",".join(ALL_STAGES), help="comma list of perceive,map,render")
    ap.add_argument("--limit", type=int, default=None, help="cases per corpus")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out", default=DEFAULT_OUT_ROOT, help="report root (use '' for no files)")
    ap.add_argument("--gate", "--baseline", dest="gate", nargs="?", const=BASELINE_PATH, default=None,
                    help=f"gate against a baseline (default {os.path.relpath(BASELINE_PATH, ROOT)}); exit 1 on regression")
    ap.add_argument("--set-baseline", "--write-baseline", dest="set_baseline", nargs="?", const=BASELINE_PATH,
                    default=None, help=f"write this run as the baseline (default {os.path.relpath(BASELINE_PATH, ROOT)})")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    report = run(a.corpus, stages=tuple(s.strip() for s in a.stages.split(",") if s.strip()), limit=a.limit,
                 workers=a.workers, run_id=a.run_id, out_root=(a.out or None), verbose=not a.quiet)
    print(f"composite {report['composite']:.4f} over {report['n_cases']} cases "
          f"({report['n_errors']} errors, {report['n_dropped']} dropped) in {report['timing']['wall_s']:.1f}s "
          f"-> {report['out_dir']}")
    for m in METRICS:
        agg = report["metrics"].get(m.name)
        if agg:
            print(f"  {m.name:20s} mean={agg['mean']:.4f} p50={agg['p50']:.4f} worst={agg['worst']:.4f} goodness={agg['goodness']:.3f}")
    for k in ("component_name_acc", "text_line_recall"):
        vals = [r["metrics"][k] for r in report["cases"] if r.get("metrics", {}).get(k) is not None]
        if vals:
            print(f"  {k:20s} mean={sum(vals) / len(vals):.4f} (info)")
    gate, msg = gate_and_baseline(report, a.gate, a.set_baseline)
    if msg:
        print(msg)
    return 1 if (gate is not None and not gate["pass"]) else 0


def gate_and_baseline(report: dict, gate_path: Optional[str] = None,
                      set_path: Optional[str] = None) -> tuple[Optional[dict], str]:
    """Shared by ``dt bench`` and ``python -m dt.selftest.bench``: gate ``report`` against the
    baseline at ``gate_path`` (a missing baseline fails the gate) and/or write it as the new
    baseline at ``set_path`` (after gating, so ``--gate --set-baseline`` compares with the old one).

    Returns ``(gate, text)``: ``gate = {baseline, found, pass, failures, table}`` (``None`` when
    not gating) and a human-readable summary with the per-metric delta table."""
    lines: list[str] = []
    gate: Optional[dict] = None
    if gate_path:
        base = load_baseline(gate_path)
        if base is None:
            fails, table = [f"no baseline at {gate_path} (create one with --set-baseline)"], []
        else:
            fails, table = check_regression(report, base), gate_table(report, base)
        gate = {"baseline": gate_path, "found": base is not None, "pass": not fails, "failures": fails, "table": table}
        if table:
            lines.append(format_gate_table(table))
        lines.append("regression gate: " + ("PASS" if not fails else "FAIL"))
        lines += ["  " + f_ for f_ in fails]
    if set_path:
        lines.append(f"baseline written to {save_baseline(report, set_path)}")
    return gate, "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
