"""Run a pipeline stage on a scenario case and measure it with the existing measures.

* structural (needs gt): :func:`dt.compare.structural_metrics` restricted to the ROI
  (node recall/precision, colour ΔE, text CER/recall, component+variant, ...);
* fidelity: the independent validator's measures (:mod:`dt.validate.fidelity`) within the ROI:
  ``jnd_frac_nontext``, ``mean_de``, ``chamfer``, ``chamfer_tile_max``, ``edge_within1``,
  ``edge_f1``, ``region_de_worst``, ``region_de_mean``;
* family-specific measures (``family.metrics``).

Metrics come back as ``{"full": {...}, "roi": {...}}`` (``roi`` only when the case has one);
criteria look their metric up in the matching scope (family metrics live in ``full``).
"""
from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from dt.ir import Box, Document, Node
from dt.params import P, register
from dt.scenarios.spec import ScenarioCase, ScenarioFamily

register("scenarios.roi.min_overlap", 0.5,
         "a node belongs to a scenario ROI when at least this fraction of its box lies inside it", (0.1, 1.0))
register("scenarios.translate.patience", 2,
         "refine patience for scenario families run at stage 'translate' (iterations come from the family)", (1, 10))

_DS = None


def _design_system():
    global _DS
    if _DS is None:
        from dt.mapping import material3
        _DS = material3()
    return _DS


# ----------------------------------------------------------------------------- pipeline stages
def run_stage(case: ScenarioCase, stage: str, refine_iters: int = 0) -> tuple[Document, np.ndarray, dict]:
    """``(prediction, rendered_rgb, info)`` for ``stage`` in perceive | map | translate.

    ``translate`` = perceive -> map -> refine (``refine_iters``, no wall-clock budget, so the run is
    deterministic) -> re-map, like :func:`dt.pipeline.translate` without the file outputs."""
    from dt.perceive import perceive
    from dt.render.screenshot import render_doc
    info: dict[str, Any] = {"stage": stage, "timing": {}}
    t = time.perf_counter()
    doc = perceive(case.target_rgb)
    info["timing"]["perceive_s"] = round(time.perf_counter() - t, 3)
    if stage in ("map", "translate"):
        from dt.mapping import map_document
        t = time.perf_counter()
        doc = map_document(doc, _design_system())
        info["timing"]["map_s"] = round(time.perf_counter() - t, 3)
    if stage == "translate" and refine_iters > 0:
        from dt.mapping import map_document
        from dt.pipeline import unmap_document
        from dt.refine.optimizer import accepted_moves, refine
        t = time.perf_counter()
        doc, hist = refine(doc, case.target_rgb, max_iters=int(refine_iters),
                           patience=int(P["scenarios.translate.patience"]), time_budget_s=None)
        acc = accepted_moves(hist)
        info["refine"] = {"accepted": len(acc), "kinds": sorted({h.get("kind") for h in acc})}
        unmap_document(doc)
        doc = map_document(doc, _design_system())
        info["timing"]["refine_s"] = round(time.perf_counter() - t, 3)
    t = time.perf_counter()
    rendered = render_doc(doc)
    info["timing"]["render_s"] = round(time.perf_counter() - t, 3)
    return doc, rendered, info


# ----------------------------------------------------------------------------- ROI helpers
def roi_int(roi: Optional[Box], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    if roi is None:
        return 0, 0, w, h
    x0, y0, x1, y1 = roi.as_int()
    return max(0, x0), max(0, y0), min(w, max(x0 + 1, x1)), min(h, max(y0 + 1, y1))


def visible_nodes(doc: Document) -> list[Node]:
    from dt.compare.structural import _nodes
    return _nodes(doc)


def nodes_in_roi(doc: Document, roi: Optional[Box]) -> list[Node]:
    """Visible nodes of ``doc`` whose box lies in ``roi`` (by ``scenarios.roi.min_overlap``);
    all visible nodes when ``roi`` is None."""
    nodes = visible_nodes(doc)
    if roi is None:
        return nodes
    frac = float(P["scenarios.roi.min_overlap"])
    return [n for n in nodes if n.box.area > 0 and n.box.intersect(roi).area >= frac * n.box.area]


def structural_in_roi(pred: Document, gt: Document, roi: Optional[Box]) -> dict:
    from dt.compare import structural_metrics
    sm = structural_metrics(nodes_in_roi(pred, roi), nodes_in_roi(gt, roi))
    sm.pop("matches", None)
    return sm


def gt_text_mask(gt: Optional[Document], shape: tuple[int, int], pad: int = 2) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    if gt is None:
        return m
    for n in visible_nodes(gt):
        if n.type == "text" and (n.text or "").strip():
            x0, y0, x1, y1 = n.box.expand(pad).as_int()
            m[max(0, y0):max(0, min(h, y1)), max(0, x0):max(0, min(w, x1))] = True
    return m


def ocr_text_mask(rgb: np.ndarray, pad: int = 2) -> np.ndarray:
    from dt.validate.fidelity import _ocr_lines, text_mask_from_lines
    lines = _ocr_lines(rgb) or []
    return text_mask_from_lines(rgb.shape[:2], lines, pad)


def fidelity(target: np.ndarray, rendered: np.ndarray, roi: Optional[Box] = None,
             text_mask: Optional[np.ndarray] = None) -> dict[str, Any]:
    """The validator's measures of ``rendered`` vs ``target`` on the target frame, within ``roi``."""
    from dt.validate.fidelity import (conform_to_target, delta_e2000_map, edge_f1, edge_metrics, flat_regions,
                                      region_scores)
    t = target[..., :3]
    r, size_ok = conform_to_target(t, rendered)
    x0, y0, x1, y1 = roi_int(roi, t.shape[:2])
    tc, rc = np.ascontiguousarray(t[y0:y1, x0:x1]), np.ascontiguousarray(r[y0:y1, x0:x1])
    de = delta_e2000_map(tc, rc)
    jnd = float(P["validate.jnd_de"])
    m = None if text_mask is None else text_mask[y0:y1, x0:x1]
    nontext = ~m if (m is not None and m.any()) else np.ones(de.shape, dtype=bool)
    _iou, chamfer, within1, _nt, _nr, tile_max, _arg = edge_metrics(tc, rc)
    scores = region_scores(tc, rc, flat_regions(tc))
    des = [s.delta_e for s in scores]
    return {"jnd_frac_nontext": float((de[nontext] > jnd).mean()) if nontext.any() else 0.0,
            "jnd_frac": float((de > jnd).mean()), "mean_de": float(de.mean()),
            "mean_de_nontext": float(de[nontext].mean()) if nontext.any() else 0.0,
            "chamfer": float(chamfer), "chamfer_tile_max": float(tile_max), "edge_within1": float(within1),
            "edge_f1": float(edge_f1(tc, rc)), "region_de_worst": float(max(des)) if des else 0.0,
            "region_de_mean": float(np.mean(des)) if des else 0.0, "size_match": bool(size_ok)}


def raster_share(pred: Document, shape: tuple[int, int], roi: Optional[Box] = None) -> float:
    """Share of the frame (or ROI) painted by raster crops of ``pred`` (editability)."""
    from dt.validate.fidelity import raster_mask, raster_nodes
    rasters = raster_nodes(pred)
    if not rasters:
        return 0.0
    x0, y0, x1, y1 = roi_int(roi, shape)
    return float(raster_mask(rasters, shape)[y0:y1, x0:x1].mean())


# ----------------------------------------------------------------------------- case evaluation
def measure(family: ScenarioFamily, case: ScenarioCase, pred: Document, rendered: np.ndarray) -> dict[str, dict]:
    """All metrics of one case: ``{"full": {...}, "roi": {...}?}``."""
    shape = case.target_rgb.shape[:2]
    if family.fidelity_text == "ocr" or case.gt is None:
        tmask = ocr_text_mask(case.target_rgb)
    else:
        tmask = gt_text_mask(case.gt, shape)
    scopes: dict[str, Optional[Box]] = {"full": None}
    if case.roi is not None:
        scopes["roi"] = case.roi
    out: dict[str, dict] = {}
    for name, roi in scopes.items():
        m: dict[str, Any] = fidelity(case.target_rgb, rendered, roi, tmask)
        m["raster_frac"] = raster_share(pred, shape, roi)
        if case.gt is not None:
            m.update({k: v for k, v in structural_in_roi(pred, case.gt, roi).items()})
        out[name] = m
    if family.metrics is not None:
        out["full"].update(family.metrics(case, pred, rendered) or {})
    return out


def lookup(metrics: dict[str, dict], key: str) -> Optional[float]:
    if key.startswith("roi."):
        v = metrics.get("roi", {}).get(key[4:])
        return v if v is not None else metrics.get("full", {}).get(key[4:])
    return metrics.get("full", {}).get(key)


def judge(family: ScenarioFamily, metrics: dict[str, dict]) -> tuple[bool, list[dict]]:
    """``(passed, per-criterion results)``."""
    res = [c.evaluate(lookup(metrics, c.key())) for c in family.criteria]
    return all(r["passed"] for r in res), res
