"""Bench metric definitions: normalisation to a [0, 1] "goodness", the composite score,
aggregation over cases and the worst-case ranking.

Every metric the bench reports is described by a :class:`MetricSpec`:

* ``higher_better`` -- direction (``node_recall`` up, ``color_de`` down);
* ``norm`` -- for unbounded error metrics, the value that maps to goodness 0 (read from
  ``dt.params`` under ``bench.norm.<metric>``); metrics already in [0, 1] have ``norm=None``;
* ``stage`` -- the pipeline stage that is *usually* responsible when the metric is bad
  (``perceive`` / ``map`` / ``render``), used for the suspected-stage column of the report;
* weight ``bench.w.<metric>`` (dt.params) in the composite.

The composite of a case is the weighted mean of the goodness of every metric that is defined
for that case (``None`` metrics are skipped and their weight is dropped); the composite of a
run is the mean of the case composites. Both are in [0, 1]; 1 means every metric is perfect.

Goodness normalisers (``g`` = goodness in [0, 1], ``clamp`` to [0, 1]):

==================== ========= ==================================== ============================
metric               source    goodness                             normaliser (default)
==================== ========= ==================================== ============================
node_precision       structure ``v``                                (already a fraction)
node_recall          structure ``v``                                (already a fraction)
mean_iou             structure ``v``                                (already a fraction)
color_de             structure ``1 - clamp(v / norm)``              ``bench.norm.color_de`` 20 ΔE
text_cer             structure ``1 - clamp(v)``                     (CER is a fraction, capped 1)
text_recall          structure ``v``                                (line-fair fraction)
radius_mae           structure ``1 - clamp(v / norm)``              ``bench.norm.radius_mae`` 8 px
type_acc             structure ``v``                                (already a fraction)
component_acc        structure ``v``                                (already a fraction)
token_acc            structure ``v``                                (already a fraction)
mean_de              pixel     ``1 - clamp(v / norm)``              ``bench.norm.mean_de`` 20 ΔE
frac_bad             pixel     ``1 - clamp(v / norm)``              ``bench.norm.frac_bad`` 0.3
ssim                 pixel     ``clamp(v)``                         (already <= 1)
layout_consistency   layout    ``v``                                (already a fraction)
jnd_frac_nontext     fidelity  ``1 - clamp(v / norm)``              ``bench.norm.jnd_frac_nontext`` 0.3
chamfer_tile_max     fidelity  ``1 - clamp(v / norm)``              ``bench.norm.chamfer_tile_max`` 128 px
edge_within1         fidelity  ``v``                                (already a fraction)
==================== ========= ==================================== ============================

The *fidelity* metrics come from the independent validator :mod:`dt.validate.fidelity`
(ΔE2000 / Canny edges of the gt screenshot vs the render, text regions masked with the gt
text boxes, no OCR): ``jnd_frac_nontext`` = fraction of non-text pixels with ΔE2000 above the
JND (``validate.jnd_de``), ``chamfer_tile_max`` = worst 32 px tile symmetric edge chamfer (px;
catches one displaced element the global mean would dilute), ``edge_within1`` = fraction of
target edge pixels within 1 px of a render edge. They make the composite agree with the
validation gates in docs/VALIDATION.md, so tuning cannot win by gaming the IR metrics alone.

**Editability.** Pixel and fidelity metrics cannot tell a design from a screenshot of the
target, so the case composite is multiplied by ``1 - clamp(bench.raster_penalty * raster_frac)``
(:func:`editability`), ``raster_frac`` being the share of the screen painted by raster image
nodes outside gt images (0 for every output that has no crops, so such outputs score exactly
as without the factor), and by ``frame_iou``, the IoU of the output canvas with the gt canvas
(renders are measured on the gt frame: missing canvas counts as wrong pixels, extra canvas is
only visible through this factor). Structural metrics only see nodes that paint (hidden subtrees,
zero-opacity nodes and transparent text are left out, ``compare.struct.min_paint_alpha``).

``METRICS_VERSION`` identifies this set of definitions; a regression baseline written under a
different version is not comparable (the gate refuses it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from dt.params import P, register

# ----------------------------------------------------------------------------- params
register("bench.w.node_precision", 1.0, "Composite weight of node_precision.", (0.0, 5.0))
register("bench.w.node_recall", 1.5, "Composite weight of node_recall.", (0.0, 5.0))
register("bench.w.mean_iou", 1.0, "Composite weight of mean_iou (matched pairs).", (0.0, 5.0))
register("bench.w.color_de", 1.0, "Composite weight of color_de (matched fills).", (0.0, 5.0))
register("bench.w.text_cer", 1.0, "Composite weight of text_cer.", (0.0, 5.0))
register("bench.w.text_recall", 1.5, "Composite weight of text_recall.", (0.0, 5.0))
register("bench.w.radius_mae", 0.5, "Composite weight of radius_mae.", (0.0, 5.0))
register("bench.w.type_acc", 0.5, "Composite weight of type_acc.", (0.0, 5.0))
register("bench.w.component_acc", 1.5, "Composite weight of component_acc (mapping).", (0.0, 5.0))
register("bench.w.token_acc", 0.5, "Composite weight of token_acc (mapping).", (0.0, 5.0))
register("bench.w.mean_de", 1.5, "Composite weight of the pixel mean ΔE (render vs target).", (0.0, 5.0))
register("bench.w.frac_bad", 1.0, "Composite weight of the bad-pixel fraction.", (0.0, 5.0))
register("bench.w.ssim", 0.5, "Composite weight of SSIM.", (0.0, 5.0))
register("bench.w.layout_consistency", 0.5, "Composite weight of layout_consistency.", (0.0, 5.0))
register("bench.w.jnd_frac_nontext", 1.0,
         "Composite weight of the validator's non-text JND fraction (ΔE2000 > validate.jnd_de).", (0.0, 5.0))
register("bench.w.chamfer_tile_max", 0.5,
         "Composite weight of the validator's worst-tile edge chamfer (px).", (0.0, 5.0))
register("bench.w.edge_within1", 1.0,
         "Composite weight of the validator's fraction of target edges within 1 px of a render edge.", (0.0, 5.0))

register("bench.norm.color_de", 20.0, "Matched-fill ΔE that maps to goodness 0.", (5.0, 60.0))
register("bench.norm.radius_mae", 8.0, "Radius MAE (px) that maps to goodness 0.", (2.0, 30.0))
register("bench.norm.mean_de", 20.0, "Pixel mean ΔE that maps to goodness 0.", (5.0, 60.0))
register("bench.norm.frac_bad", 0.3, "Bad-pixel fraction that maps to goodness 0.", (0.05, 1.0))
register("bench.norm.jnd_frac_nontext", 0.3,
         "Non-text JND pixel fraction (validator) that maps to goodness 0.", (0.05, 1.0))
register("bench.norm.chamfer_tile_max", 128.0,
         "Worst-tile symmetric edge chamfer (px, validator) that maps to goodness 0. Large because a "
         "tile whose element is missing in the render scores the distance to the nearest render edge "
         "anywhere (40-125 px on the corpora today, 1-2 px at the validation gates); a small norm would "
         "saturate every case at goodness 0 and give the composite no gradient.", (8.0, 256.0))
register("bench.fidelity", True,
         "Compute the validator fidelity metrics (jnd_frac_nontext, chamfer_tile_max, edge_within1) "
         "per bench case when the render stage runs.")

register("bench.raster_penalty", 1.0,
         "Editability factor of the case composite: it is multiplied by 1 - clamp(raster_penalty * raster_frac), "
         "raster_frac = share of the screen painted by raster image nodes (image_ref crops) outside gt images. "
         "Without it a screenshot crop over gt-shaped invisible structure beats the gt IR itself.", (0.0, 3.0))
register("bench.gate.case_eps", 0.05,
         "Regression gate: largest tolerated drop of any single case composite versus knowledge/baseline.json "
         "(improvements elsewhere must not hide one case collapsing).", (0.0, 0.5))
register("bench.stage_struct_ok", 0.8,
         "node_recall goodness above which a bad *pixel* metric is blamed on 'render' "
         "rather than 'perceive' in the suspected-stage column.", (0.3, 1.0))
register("bench.worst_k", 10, "Rows in the worst-cases table of the report.", (3, 50))
register("bench.gt.opaque_alpha", 0.99,
         "Effective fill alpha below which a gt node counts as translucent and is dropped from the "
         "colour metric (a perceiver cannot observe its true paint).", (0.5, 1.0))
register("bench.gate.eps", 0.003,
         "Regression gate: largest tolerated drop of the run composite versus knowledge/baseline.json.",
         (0.0, 0.05))
register("bench.gate.metric_eps", 0.01,
         "Regression gate: largest tolerated drop of any single metric's goodness (mean over cases) "
         "versus knowledge/baseline.json.", (0.0, 0.1))

METRICS_VERSION = 2
"""Version of the metric definitions (1 = before geometry-fair matching / line-fair text /
fidelity metrics). Bump whenever a metric's meaning changes; the gate refuses old baselines."""


# ----------------------------------------------------------------------------- specs
@dataclass(frozen=True)
class MetricSpec:
    """How one bench metric is interpreted (see module docstring)."""

    name: str
    higher_better: bool
    stage: str  # perceive | map | render
    norm: Optional[str] = None  # params key of the value mapping to goodness 0 (error metrics)
    source: str = "structure"  # structure | pixel | layout | fidelity

    @property
    def weight_key(self) -> str:
        return f"bench.w.{self.name}"


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("node_precision", True, "perceive"),
    MetricSpec("node_recall", True, "perceive"),
    MetricSpec("mean_iou", True, "perceive"),
    MetricSpec("color_de", False, "perceive", "bench.norm.color_de"),
    MetricSpec("text_cer", False, "perceive"),
    MetricSpec("text_recall", True, "perceive"),
    MetricSpec("radius_mae", False, "perceive", "bench.norm.radius_mae"),
    MetricSpec("type_acc", True, "perceive"),
    MetricSpec("component_acc", True, "map"),
    MetricSpec("token_acc", True, "map"),
    MetricSpec("mean_de", False, "render", "bench.norm.mean_de", "pixel"),
    MetricSpec("frac_bad", False, "render", "bench.norm.frac_bad", "pixel"),
    MetricSpec("ssim", True, "render", None, "pixel"),
    MetricSpec("layout_consistency", True, "perceive", None, "layout"),
    MetricSpec("jnd_frac_nontext", False, "render", "bench.norm.jnd_frac_nontext", "fidelity"),
    MetricSpec("chamfer_tile_max", False, "render", "bench.norm.chamfer_tile_max", "fidelity"),
    MetricSpec("edge_within1", True, "render", None, "fidelity"),
)
PIXEL_SOURCES: frozenset[str] = frozenset({"pixel", "fidelity"})
"""Metric sources measured on the render (blamed on ``perceive`` when structure is missing)."""
SPEC: dict[str, MetricSpec] = {m.name: m for m in METRICS}
INFO_KEYS: tuple[str, ...] = ("n_pred", "n_gt", "n_matched", "mse", "psnr", "max_de",
                              "component_name_acc", "text_line_recall", "size_match")
"""Reported for information only; never part of the composite (see ``bench.extra_metrics``)."""
EDITABILITY_KEY = "raster_frac"
"""Per-case share of the screen painted by raster crops; scales the case composite down
(``bench.raster_penalty``) instead of being a weighted term, so outputs without rasters score
exactly as before."""
FRAME_KEY = "frame_iou"
"""Per-case IoU of the output canvas with the gt canvas; also scales the case composite."""
DEFINITION_PREFIXES: tuple[str, ...] = ("bench.norm.", "bench.gt.", "bench.raster_penalty", "compare.struct.",
                                        "compare.pixel.bad_de", "validate.jnd_de", "validate.edge.")
"""``dt.params`` keys that define what a metric value *means* (normalisers, match thresholds,
ΔE thresholds, edge detector). A run under different values is not comparable with a baseline
even when the weights agree (see :func:`definition_params`)."""


def definition_params() -> dict:
    """Current values of every metric-definition param (:data:`DEFINITION_PREFIXES`)."""
    import dt.compare  # noqa: F401 -- registers compare.*
    import dt.validate.fidelity  # noqa: F401 -- registers validate.*
    allp = {**P.defaults(), **P.all()}
    return {k: allp[k] for k in sorted(allp) if k.startswith(DEFINITION_PREFIXES)}


# ----------------------------------------------------------------------------- normalisation
def goodness(name: str, value: Optional[float]) -> Optional[float]:
    """Map a raw metric value to [0, 1] (1 = perfect). ``None`` stays ``None`` (undefined)."""
    if value is None or name not in SPEC:
        return None
    v = float(value)
    if not np.isfinite(v):
        return 0.0
    spec = SPEC[name]
    if spec.norm is not None:
        v = v / max(float(P[spec.norm]), 1e-9)
    v = min(1.0, max(0.0, v))
    return v if spec.higher_better else 1.0 - v


def editability(metrics: dict) -> float:
    """Composite factor in [0, 1]: ``(1 - clamp(bench.raster_penalty * raster_frac)) * frame_iou``
    -- 1 for an output without raster crops on the target's canvas size. ``frame_iou`` is the IoU
    of the output canvas with the gt canvas (top-left anchored): pixel metrics are measured on the
    gt frame, so without it extra canvas would never cost anything."""
    f = 1.0
    rf = metrics.get(EDITABILITY_KEY)
    if rf is not None:
        rf = float(rf) if np.isfinite(float(rf)) else 1.0
        f *= 1.0 - min(1.0, max(0.0, float(P["bench.raster_penalty"]) * rf))
    fi = metrics.get(FRAME_KEY)
    if fi is not None:
        f *= min(1.0, max(0.0, float(fi))) if np.isfinite(float(fi)) else 0.0
    return f


def case_composite(metrics: dict) -> Optional[float]:
    """Weighted mean goodness over the metrics defined in ``metrics`` (``None`` if none are),
    scaled by :func:`editability` (a picture of the screen is not a translation)."""
    num, den = 0.0, 0.0
    for spec in METRICS:
        g = goodness(spec.name, metrics.get(spec.name))
        if g is None:
            continue
        w = float(P[spec.weight_key])
        num += w * g
        den += w
    return (num / den) * editability(metrics) if den > 0 else None


def suspect_stage(name: str, metrics: dict, stages: Optional[Sequence[str]] = None) -> str:
    """Stage most likely responsible for a bad value of ``name`` in a case.

    Pixel metrics are blamed on ``render`` only when the structure was recovered well
    (``node_recall`` goodness >= ``bench.stage_struct_ok``); otherwise the pixels are wrong
    because perception missed or misplaced nodes. When ``stages`` (the stages that actually
    ran for the case) is given, a stage that did not run is never blamed: the blame moves to
    the first stage that did run (e.g. ``map`` when the gt IR was used as the prediction).
    """
    spec = SPEC.get(name)
    if spec is None:
        return "unknown"
    stage = spec.stage
    if spec.source in PIXEL_SOURCES:
        rec = goodness("node_recall", metrics.get("node_recall"))
        if rec is not None and rec < float(P["bench.stage_struct_ok"]):
            stage = "perceive"
    if stages is not None and stage not in stages:
        ran = [s for s in ("perceive", "map", "render") if s in stages]
        return ran[0] if ran else "gt"
    return stage


# ----------------------------------------------------------------------------- aggregation
def _p50(vals: list[float]) -> float:
    return float(np.median(vals))


def aggregate(case_rows: list[dict]) -> dict[str, dict]:
    """Per-metric ``{mean, p50, worst, best, n, goodness}`` over cases.

    ``case_rows`` are bench case dicts with a ``metrics`` sub-dict. ``worst`` honours the
    metric direction (min for higher-better, max for lower-better); ``goodness`` is the mean
    normalised score. Metrics undefined in every case are omitted.
    """
    out: dict[str, dict] = {}
    for spec in METRICS:
        vals = [float(r["metrics"][spec.name]) for r in case_rows
                if r.get("metrics") and r["metrics"].get(spec.name) is not None]
        if not vals:
            continue
        gs = [goodness(spec.name, v) for v in vals]
        out[spec.name] = {
            "mean": float(np.mean(vals)),
            "p50": _p50(vals),
            "worst": float(min(vals) if spec.higher_better else max(vals)),
            "best": float(max(vals) if spec.higher_better else min(vals)),
            "n": len(vals),
            "goodness": float(np.mean([g for g in gs if g is not None])),
        }
    return out


def worst_entries(case_rows: list[dict], k: Optional[int] = None) -> list[dict]:
    """The ``k`` (case, metric) pairs with the largest weighted deficit ``w * (1 - goodness)``.

    Each entry: ``{case, corpus, metric, value, goodness, deficit, stage}``. Cases that
    errored contribute one entry with ``metric='error'`` and deficit = total weight.
    """
    k = int(P["bench.worst_k"] if k is None else k)
    entries: list[dict] = []
    total_w = sum(float(P[s.weight_key]) for s in METRICS)
    for r in case_rows:
        if r.get("error"):
            entries.append({"case": r["id"], "corpus": r.get("corpus", ""), "metric": "error",
                            "value": None, "goodness": 0.0, "deficit": total_w,
                            "stage": r.get("error_stage", "unknown")})
            continue
        m = r.get("metrics") or {}
        for spec in METRICS:
            g = goodness(spec.name, m.get(spec.name))
            if g is None:
                continue
            w = float(P[spec.weight_key])
            entries.append({"case": r["id"], "corpus": r.get("corpus", ""), "metric": spec.name,
                            "value": float(m[spec.name]), "goodness": g, "deficit": w * (1.0 - g),
                            "stage": suspect_stage(spec.name, m, r.get("stages"))})
    entries.sort(key=lambda e: (-e["deficit"], e["case"], e["metric"]))
    return entries[:k]


__all__ = ["MetricSpec", "METRICS", "METRICS_VERSION", "PIXEL_SOURCES", "SPEC", "INFO_KEYS", "EDITABILITY_KEY", "FRAME_KEY",
           "DEFINITION_PREFIXES", "definition_params", "editability", "goodness", "case_composite", "suspect_stage",
           "aggregate", "worst_entries"]
