"""Scenario families: a failure turned into a parameterised, seeded generator of cases + criteria.

    fam = ScenarioFamily(name="pale_surface", stage="perceive", param_space={"de": (1.5, 5.0), ...},
                         generate=make_case, criteria=[Criterion("surface_recall", ">=", 1.0)], ...)
    case = fam.case(seed=7)          # deterministic: same seed -> same params -> same pixels

A :class:`ScenarioCase` carries the target pixels and, when the source can provide it, the
ground-truth :class:`dt.ir.Document` (IR-generated and HTML/DOM-generated cases) and a region
of interest. Real crops have no ground truth: their criteria are fidelity measures only.

Criteria are thresholds on measured metrics (see :mod:`dt.scenarios.measures`). A case passes
when every criterion passes. Each criterion also yields a signed, normalised *margin*
(> 0 = passing with room, < 0 = failing by that much) used as the tuning objective.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np

from dt.ir import Box, Document
from dt.params import P

STAGES = ("perceive", "map", "translate")
"""Pipeline subsets a family exercises: perceive only; perceive + map; full translate
(perceive -> map -> refine k iterations -> remap). Every stage renders its output."""

SOURCES = ("ir", "html", "real")

_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": lambda v, t: v >= t, ">": lambda v, t: v > t,
    "<=": lambda v, t: v <= t, "<": lambda v, t: v < t,
    "==": lambda v, t: abs(v - t) <= 1e-9,
}

Threshold = Union[float, str]
"""A number, or ``"param:<key>"`` resolved from :data:`dt.params.P` at evaluation time (gate
thresholds such as ``validate.gate.ident.jnd_frac``; ``validate.*`` is never tuned)."""


@dataclass
class Criterion:
    """``metric op threshold`` measured on the whole frame (``roi=None``) or on the case ROI
    (``roi="roi"``). ``scale`` normalises the margin: ``(value - threshold) / scale`` for ``>``/``>=``,
    ``(threshold - value) / scale`` for ``<``/``<=``, ``-|value - threshold| / scale`` for ``==``."""
    metric: str
    op: str
    threshold: Threshold
    roi: Optional[str] = None
    scale: float = 1.0
    doc: str = ""

    def __post_init__(self) -> None:
        if self.op not in _OPS:
            raise ValueError(f"criterion op {self.op!r} not in {sorted(_OPS)}")
        if self.roi not in (None, "roi"):
            raise ValueError("criterion roi must be None (whole frame) or 'roi'")

    def resolved_threshold(self) -> float:
        t = self.threshold
        if isinstance(t, str):
            if not t.startswith("param:"):
                raise ValueError(f"threshold {t!r} must be a number or 'param:<key>'")
            import dt.validate.fidelity  # noqa: F401 - registers the validate.* gate thresholds
            return float(P[t[len("param:"):]])
        return float(t)

    def key(self) -> str:
        return f"{'roi.' if self.roi else ''}{self.metric}"

    def evaluate(self, value: Optional[float]) -> dict:
        """``{criterion, value, threshold, passed, margin}``; a missing/NaN value fails with margin -1."""
        thr = self.resolved_threshold()
        label = f"{self.key()} {self.op} {thr:g}"
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return {"criterion": label, "metric": self.key(), "value": None, "threshold": thr, "passed": False, "margin": -1.0}
        v = float(value)
        s = float(self.scale) or 1.0
        if self.op in (">=", ">"):
            m = (v - thr) / s
        elif self.op in ("<=", "<"):
            m = (thr - v) / s
        else:
            m = -abs(v - thr) / s
        return {"criterion": label, "metric": self.key(), "value": v, "threshold": thr,
                "passed": bool(_OPS[self.op](v, thr)), "margin": float(m)}

    def to_dict(self) -> dict:
        return {"metric": self.metric, "op": self.op, "threshold": self.threshold, "roi": self.roi,
                "scale": self.scale, "doc": self.doc}

    @staticmethod
    def from_dict(d: dict) -> "Criterion":
        return Criterion(d["metric"], d["op"], d["threshold"], d.get("roi"), float(d.get("scale", 1.0)), d.get("doc", ""))


@dataclass
class ScenarioCase:
    """One generated (or mined) instance of a family."""
    id: str
    family: str
    params: dict[str, Any]
    target_rgb: np.ndarray
    gt: Optional[Document] = None
    roi: Optional[Box] = None
    html: Optional[str] = None
    seed: Optional[int] = None
    meta: dict[str, Any] = field(default_factory=dict)
    """Family-specific facts the metrics need (e.g. glyph boxes, row boxes, the surface box)."""

    @property
    def width(self) -> int:
        return int(self.target_rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.target_rgb.shape[0])

    def fingerprint(self) -> str:
        """Hash of the target pixels (determinism checks)."""
        return hashlib.sha1(np.ascontiguousarray(self.target_rgb).tobytes()).hexdigest()[:16]


ParamSpace = dict[str, Union[tuple, list]]
"""``{name: (lo, hi)}`` (ints when both bounds are ints, else floats) or ``{name: [choices]}``."""


def sample_params(space: ParamSpace, seed: int) -> dict[str, Any]:
    """Deterministic draw of one parameter set from ``space`` (sorted key order)."""
    rng = random.Random(f"params:{seed}")
    out: dict[str, Any] = {}
    for k in sorted(space):
        v = space[k]
        if isinstance(v, list):
            out[k] = v[rng.randrange(len(v))]
        elif isinstance(v, tuple) and len(v) == 2:
            lo, hi = v
            if isinstance(lo, int) and isinstance(hi, int) and not isinstance(lo, bool):
                out[k] = rng.randint(lo, hi)
            else:
                out[k] = round(rng.uniform(float(lo), float(hi)), 4)
        else:
            raise ValueError(f"param space entry {k!r} must be (lo, hi) or [choices], got {v!r}")
    return out


@dataclass
class ScenarioFamily:
    """A failure mode as a family of generated cases.

    * ``stage`` -- pipeline subset (:data:`STAGES`); ``refine_iters`` applies to ``translate``.
    * ``failure_refs`` -- the ``knowledge/failures.jsonl`` cases/symptoms this family generalises.
    * ``param_space`` -- wide priors (the family must cover the failure *mode*, not one page).
    * ``generate(params, seed) -> ScenarioCase`` -- deterministic for a given ``(params, seed)``.
    * ``criteria`` -- all must pass for a case to pass.
    * ``metrics(case, pred, rendered) -> dict`` -- optional family-specific measures (added to
      the whole-frame metrics; criteria reference them by name).
    * ``tune_prefixes`` -- the ``dt.params`` key prefixes relevant to this failure (the default
      search space of ``dt train --family``).
    * ``sample`` -- optional ``seed -> params`` override (e.g. real crops pick a manifest entry).
    """
    name: str
    description: str
    stage: str
    failure_refs: list[str]
    param_space: ParamSpace
    generate: Callable[[dict, int], ScenarioCase]
    criteria: list[Criterion]
    source: str = "ir"
    refine_iters: int = 0
    tune_prefixes: tuple[str, ...] = ()
    metrics: Optional[Callable[[ScenarioCase, Document, np.ndarray], dict]] = None
    sample: Optional[Callable[[int], dict]] = None
    fidelity_text: str = "gt"
    """How fidelity measures find text to exclude: ``gt`` (gt text boxes) or ``ocr`` (target OCR)."""
    version: int = 1
    available: Callable[[], bool] = lambda: True
    """False when the family cannot produce cases on this machine (e.g. no mined crops)."""
    n_cases: Optional[Callable[[], int]] = None
    """For finite families (mined real crops): the number of distinct cases; seed ``s`` is case
    ``s % n``, and ``promote`` splits the indices into train / holdout instead of drawing seeds."""

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"family {self.name}: stage {self.stage!r} not in {STAGES}")
        if self.source not in SOURCES:
            raise ValueError(f"family {self.name}: source {self.source!r} not in {SOURCES}")
        if not self.criteria:
            raise ValueError(f"family {self.name}: at least one criterion is required")

    def params_for(self, seed: int) -> dict[str, Any]:
        return self.sample(seed) if self.sample else sample_params(self.param_space, seed)

    def case(self, seed: int) -> ScenarioCase:
        params = self.params_for(seed)
        c = self.generate(params, seed)
        c.seed = seed
        if not c.id:
            c.id = f"{self.name}_{seed}"
        return c

    def summary(self) -> dict:
        return {"name": self.name, "description": self.description, "stage": self.stage, "source": self.source,
                "refine_iters": self.refine_iters, "failure_refs": list(self.failure_refs),
                "param_space": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.param_space.items()},
                "criteria": [c.to_dict() for c in self.criteria], "tune_prefixes": list(self.tune_prefixes),
                "version": self.version}


def params_hash(params: dict) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:10]
