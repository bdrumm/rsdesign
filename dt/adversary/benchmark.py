"""The adversary benchmark: a deterministic, render-verified yardstick for comparison methods.

An *adversary* is any function ``adversary_fn(target_rgb, candidate_rgb, candidate_ir | None) ->
list[Finding]`` that reports typed, localised discrepancies of a candidate render w.r.t. a target
screenshot (see ``dt.adversary.taxonomy``). This module

1. **builds** a seeded benchmark: clean ground-truth IR documents x perturbation mixes (1-4 typed
   perturbations per case, magnitudes from just-visible to large) plus pure-noise cases. Each case is
   ``(target_rgb, candidate_rgb, candidate_ir, gt_findings)``: the target is the real render of the
   clean document (optionally with nuisance noise), the candidate is the render of the perturbed IR.
   Every ground-truth finding is verified against the real renderer (visible, and no changed pixel
   outside the ground-truth boxes); cases that fail are redrawn.
2. **splits** it into CALIBRATION and EVALUATION by document *and* seed: the two sets never share a
   source document or a perturbation seed, and EVALUATION additionally holds out a generator seed
   family (``synth_gen``) that calibration never sees. Approaches may only tune on calibration.
3. **scores** an adversary: per-type detection precision/recall with localisation (IoU >= 0.3 or the
   prediction's centre inside the ground-truth box), type accuracy at parent and leaf level, magnitude
   error where the type defines one, false findings on pure-noise cases, average precision by
   confidence, recall per magnitude bin, runtime; macro-averaged report + per-type confusion matrix.
4. ships two **floor** adversaries: ``null_adversary`` (always silent) and ``baseline_residual``
   (threshold the ``dt.compare`` residual regions, call everything ``structure.missing``).

CLI: ``python -m dt.adversary.benchmark [--quick] [--out out/adversary]`` builds (cached under
``out/adversary/bench/<key>/``), scores the baselines and writes ``benchmark_baseline.md``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from dt.adversary import perturb as PT
from dt.adversary.taxonomy import (ERROR_TYPES, GLOBAL_TYPES, NOISE_TYPES, PARENTS, TAXONOMY_VERSION, TYPES, Finding,
                                   is_noise, parent_of)
from dt.ir import Box, Document
from dt.params import P, register

BENCH_VERSION = "1"
ROOT = PT.ROOT
CORPUS = os.path.join(ROOT, "fixtures", "corpus")
DEFAULT_OUT = os.path.join(ROOT, "out", "adversary")

register("adversary.bench.loc_iou", 0.3, "a prediction localises a ground-truth finding at IoU >= this (or with its centre inside the box)", (0.05, 0.9))
register("adversary.bench.centre_max_frac", 0.25,
         "centre-in-box localisation only for ground-truth boxes up to this fraction of the page (page-sized truths need IoU)", (0.01, 1.0))
register("adversary.bench.noise_case_frac", 0.2, "fraction of cases that are pure noise (no ground-truth finding)", (0.0, 0.6))
register("adversary.bench.noise_in_perturbed", 0.5, "probability that a perturbed case's target also carries one nuisance", (0.0, 1.0))
register("adversary.bench.max_attempts", 8, "redraws of a case whose ground truth fails render verification", (1, 32))
register("adversary.bench.w_detect", 0.4, "headline weight: type-agnostic detection F1", (0.0, 1.0))
register("adversary.bench.w_typed", 0.3, "headline weight: macro leaf-typed F1", (0.0, 1.0))
register("adversary.bench.w_noise", 0.3, "headline weight: (1 - false-case rate on pure-noise cases) x detection recall", (0.0, 1.0))
register("adversary.baseline.conf_de", 50.0, "baseline confidence = mean ΔE in the residual region / this (clipped to 1)", (5.0, 200.0))

DEFAULT_CALIBRATION = tuple([f"synth:synth_1_{i:03d}" for i in range(6)] + [f"mwc:mwc_1_{i:03d}" for i in range(6)])
DEFAULT_EVALUATION = tuple([f"synth:synth_1_{i:03d}" for i in range(6, 12)] + [f"mwc:mwc_1_{i:03d}" for i in range(6, 12)]
                           + [f"gen:9001:{i}" for i in range(4)])
SPLITS = ("calibration", "evaluation")


# --------------------------------------------------------------------------- config / sources
@dataclass
class BenchConfig:
    seed: int = 0
    cases_per_doc: int = 4
    max_perturbations: int = 4
    calibration_docs: tuple[str, ...] = DEFAULT_CALIBRATION
    evaluation_docs: tuple[str, ...] = DEFAULT_EVALUATION
    include_browser_noise: bool = True

    def to_dict(self) -> dict:
        return {"seed": self.seed, "cases_per_doc": self.cases_per_doc, "max_perturbations": self.max_perturbations,
                "calibration_docs": list(self.calibration_docs), "evaluation_docs": list(self.evaluation_docs),
                "include_browser_noise": self.include_browser_noise}

    @staticmethod
    def from_dict(d: dict) -> "BenchConfig":
        return BenchConfig(int(d["seed"]), int(d["cases_per_doc"]), int(d["max_perturbations"]), tuple(d["calibration_docs"]),
                           tuple(d["evaluation_docs"]), bool(d.get("include_browser_noise", True)))

    def key(self, extra: Optional[dict] = None) -> str:
        """Cache key: config + every param that shapes the benchmark + taxonomy/bench versions + the source of
        the three benchmark modules (any code change rebuilds instead of silently reusing a stale cache)."""
        params = {k: v for k, v in sorted(P.all().items()) if k.startswith(("adversary.perturb.", "adversary.noise.", "adversary.bench.noise", "adversary.bench.max_attempts"))}
        code = hashlib.sha1(b"".join(open(os.path.join(os.path.dirname(__file__), f), "rb").read()
                                     for f in ("taxonomy.py", "perturb.py", "benchmark.py"))).hexdigest()[:12]
        blob = json.dumps({"cfg": self.to_dict(), "params": params, "tax": TAXONOMY_VERSION, "bench": BENCH_VERSION, "code": code, **(extra or {})},
                          sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


def family_of(spec: str) -> str:
    kind = spec.split(":", 1)[0]
    return {"synth": "synth", "mwc": "mwc", "gen": "synth_gen"}.get(kind, kind)


def _sub_seed(*parts: object) -> int:
    return int(hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def load_doc(spec: str) -> Document:
    """``synth:<id>`` / ``mwc:<id>`` (fixtures/corpus ground truth) or ``gen:<seed>:<index>`` (a fresh
    ``dt.selftest.synth`` document, text boxes measured in the real renderer)."""
    kind, rest = spec.split(":", 1)
    if kind in ("synth", "mwc"):
        doc = Document.load(os.path.join(CORPUS, kind, rest + ".gt.json"))
        doc.source_image = None
        return doc
    if kind == "gen":
        seed, idx = (int(v) for v in rest.split(":"))
        from dt.selftest import synth
        rng = random.Random(_sub_seed("gen", seed, idx))
        width = rng.choice(synth.WIDTHS)
        height = rng.randint(*synth.HEIGHT_RANGE)
        os.makedirs(PT.WORK_DIR, exist_ok=True)
        doc = synth.make_document(random.Random(rng.random()), width, height, PT.WORK_DIR)
        doc.meta = {"corpus": "synth_gen", "seed": seed, "index": idx}
        return doc
    raise ValueError(f"unknown document spec {spec!r}")


# --------------------------------------------------------------------------- cases
@dataclass
class Case:
    id: str
    split: str
    family: str
    doc: str
    kind: str                       # "perturbed" | "noise"
    gt: list[Finding]
    noise: dict                     # NoiseRecipe.to_dict() applied to the target
    meta: dict = field(default_factory=dict)
    _target: Optional[np.ndarray] = field(default=None, repr=False)
    _candidate: Optional[np.ndarray] = field(default=None, repr=False)
    _ir: Optional[Document] = field(default=None, repr=False)
    paths: dict = field(default_factory=dict, repr=False)

    @property
    def target_rgb(self) -> np.ndarray:
        if self._target is not None:
            return self._target
        from dt.common.image import load_rgb
        return load_rgb(self.paths["target"])

    @property
    def candidate_rgb(self) -> np.ndarray:
        if self._candidate is not None:
            return self._candidate
        from dt.common.image import load_rgb
        return load_rgb(self.paths["candidate"])

    @property
    def candidate_ir(self) -> Optional[Document]:
        if self._ir is not None:
            return copy.deepcopy(self._ir)
        return Document.load(self.paths["ir"]) if self.paths.get("ir") else None

    @property
    def page(self) -> Box:
        return Box(0, 0, self.meta["width"], self.meta["height"])

    def record(self) -> dict:
        return {"id": self.id, "split": self.split, "family": self.family, "doc": self.doc, "kind": self.kind,
                "gt": [f.to_dict() for f in self.gt], "noise": self.noise, "meta": self.meta}


@dataclass
class Benchmark:
    config: BenchConfig
    cases: list[Case]
    meta: dict = field(default_factory=dict)

    def split(self, name: str) -> list[Case]:
        if name not in SPLITS and name != "all":
            raise ValueError(f"split must be one of {SPLITS} or 'all'")
        return list(self.cases) if name == "all" else [c for c in self.cases if c.split == name]

    def summary(self) -> dict:
        out: dict = {}
        for s in SPLITS:
            cs = self.split(s)
            types: dict[str, int] = {}
            for c in cs:
                for f in c.gt:
                    types[f.type] = types.get(f.type, 0) + 1
            out[s] = {"cases": len(cs), "perturbed": sum(c.kind == "perturbed" for c in cs), "noise": sum(c.kind == "noise" for c in cs),
                      "findings": sum(len(c.gt) for c in cs), "docs": len({c.doc for c in cs}),
                      "families": sorted({c.family for c in cs}), "types": dict(sorted(types.items()))}
        return out

    # ---- persistence
    def save(self, out_dir: str) -> str:
        from dt.common.image import save_rgb
        os.makedirs(out_dir, exist_ok=True)
        for c in self.cases:
            for key, arr, ext in (("target", c._target, ".target.png"), ("candidate", c._candidate, ".candidate.png")):
                p = os.path.join(out_dir, c.id + ext)
                if arr is not None:
                    save_rgb(arr, p)
                c.paths[key] = p
            p = os.path.join(out_dir, c.id + ".candidate.json")
            if c._ir is not None:
                c._ir.save(p)
            c.paths["ir"] = p
        manifest = {"version": BENCH_VERSION, "taxonomy": TAXONOMY_VERSION, "config": self.config.to_dict(), "meta": self.meta,
                    "cases": [c.record() for c in self.cases]}
        path = os.path.join(out_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=1, sort_keys=True)
        return path

    @staticmethod
    def load(out_dir: str) -> "Benchmark":
        with open(os.path.join(out_dir, "manifest.json")) as f:
            m = json.load(f)
        cases = []
        for r in m["cases"]:
            c = Case(r["id"], r["split"], r["family"], r["doc"], r["kind"], [Finding.from_dict(g) for g in r["gt"]], r["noise"], r["meta"])
            c.paths = {"target": os.path.join(out_dir, c.id + ".target.png"), "candidate": os.path.join(out_dir, c.id + ".candidate.png"),
                       "ir": os.path.join(out_dir, c.id + ".candidate.json")}
            cases.append(c)
        return Benchmark(BenchConfig.from_dict(m["config"]), cases, m.get("meta", {}))

    def release_images(self) -> None:
        """Drop in-memory arrays of saved cases (they reload lazily from disk)."""
        for c in self.cases:
            if c.paths:
                c._target = c._candidate = c._ir = None


# --------------------------------------------------------------------------- build
def _pick_types(rng: random.Random, counts: dict[str, int], n: int) -> list[str]:
    """Balanced draw: types seen less often in this split are more likely (deterministic given rng)."""
    out: list[str] = []
    for _ in range(n):
        pool = [t for t in ERROR_TYPES if t not in out]
        w = [1.0 / (1.0 + counts.get(t, 0)) ** 2 for t in pool]
        out.append(rng.choices(pool, w)[0])
    return out


def _is_noise_slot(k: int) -> bool:
    """Exactly ``noise_case_frac`` of a split's cases are pure noise, spread evenly over its case sequence."""
    f = float(P["adversary.bench.noise_case_frac"])
    return math.floor((k + 1) * f + 1e-9) > math.floor(k * f + 1e-9)


def _make_case(cfg: BenchConfig, split: str, spec: str, j: int, doc: Document, clean: np.ndarray, counts: dict[str, int],
               kinds: list[str], noise_slot: bool = False) -> Optional[Case]:
    from dt.render.screenshot import render_doc
    rng = random.Random(_sub_seed(cfg.seed, split, spec, j))
    cid = f"{split[:3]}_{spec.replace(':', '-')}_{j}"
    meta = {"width": doc.width, "height": doc.height}
    if noise_slot:
        recipe = PT.sample_noise(rng, kinds, n=rng.choice([1, 1, 2]))
        cand_ir, _ = PT.sanitize(copy.deepcopy(doc))
        target = PT.render_target(doc, recipe, clean)
        from dt.compare.pixel import diff_map
        dm = diff_map(clean, target)
        meta.update(noise_mean_de=float(dm.mean()), noise_px=int((dm > float(P["adversary.perturb.visible_de"])).sum()))
        return Case(cid, split, family_of(spec), spec, "noise", [], recipe.to_dict(), meta, target, clean.copy(), cand_ir)
    for attempt in range(int(P["adversary.bench.max_attempts"])):
        n = rng.randint(1, cfg.max_perturbations)
        plan = [(t, rng.random()) for t in _pick_types(rng, counts, n)]
        cand_ir, gt = PT.perturb(doc, plan, rng)
        if not gt:
            continue
        cand = render_doc(cand_ir)
        v = PT.verify(clean, cand, gt)
        if not v["ok"]:
            continue
        for f, vis, ink in zip(gt, v["visible"], v["ink_boxes"]):
            f.evidence["visible_px"] = vis
            f.evidence["ink_box"] = ink
        for f in gt:
            counts[f.type] = counts.get(f.type, 0) + 1
        recipe = PT.sample_noise(rng, kinds, 1) if rng.random() < float(P["adversary.bench.noise_in_perturbed"]) else PT.NoiseRecipe()
        target = PT.render_target(doc, recipe, clean)
        meta.update(attempts=attempt + 1, plan=[t for t, _ in plan], changed_px=v["changed_px"])
        return Case(cid, split, family_of(spec), spec, "perturbed", gt, recipe.to_dict(), meta, target, cand, cand_ir)
    return None


def build(config: Optional[BenchConfig] = None, out_dir: Optional[str] = None, cache: bool = True,
          progress: Optional[Callable[[str], None]] = None) -> Benchmark:
    """Build (or load from cache) the benchmark. Deterministic for a given config, params, renderer and
    available noise kinds. With ``out_dir`` the cases are written to ``<out_dir>/<key>/`` and reloaded
    lazily; without it everything stays in memory."""
    from dt.render import screenshot as S
    from dt.render.screenshot import render_doc
    cfg = config or BenchConfig()
    kinds = PT.noise_kinds_available(cfg.include_browser_noise)
    key = cfg.key({"noise_kinds": kinds})
    target_dir = os.path.join(out_dir, key) if out_dir else None
    if cache and target_dir and os.path.exists(os.path.join(target_dir, "manifest.json")):
        return Benchmark.load(target_dir)
    t0 = time.time()
    cases: list[Case] = []
    skipped = 0
    for split, specs in (("calibration", cfg.calibration_docs), ("evaluation", cfg.evaluation_docs)):
        counts: dict[str, int] = {}
        k = 0
        for spec in specs:
            doc = load_doc(spec)
            clean = render_doc(doc)
            for j in range(cfg.cases_per_doc):
                c = _make_case(cfg, split, spec, j, doc, clean, counts, kinds, _is_noise_slot(k))
                k += 1
                if c is None:
                    skipped += 1
                    continue
                cases.append(c)
            if progress:
                progress(f"{split} {spec}: {len(cases)} cases")
    alt = PT.alt_browser()
    meta = {"key": key, "browser": S.BROWSER_USED, "browser_version": S._brw().version if S._brw() else None,
            "alt_browser": alt.name, "alt_browser_version": alt.version, "noise_kinds": kinds, "skipped_cases": skipped,
            "build_seconds": round(time.time() - t0, 1)}
    bench = Benchmark(cfg, cases, meta)
    if target_dir:
        bench.save(target_dir)
        bench.release_images()
    return bench


# --------------------------------------------------------------------------- scoring
AdversaryFn = Callable[[np.ndarray, np.ndarray, Optional[Document]], list[Finding]]


def localises(pred: Box, gt: Finding, page: Box) -> bool:
    if pred.iou(gt.box) >= float(P["adversary.bench.loc_iou"]):
        return True
    if gt.type in GLOBAL_TYPES or gt.box.area > float(P["adversary.bench.centre_max_frac"]) * page.area:
        return False
    return gt.box.contains_point(pred.cx, pred.cy)


def match(preds: list[Finding], gts: list[Finding], page: Box) -> list[tuple[int, int]]:
    """One-to-one assignment of predictions to ground truth among localising pairs, preferring IoU and
    type agreement (Hungarian)."""
    if not preds or not gts:
        return []
    from scipy.optimize import linear_sum_assignment
    S = np.full((len(preds), len(gts)), -1e6)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            if localises(p.box, g, page):
                S[i, j] = 1.0 + p.box.iou(g.box) + 0.5 * (parent_of(p.type) == parent_of(g.type)) + 0.5 * (p.type == g.type)
    r, c = linear_sum_assignment(-S)
    return [(int(i), int(j)) for i, j in zip(r, c) if S[i, j] > 0]


def _prf(tp: int, n_pred: int, n_gt: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "pred": n_pred, "gt": n_gt}


def _coerce(out: object) -> list[Finding]:
    res = []
    for f in (out or []):  # type: ignore[union-attr]
        res.append(f if isinstance(f, Finding) else Finding.from_dict(f))
    return res


def run(adversary_fn: AdversaryFn, cases: Iterable[Case], use_ir: bool = True) -> list[dict]:
    """Run an adversary over cases; returns per-case raw records (predictions, matches, runtime)."""
    rows = []
    for c in cases:
        target, cand, ir = c.target_rgb, c.candidate_rgb, (c.candidate_ir if use_ir else None)
        t0 = time.perf_counter()
        preds = _coerce(adversary_fn(target, cand, ir))
        dt_s = time.perf_counter() - t0
        errs = [p for p in preds if not is_noise(p.type)]
        rows.append({"case": c, "preds": errs, "abstained": len(preds) - len(errs), "pairs": match(errs, c.gt, c.page), "seconds": dt_s})
    return rows


def aggregate(rows: list[dict], name: str = "adversary", split: str = "") -> dict:
    """Turn per-case records into the benchmark report (all metrics documented in the module docstring)."""
    pert = [r for r in rows if r["case"].kind == "perturbed"]
    noise = [r for r in rows if r["case"].kind == "noise"]
    n_gt = sum(len(r["case"].gt) for r in pert)
    n_pred = sum(len(r["preds"]) for r in rows)
    tp = sum(len(r["pairs"]) for r in pert)
    det = _prf(tp, n_pred, n_gt)
    # per type (leaf and parent)
    gt_t: dict[str, int] = {}
    pred_t: dict[str, int] = {}
    tp_t: dict[str, int] = {}
    loc_t: dict[str, int] = {}
    gt_p: dict[str, int] = {}
    pred_p: dict[str, int] = {}
    tp_p: dict[str, int] = {}
    confusion: dict[str, dict[str, int]] = {}
    mag: dict[str, list[tuple[float, float]]] = {}
    bins: dict[str, list[int]] = {}
    fam: dict[str, list[int]] = {}
    leaf_ok = parent_ok = 0
    scored: list[tuple[float, bool]] = []
    for r in rows:
        c = r["case"]
        matched_p = {i: j for i, j in r["pairs"]}
        matched_g = {j: i for i, j in r["pairs"]}
        for i, p in enumerate(r["preds"]):
            pred_t[p.type] = pred_t.get(p.type, 0) + 1
            pred_p[parent_of(p.type)] = pred_p.get(parent_of(p.type), 0) + 1
            scored.append((float(p.confidence), i in matched_p))
            if i not in matched_p:
                row = confusion.setdefault("(spurious)", {})
                row[p.type] = row.get(p.type, 0) + 1
        for j, g in enumerate(c.gt):
            gt_t[g.type] = gt_t.get(g.type, 0) + 1
            gt_p[g.parent] = gt_p.get(g.parent, 0) + 1
            b = bins.setdefault(str(g.evidence.get("bin", "?")), [0, 0])
            fm = fam.setdefault(c.family, [0, 0, 0])
            b[1] += 1
            fm[1] += 1
            row = confusion.setdefault(g.type, {})
            if j not in matched_g:
                row["(missed)"] = row.get("(missed)", 0) + 1
                continue
            p = r["preds"][matched_g[j]]
            row[p.type] = row.get(p.type, 0) + 1
            b[0] += 1
            fm[0] += 1
            loc_t[g.type] = loc_t.get(g.type, 0) + 1
            if p.type == g.type:
                leaf_ok += 1
                tp_t[g.type] = tp_t.get(g.type, 0) + 1
                spec = TYPES.get(g.type)
                if spec and spec.magnitude_scored and g.magnitude is not None and p.magnitude is not None:
                    mag.setdefault(g.type, []).append((float(p.magnitude), float(g.magnitude)))
            if parent_of(p.type) == g.parent:
                parent_ok += 1
                tp_p[g.parent] = tp_p.get(g.parent, 0) + 1
        fam.setdefault(c.family, [0, 0, 0])[2] += len(r["preds"])
    per_type = {}
    for t in ERROR_TYPES:
        if gt_t.get(t, 0) or pred_t.get(t, 0):
            d = _prf(tp_t.get(t, 0), pred_t.get(t, 0), gt_t.get(t, 0))
            d["loc_recall"] = loc_t.get(t, 0) / gt_t[t] if gt_t.get(t) else 0.0
            pm = mag.get(t, [])
            if pm:
                ae = [abs(p - g) for p, g in pm]
                d["mag_mae"] = float(np.mean(ae))
                d["mag_rel_median"] = float(np.median([abs(p - g) / max(abs(g), 1e-6) for p, g in pm]))
                d["mag_n"] = len(pm)
            per_type[t] = d
    present = [t for t in ERROR_TYPES if gt_t.get(t, 0)]
    macro = {k: float(np.mean([per_type[t][k] for t in present])) if present else 0.0 for k in ("precision", "recall", "f1", "loc_recall")}
    per_parent = {p: _prf(tp_p.get(p, 0), pred_p.get(p, 0), gt_p.get(p, 0)) for p in PARENTS if p != "noise" and (gt_p.get(p) or pred_p.get(p))}
    pp = [p for p in per_parent if gt_p.get(p)]
    macro_parent = {k: float(np.mean([per_parent[p][k] for p in pp])) if pp else 0.0 for k in ("precision", "recall", "f1")}
    # average precision (confidence-ranked, non-interpolated)
    scored.sort(key=lambda t: -t[0])
    ap, hits = 0.0, 0
    for k, (_, hit) in enumerate(scored, 1):
        if hit:
            hits += 1
            ap += hits / k
    ap = ap / n_gt if n_gt else 0.0
    # noise
    false_cases = sum(1 for r in noise if r["preds"])
    by_noise: dict[str, list[int]] = {}
    for r in noise:
        for t in PT.NoiseRecipe(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in r["case"].noise.items()}).types():
            e = by_noise.setdefault(t, [0, 0])
            e[0] += bool(r["preds"])
            e[1] += 1
    noisy_pert = [r for r in pert if r["case"].noise]
    clean_pert = [r for r in pert if not r["case"].noise]

    def _fp(rs: list[dict]) -> float:
        return float(np.mean([len(r["preds"]) - len(r["pairs"]) for r in rs])) if rs else 0.0

    noise_rep = {"cases": len(noise), "cases_with_false": false_cases,
                 "false_case_rate": false_cases / len(noise) if noise else 0.0,
                 "false_per_case": float(np.mean([len(r["preds"]) for r in noise])) if noise else 0.0,
                 "by_type": {t: {"false_case_rate": v[0] / v[1], "cases": v[1]} for t, v in sorted(by_noise.items())},
                 "spurious_per_perturbed_case_clean_target": _fp(clean_pert),
                 "spurious_per_perturbed_case_noisy_target": _fp(noisy_pert)}
    secs = [r["seconds"] for r in rows]
    runtime = {"mean_s": float(np.mean(secs)) if secs else 0.0, "p95_s": float(np.percentile(secs, 95)) if secs else 0.0,
               "max_s": float(np.max(secs)) if secs else 0.0, "total_s": float(np.sum(secs)) if secs else 0.0}
    # silence on noise only earns credit in proportion to what the adversary does find (the always-silent
    # adversary must not out-score a detector)
    quiet = (1.0 - noise_rep["false_case_rate"]) if noise else 1.0
    headline = (float(P["adversary.bench.w_detect"]) * det["f1"] + float(P["adversary.bench.w_typed"]) * macro["f1"]
                + float(P["adversary.bench.w_noise"]) * quiet * det["recall"])
    return {
        "adversary": name, "split": split, "cases": len(rows), "perturbed_cases": len(pert), "gt_findings": n_gt,
        "headline": headline, "detection": det, "ap": ap,
        "type_accuracy": {"leaf": leaf_ok / tp if tp else 0.0, "parent": parent_ok / tp if tp else 0.0},
        "macro_leaf": macro, "macro_parent": macro_parent, "per_type": per_type, "per_parent": per_parent,
        "recall_by_bin": {b: v[0] / v[1] for b, v in sorted(bins.items()) if v[1]},
        "by_family": {f: {"recall": v[0] / v[1] if v[1] else 0.0, "gt": v[1], "preds": v[2]} for f, v in sorted(fam.items())},
        "noise": noise_rep, "runtime": runtime, "confusion": confusion,
        "abstentions": sum(r["abstained"] for r in rows), "taxonomy": TAXONOMY_VERSION,
    }


def score(adversary_fn: AdversaryFn, bench: Benchmark | list[Case], split: str = "evaluation", use_ir: bool = True,
          name: Optional[str] = None) -> dict:
    """Score ``adversary_fn`` on one split (default EVALUATION; tune only on ``"calibration"``)."""
    cases = bench.split(split) if isinstance(bench, Benchmark) else list(bench)
    rows = run(adversary_fn, cases, use_ir)
    return aggregate(rows, name or getattr(adversary_fn, "__name__", "adversary"), split)


# --------------------------------------------------------------------------- floor adversaries
def null_adversary(target_rgb: np.ndarray, candidate_rgb: np.ndarray, candidate_ir: Optional[Document] = None) -> list[Finding]:
    """Always silent: perfect on noise, zero recall. The absolute floor."""
    return []


def baseline_residual(target_rgb: np.ndarray, candidate_rgb: np.ndarray, candidate_ir: Optional[Document] = None) -> list[Finding]:
    """Trivial baseline: ``dt.compare`` residual regions (ΔE > compare.pixel.bad_de, dilated connected
    components), every region typed ``structure.missing``; confidence from its mean ΔE."""
    from dt.compare.pixel import diff_map, residual_regions
    d = diff_map(target_rgb, candidate_rgb)
    out = []
    for b in residual_regions(d):
        x0, y0, x1, y1 = b.as_int()
        m = float(d[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else 0.0
        out.append(Finding("structure.missing", b, None, min(1.0, m / float(P["adversary.baseline.conf_de"])), {"mean_de": m}))
    return out


# --------------------------------------------------------------------------- report
def _pct(v: float) -> str:
    return f"{100 * v:.1f}"


def format_report(reports: list[dict], bench: Benchmark, title: str = "Adversary benchmark") -> str:
    """Markdown report for one or more scored adversaries (same benchmark)."""
    L = [f"# {title}", ""]
    s = bench.summary()
    m = bench.meta
    L += [f"Benchmark v{BENCH_VERSION}, taxonomy v{TAXONOMY_VERSION}, key `{m.get('key')}`, seed {bench.config.seed}. "
          f"Renderer {m.get('browser')} {m.get('browser_version')}; alternate build for `noise.browser`: "
          f"{m.get('alt_browser') or 'none'} {m.get('alt_browser_version') or ''}. Noise kinds: {', '.join(m.get('noise_kinds', []))}.", "",
          "| split | cases | perturbed | pure noise | gt findings | docs | families |", "|---|---|---|---|---|---|---|"]
    for k in SPLITS:
        v = s[k]
        L.append(f"| {k} | {v['cases']} | {v['perturbed']} | {v['noise']} | {v['findings']} | {v['docs']} | {', '.join(v['families'])} |")
    L += ["", "Ground-truth findings per type: " + "; ".join(f"`{t}` {s['calibration']['types'].get(t, 0)}/{s['evaluation']['types'].get(t, 0)}" for t in ERROR_TYPES)
          + " (calibration/evaluation).", ""]
    L += ["## Headline", "", "| adversary | split | headline | det P | det R | det F1 | AP | macro leaf F1 | macro parent F1 | type acc leaf / parent | noise false-case rate | s/case |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in reports:
        d = r["detection"]
        L.append(f"| {r['adversary']} | {r['split']} | {r['headline']:.3f} | {_pct(d['precision'])} | {_pct(d['recall'])} | {_pct(d['f1'])} | {_pct(r['ap'])} | "
                 f"{_pct(r['macro_leaf']['f1'])} | {_pct(r['macro_parent']['f1'])} | {_pct(r['type_accuracy']['leaf'])} / {_pct(r['type_accuracy']['parent'])} | "
                 f"{_pct(r['noise']['false_case_rate'])} | {r['runtime']['mean_s']:.3f} |")
    L += ["", "headline = " + f"{P['adversary.bench.w_detect']}·detection F1 + {P['adversary.bench.w_typed']}·macro leaf F1 + "
          f"{P['adversary.bench.w_noise']}·(1 − noise false-case rate)·detection recall (silence earns credit only alongside detection). "
          "Percentages unless noted; detection = localised (IoU ≥ " + f"{P['adversary.bench.loc_iou']}" + " or prediction centre inside the truth box), any type.", ""]
    for r in reports:
        L += [f"## {r['adversary']} on {r['split']}", "",
              "| type | gt | pred | typed P | typed R | typed F1 | loc recall | magnitude MAE | median rel. err |", "|---|---|---|---|---|---|---|---|---|"]
        for t, d in r["per_type"].items():
            mae = f"{d['mag_mae']:.2f}" if "mag_mae" in d else "–"
            rel = f"{d['mag_rel_median']:.2f}" if "mag_rel_median" in d else "–"
            L.append(f"| `{t}` | {d['gt']} | {d['pred']} | {_pct(d['precision'])} | {_pct(d['recall'])} | {_pct(d['f1'])} | {_pct(d['loc_recall'])} | {mae} | {rel} |")
        n = r["noise"]
        L += ["", f"Recall (localised, any type) by magnitude bin: " + ", ".join(f"{b} {_pct(v)}" for b, v in r["recall_by_bin"].items()) + ".",
              "By family: " + ", ".join(f"{f} recall {_pct(v['recall'])} ({v['gt']} gt, {v['preds']} preds)" for f, v in r["by_family"].items()) + ".",
              f"Noise: {n['cases_with_false']}/{n['cases']} pure-noise cases drew a finding ({n['false_per_case']:.2f} findings per case); "
              + ", ".join(f"{t} {_pct(v['false_case_rate'])} of {v['cases']}" for t, v in n["by_type"].items())
              + f". Spurious findings per perturbed case: {n['spurious_per_perturbed_case_clean_target']:.2f} (clean target) vs "
              f"{n['spurious_per_perturbed_case_noisy_target']:.2f} (noisy target).",
              f"Runtime: mean {r['runtime']['mean_s']:.3f} s, p95 {r['runtime']['p95_s']:.3f} s per case.", ""]
        cols = sorted({k for row in r["confusion"].values() for k in row}, key=lambda k: (k.startswith("("), k))
        if cols:
            L += ["Confusion (rows: ground truth, columns: matched prediction type):", "",
                  "| gt \\ pred | " + " | ".join(f"`{c}`" for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
            for g in [t for t in ERROR_TYPES if t in r["confusion"]] + (["(spurious)"] if "(spurious)" in r["confusion"] else []):
                L.append(f"| `{g}` | " + " | ".join(str(r["confusion"][g].get(c, "")) for c in cols) + " |")
            L.append("")
    return "\n".join(L)


def save_examples(bench: Benchmark, adversary_fn: AdversaryFn, out_dir: str, n: int = 4, split: str = "evaluation") -> list[str]:
    """Evidence panels: target | candidate | ΔE heatmap with ground truth (green) and predictions (red)."""
    import cv2
    from dt.compare.pixel import diff_map
    from dt.compare.visualize import draw_boxes, heatmap
    from dt.common.image import save_rgb
    os.makedirs(out_dir, exist_ok=True)
    cases = bench.split(split)
    picks = [c for c in cases if c.kind == "perturbed"][: max(1, n - 1)] + [c for c in cases if c.kind == "noise"][:1]
    paths = []
    for c in picks:
        t, k = c.target_rgb, c.candidate_rgb
        preds = _coerce(adversary_fn(t, k, c.candidate_ir))
        hm = heatmap(diff_map(t, k))
        panels = []
        for img in (t, k, hm):
            img = draw_boxes(img, [f.box for f in c.gt], (0, 170, 0), 2)
            img = draw_boxes(img, [p.box for p in preds if not is_noise(p.type)], (220, 0, 0), 1)
            panels.append(img)
        sep = np.full((t.shape[0], 6, 3), 255, np.uint8)
        im = np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)
        scale = min(1.0, 2400 / im.shape[1])
        if scale < 1.0:
            im = cv2.resize(im, (int(im.shape[1] * scale), int(im.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        p = os.path.join(out_dir, f"{c.id}.png")
        save_rgb(im, p)
        paths.append(p)
    return paths


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Build the adversary benchmark and score the floor adversaries.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="2+2 documents, 3 cases each (smoke run)")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args(argv)
    cfg = BenchConfig(seed=a.seed)
    if a.quick:
        cfg = BenchConfig(seed=a.seed, cases_per_doc=3, calibration_docs=DEFAULT_CALIBRATION[:1] + DEFAULT_CALIBRATION[6:7],
                          evaluation_docs=DEFAULT_EVALUATION[:1] + DEFAULT_EVALUATION[6:7])
    bench = build(cfg, os.path.join(a.out, "bench"), cache=not a.no_cache, progress=print)
    reports = []
    for fn in (null_adversary, baseline_residual):
        for split in SPLITS:
            reports.append(score(fn, bench, split))
    md = format_report(reports, bench, "Adversary benchmark: floor adversaries")
    ex = save_examples(bench, baseline_residual, os.path.join(a.out, "examples"))
    md += "\n## Evidence\n\n" + "\n".join(f"* `{os.path.relpath(p, ROOT)}` (green = ground truth, red = baseline findings)" for p in ex) + "\n"
    with open(os.path.join(a.out, "benchmark_baseline.md"), "w") as f:
        f.write(md)
    with open(os.path.join(a.out, "benchmark_baseline.json"), "w") as f:
        json.dump({"summary": bench.summary(), "meta": bench.meta, "reports": reports}, f, indent=1, default=str)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
