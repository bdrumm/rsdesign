"""Type-routed, Bayesian fusion of the four adversary approaches (the synthesis ensemble).

The four research approaches (``acontrario``, ``decompose``, ``irspace``, ``metamorphic``) are strong
on different finding types and fail in different ways: a-contrario registers nuisance in the real
renderer and almost never fires on noise, decomposition explains colour/geometry causes cheaply,
ir-space and the metamorphic differ type from IR readings. This module combines them.

Three combiners, all fitted ONLY on a *fusion-calibration* set: fresh perturbation seeds (101-103)
drawn by the benchmark's own builder on the 12 CALIBRATION documents. No approach was tuned on
those seeds, so the approaches' reliabilities measured there are honest (measuring them on the
seed-0 calibration split, which every approach was tuned on, over-trusts the approaches; the
report shows by how much).

* ``route``  -- literal "use each approach where it is strongest": per predicted type, keep only
  the predictions of the approach with the best fusion-calibration F1 for that type, then
  non-maximum suppression across types.
* ``fuse``   -- **supervised Dawid-Skene fusion**. Findings of all approaches are clustered into
  candidate *causes* (one vote per approach per cluster; co-localised by the benchmark's own
  localisation rule). Each approach is a noisy annotator with a per-class firing rate
  ``P(fire_a | y)``, a confidence-bucket likelihood and a confusion row ``P(type_a = t | fire, y)``,
  all Beta/Dirichlet-smoothed towards approach-level and parent-level priors (hierarchical
  shrinkage: 18 leaf types x 4 approaches would otherwise over-fit ~250 labelled clusters). The
  posterior over ``y in types + {none}`` gives the fused type (argmax over error types) and a
  *calibrated* confidence ``1 - P(none | votes)``; a cluster is emitted when that is >= ``emit_min``.
  Silence is evidence too: an approach that reliably fires on a type and stayed silent lowers it.
* ``oracle`` variants (fitted on EVALUATION) are reported only as an optimistic upper bound: the
  gap between the honest and the oracle fit is the over-fitting a selection-on-evaluation would hide.

The box of a fused finding comes from the member whose approach localises that type best, the
magnitude from the member with the lowest fusion-calibration magnitude error for that type.

Predictions of every approach are cached per case under ``out/adversary/synthesis/preds/<approach>/
<approach version>/<bench key>/<case>.json`` so fitting and scoring never re-run an approach.

CLI: ``python -m dt.adversary.ensemble [--workers N]`` collects predictions, fits on the
fusion-calibration set, scores every approach / combiner on EVALUATION and writes
``out/adversary/synthesis/ensemble.json`` (the report is part of docs/ADVERSARIAL.md).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from dt.adversary.taxonomy import ERROR_TYPES, GLOBAL_TYPES, TYPES, Finding, is_noise, parent_of
from dt.ir import Box, Document
from dt.params import P, register

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "out", "adversary", "synthesis")
PRED_DIR = os.path.join(OUT, "preds")
MAIN_BENCH = os.path.join(ROOT, "out", "adversary", "bench", "f3bd306e2d77")
FCAL_DIR = os.path.join(OUT, "fcal")
FCAL_SEEDS = (101, 102, 103)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ensemble_model.json")

APPROACHES = ("acontrario", "decompose", "irspace", "metamorphic")
NONE = "none"
CLASSES = tuple(ERROR_TYPES) + (NONE,)

_K = "adversary.ensemble."
register(_K + "link_iou", 0.3, "two findings of different approaches are one cause at box IoU >= this (or one's centre inside the other's box)", (0.05, 0.9))
register(_K + "centre_max_frac", 0.25, "centre-in-box linking only for boxes up to this fraction of the page (as the benchmark's localisation)", (0.01, 1.0))
register(_K + "fire_prior", 4.0, "Beta prior strength of P(fire_a | y) towards the approach's overall firing rate", (0.5, 50.0))
register(_K + "confusion_prior", 3.0, "Dirichlet prior strength of P(type_a | fire, y) towards the approach-level confusion prior", (0.5, 50.0))
register(_K + "same_parent_share", 0.6, "prior share of an approach's typing errors that stay inside the true parent", (0.0, 1.0))
register(_K + "class_prior", 1.0, "add-k smoothing of the class prior P(y)", (0.1, 20.0))
register(_K + "conf_buckets", 3, "confidence buckets per approach (quantiles on the fusion-calibration set)", (1, 6))
register(_K + "emit_min", 0.5, "emit a fused cluster when 1 - P(none | votes) >= this (Bayes decision at 0.5)", (0.05, 0.95))
register(_K + "nms_iou", 0.5, "route: drop a lower-precision prediction that localises onto an emitted one of another approach at IoU >= this", (0.1, 0.9))


# --------------------------------------------------------------------------- approaches + prediction cache
def finder(name: str) -> Callable:
    """The ``find`` function of an approach (lazy import: the renderer and perceiver are heavy)."""
    if name not in APPROACHES:
        raise ValueError(f"unknown approach {name!r}; one of {APPROACHES}")
    return importlib.import_module(f"dt.adversary.{name}").find


def approach_version(name: str) -> str:
    """Hash of everything that determines an approach's output: its source, its learned model and its
    registered params (so the prediction cache can never be stale)."""
    mod = importlib.import_module(f"dt.adversary.{name}")
    h = hashlib.sha1()
    h.update(open(mod.__file__, "rb").read())
    if name == "acontrario":
        mp = getattr(mod, "MODEL_PATH", None)
        if mp and os.path.exists(mp):
            h.update(open(mp, "rb").read())
    prefix = {"acontrario": "adversary.acontrario.", "decompose": "adversary.decompose.", "irspace": "adversary.irspace.",
              "metamorphic": "adversary.meta."}[name]
    h.update(json.dumps({k: v for k, v in sorted(P.all().items()) if k.startswith(prefix)}, sort_keys=True, default=str).encode())
    return h.hexdigest()[:10]


def bench_key(case) -> str:
    k = getattr(case, "bench_key", None)
    if isinstance(k, str):
        return k
    p = case.paths.get("target") if case.paths else None
    return os.path.basename(os.path.dirname(p)) if p else "memory"


def uid(case) -> str:
    """Unique case key across benchmarks (case ids repeat across seeds of the same documents)."""
    return f"{bench_key(case)}:{case.id}"


def _pred_path(name: str, version: str, case) -> str:
    return os.path.join(PRED_DIR, name, version, bench_key(case), case.id + ".json")


def _run_one(args: tuple) -> tuple[str, list[dict], float]:
    name, target_p, cand_p, ir_p, cid = args  # cid = uid(case)
    from dt.common.image import load_rgb
    t, c = load_rgb(target_p), load_rgb(cand_p)
    ir = Document.load(ir_p) if ir_p and os.path.exists(ir_p) else None
    t0 = time.perf_counter()
    fs = finder(name)(t, c, ir)
    sec = time.perf_counter() - t0
    out = [(f if isinstance(f, Finding) else Finding.from_dict(f)).to_dict() for f in fs]
    return cid, out, sec


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Box):
        return o.to_dict()
    return o


def predictions(name: str, cases: list, workers: int = 1, progress: Optional[Callable[[str], None]] = None) -> dict[str, dict]:
    """``{uid(case): {"findings": [Finding], "seconds": s}}`` for approach ``name``, cached on disk."""
    ver = approach_version(name)
    out: dict[str, dict] = {}
    todo = []
    for c in cases:
        p = _pred_path(name, ver, c)
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            out[uid(c)] = {"findings": [Finding.from_dict(x) for x in d["findings"]], "seconds": d["seconds"]}
        else:
            todo.append(c)
    if todo:
        jobs = [(name, c.paths["target"], c.paths["candidate"], c.paths.get("ir"), uid(c)) for c in todo]
        by_id = {uid(c): c for c in todo}
        if len(by_id) != len(todo):
            raise ValueError("duplicate case uids")

        def _store(cid: str, fs: list[dict], sec: float) -> None:
            c = by_id[cid]
            p = _pred_path(name, ver, c)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                json.dump({"approach": name, "version": ver, "case": c.id, "uid": cid, "seconds": sec, "findings": _jsonable(fs)}, f)
            out[cid] = {"findings": [Finding.from_dict(x) for x in fs], "seconds": sec}
            if progress:
                progress(f"{name} {cid} {len(fs)} findings {sec:.2f}s ({len(out)}/{len(cases)})")

        if workers > 1 and len(jobs) > 1:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers) as pool:
                for cid, fs, sec in pool.imap_unordered(_run_one, jobs):
                    _store(cid, fs, sec)
        else:
            for j in jobs:
                _store(*_run_one(j))
    return out


# --------------------------------------------------------------------------- benchmarks
def main_bench():
    from dt.adversary.benchmark import Benchmark
    return Benchmark.load(MAIN_BENCH)


def fusion_calibration(seeds: Iterable[int] = FCAL_SEEDS) -> list:
    """The fusion-calibration cases: the benchmark builder on the 12 CALIBRATION documents with fresh
    perturbation/noise seeds no approach was tuned on (built once, cached)."""
    from dt.adversary.benchmark import BenchConfig, build
    cases = []
    for s in seeds:
        b = build(BenchConfig(seed=int(s), evaluation_docs=()), FCAL_DIR)
        for c in b.split("calibration"):
            c.meta = dict(c.meta, fcal_seed=int(s))
            cases.append(c)
    return cases


# --------------------------------------------------------------------------- scoring from cached predictions
def rows_from(preds: dict[str, list[Finding]], cases: list, seconds: Optional[dict[str, float]] = None) -> list[dict]:
    """Benchmark rows (``dt.adversary.benchmark.run`` format) from already-computed predictions."""
    from dt.adversary.benchmark import match
    rows = []
    for c in cases:
        fs = preds.get(uid(c), [])
        errs = [f for f in fs if not is_noise(f.type)]
        rows.append({"case": c, "preds": errs, "abstained": len(fs) - len(errs), "pairs": match(errs, c.gt, c.page),
                     "seconds": float((seconds or {}).get(uid(c), 0.0))})
    return rows


def score_preds(preds: dict[str, list[Finding]], cases: list, name: str, split: str = "", seconds: Optional[dict] = None) -> dict:
    from dt.adversary.benchmark import aggregate
    return aggregate(rows_from(preds, cases, seconds), name, split)


def _page_of(case) -> Box:
    return case.page


# --------------------------------------------------------------------------- clustering into causes
def _colocated(a: Finding, b: Finding, page: Box) -> float:
    """Link score of two findings (0 = not one cause): IoU, or one's centre inside the other's box
    (small boxes only, like the benchmark's localisation). Page-wide types only link to each other."""
    ga, gb = a.type in GLOBAL_TYPES, b.type in GLOBAL_TYPES
    if ga or gb:
        return 1.0 if ga and gb else 0.0
    iou = a.box.iou(b.box)
    if iou >= float(P[_K + "link_iou"]):
        return 1.0 + iou
    lim = float(P[_K + "centre_max_frac"]) * page.area
    if (a.box.area <= lim and a.box.contains_point(b.box.cx, b.box.cy)) or (b.box.area <= lim and b.box.contains_point(a.box.cx, a.box.cy)):
        return 0.5 + iou
    return 0.0


@dataclass
class Cluster:
    """One candidate cause: at most one voting member per approach (``votes``) plus same-approach
    fragments that co-localise with that approach's member (``fragments``, kept as evidence)."""
    votes: dict[str, Finding] = field(default_factory=dict)
    fragments: list[tuple[str, Finding]] = field(default_factory=list)

    def members(self) -> list[tuple[str, Finding]]:
        return list(self.votes.items())

    def box(self) -> Box:
        bs = [f.box for f in self.votes.values()]
        out = bs[0]
        for b in bs[1:]:
            out = out.union(b)
        return out


def cluster(by_approach: dict[str, list[Finding]], page: Box, order: Iterable[str] = APPROACHES) -> list[Cluster]:
    """Greedy multi-way matching of the approaches' findings into causes. Approaches are visited in
    ``order`` (most reliable first) and each finding in confidence order; a finding joins the cluster
    it co-localises with best that has no vote from its approach yet, else it is a fragment of a
    cluster where its approach already voted (same approach, same place), else it opens a cluster."""
    clusters: list[Cluster] = []
    for a in order:
        fs = sorted([f for f in by_approach.get(a, []) if not is_noise(f.type)], key=lambda f: -float(f.confidence))
        for f in fs:
            best, best_s, frag, frag_s = None, 0.0, None, 0.0
            for cl in clusters:
                s = max((_colocated(f, g, page) for g in cl.votes.values()), default=0.0)
                if s <= 0:
                    continue
                if a in cl.votes:
                    if s > frag_s:
                        frag, frag_s = cl, s
                elif s > best_s:
                    best, best_s = cl, s
            if best is not None:
                best.votes[a] = f
            elif frag is not None and frag.votes[a].type == f.type:
                frag.fragments.append((a, f))
            else:
                clusters.append(Cluster({a: f}))
    return clusters


def label_clusters(clusters: list[Cluster], gt: list[Finding], page: Box) -> list[str]:
    """Training labels: the gt type a cluster localises (one-to-one, Hungarian over the members' best
    localisation), else ``none``."""
    from dt.adversary.benchmark import localises
    labels = [NONE] * len(clusters)
    if not clusters or not gt:
        return labels
    from scipy.optimize import linear_sum_assignment
    S = np.full((len(clusters), len(gt)), -1e6)
    for i, cl in enumerate(clusters):
        for j, g in enumerate(gt):
            best = -1e6
            for f in cl.votes.values():
                if localises(f.box, g, page):
                    best = max(best, 1.0 + f.box.iou(g.box) + 0.5 * (f.type == g.type))
            S[i, j] = best
    r, c = linear_sum_assignment(-S)
    for i, j in zip(r, c):
        if S[i, j] > 0:
            labels[i] = gt[j].type
    return labels


# --------------------------------------------------------------------------- the fusion model
def _parent_peers(t: str) -> list[str]:
    return [u for u in ERROR_TYPES if u != t and parent_of(u) == parent_of(t)]


@dataclass
class FusionModel:
    """Supervised Dawid-Skene with hierarchical smoothing (see module docstring). All tables are plain
    dicts so the model is JSON-serialisable and printable."""
    approaches: tuple[str, ...]
    prior: dict[str, float]                               # P(y)
    fire: dict[str, dict[str, float]]                     # a -> y -> P(fire | y)
    conf_edges: dict[str, list[float]]                    # a -> bucket edges
    conf_lik: dict[str, dict[str, list[float]]]           # a -> {"err": [...], "none": [...]} P(bucket | fire, y is err/none)
    confusion: dict[str, dict[str, dict[str, float]]]     # a -> y -> t -> P(type t | fire, y)
    box_rank: dict[str, list[str]]                        # type -> approaches by localisation precision
    mag_rank: dict[str, list[str]]                        # type -> approaches by magnitude MAE
    order: list[str]                                      # clustering order
    emit_min: float = 0.5
    meta: dict = field(default_factory=dict)

    # ---- inference
    def _bucket(self, a: str, conf: float) -> int:
        return int(np.searchsorted(np.asarray(self.conf_edges.get(a, []), float), float(conf), side="right"))

    def log_lik(self, cl: Cluster) -> dict[str, float]:
        out = {}
        for y in CLASSES:
            s = math.log(max(self.prior.get(y, 1e-9), 1e-12))
            for a in self.approaches:
                pf = min(max(self.fire[a].get(y, 1e-6), 1e-6), 1 - 1e-6)
                f = cl.votes.get(a)
                if f is None:
                    s += math.log(1 - pf)
                    continue
                s += math.log(pf)
                b = self._bucket(a, f.confidence)
                lk = self.conf_lik[a]["none" if y == NONE else "err"]
                s += math.log(max(lk[min(b, len(lk) - 1)], 1e-9))
                row = self.confusion[a].get(y, {})
                s += math.log(max(row.get(f.type, row.get("_other", 1e-6)), 1e-9))
            out[y] = s
        return out

    def posterior(self, cl: Cluster) -> dict[str, float]:
        ll = self.log_lik(cl)
        m = max(ll.values())
        e = {k: math.exp(v - m) for k, v in ll.items()}
        z = sum(e.values())
        return {k: v / z for k, v in e.items()}

    def fuse_cluster(self, cl: Cluster) -> Optional[Finding]:
        post = self.posterior(cl)
        p_err = 1.0 - post[NONE]
        y = max((t for t in ERROR_TYPES), key=lambda t: post.get(t, 0.0))
        voters = {a: f for a, f in cl.votes.items()}
        rank = self.box_rank.get(y, list(self.order))
        same = [a for a in rank if a in voters and voters[a].type == y]
        src = same[0] if same else next((a for a in rank if a in voters), next(iter(voters)))
        f0 = voters[src]
        mag = None
        for a in self.mag_rank.get(y, []):
            if a in voters and voters[a].type == y and voters[a].magnitude is not None:
                mag = voters[a].magnitude
                break
        if mag is None and f0.type == y:
            mag = f0.magnitude
        node = f0.node_id or next((f.node_id for f in voters.values() if f.node_id), None)
        top = sorted(post.items(), key=lambda kv: -kv[1])[:3]
        ev = {"method": "ensemble.fuse", "p_error": round(p_err, 4), "p_type": round(post[y] / max(p_err, 1e-9), 4),
              "posterior_top": [[k, round(v, 4)] for k, v in top], "box_from": src,
              "members": [{"approach": a, "type": f.type, "confidence": round(float(f.confidence), 3), "box": f.box.to_dict()}
                          for a, f in voters.items()],
              "fragments": len(cl.fragments)}
        return Finding(y, f0.box, mag, float(p_err), ev, node)

    def fuse(self, by_approach: dict[str, list[Finding]], page: Box) -> list[Finding]:
        cls = cluster(by_approach, page, self.order)
        out = []
        for cl in cls:
            f = self.fuse_cluster(cl)
            if f is not None and f.confidence >= self.emit_min:
                out.append(f)
        noise = [f for a in self.approaches for f in by_approach.get(a, []) if is_noise(f.type)]
        if noise:  # keep one page-level abstention so the evidence that a nuisance was explained survives
            n = noise[0]
            out.append(Finding(n.type, n.box, None, float(n.confidence), {"method": "ensemble.fuse", "abstention_from": len(noise)}))
        return out

    # ---- persistence
    def to_dict(self) -> dict:
        return {"approaches": list(self.approaches), "prior": self.prior, "fire": self.fire, "conf_edges": self.conf_edges,
                "conf_lik": self.conf_lik, "confusion": self.confusion, "box_rank": self.box_rank, "mag_rank": self.mag_rank,
                "order": self.order, "emit_min": self.emit_min, "meta": self.meta}

    @staticmethod
    def from_dict(d: dict) -> "FusionModel":
        return FusionModel(tuple(d["approaches"]), d["prior"], d["fire"], d["conf_edges"], d["conf_lik"], d["confusion"],
                           d["box_rank"], d["mag_rank"], list(d["order"]), float(d.get("emit_min", 0.5)), d.get("meta", {}))

    def save(self, path: str = MODEL_PATH) -> str:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1, sort_keys=True)
        return path

    @staticmethod
    def load(path: str = MODEL_PATH) -> "FusionModel":
        with open(path) as f:
            return FusionModel.from_dict(json.load(f))


def _approach_stats(preds: dict[str, dict[str, list[Finding]]], cases: list, approaches: Iterable[str]) -> dict:
    """Per approach: per-type typed precision (box rank), magnitude MAE, overall precision (cluster order)."""
    stats: dict = {}
    for a in approaches:
        rep = score_preds(preds[a], cases, a)
        stats[a] = rep
    return stats


def fit(preds: dict[str, dict[str, list[Finding]]], cases: list, approaches: Iterable[str] = APPROACHES,
        emit_min: Optional[float] = None) -> FusionModel:
    """Fit the fusion model on labelled cases (``preds[approach][case_id] -> findings``)."""
    approaches = tuple(approaches)
    stats = _approach_stats(preds, cases, approaches)
    order = sorted(approaches, key=lambda a: -stats[a]["detection"]["precision"])
    # clusters + labels
    samples: list[tuple[Cluster, str]] = []
    for c in cases:
        by = {a: preds[a].get(uid(c), []) for a in approaches}
        cls = cluster(by, c.page, order)
        for cl, y in zip(cls, label_clusters(cls, c.gt, c.page)):
            samples.append((cl, y))
    n_y = Counter(y for _, y in samples)
    # gt findings never touched by any cluster still count for the class prior of "silent" evidence
    k = float(P[_K + "class_prior"])
    tot = sum(n_y.values()) + k * len(CLASSES)
    prior = {y: (n_y.get(y, 0) + k) / tot for y in CLASSES}
    fire: dict = {}
    confusion: dict = {}
    conf_edges: dict = {}
    conf_lik: dict = {}
    a_fire = float(P[_K + "fire_prior"])
    a_conf = float(P[_K + "confusion_prior"])
    rho = float(P[_K + "same_parent_share"])
    nb = int(P[_K + "conf_buckets"])
    for a in approaches:
        err_samples = [(cl, y) for cl, y in samples if y != NONE]
        none_samples = [(cl, y) for cl, y in samples if y == NONE]
        base_err = (sum(a in cl.votes for cl, _ in err_samples) + 1) / (len(err_samples) + 2)
        base_none = (sum(a in cl.votes for cl, _ in none_samples) + 1) / (len(none_samples) + 2)
        fire[a] = {}
        for y in CLASSES:
            ss = [cl for cl, yy in samples if yy == y]
            m = base_none if y == NONE else base_err
            fire[a][y] = (sum(a in cl.votes for cl in ss) + a_fire * m) / (len(ss) + a_fire)
        # confidence buckets: quantiles of this approach's confidences
        confs = [float(cl.votes[a].confidence) for cl, _ in samples if a in cl.votes]
        edges = sorted(set(float(np.quantile(confs, q)) for q in np.linspace(0, 1, nb + 1)[1:-1])) if len(confs) >= 2 * nb and nb > 1 else []
        conf_edges[a] = edges
        lik = {}
        for grp, ss in (("err", err_samples), ("none", none_samples)):
            cnt = np.ones(len(edges) + 1)
            for cl, _ in ss:
                if a in cl.votes:
                    cnt[int(np.searchsorted(np.asarray(edges, float), float(cl.votes[a].confidence), side="right"))] += 1
            lik[grp] = (cnt / cnt.sum()).tolist()
        conf_lik[a] = lik
        # confusion rows with an approach-level prior
        typed = [(cl.votes[a].type, y) for cl, y in err_samples if a in cl.votes]
        acc = (sum(t == y for t, y in typed) + 1) / (len(typed) + 2)
        confusion[a] = {}
        for y in CLASSES:
            row_cnt = Counter(cl.votes[a].type for cl, yy in samples if yy == y and a in cl.votes)
            n = sum(row_cnt.values())
            if y == NONE:
                spur = Counter(cl.votes[a].type for cl, yy in none_samples if a in cl.votes)
                tot_s = sum(spur.values())
                pri = {t: (spur.get(t, 0) + 1) / (tot_s + len(ERROR_TYPES)) for t in ERROR_TYPES}
            else:
                peers = _parent_peers(y)
                others = [t for t in ERROR_TYPES if t != y and t not in peers]
                pri = {y: acc}
                for t in peers:
                    pri[t] = (1 - acc) * rho / len(peers)
                for t in others:
                    pri[t] = (1 - acc) * ((1 - rho) if peers else 1.0) / len(others)
            row = {t: (row_cnt.get(t, 0) + a_conf * pri[t]) / (n + a_conf) for t in ERROR_TYPES}
            row["_other"] = min(row.values())
            confusion[a][y] = row
    # box / magnitude ranks per type
    box_rank: dict = {}
    mag_rank: dict = {}
    for t in ERROR_TYPES:
        def prec(a: str) -> float:
            d = stats[a]["per_type"].get(t)
            return ((d["tp"] + 1) / (d["pred"] + 2)) if d else 0.5 * stats[a]["detection"]["precision"]
        box_rank[t] = sorted(approaches, key=lambda a: -prec(a))

        def mae(a: str) -> float:
            d = stats[a]["per_type"].get(t) or {}
            return float(d.get("mag_rel_median", d.get("mag_mae", 1e9)) if d.get("mag_n", 0) >= 2 else 1e9)
        mag_rank[t] = [a for a in sorted(approaches, key=mae) if mae(a) < 1e9]
    model = FusionModel(approaches, prior, fire, conf_edges, conf_lik, confusion, box_rank, mag_rank, order,
                        float(P[_K + "emit_min"] if emit_min is None else emit_min),
                        {"cases": len(cases), "clusters": len(samples), "labels": dict(n_y),
                         "case_ids_hash": hashlib.sha1(",".join(sorted(uid(c) for c in cases)).encode()).hexdigest()[:10]})
    return model


def fuse_preds(model: FusionModel, preds: dict[str, dict[str, list[Finding]]], cases: list) -> dict[str, list[Finding]]:
    return {uid(c): model.fuse({a: preds[a].get(uid(c), []) for a in model.approaches}, c.page) for c in cases}


def tune_emit(model: FusionModel, preds: dict, cases: list, grid: Iterable[float] = (0.3, 0.4, 0.5, 0.6, 0.7)) -> tuple[float, list]:
    """Choose ``emit_min`` on the (fusion-calibration) cases by the benchmark headline; one scalar."""
    hist = []
    best, best_h = model.emit_min, -1.0
    for e in grid:
        model.emit_min = float(e)
        h = score_preds(fuse_preds(model, preds, cases), cases, "fuse")["headline"]
        hist.append((float(e), h))
        if h > best_h + 1e-9:
            best, best_h = float(e), h
    model.emit_min = best
    return best, hist


# --------------------------------------------------------------------------- routing ("each where strongest")
@dataclass
class Router:
    route: dict[str, str]          # predicted type -> approach
    precision: dict[str, float]    # predicted type -> routed approach's typed precision (NMS order)

    def apply(self, by_approach: dict[str, list[Finding]], page: Box) -> list[Finding]:
        from dt.adversary.benchmark import localises
        cand = []
        for t, a in self.route.items():
            for f in by_approach.get(a, []):
                if f.type == t:
                    g = Finding(f.type, f.box, f.magnitude, float(f.confidence), dict(f.evidence, routed_from=a), f.node_id)
                    cand.append((self.precision.get(t, 0.0), g))
        cand.sort(key=lambda t: (-t[0], -t[1].confidence))
        out: list[Finding] = []
        thr = float(P[_K + "nms_iou"])
        for _, f in cand:
            if any(f.box.iou(g.box) >= thr or (f.type not in GLOBAL_TYPES and g.type not in GLOBAL_TYPES and localises(f.box, g, page)
                                               and f.box.iou(g.box) > 0.1) for g in out):
                continue
            out.append(f)
        return out

    def to_dict(self) -> dict:
        return {"route": self.route, "precision": self.precision}


def fit_router(preds: dict, cases: list, approaches: Iterable[str] = APPROACHES) -> Router:
    stats = _approach_stats(preds, cases, approaches)
    route, prec = {}, {}
    for t in ERROR_TYPES:
        best = max(approaches, key=lambda a: (stats[a]["per_type"].get(t, {}).get("f1", 0.0), stats[a]["per_type"].get(t, {}).get("precision", 0.0)))
        d = stats[best]["per_type"].get(t)
        if d and d["f1"] > 0:
            route[t], prec[t] = best, d["precision"]
    return Router(route, prec)


def route_preds(router: Router, preds: dict, cases: list, approaches: Iterable[str] = APPROACHES) -> dict[str, list[Finding]]:
    return {uid(c): router.apply({a: preds[a].get(uid(c), []) for a in approaches}, c.page) for c in cases}


# --------------------------------------------------------------------------- the shipped adversary
_MODEL: Optional[FusionModel] = None


def load_model(path: str = MODEL_PATH) -> FusionModel:
    global _MODEL
    if _MODEL is None or path != MODEL_PATH:
        m = FusionModel.load(path)
        if path != MODEL_PATH:
            return m
        _MODEL = m
    return _MODEL


def find(target_rgb: np.ndarray, candidate_rgb: np.ndarray, candidate_ir: Optional[Document] = None,
         approaches: Optional[Iterable[str]] = None) -> list[Finding]:
    """The fused adversary (benchmark ``AdversaryFn``): run every approach in the shipped model and fuse.
    Approaches missing from ``approaches`` are treated as silent (lower recall, still calibrated)."""
    import copy
    model = load_model()
    use = tuple(approaches) if approaches is not None else model.approaches
    by = {}
    for a in model.approaches:
        if a in use:
            by[a] = [f if isinstance(f, Finding) else Finding.from_dict(f)
                     for f in finder(a)(target_rgb, candidate_rgb, copy.deepcopy(candidate_ir) if candidate_ir is not None else None)]
    h, w = target_rgb.shape[:2]
    return model.fuse(by, Box(0, 0, w, h))


find.__name__ = "ensemble"


# --------------------------------------------------------------------------- evidence
def save_panels(cases: list, runs: dict[str, dict[str, list[Finding]]], out_dir: str, max_w: int = 2400) -> list[str]:
    """target | candidate | ΔE heatmap per case, ground truth in green; one row per adversary in ``runs``
    (``{name: {uid: findings}}``), its findings in red on the heatmap."""
    import cv2
    from dt.common.image import save_rgb
    from dt.compare.pixel import diff_map
    from dt.compare.visualize import draw_boxes, heatmap
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for c in cases:
        t, k = c.target_rgb, c.candidate_rgb
        hm = heatmap(diff_map(t, k))
        rows = []
        sep = np.full((t.shape[0], 6, 3), 255, np.uint8)
        for name, by in runs.items():
            fs = [f for f in by.get(uid(c), []) if not is_noise(f.type)]
            panels = []
            for img in (t, k, hm):
                img = draw_boxes(img, [g.box for g in c.gt], (0, 170, 0), 2)
                img = draw_boxes(img, [f.box for f in fs], (220, 0, 0), 1)
                panels.append(img)
            row = np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)
            label = np.full((22, row.shape[1], 3), 255, np.uint8)
            txt = f"{name}: " + ", ".join(sorted(f.type for f in fs))[:180]
            cv2.putText(label, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            rows.append(np.concatenate([label, row], axis=0))
        im = np.concatenate(rows, axis=0)
        sc = min(1.0, max_w / im.shape[1])
        if sc < 1.0:
            im = cv2.resize(im, (int(im.shape[1] * sc), int(im.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        p = os.path.join(out_dir, f"{c.id}.png")
        save_rgb(np.ascontiguousarray(im), p)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- experiment
def _load_all(cases: list, approaches: Iterable[str], workers: int, progress=None) -> tuple[dict, dict]:
    preds, secs = {}, {}
    for a in approaches:
        d = predictions(a, cases, workers=workers, progress=progress)
        preds[a] = {k: v["findings"] for k, v in d.items()}
        secs[a] = {k: v["seconds"] for k, v in d.items()}
    return preds, secs


def _summ(r: dict) -> dict:
    keep = ("adversary", "split", "headline", "ap")
    out = {k: r[k] for k in keep}
    out.update(det=r["detection"], macro_leaf=r["macro_leaf"], macro_parent=r["macro_parent"], type_acc=r["type_accuracy"],
               noise=r["noise"], runtime=r["runtime"], per_type={t: {k: d[k] for k in ("precision", "recall", "f1", "gt", "pred")}
                                                                for t, d in r["per_type"].items()},
               by_family=r["by_family"], recall_by_bin=r["recall_by_bin"])
    return out


def experiment(workers: int = 3, progress=print, save_model: bool = True) -> dict:
    """Collect predictions, fit on the fusion-calibration set, score everything on EVALUATION."""
    bench = main_bench()
    cal, ev = bench.split("calibration"), bench.split("evaluation")
    fcal = fusion_calibration()
    all_cases = cal + ev + fcal
    preds, secs = _load_all(all_cases, APPROACHES, workers, progress)
    res: dict = {"fcal": {"cases": len(fcal), "perturbed": sum(c.kind == "perturbed" for c in fcal),
                          "gt": sum(len(c.gt) for c in fcal), "seeds": list(FCAL_SEEDS)}, "reports": {}, "fits": {}}

    def total_secs(cid: str, aps=APPROACHES) -> float:
        return float(sum(secs[a].get(cid, 0.0) for a in aps))

    rep = res["reports"]
    from dt.adversary.benchmark import baseline_residual, null_adversary, score
    rep["null/evaluation"] = _summ(score(null_adversary, ev, "evaluation"))
    rep["baseline_residual/evaluation"] = _summ(score(baseline_residual, ev, "evaluation"))
    for a in APPROACHES:
        for split, cs in (("evaluation", ev), ("fcal", fcal), ("calibration", cal)):
            rep[f"{a}/{split}"] = _summ(score_preds(preds[a], cs, a, split, secs[a]))
    # honest fit on fusion-calibration
    model = fit(preds, fcal)
    emit, hist = tune_emit(model, preds, fcal)
    res["fits"]["fuse_fcal"] = {"emit_min": emit, "emit_grid": hist, "meta": model.meta, "order": model.order}
    esec = {uid(c): total_secs(uid(c)) for c in ev}
    rep["fuse/evaluation"] = _summ(score_preds(fuse_preds(model, preds, ev), ev, "ensemble (fuse, fit on fusion-calibration)", "evaluation", esec))
    rep["fuse/fcal"] = _summ(score_preds(fuse_preds(model, preds, fcal), fcal, "ensemble (fuse) in-sample", "fcal"))
    router = fit_router(preds, fcal)
    res["fits"]["route_fcal"] = router.to_dict()
    rep["route/evaluation"] = _summ(score_preds(route_preds(router, preds, ev), ev, "ensemble (route, fit on fusion-calibration)", "evaluation", esec))
    # fit on the tuned calibration split (every approach was tuned on it: over-trusting)
    m_cal = fit(preds, cal)
    tune_emit(m_cal, preds, cal)
    rep["fuse_fit_on_tuned_cal/evaluation"] = _summ(score_preds(fuse_preds(m_cal, preds, ev), ev, "fuse fit on seed-0 calibration (tuned on)", "evaluation"))
    # oracle: fit on evaluation itself (optimistic bound, never shipped)
    m_or = fit(preds, ev)
    tune_emit(m_or, preds, ev)
    rep["fuse_oracle/evaluation"] = _summ(score_preds(fuse_preds(m_or, preds, ev), ev, "fuse ORACLE (fit on evaluation)", "evaluation"))
    r_or = fit_router(preds, ev)
    res["fits"]["route_oracle"] = r_or.to_dict()
    rep["route_oracle/evaluation"] = _summ(score_preds(route_preds(r_or, preds, ev), ev, "route ORACLE (fit on evaluation)", "evaluation"))
    # without the expensive approaches (renderer-free / cheap subsets)
    for sub in (("decompose", "irspace", "metamorphic"), ("acontrario", "decompose"), ("decompose",)):
        m = fit({a: preds[a] for a in sub}, fcal, sub)
        tune_emit(m, preds, fcal)
        rep[f"fuse[{'+'.join(sub)}]/evaluation"] = _summ(score_preds(fuse_preds(m, preds, ev), ev, f"fuse[{'+'.join(sub)}]", "evaluation",
                                                                     {uid(c): total_secs(uid(c), sub) for c in ev}))
    # leave-one-family-out inside the fusion-calibration set (fit on one family, test on the other)
    lofo = {}
    for fam in sorted({c.family for c in fcal}):
        tr = [c for c in fcal if c.family != fam]
        te_fc = [c for c in fcal if c.family == fam]
        te_ev = [c for c in ev if c.family == fam]
        m = fit(preds, tr)
        tune_emit(m, preds, tr)
        full = fit(preds, fcal)
        full.emit_min = model.emit_min
        lofo[fam] = {"train_cases": len(tr),
                     "heldout_fcal": score_preds(fuse_preds(m, preds, te_fc), te_fc, "lofo")["headline"],
                     "heldout_fcal_full_fit": score_preds(fuse_preds(full, preds, te_fc), te_fc, "in")["headline"],
                     "eval_family": score_preds(fuse_preds(m, preds, te_ev), te_ev, "lofo")["headline"] if te_ev else None,
                     "eval_family_full_fit": score_preds(fuse_preds(full, preds, te_ev), te_ev, "in")["headline"] if te_ev else None,
                     "best_single_eval_family": max((score_preds(preds[a], te_ev, a)["headline"], a) for a in APPROACHES) if te_ev else None}
    res["lofo"] = lofo
    # seed stability: fit on one fresh seed, test on the others
    seeds = {}
    for s in FCAL_SEEDS:
        tr = [c for c in fcal if c.meta.get("fcal_seed") == s]
        te = [c for c in fcal if c.meta.get("fcal_seed") != s]
        m = fit(preds, tr)
        tune_emit(m, preds, tr)
        seeds[str(s)] = {"in_seed": score_preds(fuse_preds(m, preds, tr), tr, "x")["headline"],
                         "other_seeds": score_preds(fuse_preds(m, preds, te), te, "x")["headline"],
                         "evaluation": score_preds(fuse_preds(m, preds, ev), ev, "x")["headline"], "emit_min": m.emit_min}
    res["seed_stability"] = seeds
    res["model"] = {"approaches": list(model.approaches), "prior": model.prior, "fire": model.fire, "order": model.order,
                    "emit_min": model.emit_min}
    # evidence: the evaluation cases where the fused and the best single approach disagree most
    fused_ev = fuse_preds(model, preds, ev)
    from dt.adversary.benchmark import match as _match
    gain = []
    for c in ev:
        if c.kind != "perturbed":
            continue
        ff = [f for f in fused_ev[uid(c)] if not is_noise(f.type)]
        fa = [f for f in preds["acontrario"][uid(c)] if not is_noise(f.type)]
        tp_f = sum(ff[i].type == c.gt[j].type for i, j in _match(ff, c.gt, c.page))
        tp_a = sum(fa[i].type == c.gt[j].type for i, j in _match(fa, c.gt, c.page))
        gain.append((tp_f - tp_a, c.id, c))
    gain.sort(key=lambda t: (-t[0], t[1]))
    picks = [g[2] for g in gain[:3]] + [g[2] for g in gain[-1:]]
    res["examples"] = [os.path.relpath(p, ROOT) for p in save_panels(
        picks, {"fused": fused_ev, **{a: preds[a] for a in APPROACHES}}, os.path.join(OUT, "examples"))]
    if save_model:
        model.meta.update(fit_on="fusion-calibration seeds " + ",".join(map(str, FCAL_SEEDS)), evaluation_headline=rep["fuse/evaluation"]["headline"])
        model.save()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ensemble.json"), "w") as f:
        json.dump(_jsonable(res), f, indent=1, default=str)
    return res


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Collect approach predictions, fit the fusion ensemble, score on EVALUATION.")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--approach", action="append", help="collect only these approaches")
    a = ap.parse_args(argv)
    if a.collect_only:
        bench = main_bench()
        cases = bench.split("calibration") + bench.split("evaluation") + fusion_calibration()
        for name in (a.approach or APPROACHES):
            predictions(name, cases, workers=a.workers, progress=print)
        return 0
    res = experiment(a.workers)
    for k, r in res["reports"].items():
        print(f"{k:45s} {r['headline']:.3f}  detF1 {r['det']['f1']:.3f}  leafF1 {r['macro_leaf']['f1']:.3f}  noiseFP {r['noise']['false_case_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
