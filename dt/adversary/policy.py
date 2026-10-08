"""Finding type -> refine operator policy, learned by replaying the real refine operators.

The critic (``dt.refine.critic``) proposes ~10 kinds of hypotheses for the worst nodes and the
optimizer tries them in expected-gain order, rendering every batch. A typed finding says *what* is
wrong, so it should say *which operator* to try first and which are pointless. This module learns
that mapping from evidence instead of hard-coding it:

1. **Replay** (``replay_case``): on a benchmark case (target = clean render + nuisance, candidate IR
   = the perturbed document, every error known), the critic runs once on the candidate. For every
   ground-truth finding and every operator kind, the critic's own hypotheses of that kind that touch
   the finding (region overlap or the finding's node) are applied and rendered in the real renderer.
   The operator *fixes* the finding when the total loss drops and the residual ΔE mass inside the
   finding's box drops by at least ``fix_frac``. Records: proposed?, fixed?, renders used, loss gain.
2. **Estimate** (``Policy.fit``): ``P(proposed | type, kind)`` and ``P(fix | proposed, type,
   magnitude tercile, kind)`` with Beta priors centred on the taxonomy's own ``refine_kinds`` (the
   prior says "the operators the taxonomy lists probably work"; data moves it) and back-off
   (type x magnitude -> type -> prior), plus the mean relative loss gain of a fix and renders per try.
3. **Suggest** (``Policy.suggest(findings, doc)``): ordered ``(kind, target node, box, value)``
   hypotheses for the critic: value = confidence x P(proposed) x P(fix) x (gain + floor) / renders.
4. **Guided critic** (``guided``): a context manager that wraps ``dt.refine.optimizer.critique``
   (no file in ``dt/refine`` is edited): it reorders the critic's hypotheses by the policy (matched
   suggestions first), demotes or prunes operators the policy says do not fix the overlapping
   finding type, and keeps a bounded number of hypotheses that overlap no finding (the adversary can
   miss). Findings come from a cheap, renderer-free adversary (``decompose``) every iteration.
5. **Measure** (``compare_refine``): renders-to-converge, final loss and loss recovered by the
   current critic vs the guided critic on held-out cases (EVALUATION split).

CLI: ``python -m dt.adversary.policy learn|evaluate [--workers N] [--limit N]``.
"""
from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from dt.adversary.taxonomy import CRITIC_KINDS, ERROR_TYPES, TYPES, Finding, is_noise
from dt.ir import Box, Document, Node
from dt.params import P, register

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "out", "adversary", "synthesis", "policy")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "policy_model.json")

_K = "adversary.policy."
register(_K + "fix_frac", 0.5, "an operator fixes a finding when the residual ΔE mass in the finding's box drops by >= this fraction (and the total loss drops)", (0.1, 0.95))
register(_K + "box_pad", 4, "px around a finding box when measuring its residual mass and matching hypotheses", (0, 16))
register(_K + "top_per_kind", 2, "replay: try at most this many of the critic's hypotheses per (finding, operator kind)", (1, 8))
register(_K + "prior_strength", 4.0, "Beta prior strength of P(fix | type, kind)", (0.5, 50.0))
register(_K + "prior_listed", 0.35, "prior P(fix) of an operator the taxonomy lists in TypeSpec.refine_kinds for the type", (0.01, 0.99))
register(_K + "prior_other", 0.03, "prior P(fix) of any other operator", (0.0, 0.5))
register(_K + "gain_floor", 0.002, "relative loss gain added to every suggestion's value (keeps zero-gain-history fixes orderable)", (0.0, 0.1))
register(_K + "min_p", 0.06, "guided critic: an operator whose P(proposed) x P(fix) for every finding it overlaps is below this is pruned", (0.0, 0.5))
register(_K + "background_keep", 12, "guided critic: keep at most this many hypotheses that overlap no finding, per iteration", (0, 64))
register(_K + "background_weight", 0.25, "guided critic: background hypotheses sort after policy-matched ones with this weight on their expected gain", (0.0, 1.0))


# --------------------------------------------------------------------------- helpers
def _clip(b: Box, W: int, H: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = b.as_int()
    return max(0, x0), max(0, y0), min(W, x1), min(H, y1)


def _mass(d: np.ndarray, b: Box) -> float:
    H, W = d.shape[:2]
    x0, y0, x1, y1 = _clip(b, W, H)
    return float(d[y0:y1, x0:x1].sum()) if x1 > x0 and y1 > y0 else 0.0


def _overlaps(a: Box, b: Box) -> bool:
    return a.intersect(b).area > 0


def _flat_hyps(hyps: list) -> list:
    out = []
    for h in hyps:
        out.append(h)
        out.extend(getattr(h, "alternatives", []) or [])
    return out


# --------------------------------------------------------------------------- replay
def replay_case(case, lines: Optional[list[str]] = None) -> list[dict]:
    """Per (gt finding, operator kind): did the critic propose it, and did its best hypothesis fix it?"""
    from dt.compare.pixel import diff_map
    from dt.refine.critic import critique
    from dt.refine.optimizer import loss_of, target_text_lines
    from dt.render.screenshot import render_doc
    doc = case.candidate_ir
    target = np.ascontiguousarray(case.target_rgb[..., :3])
    rendered = render_doc(doc)
    if lines is None:
        lines = target_text_lines(target)
    loss0, rep = loss_of(doc, target, rendered, lines)
    d0 = rep.diff_map if rep.diff_map is not None else diff_map(target, rendered)
    t0 = time.perf_counter()
    hyps = critique(doc, target, rendered, rep)
    crit_s = time.perf_counter() - t0
    order = {id(h): i for i, h in enumerate(hyps)}
    flat = _flat_hyps(hyps)
    pad = float(P[_K + "box_pad"])
    top = int(P[_K + "top_per_kind"])
    tried: dict[int, tuple[float, np.ndarray]] = {}
    recs = []
    for gi, g in enumerate(case.gt):
        gbox = g.box.expand(pad)
        m0 = _mass(d0, gbox)
        by_kind: dict[str, list] = defaultdict(list)
        for h in flat:
            if (h.node_id is not None and h.node_id == g.node_id) or _overlaps(h.region, gbox):
                by_kind[h.kind].append(h)
        for kind in CRITIC_KINDS:
            hs = sorted(by_kind.get(kind, []), key=lambda h: -float(h.expected_gain))[:top]
            rec = {"case": case.id, "uid": f"{_bench_key(case)}:{case.id}", "family": case.family, "gt": gi, "type": g.type, "magnitude": g.magnitude,
                   "bin": g.evidence.get("bin"), "kind": kind, "proposed": bool(hs), "tries": len(hs), "fixed": False,
                   "fix_frac": 0.0, "gain": 0.0, "rank": None, "loss0": loss0}
            for h in hs:
                if id(h) not in tried:
                    cand = h.apply(doc)
                    r = render_doc(cand)
                    L, rep2 = loss_of(cand, target, r, lines)
                    tried[id(h)] = (L, rep2.diff_map if rep2.diff_map is not None else diff_map(target, r))
                L, d1 = tried[id(h)]
                ff = 1.0 - _mass(d1, gbox) / m0 if m0 > 0 else 0.0
                gain = (loss0 - L) / loss0 if loss0 > 0 else 0.0
                fixed = ff >= float(P[_K + "fix_frac"]) and L < loss0 - float(P["refine.opt.eps"])
                if rec["rank"] is None or (fixed, gain) > (rec["fixed"], rec["gain"]):
                    rec.update(fixed=bool(fixed), fix_frac=float(ff), gain=float(gain), rank=order.get(id(h)))
            recs.append(rec)
    for r in recs:
        r["critique_s"] = crit_s
        r["n_hyps"] = len(hyps)
    return recs


def _replay_job(args: tuple) -> list[dict]:
    tp, cp, irp, meta = args
    from dt.adversary.benchmark import Case
    c = Case(meta["id"], meta["split"], meta["family"], meta["doc"], meta["kind"], [Finding.from_dict(g) for g in meta["gt"]],
             meta["noise"], meta["meta"])
    c.paths = {"target": tp, "candidate": cp, "ir": irp}
    try:
        return replay_case(c)
    except Exception as e:  # one broken case must not stop the replay
        return [{"case": c.id, "uid": f"{_bench_key(c)}:{c.id}", "error": repr(e)}]


def _case_job(c) -> tuple:
    return (c.paths["target"], c.paths["candidate"], c.paths["ir"], c.record())


def replay(cases: list, workers: int = 1, cache: Optional[str] = None, progress=None) -> list[dict]:
    """Replay every perturbed case (cached per case under ``cache``)."""
    cases = [c for c in cases if c.kind == "perturbed"]
    out: list[dict] = []
    todo = []
    for c in cases:
        p = os.path.join(cache, f"{_bench_key(c)}_{c.id}.json") if cache else None
        if p and os.path.exists(p):
            got = json.load(open(p))
            for r in got:
                r.setdefault("uid", f"{_bench_key(c)}:{c.id}")
            out += got
        else:
            todo.append((c, p))
    jobs = [_case_job(c) for c, _ in todo]
    paths = {f"{_bench_key(c)}:{c.id}": p for c, p in todo}
    if len(paths) != len(todo):
        raise ValueError("duplicate case uids")

    def _store(recs: list[dict]) -> None:
        if not recs:
            return
        cid = recs[0].get("uid") or recs[0]["case"]
        if paths.get(cid):
            os.makedirs(os.path.dirname(paths[cid]), exist_ok=True)
            with open(paths[cid], "w") as f:
                json.dump(recs, f)
        out.extend(recs)
        if progress:
            progress(f"replay {cid}: {sum(r.get('fixed', False) for r in recs)} fixes / {len(recs)} records")

    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            for recs in pool.imap_unordered(_replay_job, jobs):
                _store(recs)
    else:
        for j in jobs:
            _store(_replay_job(j))
    return out


def _bench_key(c) -> str:
    p = (c.paths or {}).get("target")
    return os.path.basename(os.path.dirname(p)) if p else "memory"


# --------------------------------------------------------------------------- the policy
def _beta(k: float, n: float, prior: float, strength: float) -> float:
    return (k + strength * prior) / (n + strength)


@dataclass
class Policy:
    """Tables learned by :func:`Policy.fit`; ``p_fix`` / ``p_proposed`` / ``gain`` back off to priors."""
    counts: dict = field(default_factory=dict)       # type -> kind -> {"n", "proposed", "fixed", "gain_sum", "tries"}
    counts_mag: dict = field(default_factory=dict)   # type -> tercile -> kind -> {...}
    mag_edges: dict = field(default_factory=dict)    # type -> [e1, e2]
    meta: dict = field(default_factory=dict)
    confusion: dict = field(default_factory=dict)    # finder's predicted type -> {true type: P(true | predicted)} ("_none" = spurious)

    # ---- estimates
    @staticmethod
    def prior_fix(type_: str, kind: str) -> float:
        spec = TYPES.get(type_)
        listed = spec is not None and kind in spec.refine_kinds
        return float(P[_K + "prior_listed"] if listed else P[_K + "prior_other"])

    def tercile(self, type_: str, magnitude: Optional[float]) -> Optional[str]:
        e = self.mag_edges.get(type_)
        if magnitude is None or not e:
            return None
        return "lo" if magnitude <= e[0] else ("mid" if magnitude <= e[1] else "hi")

    def p_proposed(self, type_: str, kind: str) -> float:
        c = self.counts.get(type_, {}).get(kind)
        prior = 0.7 if self.prior_fix(type_, kind) >= float(P[_K + "prior_listed"]) else 0.3
        if not c:
            return prior
        return _beta(c["proposed"], c["n"], prior, 2.0)

    def p_fix(self, type_: str, kind: str, magnitude: Optional[float] = None) -> float:
        """P(the operator's best hypothesis fixes the finding | it proposed one, type[, magnitude])."""
        s = float(P[_K + "prior_strength"])
        prior = self.prior_fix(type_, kind)
        c = self.counts.get(type_, {}).get(kind)
        p_type = _beta(c["fixed"], c["proposed"], prior, s) if c else prior
        t = self.tercile(type_, magnitude)
        cm = self.counts_mag.get(type_, {}).get(t or "", {}).get(kind) if t else None
        return _beta(cm["fixed"], cm["proposed"], p_type, s) if cm else p_type

    def gain(self, type_: str, kind: str) -> float:
        c = self.counts.get(type_, {}).get(kind)
        g = (c["gain_sum"] / c["fixed"]) if c and c["fixed"] else 0.0
        return max(0.0, g) + float(P[_K + "gain_floor"])

    def renders_per_try(self, type_: str, kind: str) -> float:
        c = self.counts.get(type_, {}).get(kind)
        return max(1.0, c["tries"] / c["proposed"]) if c and c["proposed"] else 1.0

    def value(self, type_: str, kind: str, magnitude: Optional[float] = None, confidence: float = 1.0) -> float:
        return confidence * self.p_proposed(type_, kind) * self.p_fix(type_, kind, magnitude) * self.gain(type_, kind) / self.renders_per_try(type_, kind)

    # ---- typed findings are noisy: marginalise over the finder's confusion
    def true_types(self, predicted: str) -> dict[str, float]:
        """P(true type | the finder predicted ``predicted``) (identity when no confusion was fitted)."""
        row = self.confusion.get(predicted)
        return dict(row) if row else {predicted: 1.0}

    def p_fix_finding(self, f: Finding, kind: str) -> float:
        """P(operator ``kind`` proposes a fixing hypothesis | a finding typed ``f.type``), summed over the
        true types the finder confuses it with (a spurious finding is fixed by nothing)."""
        out = 0.0
        for y, w in self.true_types(f.type).items():
            if y not in TYPES:
                continue
            out += w * self.p_proposed(y, kind) * self.p_fix(y, kind, f.magnitude if y == f.type else None)
        return out

    def value_finding(self, f: Finding, kind: str) -> float:
        out = 0.0
        for y, w in self.true_types(f.type).items():
            if y not in TYPES:
                continue
            out += w * self.value(y, kind, f.magnitude if y == f.type else None)
        return float(f.confidence) * out

    @staticmethod
    def fit_confusion(rows: list[dict], strength: float = 2.0) -> dict:
        """Finder confusion from benchmark rows (``dt.adversary.benchmark.run`` format): for each predicted
        type, the distribution of the matched ground-truth types and of spurious predictions (``_none``),
        Dirichlet-smoothed towards "the prediction is right"."""
        cnt: dict = defaultdict(lambda: defaultdict(float))
        for r in rows:
            gt = r["case"].gt
            matched = {i: j for i, j in r["pairs"]}
            for i, p in enumerate(r["preds"]):
                cnt[p.type][gt[matched[i]].type if i in matched else "_none"] += 1
        out = {}
        for t in ERROR_TYPES:
            row = dict(cnt.get(t, {}))
            n = sum(row.values())
            row[t] = row.get(t, 0.0) + strength
            out[t] = {k: v / (n + strength) for k, v in row.items()}
        return out

    def ranking(self, type_: str, magnitude: Optional[float] = None) -> list[tuple[str, float, float]]:
        """Operators for a finding type, best first: ``(kind, P(fix), value)``."""
        rows = [(k, self.p_fix(type_, k, magnitude), self.value(type_, k, magnitude)) for k in CRITIC_KINDS]
        return sorted(rows, key=lambda r: -r[2])

    # ---- fit
    @staticmethod
    def fit(records: list[dict]) -> "Policy":
        recs = [r for r in records if "error" not in r]
        counts: dict = defaultdict(lambda: defaultdict(lambda: {"n": 0, "proposed": 0, "fixed": 0, "gain_sum": 0.0, "tries": 0}))
        counts_mag: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"n": 0, "proposed": 0, "fixed": 0, "gain_sum": 0.0, "tries": 0})))
        mags: dict = defaultdict(list)
        for r in recs:
            if r["magnitude"] is not None and r["kind"] == CRITIC_KINDS[0]:
                mags[r["type"]].append(float(r["magnitude"]))
        edges = {t: [float(np.quantile(v, 1 / 3)), float(np.quantile(v, 2 / 3))] for t, v in mags.items() if len(v) >= 6}
        pol = Policy(mag_edges=edges)
        for r in recs:
            for tab in (counts[r["type"]][r["kind"]],):
                tab["n"] += 1
                tab["proposed"] += int(r["proposed"])
                tab["fixed"] += int(r["fixed"])
                tab["gain_sum"] += float(r["gain"]) if r["fixed"] else 0.0
                tab["tries"] += int(r["tries"])
            t = pol.tercile(r["type"], r["magnitude"])
            if t:
                tab = counts_mag[r["type"]][t][r["kind"]]
                tab["n"] += 1
                tab["proposed"] += int(r["proposed"])
                tab["fixed"] += int(r["fixed"])
                tab["gain_sum"] += float(r["gain"]) if r["fixed"] else 0.0
                tab["tries"] += int(r["tries"])
        pol.counts = json.loads(json.dumps(counts))
        pol.counts_mag = json.loads(json.dumps(counts_mag))
        key = lambda r: (r.get("uid", r["case"]), r["gt"])  # noqa: E731
        gts = {key(r) for r in recs}
        fixed_any = {key(r) for r in recs if r["fixed"]}
        pol.meta = {"records": len(recs), "findings": len(gts), "fixable": len(fixed_any), "cases": len({k[0] for k in gts})}
        return pol

    # ---- suggest
    def suggest(self, findings: list[Finding], doc: Optional[Document] = None, min_p: Optional[float] = None) -> list[dict]:
        """Ordered operator hypotheses for the critic: ``[{kind, node_id, box, finding_type, p_fix, value}]``.

        The target node is the finding's ``node_id`` when the document has it, else the smallest node
        whose box best covers the finding (``None`` for ``missing``: the operator inserts)."""
        thr = float(P[_K + "min_p"] if min_p is None else min_p)
        ids = {n.id: n for n in doc.walk()} if doc is not None else {}
        out = []
        for i, f in enumerate(findings):
            if is_noise(f.type) or f.type not in TYPES:
                continue
            node = f.node_id if f.node_id in ids else _cover_node(doc, f.box) if doc is not None else f.node_id
            for kind in CRITIC_KINDS:
                pf = self.p_fix_finding(f, kind)
                if pf < thr:
                    continue
                out.append({"kind": kind, "node_id": None if kind == "missing" else node, "box": f.box, "finding": i,
                            "finding_type": f.type, "p_fix": pf, "value": self.value_finding(f, kind)})
        out.sort(key=lambda s: -s["value"])
        seen, uniq = set(), []
        for s in out:
            k = (s["kind"], s["node_id"], s["box"].as_int())
            if k not in seen:
                seen.add(k)
                uniq.append(s)
        for r, s in enumerate(uniq):
            s["rank"] = r
        return uniq

    # ---- persistence
    def to_dict(self) -> dict:
        return {"counts": self.counts, "counts_mag": self.counts_mag, "mag_edges": self.mag_edges, "meta": self.meta,
                "confusion": self.confusion}

    @staticmethod
    def from_dict(d: dict) -> "Policy":
        return Policy(d.get("counts", {}), d.get("counts_mag", {}), d.get("mag_edges", {}), d.get("meta", {}), d.get("confusion", {}))

    def save(self, path: str = MODEL_PATH) -> str:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1, sort_keys=True)
        return path

    @staticmethod
    def load(path: str = MODEL_PATH) -> "Policy":
        with open(path) as f:
            return Policy.from_dict(json.load(f))

    def table(self) -> dict[str, list[tuple[str, float, float, int]]]:
        """type -> operators by value: (kind, P(fix|proposed), P(proposed), n findings)."""
        out = {}
        for t in ERROR_TYPES:
            rows = []
            for k in CRITIC_KINDS:
                c = self.counts.get(t, {}).get(k, {})
                rows.append((k, self.p_fix(t, k), self.p_proposed(t, k), int(c.get("n", 0))))
            out[t] = sorted(rows, key=lambda r: -r[1] * r[2])
        return out


def _cover_node(doc: Document, box: Box) -> Optional[str]:
    best, best_s = None, 0.0
    for n in doc.walk():
        if n.id == doc.root.id or not n.visible or n.box.area <= 0:
            continue
        inter = n.box.intersect(box).area
        if inter <= 0:
            continue
        s = inter / max(n.box.area, box.area)  # IoU-like but prefers tight covers
        if s > best_s:
            best, best_s = n.id, s
    return best


_POLICY: Optional[Policy] = None


def load(path: str = MODEL_PATH) -> Policy:
    global _POLICY
    if _POLICY is None:
        _POLICY = Policy.load(path) if os.path.exists(path) else Policy()
    return _POLICY


def suggest(findings: list[Finding], doc: Optional[Document] = None) -> list[dict]:
    """Module-level convenience: the shipped policy's :meth:`Policy.suggest`."""
    return load().suggest(findings, doc)


# --------------------------------------------------------------------------- guided critic
def _default_finder():
    from dt.adversary.decompose import find
    return find


def reorder(hyps: list, findings: list[Finding], policy: Policy, doc: Document, prune: bool = True) -> tuple[list, dict]:
    """Reorder (and optionally prune) critic hypotheses by the policy. Returns ``(hyps, stats)``.

    * matched: same kind as a suggestion and on its node or box -> sorted by suggestion value;
    * overlapping a finding but an operator whose P(fix | finding) (marginalised over the finder's
      confusion) is below ``min_p`` -> pruned (or last when ``prune`` is False);
    * overlapping no finding -> background, at most ``background_keep``, after the matched ones."""
    sugg = policy.suggest(findings, doc, min_p=0.0)
    pad = float(P[_K + "box_pad"])
    thr = float(P[_K + "min_p"])
    boxes = [(f.box.expand(pad), f) for f in findings if not is_noise(f.type) and f.type in TYPES]
    matched, weak, background, pruned = [], [], [], 0
    for i, h in enumerate(hyps):
        best = None
        for s in sugg:
            if s["kind"] != h.kind:
                continue
            on = (h.node_id is not None and h.node_id == s["node_id"]) or _overlaps(h.region, s["box"].expand(pad))
            if on and (best is None or s["value"] > best["value"]):
                best = s
        over = [f for b, f in boxes if _overlaps(h.region, b) or (h.node_id and h.node_id == f.node_id)]
        if best is not None and best["p_fix"] >= thr:
            matched.append((best["value"], -i, h))
        elif over:
            p = max(policy.p_fix_finding(f, h.kind) for f in over)
            if prune and p < thr:
                pruned += 1
                continue
            weak.append((p * float(h.expected_gain), -i, h))
        else:
            background.append((float(h.expected_gain), -i, h))
    matched.sort(key=lambda t: (-t[0], -t[1]))
    weak.sort(key=lambda t: (-t[0], -t[1]))
    background.sort(key=lambda t: (-t[0], -t[1]))
    keep_bg = int(P[_K + "background_keep"]) if prune else len(background)
    out = [h for *_, h in matched] + [h for *_, h in weak] + [h for *_, h in background[:keep_bg]]
    return out, {"matched": len(matched), "weak": len(weak), "background": len(background), "pruned": pruned + max(0, len(background) - keep_bg),
                 "findings": len(boxes)}


@contextlib.contextmanager
def guided(policy: Optional[Policy] = None, finder: Optional[Callable] = None, prune: bool = True, log: Optional[list] = None):
    """Within the context, ``dt.refine.optimizer.refine`` uses the policy-guided critic.

    Safe pruning: when a guided iteration accepted nothing (the loss the critic sees did not move since
    the previous guided call), the next iteration gets the critic's full, unpruned list in its own order,
    so a mistyped finding can delay a fix by one iteration but never lose it."""
    import dt.refine.optimizer as O
    pol = policy or load()
    fnd = finder or _default_finder()
    orig = O.critique
    state = {"last": None}

    def critique(doc, target_rgb, rendered_rgb, report):
        hyps = orig(doc, target_rgb, rendered_rgb, report)
        loss = float(report.total)
        if prune and state["last"] is not None and abs(loss - state["last"]) <= 1e-12:
            state["last"] = None  # stalled: one full iteration, then guide again
            if log is not None:
                log.append({"fallback": True, "proposed": len(hyps), "kept": len(hyps)})
            return hyps
        try:
            fs = fnd(target_rgb, rendered_rgb, doc)
        except Exception:
            fs = []
        out, st = reorder(hyps, fs, pol, doc, prune)
        if not out:  # never starve the optimizer: an empty list would end refine ("no_hypotheses")
            out, st = hyps, dict(st, fallback=True)
        state["last"] = loss
        if log is not None:
            log.append(dict(st, proposed=len(hyps), kept=len(out)))
        return out

    O.critique = critique
    try:
        yield
    finally:
        O.critique = orig


@contextlib.contextmanager
def _trace(trace: list):
    """Record ``(renders, loss)`` at every accepted state of the optimizer (anytime curve)."""
    import dt.refine.optimizer as O
    base = O._State

    class Traced(base):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            trace.append((self.n_renders, float(self.loss)))

        def accept(self, cand, rendered, loss, report):
            super().accept(cand, rendered, loss, report)
            trace.append((self.n_renders, float(loss)))

    O._State = Traced
    try:
        yield
    finally:
        O._State = base


# --------------------------------------------------------------------------- held-out comparison
def _refine_run(case, mode: str, policy: Optional[Policy], clean_loss: Optional[float]) -> dict:
    from dt.refine.optimizer import refine
    trace: list = []
    glog: list = []
    t0 = time.perf_counter()
    target = np.ascontiguousarray(case.target_rgb[..., :3])
    with _trace(trace):
        if mode == "default":
            doc, hist = refine(case.candidate_ir, target)
        else:
            with guided(policy, prune=(mode == "guided_prune"), log=glog):
                doc, hist = refine(case.candidate_ir, target)
    stop = hist[-1]
    return {"case": case.id, "family": case.family, "mode": mode, "renders": int(stop["params"]["renders"]),
            "start": float(stop["before"]), "final": float(stop["after"]), "clean": clean_loss, "reason": stop["params"]["reason"],
            "accepted": sum(1 for h in hist if h.get("accepted") and h.get("kind") != "stop"),
            "accepted_kinds": [h["kind"] for h in hist if h.get("accepted") and h.get("kind") != "stop"],
            "trace": trace, "seconds": time.perf_counter() - t0, "guide": glog}


def _clean_loss(case) -> Optional[float]:
    """Loss of the clean (unperturbed) document against the target: the floor refine could reach."""
    try:
        from dt.adversary.benchmark import load_doc
        from dt.adversary.perturb import sanitize
        from dt.refine.optimizer import loss_of, target_text_lines
        doc, _ = sanitize(load_doc(case.doc))
        t = np.ascontiguousarray(case.target_rgb[..., :3])
        return float(loss_of(doc, t, None, target_text_lines(t))[0])
    except Exception:
        return None


def _compare_job(args: tuple) -> list[dict]:
    tp, cp, irp, meta, modes, pol_d = args
    from dt.adversary.benchmark import Case
    c = Case(meta["id"], meta["split"], meta["family"], meta["doc"], meta["kind"], [Finding.from_dict(g) for g in meta["gt"]],
             meta["noise"], meta["meta"])
    c.paths = {"target": tp, "candidate": cp, "ir": irp}
    pol = Policy.from_dict(pol_d)
    clean = _clean_loss(c)
    out = []
    for m in modes:
        try:
            r = _refine_run(c, m, pol, clean)
        except Exception as e:
            r = {"case": c.id, "mode": m, "error": repr(e)}
        r["uid"] = f"{_bench_key(c)}:{c.id}"
        out.append(r)
    return out


def policy_hash(policy: Policy) -> str:
    import hashlib
    blob = json.dumps(policy.to_dict(), sort_keys=True, default=str) + json.dumps(
        {k: v for k, v in sorted(P.all().items()) if k.startswith(_K)}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def compare_refine(cases: list, policy: Policy, modes: Iterable[str] = ("default", "guided", "guided_prune"), workers: int = 1,
                   cache: Optional[str] = None, progress=None) -> list[dict]:
    """Run refine with the current critic and with the guided critic on each perturbed case. Cached per
    (case, mode); guided runs are keyed by the policy hash, default runs are shared."""
    cases = [c for c in cases if c.kind == "perturbed"]
    modes = tuple(modes)
    ph = policy_hash(policy)

    def path(c, m: str) -> Optional[str]:
        if not cache:
            return None
        sub = "default" if m == "default" else f"{m}_{ph}"
        return os.path.join(cache, sub, f"{_bench_key(c)}_{c.id}.json")

    out, jobs = [], []
    for c in cases:
        need = []
        for m in modes:
            p = path(c, m)
            if p and os.path.exists(p):
                out.append(json.load(open(p)))
            else:
                need.append(m)
        if need:
            jobs.append((*_case_job(c), tuple(need), policy.to_dict()))
    by_id = {f"{_bench_key(c)}:{c.id}": c for c in cases}

    def _store(rs: list[dict]) -> None:
        for r in rs:
            p = path(by_id[r["uid"]], r["mode"])
            if p and "error" not in r:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w") as f:
                    json.dump(r, f)
            out.append(r)
        if progress and rs:
            progress(f"refine {rs[0]['case']}: " + ", ".join(f"{r['mode']} {r.get('renders')}r {r.get('final', float('nan')):.4f}" for r in rs))

    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            for rs in pool.imap_unordered(_compare_job, jobs):
                _store(rs)
    else:
        for j in jobs:
            _store(_compare_job(j))
    return out


def renders_to_reach(trace: list, level: float) -> Optional[int]:
    """First render count at which the accepted loss is <= ``level``."""
    for n, L in trace:
        if L <= level + 1e-12:
            return int(n)
    return None


def summarise(runs: list[dict]) -> dict:
    """Paired comparison per mode vs ``default``: renders, final loss, recovered fraction, anytime."""
    by = defaultdict(dict)
    for r in runs:
        if "error" not in r:
            by[r.get("uid", r["case"])][r["mode"]] = r
    modes = sorted({m for d in by.values() for m in d})
    out: dict = {"cases": len(by), "modes": {}}
    for m in modes:
        rows = [d for d in by.values() if m in d and "default" in d]
        if not rows:
            continue
        ren = [d[m]["renders"] for d in rows]
        ren0 = [d["default"]["renders"] for d in rows]
        fin = [d[m]["final"] for d in rows]
        fin0 = [d["default"]["final"] for d in rows]

        def rec(r):
            c = r["clean"] if r.get("clean") is not None else 0.0
            span = r["start"] - c
            return (r["start"] - r["final"]) / span if span > 1e-9 else 1.0
        rc = [rec(d[m]) for d in rows]
        # renders the mode needs to match the default's final loss (anytime efficiency)
        reach = []
        for d in rows:
            n = renders_to_reach(d[m]["trace"], d["default"]["final"])
            reach.append(n)
        reached = [n for n in reach if n is not None]
        out["modes"][m] = {
            "n": len(rows), "renders_mean": float(np.mean(ren)), "renders_median": float(np.median(ren)),
            "renders_ratio_vs_default": float(np.sum(ren) / max(1, np.sum(ren0))),
            "final_loss_mean": float(np.mean(fin)), "final_loss_ratio_vs_default": float(np.sum(fin) / max(1e-12, np.sum(fin0))),
            "recovered_mean": float(np.mean(rc)),
            "worse_than_default": int(sum(f > f0 + 1e-6 for f, f0 in zip(fin, fin0))),
            "better_than_default": int(sum(f < f0 - 1e-6 for f, f0 in zip(fin, fin0))),
            "reach_default_final": len(reached), "renders_to_default_final_mean": float(np.mean(reached)) if reached else None,
            "seconds_mean": float(np.mean([d[m]["seconds"] for d in rows])),
        }
    return out


def per_type_value(policy: Policy) -> dict:
    """The learned type -> operator ordering, for the report."""
    out = {}
    for t, rows in policy.table().items():
        out[t] = [{"kind": k, "p_fix": round(pf, 3), "p_proposed": round(pp, 3), "n": n} for k, pf, pp, n in rows[:4]]
    return out


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Learn the finding-type -> operator policy and compare guided vs default refine.")
    ap.add_argument("cmd", choices=["learn", "evaluate", "both"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    from dt.adversary import ensemble as E
    if a.cmd in ("learn", "both"):
        fcal = E.fusion_calibration()
        cases = [c for c in fcal if c.kind == "perturbed"]
        if a.limit:
            cases = cases[: a.limit]
        recs = replay(cases, a.workers, os.path.join(OUT, "replay"), progress=print)
        pol = Policy.fit(recs)
        # the guided critic's finder (decompose) mistypes: learn its confusion on the same fresh seeds
        dp = E.predictions("decompose", fcal)
        pol.confusion = Policy.fit_confusion(E.rows_from({k: v["findings"] for k, v in dp.items()}, fcal))
        pol.meta["fit_on"] = "fusion-calibration seeds " + ",".join(map(str, E.FCAL_SEEDS))
        pol.meta["finder"] = "decompose"
        pol.save()
        with open(os.path.join(OUT, "policy_table.json"), "w") as f:
            json.dump({"meta": pol.meta, "table": per_type_value(pol)}, f, indent=1)
        print(json.dumps(pol.meta))
    if a.cmd in ("evaluate", "both"):
        pol = Policy.load()
        ev = [c for c in E.main_bench().split("evaluation") if c.kind == "perturbed"]
        if a.limit:
            ev = ev[: a.limit]
        runs = compare_refine(ev, pol, workers=a.workers, cache=os.path.join(OUT, "refine_eval"), progress=print)
        s = summarise(runs)
        with open(os.path.join(OUT, "compare.json"), "w") as f:
            json.dump({"summary": s, "runs": runs}, f, indent=1, default=str)
        print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
