"""The anti-overfitting protocol of the recursive improvement loop.

Every proposed change to a learned component (a parameter set, a rule, a fitted model) is judged by
``check(change_fn)``, which returns a verdict with evidence and a ledger entry. The protocol, not the
proposer, controls which data the change may see:

1. **Bounded priors.** Every changed parameter must be registered, stay inside its registered range,
   move at most ``max_step`` of that range per change, and never touch the yardstick
   (``validate.*``, ``bench.*``, ``adversary.bench.*``). At most ``max_params`` parameters per change,
   and the required calibration gain grows with the number of parameters changed (an MDL-style price:
   a 5-parameter change must buy five times the improvement of a 1-parameter one).
2. **Calibration gain.** The change is fitted on the CALIBRATION split (``change_fn`` receives the
   calibration cases, nothing else) and must raise the benchmark headline there by ``min_gain`` per
   parameter, without raising the pure-noise false-case rate.
3. **Leave-one-family-out.** For every benchmark family in calibration, the change is re-fitted
   without that family and scored on it. No family may lose more than ``lofo_tol``. A change that only
   helps the family it was fitted on is a family-specific overfit.
4. **Fresh seeds.** The change (fitted on calibration) is scored on the fusion-calibration cases:
   the same calibration documents with perturbation/noise seeds nobody tuned on. The gain must keep
   its sign there (``seed_tol``).
5. **Metamorphic regulariser.** Exact relations the adversary must satisfy whatever its parameters:
   identity (``find(T, T)`` is silent), nuisance invariance (a JPEG / blur / sub-pixel nuisance on the
   target changes nothing), translation equivariance (padding both images moves every finding by the
   pad) and swap symmetry (``find(C, T)`` reports the same places as ``find(T, C)``). A change that
   wins the benchmark but adds metamorphic violations is rejected.
6. **Holdouts that are never tuned on.** EVALUATION is scored only after the decision and only when
   asked (``report_evaluation``); every look is counted in ``eval_looks.json`` and the verdict warns
   when the budget is spent (rotate the evaluation seed). Real pages are never passed to
   ``change_fn``; their finding counts are reported (``real_pages``) as a one-sided flood guard only.
7. **Ledger provenance.** The verdict carries a ``knowledge/ledger.jsonl``-schema entry (kind,
   scope, source ids, exact diff, before/after metrics, gates, accepted). ``record`` appends it to a
   draft ledger (``out/adversary/synthesis/ledger_draft.jsonl`` by default: the global ledger is the
   lead's to write).

``change_fn`` may be a ``Change``, a ``{param: value}`` dict, or a callable ``(train_cases) ->
context manager`` that activates the change (fitted on ``train_cases`` only).
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, Iterable, Optional, Union

import numpy as np

from dt.adversary.taxonomy import Finding, is_noise
from dt.ir import Box, Document
from dt.params import P, register

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "out", "adversary", "synthesis", "protocol")
LEDGER_DRAFT = os.path.join(ROOT, "out", "adversary", "synthesis", "ledger_draft.jsonl")
EVAL_LOOKS = os.path.join(ROOT, "out", "adversary", "synthesis", "eval_looks.json")

_K = "adversary.protocol."
register(_K + "min_gain", 0.003, "required calibration headline gain per changed parameter (MDL-style price)", (0.0, 0.05))
register(_K + "max_params", 5, "a change may touch at most this many parameters", (1, 50))
register(_K + "max_step", 0.25, "a parameter may move at most this fraction of its registered range per change", (0.01, 1.0))
register(_K + "lofo_tol", 0.01, "leave-one-family-out: no held-out family may lose more headline than this", (0.0, 0.1))
register(_K + "seed_tol", 0.005, "fresh seeds: the change may lose at most this much headline on unseen perturbation seeds", (0.0, 0.1))
register(_K + "noise_tol", 0.0, "the pure-noise false-case rate on calibration may rise by at most this", (0.0, 0.5))
register(_K + "meta_tol", 0.05, "metamorphic: violations may grow by at most this fraction of the baseline count (and at least 1)", (0.0, 1.0))
register(_K + "meta_cases", 12, "metamorphic relations run on this many calibration cases (one per document first)", (1, 200))
register(_K + "meta_pad", [3, 5], "translation-equivariance pad (dx, dy) in px, off the 4 px grid")
register(_K + "eval_budget", 5, "looks at EVALUATION per benchmark key before the verdict warns to rotate the seed", (1, 100))
register(_K + "flood_ratio", 3.0, "real-page guard: findings per page growing by this factor (and by flood_min) is a flood", (1.0, 20.0))
register(_K + "flood_min", 20, "real-page guard: minimum absolute growth of findings per page for a flood", (1, 1000))

FORBIDDEN_PREFIXES = ("validate.", "bench.", "adversary.bench.", "adversary.protocol.")


# --------------------------------------------------------------------------- changes
@dataclass
class Change:
    """A proposed change. ``make(train_cases)`` returns a context manager in which it is active."""
    name: str
    kind: str = "params"                      # ledger kind: params | rule
    params: dict = field(default_factory=dict)
    make: Optional[Callable[[list], ContextManager]] = None
    scope: str = "global"
    source: dict = field(default_factory=lambda: {"type": "scenario", "ids": []})
    description: str = ""

    def activate(self, train_cases: list) -> ContextManager:
        if self.make is not None:
            return self.make(train_cases)
        return override(self.params)


@contextlib.contextmanager
def override(params: dict):
    """Temporarily set ``dt.params`` values (restores the previous override state)."""
    import dt.params as PM  # the registry's runtime layer (looked up each time: reset() rebinds it)
    saved = {k: PM._OVERRIDES.get(k, _MISSING) for k in params}
    try:
        for k, v in params.items():
            P.set(k, v)
        yield
    finally:
        for k, v in saved.items():
            if v is _MISSING:
                PM._OVERRIDES.pop(k, None)
            else:
                PM._OVERRIDES[k] = v


_MISSING = object()


def as_change(change_fn: Union[Change, dict, Callable], name: Optional[str] = None) -> Change:
    if isinstance(change_fn, Change):
        return change_fn
    if isinstance(change_fn, dict):
        return Change(name or "params:" + ",".join(sorted(change_fn)), "params", dict(change_fn))
    if callable(change_fn):
        return Change(name or getattr(change_fn, "__name__", "change"), "rule", {}, change_fn)
    raise TypeError("change_fn must be a Change, a {param: value} dict or a callable(train_cases) -> context manager")


# --------------------------------------------------------------------------- checks
def bounded_priors(change: Change) -> dict:
    """Registered, in range, bounded step, not the yardstick, few parameters."""
    rng = P.ranges()
    known = P.defaults()
    problems = []
    steps = {}
    for k, new in change.params.items():
        if k.startswith(FORBIDDEN_PREFIXES):
            problems.append(f"{k}: the yardstick ({k.split('.')[0]}.*) is never tuned")
            continue
        if k not in known:
            problems.append(f"{k}: not a registered parameter")
            continue
        old = P[k]
        if k in rng and isinstance(new, (int, float)) and not isinstance(new, bool):
            lo, hi = rng[k]
            if not (lo <= float(new) <= hi):
                problems.append(f"{k}={new} outside its registered range [{lo}, {hi}]")
            if isinstance(old, (int, float)) and hi > lo:
                steps[k] = abs(float(new) - float(old)) / (hi - lo)
                if steps[k] > float(P[_K + "max_step"]) + 1e-12:
                    problems.append(f"{k}: step {steps[k]:.2f} of its range > max_step {P[_K + 'max_step']}")
        elif type(new) is not type(old) and not (isinstance(new, (int, float)) and isinstance(old, (int, float))):
            problems.append(f"{k}: type {type(new).__name__} != registered {type(old).__name__}")
    if len(change.params) > int(P[_K + "max_params"]):
        problems.append(f"{len(change.params)} parameters > max_params {P[_K + 'max_params']} (one mechanism per change)")
    return {"passed": not problems, "problems": problems, "steps": steps, "n_params": len(change.params)}


def _score(adversary: Callable, cases: list, name: str = "x") -> dict:
    from dt.adversary.benchmark import run, aggregate
    return aggregate(run(adversary, cases), name)


def _brief(r: dict) -> dict:
    return {"headline": round(r["headline"], 4), "det_f1": round(r["detection"]["f1"], 4), "macro_leaf_f1": round(r["macro_leaf"]["f1"], 4),
            "noise_false_case_rate": round(r["noise"]["false_case_rate"], 4), "cases": r["cases"]}


# ---- metamorphic relations on the adversary
def _pad(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    border = np.median(np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]), axis=0).astype(np.uint8)
    out = np.empty((img.shape[0] + dy, img.shape[1] + dx, 3), np.uint8)
    out[:] = border
    out[dy:, dx:] = img[..., :3]
    return out


def _shift_doc(doc: Optional[Document], dx: int, dy: int) -> Optional[Document]:
    if doc is None:
        return None
    import copy
    d = copy.deepcopy(doc)
    d.width, d.height = d.width + dx, d.height + dy
    for n in d.walk():
        if n is d.root:
            n.box = Box(0, 0, n.box.w + dx, n.box.h + dy)
        else:
            n.box = Box(n.box.x + dx, n.box.y + dy, n.box.w, n.box.h)
    return d


def _jpeg(img: np.ndarray, q: int = 92) -> np.ndarray:
    import cv2
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _subpixel(img: np.ndarray, dx: float = 0.25, dy: float = 0.0) -> np.ndarray:
    import cv2
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(np.ascontiguousarray(img[..., :3]), M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _errs(fs: Iterable) -> list[Finding]:
    return [f if isinstance(f, Finding) else Finding.from_dict(f) for f in fs if not is_noise((f.type if isinstance(f, Finding) else f["type"]))]


def _unmatched(a: list[Finding], b: list[Finding], page: Box, typed: bool) -> int:
    """Findings of ``a`` and ``b`` left over after a one-to-one localising match (optionally same type)."""
    from dt.adversary.benchmark import match
    if not a or not b:
        return len(a) + len(b)
    pairs = match(a, b, page)
    if typed:
        pairs = [(i, j) for i, j in pairs if a[i].type == b[j].type]
    return len(a) + len(b) - 2 * len(pairs)


SWAP = {"structure.missing": "structure.extra", "structure.extra": "structure.missing", "icon.missing": "structure.extra",
        "structure.split": "structure.merge", "structure.merge": "structure.split"}


def metamorphic_violations(adversary: Callable, cases: list, relations: Iterable[str] = ("identity", "nuisance", "translate", "swap")) -> dict:
    """Violations of exact relations of the adversary itself (no ground truth needed)."""
    relations = tuple(relations)
    counts = {r: 0 for r in relations}
    per_case = []
    dx, dy = (int(v) for v in P[_K + "meta_pad"])
    for c in cases:
        T, C, ir = c.target_rgb, c.candidate_rgb, c.candidate_ir
        page = Box(0, 0, T.shape[1], T.shape[0])
        base = _errs(adversary(T, C, ir))
        row = {"case": getattr(c, "id", "?")}
        if "identity" in relations:
            v = len(_errs(adversary(T, T.copy(), ir)))
            row["identity"] = v
        if "nuisance" in relations:
            v = 0
            for N in (lambda x: _jpeg(x, 92), lambda x: _subpixel(x, 0.25)):
                v += _unmatched(base, _errs(adversary(N(T), C, ir)), page, typed=True)
            row["nuisance"] = v
        if "translate" in relations:
            got = _errs(adversary(_pad(T, dx, dy), _pad(C, dx, dy), _shift_doc(ir, dx, dy)))
            back = [Finding(f.type, Box(f.box.x - dx, f.box.y - dy, f.box.w, f.box.h), f.magnitude, f.confidence, f.evidence, f.node_id)
                    for f in got]
            row["translate"] = _unmatched(base, back, page, typed=True)
        if "swap" in relations:
            fw = _errs(adversary(T, C, None))
            bw = _errs(adversary(C, T, None))
            row["swap"] = _unmatched(fw, bw, page, typed=False)
        for r in relations:
            counts[r] += int(row.get(r, 0))
        per_case.append(row)
    return {"total": int(sum(counts.values())), "by_relation": counts, "cases": len(cases), "per_case": per_case}


def meta_cases(cases: list, n: int) -> list:
    """One perturbed case per document first (diverse), then the rest, deterministic."""
    by_doc: dict = {}
    for c in cases:
        if c.kind == "perturbed":
            by_doc.setdefault(c.doc, []).append(c)
    first = [v[0] for _, v in sorted(by_doc.items())]
    rest = [c for _, v in sorted(by_doc.items()) for c in v[1:]]
    noise = [c for c in cases if c.kind == "noise"]
    return (first + noise[:2] + rest)[:n]


# --------------------------------------------------------------------------- evaluation budget
def _eval_look(key: str) -> int:
    d = {}
    if os.path.exists(EVAL_LOOKS):
        try:
            d = json.load(open(EVAL_LOOKS))
        except Exception:
            d = {}
    d[key] = int(d.get(key, 0)) + 1
    os.makedirs(os.path.dirname(EVAL_LOOKS), exist_ok=True)
    with open(EVAL_LOOKS, "w") as f:
        json.dump(d, f, indent=1)
    return d[key]


# --------------------------------------------------------------------------- ledger
def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def ledger_entry(change: Change, evidence: dict, accepted: bool) -> dict:
    """An entry in the shared ledger schema (``knowledge/ledger.jsonl``)."""
    diff = {"params": {k: [P.get(k), v] for k, v in change.params.items()},
            "layer": change.scope} if change.params else {"rule": change.name, "description": change.description}
    kind = change.kind if change.kind in ("params", "rule", "scenario_baseline", "feedback_promotion", "revert") else "rule"
    body = {"ts": _now(), "kind": kind, "scope": change.scope, "source": dict(change.source), "change": diff,
            "evidence": evidence, "accepted": bool(accepted), "reverts": None, "protocol": "dt.adversary.protocol/1"}
    try:  # the scenario harness's ledger module, when installed, validates and ids the entry
        from dt.learn.ledger import make_entry
        return make_entry(kind, change.scope, body["source"].get("type", "scenario"), body["source"].get("ids", []), diff,
                          evidence=evidence, accepted=bool(accepted), protocol=body["protocol"])
    except Exception:
        h = hashlib.sha1(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:10]
        return {"id": f"{kind}-{h}", **body}


def record(verdict: dict, path: str = LEDGER_DRAFT) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(verdict["ledger"], sort_keys=True, default=str) + "\n")
    return path


# --------------------------------------------------------------------------- check
def _resolve_adversary(adversary: Union[str, Callable]) -> Callable:
    if callable(adversary):
        return adversary
    from dt.adversary import ensemble as E
    if adversary == "ensemble":
        return E.find
    return E.finder(adversary)


def check(change_fn: Union[Change, dict, Callable], *, adversary: Union[str, Callable] = "decompose", bench=None,
          calibration: Optional[list] = None, fresh: Optional[list] = None, evaluation: Optional[list] = None,
          real: Optional[list] = None, families: Optional[Iterable[str]] = None, metamorphic: bool = True,
          report_evaluation: bool = False, name: Optional[str] = None, progress: Optional[Callable[[str], None]] = None) -> dict:
    """Judge a proposed change. Returns ``{accepted, reasons, checks, ledger, ...}``.

    ``adversary`` is resolved *inside* each activation so parameter changes take effect. ``calibration``
    defaults to the benchmark's calibration split, ``fresh`` to the fusion-calibration cases (fresh seeds,
    calibration documents), ``evaluation`` to the evaluation split (scored only with ``report_evaluation``).
    """
    from dt.adversary import ensemble as E
    say = progress or (lambda s: None)
    change = as_change(change_fn, name)
    t0 = time.time()
    bench = bench or (E.main_bench() if calibration is None or (report_evaluation and evaluation is None) else None)
    cal = calibration if calibration is not None else bench.split("calibration")
    checks: dict[str, Any] = {}
    reasons: list[str] = []

    def adv():
        return _resolve_adversary(adversary)

    # 1. priors
    checks["bounded_priors"] = bp = bounded_priors(change)
    if not bp["passed"]:
        reasons += bp["problems"]
    n_par = max(1, len(change.params))
    # 2. calibration
    say("calibration before/after")
    before = _score(adv(), cal)
    with change.activate(cal):
        after = _score(adv(), cal)
    need = float(P[_K + "min_gain"]) * n_par
    gain = after["headline"] - before["headline"]
    noise_up = after["noise"]["false_case_rate"] - before["noise"]["false_case_rate"]
    ok = gain >= need and noise_up <= float(P[_K + "noise_tol"]) + 1e-12
    checks["calibration"] = {"passed": ok, "before": _brief(before), "after": _brief(after), "gain": round(gain, 4), "required": need,
                             "noise_rate_change": round(noise_up, 4)}
    if not ok:
        reasons.append(f"calibration gain {gain:+.4f} < required {need:.4f}" if gain < need else f"noise false-case rate +{noise_up:.3f}")
    # 3. leave-one-family-out
    fams = sorted(set(families) if families else {c.family for c in cal})
    lofo = {}
    worst = 0.0
    for fam in fams:
        tr = [c for c in cal if c.family != fam]
        te = [c for c in cal if c.family == fam]
        if not te or not tr:
            continue
        say(f"leave out {fam}")
        b = _score(adv(), te)["headline"]
        with change.activate(tr):
            a = _score(adv(), te)["headline"]
        lofo[fam] = {"before": round(b, 4), "after": round(a, 4), "delta": round(a - b, 4), "train_cases": len(tr), "test_cases": len(te)}
        worst = min(worst, a - b)
    ok = worst >= -float(P[_K + "lofo_tol"])
    checks["leave_one_family_out"] = {"passed": ok, "families": lofo, "worst": round(worst, 4)}
    if not ok:
        reasons.append(f"leave-one-family-out: a held-out family lost {-worst:.4f}")
    # 4. fresh seeds
    if fresh is None:
        try:
            fresh = E.fusion_calibration()
        except Exception:
            fresh = []
    if fresh:
        say("fresh seeds")
        b = _score(adv(), fresh)
        with change.activate(cal):
            a = _score(adv(), fresh)
        d = a["headline"] - b["headline"]
        ok = d >= -float(P[_K + "seed_tol"])
        checks["fresh_seeds"] = {"passed": ok, "before": _brief(b), "after": _brief(a), "delta": round(d, 4), "cases": len(fresh)}
        if not ok:
            reasons.append(f"fresh seeds: headline {d:+.4f} on unseen perturbation seeds")
    # 5. metamorphic regulariser
    if metamorphic:
        mc = meta_cases(cal, int(P[_K + "meta_cases"]))
        say(f"metamorphic relations on {len(mc)} cases")
        mb = metamorphic_violations(adv(), mc)
        with change.activate(cal):
            ma = metamorphic_violations(adv(), mc)
        allow = max(1, int(math.ceil(float(P[_K + "meta_tol"]) * mb["total"])))
        ok = ma["total"] <= mb["total"] + allow
        checks["metamorphic"] = {"passed": ok, "before": {k: mb[k] for k in ("total", "by_relation")},
                                 "after": {k: ma[k] for k in ("total", "by_relation")}, "allowed_growth": allow, "cases": len(mc)}
        if not ok:
            reasons.append(f"metamorphic violations {mb['total']} -> {ma['total']} (allowed +{allow})")
    accepted = not reasons
    # 6. holdouts (never part of the decision)
    if real:
        say("real pages (report only)")
        rp = {}
        for rc in real:
            t, c, ir = rc.target_rgb, rc.candidate_rgb, rc.candidate_ir
            nb = len(_errs(adv()(t, c, ir)))
            with change.activate(cal):
                na = len(_errs(adv()(t, c, ir)))
            flood = na >= nb * float(P[_K + "flood_ratio"]) and na - nb >= int(P[_K + "flood_min"])
            rp[rc.id] = {"before": nb, "after": na, "flood": bool(flood)}
        checks["real_pages"] = {"report_only": True, "pages": rp, "floods": sum(v["flood"] for v in rp.values())}
    if report_evaluation:
        ev = evaluation if evaluation is not None else bench.split("evaluation")
        key = (bench.meta.get("key") if bench is not None else None) or "evaluation"
        looks = _eval_look(key)
        b = _score(adv(), ev)
        with change.activate(cal):
            a = _score(adv(), ev)
        checks["evaluation"] = {"report_only": True, "before": _brief(b), "after": _brief(a), "delta": round(a["headline"] - b["headline"], 4),
                                "looks": looks, "budget": int(P[_K + "eval_budget"]),
                                "warning": "evaluation budget spent: rotate the evaluation seed" if looks > int(P[_K + "eval_budget"]) else None}
    gates = {k: bool(v["passed"]) for k, v in checks.items() if "passed" in v}
    evidence = {"before": {"calibration": checks["calibration"]["before"]}, "after": {"calibration": checks["calibration"]["after"]},
                "gates": gates, "checks": {k: {kk: vv for kk, vv in v.items() if kk != "per_case"} for k, v in checks.items()},
                "adversary": adversary if isinstance(adversary, str) else getattr(adversary, "__name__", "fn")}
    verdict = {"change": {"name": change.name, "kind": change.kind, "params": change.params, "description": change.description},
               "accepted": accepted, "reasons": reasons, "checks": checks, "seconds": round(time.time() - t0, 1)}
    verdict["ledger"] = ledger_entry(change, evidence, accepted)
    return verdict


# --------------------------------------------------------------------------- demonstration
def naive_proposals(adversary: str = "decompose", keys: Optional[list[str]] = None, cases: Optional[list] = None,
                    steps: Iterable[float] = (-0.2, 0.2), progress=None) -> list[dict]:
    """What an unprotected tuner would do: try each parameter +-step (fraction of range) on the seed-0
    calibration split and propose every change that raises its headline. Returns proposals sorted by
    calibration gain. The protocol then judges them."""
    from dt.adversary import ensemble as E
    cal = cases if cases is not None else E.main_bench().split("calibration")
    fn = E.finder(adversary)
    base = _score(fn, cal)["headline"]
    rng = P.ranges()
    keys = keys or [k for k in rng if k.startswith(f"adversary.{adversary}.")]
    out = []
    for k in keys:
        lo, hi = rng[k]
        cur = P[k]
        if not isinstance(cur, (int, float)) or isinstance(cur, bool):
            continue
        for s in steps:
            new = cur + s * (hi - lo)
            new = min(hi, max(lo, new))
            if isinstance(cur, int):
                new = int(round(new))
            if new == cur:
                continue
            with override({k: new}):
                h = _score(E.finder(adversary), cal)["headline"]
            if progress:
                progress(f"{k}: {cur} -> {new}: {h - base:+.4f}")
            if h > base:
                out.append({"params": {k: new}, "gain": h - base})
    out.sort(key=lambda d: -d["gain"])
    return out


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Run the anti-overfitting protocol on proposed changes.")
    ap.add_argument("--adversary", default="decompose")
    ap.add_argument("--param", action="append", default=[], help="key=value (JSON value); several make one change")
    ap.add_argument("--naive", type=int, default=0, help="judge the top-N proposals of a naive calibration tuner")
    ap.add_argument("--keys", nargs="*", help="parameters the naive tuner may touch")
    ap.add_argument("--evaluation", action="store_true", help="also report EVALUATION (counts against the look budget)")
    ap.add_argument("--real", action="store_true", help="report real-page finding counts (never used to decide)")
    ap.add_argument("--record", action="store_true", help="append the verdicts' ledger entries to the draft ledger")
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    changes = []
    if a.param:
        changes.append({k: json.loads(v) for k, v in (p.split("=", 1) for p in a.param)})
    if a.naive:
        props = naive_proposals(a.adversary, a.keys, progress=print)
        with open(os.path.join(OUT, f"naive_{a.adversary}.json"), "w") as f:
            json.dump(props, f, indent=1, default=str)
        changes += [p["params"] for p in props[: a.naive]]
        if len(props) > 1:  # what a greedy tuner ships: every winning step at once
            merged = {}
            for p in props[: max(2, a.naive)]:
                merged.update(p["params"])
            changes.append(merged)
    real = None
    if a.real:
        from dt.adversary.novelty import real_cases
        real = real_cases()
    verdicts = []
    for ch in changes:
        v = check(ch, adversary=a.adversary, report_evaluation=a.evaluation, real=real, progress=print)
        verdicts.append(v)
        print(json.dumps({"change": v["change"]["params"], "accepted": v["accepted"], "reasons": v["reasons"]}, default=str))
        if a.record:
            record(v)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(OUT, f"verdicts_{a.adversary}_{stamp}.json"), "w") as f:
        json.dump(verdicts, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
