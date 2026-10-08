"""Learned move priors: order critic hypotheses by ``expected_gain x prior``.

``knowledge/move_priors.json`` is exported by ``python -m dt.selftest.move_stats out/ --priors`` from
every ``refine_history.json`` under ``out/``. For each move kind, and for each (kind, node type, size
bucket), it holds tried / accepted counts, the realised loss gain per try, and ``prior``: the share of
the critic's promised gain (``expected_gain``) that such moves actually delivered, smoothed towards
the kind's and then the global rate. A bucket whose gain per try is near zero (with enough tries) is
``deprioritised``: its hypotheses are ordered after every other one -- never removed, they still run
when nothing better remains.

``P['refine.policy']`` selects the ordering: ``'priors'`` (default) or ``'adversary'``, which asks
``dt.adversary.policy.suggest(findings, doc)`` (built elsewhere; optional) and falls back to the priors
when that module is missing or fails.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from dt.params import P, register

register("refine.policy", "priors", "hypothesis ordering policy: 'priors' (knowledge/move_priors.json) or 'adversary' (dt.adversary.policy.suggest, falls back to priors)")
register("refine.prior.enabled", True, "multiply the critic's expected gain by the learned move prior when ordering hypotheses")
register("refine.prior.floor", 0.05, "smallest prior multiplier (a learned prior never zeroes a move)", (0.0, 1.0))
register("refine.prior.smooth", 8.0, "pseudo-tries pulling a bucket's prior towards its kind's (and the kind's towards the global) rate", (0.0, 100.0))
register("refine.prior.min_tries", 12, "a bucket needs this many tries before it can be deprioritised", (1, 200))
register("refine.prior.dead_gpt", 0.02, "deprioritise a bucket whose gain per try is below this fraction of the global gain per try", (0.0, 1.0))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRIORS_PATH = os.environ.get("DT_MOVE_PRIORS", os.path.join(ROOT, "knowledge", "move_priors.json"))
SIZE_BUCKETS = ((400.0, "xs"), (4000.0, "s"), (40000.0, "m"), (float("inf"), "l"))

_CACHE: dict[str, tuple[float, dict]] = {}


def size_bucket(area: float) -> str:
    for hi, name in SIZE_BUCKETS:
        if area < hi:
            return name
    return "l"


def bucket_key(kind: str, node_type: Optional[str], area: float) -> str:
    return f"{kind}|{node_type or '-'}|{size_bucket(area)}"


def load(path: Optional[str] = None) -> dict:
    """The priors table (``{}`` when absent or unreadable), cached by file mtime."""
    path = path or PRIORS_PATH
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return {}
    hit = _CACHE.get(path)
    if hit and hit[0] == mt:
        return hit[1]
    try:
        data = json.load(open(path))
    except Exception:
        data = {}
    _CACHE[path] = (mt, data)
    return data


def describe_node(h, doc) -> tuple[Optional[str], float]:
    """(node type, area) the hypothesis acts on; the edit region for node-less moves (missing, raster regions)."""
    n = doc.find(h.node_id) if h.node_id else None
    if n is not None:
        return n.type, float(n.box.area)
    return None, float(h.region.area)


def prior_of(kind: str, node_type: Optional[str], area: float, table: Optional[dict] = None) -> tuple[float, bool]:
    """(multiplier, deprioritised) for one move; ``(1.0, False)`` when nothing was learned about it."""
    table = load() if table is None else table
    if not table:
        return 1.0, False
    b = table.get("buckets", {}).get(bucket_key(kind, node_type, area))
    k = table.get("kinds", {}).get(kind)
    src = b or k
    if not src:
        return 1.0, False
    floor = float(P["refine.prior.floor"])
    return max(floor, float(src.get("prior", 1.0))), bool(src.get("deprioritised", False))


def _adversary_order(hyps: list, doc) -> Optional[list]:
    """Order from ``dt.adversary.policy.suggest(findings, doc)``, or None (missing module / failure / bad shape).
    Accepted return shapes: a list of the hypotheses (re-ordered; any left out are appended, never dropped),
    a list of per-hypothesis scores, or a ``{kind: multiplier}`` dict."""
    try:
        import importlib
        mod = importlib.import_module("dt.adversary.policy")
        out = mod.suggest(list(hyps), doc)
    except Exception:
        return None
    try:
        if isinstance(out, dict):
            return sorted(hyps, key=lambda h: -float(h.expected_gain) * float(out.get(h.kind, 1.0)))
        if isinstance(out, (list, tuple)) and out and all(isinstance(x, (int, float)) for x in out) and len(out) == len(hyps):
            order = sorted(range(len(hyps)), key=lambda i: -float(out[i]))
            return [hyps[i] for i in order]
        if isinstance(out, (list, tuple)):
            ids = {id(h) for h in hyps}
            picked = [h for h in out if id(h) in ids]
            seen = {id(h) for h in picked}
            return picked + [h for h in hyps if id(h) not in seen]
    except Exception:
        return None
    return None


def order(hyps: list, doc) -> list:
    """Hypotheses ordered by ``expected_gain x prior`` (deprioritised buckets last), or by the adversary policy.
    Sets ``h.prior`` / ``h.deprioritised`` on every hypothesis (used by the optimizer and the history)."""
    table = load() if bool(P["refine.prior.enabled"]) else {}
    for h in hyps:
        t, a = describe_node(h, doc)
        h.prior, h.deprioritised = prior_of(h.kind, t, a, table)
        for alt in getattr(h, "alternatives", []) or []:
            alt.prior, alt.deprioritised = h.prior, h.deprioritised
    if str(P["refine.policy"]) == "adversary":
        out = _adversary_order(hyps, doc)
        if out is not None:
            return out
    # stable: equal scores keep the critic's (expected-gain) order
    return sorted(hyps, key=lambda h: (bool(h.deprioritised), -float(h.expected_gain) * float(h.prior)))
