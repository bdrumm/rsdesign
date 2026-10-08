"""Refine-move statistics: which hypothesis kinds actually pay for their renders?

Aggregates every refine_history.json under the given roots (translate runs write one) into, per
move kind: tried, accepted, acceptance rate, total and mean loss gain of accepted moves, and gain per
attempt (the number a move-ordering policy should sort by). docs/SELF_IMPROVEMENT.md, mechanism 9.

``--priors`` also exports the learned move priors (``knowledge/move_priors.json``) that
``dt.refine.priors`` uses to order hypotheses: per kind and per (kind, node type, size bucket).

    python -m dt.selftest.move_stats out/ [--json] [--priors [knowledge/move_priors.json]]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Optional


def _node_index(dirpath: str) -> dict[str, tuple[str, float]]:
    """id -> (type, area) from the run's IR files (pre-refine ir.json first, then the refined ir.mapped.json
    for nodes refine inserted), for histories written before records carried node_type / node_area."""
    out: dict[str, tuple[str, float]] = {}
    for name in ("ir.mapped.json", "ir.json"):  # later files win: ir.json describes the nodes refine saw
        p = os.path.join(dirpath, name)
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        stack = [d.get("root")] if isinstance(d, dict) else []
        while stack:
            n = stack.pop()
            if not isinstance(n, dict):
                continue
            b = n.get("box") or {}
            try:
                area = float(b.get("w", 0)) * float(b.get("h", 0))
            except Exception:
                area = 0.0
            if n.get("id"):
                out[n["id"]] = (n.get("type") or "-", area)
            stack.extend(n.get("children") or [])
    return out


def collect(roots: list[str]) -> list[dict]:
    rows = []
    for root in roots:
        for dirpath, _, files in os.walk(root):
            if "refine_history.json" in files:
                try:
                    hist = json.load(open(os.path.join(dirpath, "refine_history.json")))
                except Exception:
                    continue
                index: Optional[dict] = None
                for h in hist if isinstance(hist, list) else hist.get("history", []):
                    if isinstance(h, dict) and h.get("kind") and h.get("kind") != "stop":
                        row = dict(h, run=os.path.relpath(dirpath))
                        if "node_type" not in row:
                            if index is None:
                                index = _node_index(dirpath)
                            t, a = index.get(h.get("node_id") or "", (None, None))
                            row["node_type"] = t
                            if a is None:  # node-less moves (missing, raster regions): the edit region
                                p = h.get("params") or {}
                                box = p.get("box") if isinstance(p.get("box"), dict) else None
                                a = float(box["w"]) * float(box["h"]) if box else 0.0
                            row["node_area"] = a
                        rows.append(row)
    return rows


def _gain(h: dict) -> float:
    """Loss gain credited to one accepted record. Tile-scored records carry their own measured gain
    (``gain_est``); otherwise members of an accepted batch share one loss drop, split so batched kinds are
    not over-credited."""
    if not h.get("accepted"):
        return 0.0
    if h.get("gain_est") is not None:
        return max(0.0, float(h["gain_est"]))
    return max(0.0, float(h.get("before", 0)) - float(h.get("after", 0))) / max(1, int(h.get("batch", 1) or 1))


def summarise(rows: list[dict]) -> dict:
    agg: dict[str, dict] = defaultdict(lambda: {"tried": 0, "accepted": 0, "gain": 0.0})
    runs = set()
    for h in rows:
        a = agg[h["kind"]]
        a["tried"] += 1
        runs.add(h.get("run"))
        if h.get("accepted"):
            a["accepted"] += 1
            a["gain"] += _gain(h)
    total_gain = sum(a["gain"] for a in agg.values()) or 1.0
    out = {}
    for k, a in agg.items():
        out[k] = {
            **a,
            "accept_rate": a["accepted"] / a["tried"] if a["tried"] else 0.0,
            "mean_gain": a["gain"] / a["accepted"] if a["accepted"] else 0.0,
            "gain_per_try": a["gain"] / a["tried"] if a["tried"] else 0.0,
            "gain_share": a["gain"] / total_gain,
        }
    return {"runs": len(runs), "moves": len(rows), "kinds": dict(sorted(out.items(), key=lambda t: -t[1]["gain_per_try"]))}


def priors(rows: list[dict], smooth: Optional[float] = None, min_tries: Optional[int] = None, dead_gpt: Optional[float] = None) -> dict:
    """The move-priors table for ``dt.refine.priors``.

    ``prior`` = realised gain / promised gain (sum of ``expected_gain``) of a bucket, smoothed with ``smooth``
    pseudo-tries towards its kind's rate, which is smoothed towards the global rate: the factor by which the
    critic's estimate should be scaled. ``deprioritised`` = gain per try below ``dead_gpt`` x the global gain
    per try after at least ``min_tries`` tries."""
    from dt.params import P
    from dt.refine.priors import bucket_key
    smooth = float(P["refine.prior.smooth"] if smooth is None else smooth)
    min_tries = int(P["refine.prior.min_tries"] if min_tries is None else min_tries)
    dead_gpt = float(P["refine.prior.dead_gpt"] if dead_gpt is None else dead_gpt)

    def acc() -> dict:
        return {"tried": 0, "accepted": 0, "gain": 0.0, "expected": 0.0}

    kinds: dict[str, dict] = defaultdict(acc)
    buckets: dict[str, dict] = defaultdict(acc)
    glob = acc()
    for h in rows:
        g, e = _gain(h), max(0.0, float(h.get("expected_gain") or 0.0))
        key = bucket_key(h["kind"], h.get("node_type"), float(h.get("node_area") or 0.0))
        for a in (glob, kinds[h["kind"]], buckets[key]):
            a["tried"] += 1
            a["accepted"] += int(bool(h.get("accepted")))
            a["gain"] += g
            a["expected"] += e
    g_rate = glob["gain"] / glob["expected"] if glob["expected"] > 0 else 1.0
    g_gpt = glob["gain"] / glob["tried"] if glob["tried"] else 0.0

    def finish(a: dict, parent_rate: float) -> dict:
        # pseudo-tries carry the parent's rate in units of this bucket's own mean expected gain
        mean_e = a["expected"] / a["tried"] if a["tried"] else 0.0
        rate = (a["gain"] + smooth * mean_e * parent_rate) / (a["expected"] + smooth * mean_e) if a["expected"] > 0 else parent_rate
        gpt = a["gain"] / a["tried"] if a["tried"] else 0.0
        return {
            "tried": a["tried"], "accepted": a["accepted"],
            "accept_rate": round(a["accepted"] / a["tried"], 4) if a["tried"] else 0.0,
            "gain": a["gain"], "gain_per_try": gpt, "expected": a["expected"],
            "prior": round(rate / g_rate, 4) if g_rate > 0 else 1.0,
            "deprioritised": bool(a["tried"] >= min_tries and gpt < dead_gpt * g_gpt),
        }

    k_out = {k: finish(a, g_rate) for k, a in sorted(kinds.items())}
    k_rate = {k: (kinds[k]["gain"] + smooth * (kinds[k]["expected"] / max(1, kinds[k]["tried"])) * g_rate)
              / (kinds[k]["expected"] + smooth * (kinds[k]["expected"] / max(1, kinds[k]["tried"])))
              if kinds[k]["expected"] > 0 else g_rate for k in kinds}
    b_out = {key: finish(a, k_rate[key.split("|", 1)[0]]) for key, a in sorted(buckets.items())}
    return {
        "doc": "learned refine move priors (dt.selftest.move_stats --priors); prior = smoothed realised/expected gain "
               "relative to the global rate; deprioritised = near-zero gain per try (ordered last, never removed)",
        "runs": len({h.get("run") for h in rows}), "moves": len(rows),
        "global": {"tried": glob["tried"], "gain": glob["gain"], "expected": glob["expected"], "gain_per_try": g_gpt, "rate": g_rate},
        "params": {"smooth": smooth, "min_tries": min_tries, "dead_gpt": dead_gpt},
        "kinds": k_out, "buckets": b_out,
    }


def markdown(s: dict) -> str:
    lines = [f"{s['moves']} tried moves across {s['runs']} refine runs.", "",
             "| move | tried | accepted | accept rate | gain share | gain per try |", "|---|---:|---:|---:|---:|---:|"]
    for k, a in s["kinds"].items():
        lines.append(f"| {k} | {a['tried']} | {a['accepted']} | {a['accept_rate']:.0%} | {a['gain_share']:.0%} | {a['gain_per_try']:.5f} |")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", default=["out"])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--priors", nargs="?", const="", default=None, help="write the move priors (default knowledge/move_priors.json)")
    a = ap.parse_args()
    rows = collect(a.roots)
    if a.priors is not None:
        from dt.refine.priors import PRIORS_PATH
        path = a.priors or PRIORS_PATH
        with open(path, "w") as f:
            json.dump(priors(rows), f, indent=1, sort_keys=True)
        print(f"wrote {path}")
    s = summarise(rows)
    print(json.dumps(s, indent=2) if a.json else markdown(s))
