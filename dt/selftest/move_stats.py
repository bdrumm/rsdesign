"""Refine-move statistics: which hypothesis kinds actually pay for their renders?

Aggregates every refine_history.json under the given roots (translate runs write one) into, per
move kind: tried, accepted, acceptance rate, total and mean loss gain of accepted moves, and gain per
attempt (the number a move-ordering policy should sort by). docs/SELF_IMPROVEMENT.md, mechanism 9.

    python -m dt.selftest.move_stats out/ [--json]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


def collect(roots: list[str]) -> list[dict]:
    rows = []
    for root in roots:
        for dirpath, _, files in os.walk(root):
            if "refine_history.json" in files:
                try:
                    hist = json.load(open(os.path.join(dirpath, "refine_history.json")))
                except Exception:
                    continue
                for h in hist if isinstance(hist, list) else hist.get("history", []):
                    if isinstance(h, dict) and h.get("kind") and h.get("kind") != "stop":
                        rows.append(dict(h, run=os.path.relpath(dirpath)))
    return rows


def summarise(rows: list[dict]) -> dict:
    agg: dict[str, dict] = defaultdict(lambda: {"tried": 0, "accepted": 0, "gain": 0.0})
    runs = set()
    for h in rows:
        a = agg[h["kind"]]
        a["tried"] += 1
        runs.add(h.get("run"))
        if h.get("accepted"):
            a["accepted"] += 1
            # members of an accepted batch share one loss drop: split it so batched kinds are not over-credited
            a["gain"] += max(0.0, float(h.get("before", 0)) - float(h.get("after", 0))) / max(1, int(h.get("batch", 1) or 1))
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
    a = ap.parse_args()
    s = summarise(collect(a.roots))
    print(json.dumps(s, indent=2) if a.json else markdown(s))
