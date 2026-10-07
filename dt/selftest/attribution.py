"""Stage attribution: where does the error budget live?

Runs the ground-truth bench several times, each time substituting ground truth for every stage
before a boundary, and reports how much composite each stage costs:

    oracle_render   = bench(stages=("render",))            gt IR -> render           (renderer / IR ceiling)
    oracle_map      = bench(stages=("map", "render"))      gt IR -> map -> render    (+ mapping)
    full            = bench(stages=("perceive","map","render"))                     (+ perception)

    cost(render)   = 1 - oracle_render
    cost(map)      = oracle_render - oracle_map
    cost(perceive) = oracle_map - full

The stage with the largest cost is where the next improvement round should go
(docs/SELF_IMPROVEMENT.md, mechanism 8). Per-metric deltas show *which* measure each stage hurts.

    python -m dt.selftest.attribution --workers 4 [--out out/attribution]
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from dt.selftest import bench

LADDER = (
    ("oracle_render", ("render",)),
    ("oracle_map", ("map", "render")),
    ("full", ("perceive", "map", "render")),
)


def run(corpus_dirs=None, workers: int = 1, limit: Optional[int] = None, out_root: Optional[str] = None) -> dict:
    corpus_dirs = list(corpus_dirs or bench.DEFAULT_CORPORA)
    runs: dict[str, dict] = {}
    for name, stages in LADDER:
        rep = bench.run(corpus_dirs, stages=stages, limit=limit, workers=workers, run_id=f"attr_{name}",
                        out_root=out_root)
        runs[name] = rep
    comp = {k: float(v["composite"]) for k, v in runs.items()}
    cost = {
        "render": 1.0 - comp["oracle_render"],
        "map": comp["oracle_render"] - comp["oracle_map"],
        "perceive": comp["oracle_map"] - comp["full"],
    }
    metrics = sorted({m for r in runs.values() for m in r.get("metrics", {})})
    per_metric = {}
    for m in metrics:
        g = {k: (runs[k]["metrics"].get(m) or {}).get("goodness") for k in runs}
        per_metric[m] = {
            "goodness": g,
            "cost_map": (g["oracle_render"] - g["oracle_map"]) if None not in (g["oracle_render"], g["oracle_map"]) else None,
            "cost_perceive": (g["oracle_map"] - g["full"]) if None not in (g["oracle_map"], g["full"]) else None,
        }
    worst = max(cost, key=cost.get)
    return {"composite": comp, "stage_cost": cost, "largest": worst, "per_metric": per_metric,
            "n_cases": runs["full"].get("n_cases")}


def markdown(res: dict) -> str:
    c, k = res["composite"], res["stage_cost"]
    lines = [
        "| pipeline | composite |", "|---|---:|",
        f"| ground-truth IR, our renderer | {c['oracle_render']:.4f} |",
        f"| ground-truth IR, our mapping + renderer | {c['oracle_map']:.4f} |",
        f"| full pipeline from pixels | {c['full']:.4f} |", "",
        "| stage | composite cost |", "|---|---:|",
        *[f"| {s} | {v:.4f} |" for s, v in sorted(k.items(), key=lambda t: -t[1])], "",
        f"Largest cost: **{res['largest']}**.", "",
        "| metric | cost of mapping | cost of perception |", "|---|---:|---:|",
    ]
    for m, d in res["per_metric"].items():
        cm, cp = d["cost_map"], d["cost_perceive"]
        lines.append(f"| {m} | {'' if cm is None else f'{cm:+.3f}'} | {'' if cp is None else f'{cp:+.3f}'} |")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=os.path.join("out", "attribution"))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    res = run(workers=a.workers, limit=a.limit, out_root=None)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "attribution.json"), "w"), indent=2)
    open(os.path.join(a.out, "attribution.md"), "w").write(markdown(res))
    print(json.dumps(res, indent=2) if a.json else markdown(res))
