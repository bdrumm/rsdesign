"""Translate a set of real (no ground truth) screenshots and summarise the independent validation.

    python tools/real_screens_eval.py --out out/real NAME [NAME ...]   (names from fixtures/screens/manifest.json)

Writes <out>/<name>/ (full translate outputs) and <out>/summary.json with gates, fidelity measures,
loss before/after, components found, icons named and wall time per screen.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_one(name: str, out_root: str, iters: int) -> dict:
    src = os.path.join(ROOT, "fixtures", "screens", f"{name}.png")
    out = os.path.join(out_root, name)
    t = time.time()
    proc = subprocess.run([sys.executable, "-m", "dt.cli", "translate", src, "--ds", "material3", "-o", out,
                           "--refine-iters", str(iters), "--json"], cwd=ROOT, capture_output=True, text=True)
    wall = time.time() - t
    row: dict = {"name": name, "exit": proc.returncode, "wall_s": round(wall, 1)}
    try:
        m = json.load(open(os.path.join(out, "metrics.json")))
        v = json.load(open(os.path.join(out, "validation", "validation.json")))
        ir = json.load(open(os.path.join(out, "ir.mapped.json")))
    except Exception as e:  # noqa: BLE001 - report, never hide
        row["error"] = f"{type(e).__name__}: {e}"
        row["stderr_tail"] = proc.stderr[-800:]
        return row

    def walk(n):
        yield n
        for c in n.get("children", []):
            yield from walk(c)

    nodes = list(walk(ir["root"]))
    row.update({
        "size": [v.get("width"), v.get("height")],
        "loss_before": m.get("loss_before"), "loss_after": m.get("loss_after"),
        "gates": v.get("gates"), "gate_failures": v.get("gate_failures"),
        "jnd_frac_nontext": v.get("jnd_frac_nontext"), "mean_de": v.get("mean_de"),
        "chamfer": v.get("chamfer"), "chamfer_tile_max": v.get("chamfer_tile_max"), "edge_within1": v.get("edge_within1"),
        "region_de_worst": v.get("region_de_worst"), "text_lines": [v.get("text_lines_matched"), v.get("text_lines_target")],
        "text_cer": v.get("text_cer"), "text_dpos_max": v.get("text_dpos_max"), "ssim": v.get("ssim"),
        "nodes": len(nodes), "components": sum(1 for n in nodes if n.get("component")),
        "icons": sum(1 for n in nodes if n.get("type") == "icon"),
        "icons_named": sum(1 for n in nodes if n.get("type") == "icon" and n.get("icon_name")),
        "errors": m.get("errors", []),
    })
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "real"))
    ap.add_argument("--iters", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for nm in a.names:
        r = run_one(nm, a.out, a.iters)
        rows.append(r)
        print(json.dumps({k: r.get(k) for k in ("name", "exit", "wall_s", "loss_before", "loss_after", "jnd_frac_nontext", "chamfer", "edge_within1", "text_cer", "components", "icons_named", "icons", "error")}), flush=True)
        json.dump(rows, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
