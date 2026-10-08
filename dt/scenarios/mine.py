"""Mine translate run dirs for real-crop scenario cases.

    dt scenario mine out/real_r3m/*          # -> $DT_HOME/scenarios/real/<id>/{target.png, meta.json}
                                             #    + knowledge/real_crops.json (manifest only)

Reads each run's ``metrics.json`` (source screenshot) and ``validation/validation.json`` (worst
flat-colour regions, worst edge tile, gate failures) and cuts a context crop around every
failure that breaks the visually-identical gate: a region with ΔE >=
``validate.gate.ident.worst_region_de`` (criterion: the crop's worst region ΔE under that gate)
and the worst chamfer tile when it is >= ``validate.gate.ident.chamfer_tile`` (criterion: the
crop's worst tile chamfer under that gate). Every crop also carries an editability guard
(rasterised area <= ``validate.gate.raster_frac``), so pasting the target back cannot pass.

Captured pages are third-party content: the crops live under ``$DT_HOME`` (never in git); the
committed manifest holds only provenance (source file + url, box, failing measure) and criteria.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Iterable, Optional

import dt.validate.fidelity  # noqa: F401 - registers the validate.gate.* thresholds used here
from dt.ir import Box
from dt.learn import state
from dt.params import P, register

register("scenarios.mine.pad", 24, "context (px) added around a mined failure box", (0, 96))
register("scenarios.mine.min_side", 96, "min side (px) of a mined real crop", (32, 400))
register("scenarios.mine.max_side", 360, "max side (px) of a mined real crop", (96, 1280))
register("scenarios.mine.per_run", 4, "max crops mined per translate run", (1, 50))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _screens_manifest() -> dict[str, dict]:
    p = os.path.join(ROOT, "fixtures", "screens", "manifest.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return {s.get("file"): s for s in json.load(f).get("screens", [])}


def crop_box(b: Box, W: int, H: int) -> Box:
    """``b`` padded, grown to ``min_side`` around its centre, capped at ``max_side``, inside the frame."""
    pad, lo, hi = float(P["scenarios.mine.pad"]), float(P["scenarios.mine.min_side"]), float(P["scenarios.mine.max_side"])
    e = b.expand(pad)
    w, h = min(hi, max(lo, e.w)), min(hi, max(lo, e.h))
    w, h = min(w, W), min(h, H)
    x = min(max(0.0, e.cx - w / 2), W - w)
    y = min(max(0.0, e.cy - h / 2), H - h)
    return Box(round(x), round(y), round(w), round(h))


def load_manifest(path: Optional[str] = None) -> dict:
    """The real-crop manifest. Every crop has a permanent ``index`` (its scenario seed): new crops get
    the next free index, existing ones never move, so suite seeds keep naming the same crops."""
    path = path or state.real_manifest_path()
    if not os.path.exists(path):
        return {"version": 1, "crops": []}
    with open(path) as f:
        m = json.load(f)
    _assign_indices(m)
    return m


def _assign_indices(m: dict) -> None:
    nxt = 1 + max((int(c["index"]) for c in m.get("crops", []) if "index" in c), default=-1)
    for c in sorted(m.get("crops", []), key=lambda c: c["id"]):
        if "index" not in c:
            c["index"] = nxt
            nxt += 1


def save_manifest(m: dict, path: Optional[str] = None) -> str:
    path = path or state.real_manifest_path()
    _assign_indices(m)
    m["crops"] = sorted(m["crops"], key=lambda c: int(c["index"]))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(m, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def candidates(val: dict) -> list[dict]:
    """Failure boxes of one validation report, worst first."""
    out: list[dict] = []
    reg_thr = float(P["validate.gate.ident.worst_region_de"])
    for r in sorted(val.get("worst_regions", []), key=lambda r: -float(r["delta_e"])):
        if float(r["delta_e"]) >= reg_thr:
            out.append({"kind": "region", "box": Box.from_dict(r["box"]), "measure": "region_de_worst",
                        "value": float(r["delta_e"]), "detail": {"target": r.get("target_color"), "render": r.get("render_color")}})
    tile = val.get("chamfer_tile_argmax") or {}
    if tile and float(val.get("chamfer_tile_max", 0.0)) >= float(P["validate.gate.ident.chamfer_tile"]):
        out.insert(min(1, len(out)), {"kind": "tile", "box": Box.from_dict(tile), "measure": "chamfer_tile_max",
                                      "value": float(val["chamfer_tile_max"]), "detail": {}})
    return out


def criteria_for(kind: str) -> list[dict]:
    guard = {"metric": "raster_frac", "op": "<=", "threshold": "param:validate.gate.raster_frac", "roi": None,
             "scale": 0.35, "doc": "editability: no pasting the target back"}
    if kind == "region":
        return [{"metric": "region_de_worst", "op": "<", "threshold": "param:validate.gate.ident.worst_region_de",
                 "roi": None, "scale": 10.0, "doc": "worst flat region of the crop under the visually-identical gate"}, guard]
    return [{"metric": "chamfer_tile_max", "op": "<", "threshold": "param:validate.gate.ident.chamfer_tile",
             "roi": None, "scale": 8.0, "doc": "worst edge tile of the crop under the visually-identical gate"}, guard]


def mine(run_dirs: Iterable[str], per_run: Optional[int] = None, manifest_path: Optional[str] = None,
         dry: bool = False) -> dict:
    """Cut real-crop cases from translate runs; returns ``{added, skipped, manifest}``."""
    from dt.common.image import load_rgb, save_rgb
    per_run = int(P["scenarios.mine.per_run"] if per_run is None else per_run)
    screens = _screens_manifest()
    man = load_manifest(manifest_path)
    known = {c["id"]: c for c in man["crops"]}
    added, skipped = [], []
    for rd in run_dirs:
        vpath = os.path.join(rd, "validation", "validation.json")
        mpath = os.path.join(rd, "metrics.json")
        if not os.path.exists(vpath):
            skipped.append({"run": rd, "reason": "no validation/validation.json"})
            continue
        with open(vpath) as f:
            val = json.load(f)
        metrics = {}
        if os.path.exists(mpath):
            with open(mpath) as f:
                metrics = json.load(f)
        src = metrics.get("source") or ""
        tpath = os.path.join(rd, "validation", "target.png")
        img_path = tpath if os.path.exists(tpath) else src
        if not img_path or not os.path.exists(img_path):
            skipped.append({"run": rd, "reason": "target image not found"})
            continue
        rgb = load_rgb(img_path)
        H, W = rgb.shape[:2]
        run_name = os.path.basename(os.path.normpath(rd))
        chosen: list[Box] = []
        for c in candidates(val):
            if len(chosen) >= per_run:
                break
            box = crop_box(c["box"], W, H)
            if any(box.iou(b) > 0.3 or b.contains(c["box"]) for b in chosen):
                continue
            chosen.append(box)
            h = hashlib.sha1(f"{run_name}:{c['kind']}:{box.as_int()}".encode()).hexdigest()[:8]
            cid = f"{run_name}_{c['kind']}_{h}"
            file = os.path.basename(src) if src else None
            entry = {"id": cid, "kind": c["kind"], "box": box.to_dict(), "failure_box": c["box"].to_dict(),
                     "measure": {"name": c["measure"], "value": round(c["value"], 3), **c["detail"]},
                     "gates_failed": sorted(k for k, v in (val.get("gates") or {}).items() if not v),
                     "source": {"file": file, "url": (screens.get(file) or {}).get("url"), "run": run_name,
                                "frame": [W, H]},
                     "criteria": criteria_for(c["kind"])}
            if not dry:
                d = os.path.join(state.real_crops_dir(), cid)
                os.makedirs(d, exist_ok=True)
                from dt.scenarios.sources import crop
                save_rgb(crop(rgb, box), os.path.join(d, "target.png"))
                with open(os.path.join(d, "meta.json"), "w") as f:
                    json.dump({**entry, "provenance": {"run_dir": os.path.abspath(rd), "image": os.path.abspath(img_path)}},
                              f, indent=2)
            if cid in known:
                entry["index"] = known[cid]["index"]
            else:
                entry["index"] = 1 + max((int(c["index"]) for c in known.values()), default=-1)
                added.append(cid)
            known[cid] = entry
    man["crops"] = list(known.values())
    path = None if dry else save_manifest(man, manifest_path)
    return {"added": added, "skipped": skipped, "n_crops": len(man["crops"]), "manifest": path}


def load_crop(entry: dict) -> Any:
    """Pixels of a manifest entry: the ``$DT_HOME`` copy, else re-cut from the source screenshot
    (``fixtures/screens/<file>`` or ``$DT_SCREENS``); ``None`` when neither exists."""
    from dt.common.image import load_rgb
    p = os.path.join(state.real_crops_dir(), entry["id"], "target.png")
    if os.path.exists(p):
        return load_rgb(p)
    file = (entry.get("source") or {}).get("file")
    for d in filter(None, (os.environ.get("DT_SCREENS"), os.path.join(ROOT, "fixtures", "screens"))):
        sp = os.path.join(d, file) if file else None
        if sp and os.path.exists(sp):
            from dt.scenarios.sources import crop
            return crop(load_rgb(sp), Box.from_dict(entry["box"]))
    return None
