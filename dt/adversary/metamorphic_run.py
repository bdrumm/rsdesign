"""Experiment driver for the metamorphic approach: calibrate, score, run relations, fuzz, report.

    python -m dt.adversary.metamorphic_run --all          # everything (≈1 h on a laptop, cached after)
    python -m dt.adversary.metamorphic_run --report       # rebuild out/adversary/metamorphic.md from saved JSON

Steps (each saves JSON under ``out/adversary/metamorphic/``):

* ``--calibrate``  coordinate descent over ``adversary.meta.*`` gates on the CALIBRATION split only
* ``--score``      score :func:`dt.adversary.metamorphic.find` on CALIBRATION and EVALUATION
* ``--synthetic``  run every relation on evaluation-split ground-truth documents (attribution vs GT)
* ``--real``       run every relation on the 8 real pages of ``out/real_r3m`` + the differential
                   adversary on (target, render, ir.mapped)
* ``--fuzz``       robustness envelope (``dt.adversary.fuzz``)
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import time
from typing import Optional

import numpy as np

from dt.adversary import metamorphic as M
from dt.adversary.taxonomy import TYPES, Finding
from dt.ir import Box, Document
from dt.params import P

OUT = M.OUT
REPORT = os.path.join(M.ROOT, "out", "adversary", "metamorphic.md")
SYNTH_DOCS = ("synth:synth_1_006", "synth:synth_1_007", "synth:synth_1_008", "mwc:mwc_1_006", "mwc:mwc_1_007", "mwc:mwc_1_008",
              "gen:9001:0", "gen:9001:1")
REAL_DIR = os.path.join(M.ROOT, "out", "real_r3m")
GRID = {"gate_de": [5.0, 10.0, 15.0, 20.0, 30.0], "gate_px": [4, 8, 16, 32], "gate_open": [0, 2, 3], "gate_blur": [0.0, 0.8, 1.5], "find_color_tol": [4.0, 6.0, 10.0], "pos_tol": [1.0, 2.0],
        "match_min": [0.35, 0.6], "fallback": [True, False], "opacity_resid": [0.1, 0.2]}
STAGE = "map"   # the translator under the relations: perceive + map onto Material 3 (the product path, minus refine)


def _say(msg: str) -> None:
    print(msg, flush=True)


def _save(name: str, obj) -> str:
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    return p


def _load(name: str):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


# --------------------------------------------------------------------------- calibrate + score
def _fn():
    def metamorphic(t, c, ir=None):
        return M.find(t, c, ir)
    return metamorphic


def calibrate(bench, sweeps: int = 2) -> dict:
    from dt.adversary.benchmark import score
    cur = {k: P[f"adversary.meta.{k}"] for k in GRID}
    hist = []

    def val(cfg):
        with M.overrides(**cfg):
            r = score(_fn(), bench, "calibration", name="metamorphic")
        hist.append({"cfg": dict(cfg), "headline": r["headline"], "det_f1": r["detection"]["f1"], "macro_leaf_f1": r["macro_leaf"]["f1"],
                     "noise_false": r["noise"]["false_case_rate"]})
        return r["headline"]

    best = val(cur)
    _say(f"calibration start {best:.4f} {cur}")
    for sweep in range(sweeps):
        changed = False
        for k, vals in GRID.items():
            for v in vals:
                if v == cur[k]:
                    continue
                cfg = dict(cur, **{k: v})
                h = val(cfg)
                if h > best + 1e-4:
                    best, cur, changed = h, cfg, True
                    _say(f"  sweep {sweep} {k}={v}: {h:.4f}")
        if not changed:
            break
    out = {"best": cur, "headline": best, "history": hist}
    _save("calibration.json", out)
    return out


def score_all(bench, cfg: dict) -> dict:
    from dt.adversary import benchmark as BM
    reps = {}
    with M.overrides(**cfg):
        for split in ("calibration", "evaluation"):
            reps[split] = BM.score(_fn(), bench, split, name="metamorphic")
        alt = not bool(cfg.get("fallback", P["adversary.meta.fallback"]))
        with M.overrides(fallback=alt):
            reps["evaluation_ir_only"] = BM.score(_fn(), bench, "evaluation", name=f"metamorphic, residual fallback {'on' if alt else 'off'} (not chosen)")
        ex = BM.save_examples(bench, _fn(), os.path.join(OUT, "examples"))
    ns = {s: M.noise_sensitivity(bench.split(s)) for s in ("calibration", "evaluation")}
    # honest runtime: uncached translation of both images of a few evaluation cases
    cases = [c for c in bench.split("evaluation") if c.kind == "perturbed"][:3]
    t0 = time.time()
    for c in cases:
        M.translate_ir(c.target_rgb, cache=False)
        M.translate_ir(c.candidate_rgb, cache=False)
    uncached = (time.time() - t0) / max(1, len(cases))
    out = {"reports": reps, "noise_sensitivity": ns, "examples": ex, "uncached_s_per_case": uncached, "cfg": cfg,
           "summary": bench.summary(), "bench_meta": bench.meta}
    _save("score.json", out)
    return out


# --------------------------------------------------------------------------- relations on inputs
def _base_jobs(inputs: list[tuple[str, np.ndarray, Optional[Document]]], workers: int) -> None:
    M.precompute([(rgb, STAGE, 1.0) for _, rgb, _ in inputs], workers, progress=_say)
    jobs = []
    for name, rgb, gt in inputs:
        base = M.translate_ir(rgb, STAGE)
        for inst in M.instances(rgb, base, M.DEFAULT_RELATIONS, gt):
            jobs.append((inst.image, STAGE, inst.dpr))
    _say(f"precomputing {len(jobs)} follow-up translations")
    M.precompute(jobs, workers, progress=_say)


def _gt_errors(gt: Document, base: Document) -> list[Finding]:
    with M.overrides(pos_tol=3.0, size_tol=4.0, text_slack=4.0, color_tol=10.0, radius_tol=4.0):
        return M.ir_diff(gt, base, painted_only=True, aspects=("geometry", "text", "color", "icon"))


def _summ(f: Finding, page_res) -> dict:
    ev = f.evidence
    v = ev.get("validator", {})
    return {"type": f.type, "aspect": ev.get("aspect"), "relation": ev.get("relation"), "box": f.box.to_dict(), "magnitude": f.magnitude,
            "confidence": f.confidence, "painted": ev.get("painted", True), "blind": v.get("blind"),
            "base_flag": v.get("base", {}).get("flagged"), "follow_flag": v.get("follow", {}).get("flagged"),
            "expected": ev.get("expected"), "actual": ev.get("actual"), "base_side": ev.get("base_side")}


def _unpainted(f: Finding, base: Document) -> bool:
    if f.evidence.get("painted") is False:
        return True
    ids = {n.id: n for n in base.walk()}
    e = ids.get(f.evidence.get("e_id") or "")
    return e is not None and not M.paints(e) and f.evidence.get("aspect") in ("geometry", "structure", "layout_gap", "layout_mode")


def run_inputs(kind: str, inputs: list[tuple[str, np.ndarray, Optional[Document]]], workers: int, renders: Optional[dict] = None) -> dict:
    from dt.common.image import save_rgb
    from dt.compare.visualize import draw_boxes
    _base_jobs(inputs, workers)
    pages, specs = [], []
    for name, rgb, gt in inputs:
        res = M.run_page(name, rgb, stage=STAGE, gt=gt, judge=True, log=_say)
        base = M.translate_ir(rgb, STAGE)
        gt_errs = _gt_errors(gt, base) if gt is not None else None
        for f in res.violations:
            f.evidence["painted"] = not _unpainted(f, base)
            if gt_errs is not None:
                f.evidence["base_side"] = any(M._covers(g, f.box) for g in gt_errs)
        by_rel: dict[str, list[Finding]] = {}
        for f in res.violations:
            by_rel.setdefault(f.evidence["relation"], []).append(f)
        src = {"doc": name} if gt is not None else {"image": os.path.relpath(os.path.join(REAL_DIR, name, "validation", "target.png"), M.ROOT)}
        for run in res.runs:
            fs = by_rel.get(run["relation"], [])
            if fs:
                specs.append(M.scenario_spec(src, run["relation"], run["params"], fs, STAGE))
        # evidence overlay
        cols = {"translate": (220, 0, 0), "recolor_tint": (230, 120, 0), "recolor_swap": (200, 160, 0), "dpr2": (0, 120, 220),
                "row_add": (0, 160, 80), "row_remove": (0, 110, 60), "crop": (150, 0, 200), "mirror": (220, 0, 140), "jpeg": (90, 90, 90)}
        img = rgb.copy()
        for rel, fs in by_rel.items():
            img = draw_boxes(img, [f.box for f in fs if f.evidence.get("painted")], cols.get(rel, (0, 0, 0)), 2)
        d = os.path.join(OUT, kind)
        os.makedirs(d, exist_ok=True)
        save_rgb(img, os.path.join(d, f"{name.replace(':', '_')}.violations.png"))
        rec = res.to_dict()
        rec["gt_base_errors"] = None if gt_errs is None else len(gt_errs)
        rec["violations"] = [_summ(f, res) for f in res.violations]
        pages.append(rec)
    out = {"kind": kind, "stage": STAGE, "translator": M.translator_version(STAGE), "pages": pages, "scenarios": specs}
    _save(f"relations_{kind}.json", out)
    return out


def synthetic(workers: int) -> dict:
    from dt.adversary.benchmark import load_doc
    from dt.render.screenshot import render_doc
    inputs = []
    for spec in SYNTH_DOCS:
        gt = load_doc(spec)
        inputs.append((spec, render_doc(gt), gt))
    return run_inputs("synthetic", inputs, workers)


def real(workers: int) -> dict:
    from dt.common.image import load_rgb
    inputs = []
    for d in sorted(glob.glob(os.path.join(REAL_DIR, "*", "validation", "target.png"))):
        name = d.split(os.sep)[-3]
        inputs.append((name, load_rgb(d), None))
    out = run_inputs("real", inputs, workers)
    # the differential adversary on the shipped translations: target vs render.png (refined), ir.mapped.json
    pair = []
    for name, rgb, _ in inputs:
        pd = os.path.join(REAL_DIR, name)
        render = load_rgb(os.path.join(pd, "render.png"))
        ir = Document.load(os.path.join(pd, "ir.mapped.json"))
        if render.shape != rgb.shape:
            from dt.validate.fidelity import conform_to_target
            render, _ = conform_to_target(rgb, render)
        t0 = time.time()
        fs = M.find(rgb, render, ir)
        val = json.load(open(os.path.join(pd, "validation", "validation.json"))) if os.path.exists(os.path.join(pd, "validation", "validation.json")) else {}
        pair.append({"name": name, "seconds": round(time.time() - t0, 1), "n": len(fs), "by_type": M._count([f.type for f in fs]),
                     "top": [{"type": f.type, "box": f.box.rounded().to_dict(), "magnitude": f.magnitude, "confidence": round(f.confidence, 3),
                              "aspect": f.evidence.get("aspect"), "expected": f.evidence.get("expected"), "actual": f.evidence.get("actual")}
                             for f in sorted(fs, key=lambda f: -f.confidence)[:12]],
                     "gates": val.get("gates"), "gate_failures": val.get("gate_failures", {}).get("visually-identical")})
        from dt.common.image import save_rgb
        from dt.compare.visualize import draw_boxes
        im = draw_boxes(rgb.copy(), [f.box for f in fs if f.evidence.get("aspect") != "residual"], (220, 0, 0), 2)
        im = draw_boxes(im, [f.box for f in fs if f.evidence.get("aspect") == "residual"], (120, 120, 120), 1)
        save_rgb(im, os.path.join(OUT, "real", f"{name}.differential.png"))
    out["differential"] = pair
    _save("relations_real.json", out)
    return out


def fuzz(workers: int) -> dict:
    from dt.adversary import fuzz as F
    rows = F.envelope(workers=workers, log=_say)
    out = {"rows": rows, "findings": [f.to_dict() for f in F.findings(rows)], "scenarios": F.scenario_specs(rows)}
    _save("fuzz.json", out)
    return out


# --------------------------------------------------------------------------- report
def _pct(v: float) -> str:
    return f"{100 * v:.1f}"


def _rel_table(rel: dict) -> list[str]:
    L = ["| input | nodes | " + " | ".join(M.RELATIONS_ORDER) + " | total | visible | validator-blind |", "|---" * (len(M.RELATIONS_ORDER) + 5) + "|"]
    for p in rel["pages"]:
        by = {}
        for v in p["violations"]:
            by[v["relation"]] = by.get(v["relation"], 0) + 1
        ran = {r["relation"] for r in p["runs"]}
        cells = [str(by.get(r, 0)) if r in ran else "–" for r in M.RELATIONS_ORDER]
        vis = [v for v in p["violations"] if v["painted"]]
        blind = [v for v in p["violations"] if v["blind"]]
        L.append(f"| {p['name']} | {p['base_nodes']} | " + " | ".join(cells) + f" | {len(p['violations'])} | {len(vis)} | {len(blind)} |")
    return L


def _type_by_relation(rel: dict) -> list[str]:
    tab: dict[str, dict[str, int]] = {}
    for p in rel["pages"]:
        for v in p["violations"]:
            t = v["type"] + ("" if v["aspect"] not in ("icon_alias", "font_family", "fill_presence", "stroke_presence", "dpr", "token_consistency", "component", "layout_mode", "layout_gap", "residual") else f" ({v['aspect']})")
            tab.setdefault(t, {})
            tab[t][v["relation"]] = tab[t].get(v["relation"], 0) + 1
    rows = sorted(tab, key=lambda t: -sum(tab[t].values()))
    L = ["| type (aspect) | in taxonomy | " + " | ".join(M.RELATIONS_ORDER) + " | total |", "|---" * (len(M.RELATIONS_ORDER) + 3) + "|"]
    for t in rows:
        base = t.split(" ")[0]
        L.append(f"| `{t}` | {'yes' if base in TYPES else '**no**'} | " + " | ".join(str(tab[t].get(r, '')) for r in M.RELATIONS_ORDER) + f" | {sum(tab[t].values())} |")
    return L


def _blind_stats(rel: dict) -> dict:
    vs = [v for p in rel["pages"] for v in p["violations"]]
    vis = [v for v in vs if v["painted"]]
    out = {"n": len(vs), "visible": len(vis), "blind": sum(1 for v in vs if v["blind"]), "visible_blind": sum(1 for v in vis if v["blind"])}
    gt = [v for v in vis if v.get("base_side") is not None]
    if gt:
        out["gt_checked"] = len(gt)
        out["base_side"] = sum(1 for v in gt if v["base_side"])
        out["base_side_blind"] = sum(1 for v in gt if v["base_side"] and v["blind"])
    return out


def write_report() -> str:
    from dt.adversary import benchmark as BM
    cal, sc, syn, rl, fz = (_load(n) for n in ("calibration.json", "score.json", "relations_synthetic.json", "relations_real.json", "fuzz.json"))
    base = json.load(open(os.path.join(M.ROOT, "out", "adversary", "benchmark_baseline.json")))
    L = ["# Adversary approach: metamorphic and robustness testing of the translator", "",
         "Approach: test the translator itself, without ground truth. A correct translator commutes with exactly-known edits of "
         "its input (`T(t(x)) == t*(T(x))`), so any disagreement beyond tolerance *proves* an error on `x` or on `t(x)`. "
         "The same typed IR differ powers a differential adversary for the shared benchmark (`find`: translate target and "
         "candidate with the same translator and diff the two IRs where the pixels differ). Code: `dt/adversary/metamorphic.py`, "
         "`dt/adversary/fuzz.py`, driver `dt/adversary/metamorphic_run.py`; tests `tests/test_adversary_metamorphic.py`.", ""]
    if sc:
        reps = sc["reports"]
        floors = {(r["adversary"], r["split"]): r for r in base["reports"]}
        L += ["## 1. Benchmark score (differential adversary `find`)", "",
              f"Tuned on CALIBRATION only (coordinate descent over {', '.join(f'`adversary.meta.{k}`' for k in GRID)}); chosen: "
              + ", ".join(f"`{k}`={v}" for k, v in sc["cfg"].items()) + (f"; calibration headline {cal['headline']:.3f} after {len(cal['history'])} configurations." if cal else "."), "",
              "| adversary | split | headline | det P | det R | det F1 | AP | macro leaf F1 | macro parent F1 | type acc leaf / parent | noise false-case rate | s/case (cached) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        rows = [(floors[("null_adversary", "evaluation")]), floors[("baseline_residual", "calibration")], floors[("baseline_residual", "evaluation")],
                reps["calibration"], reps["evaluation"], reps["evaluation_ir_only"]]
        for r in rows:
            d = r["detection"]
            L.append(f"| {r['adversary']} | {r['split']} | {r['headline']:.3f} | {_pct(d['precision'])} | {_pct(d['recall'])} | {_pct(d['f1'])} | {_pct(r['ap'])} | "
                     f"{_pct(r['macro_leaf']['f1'])} | {_pct(r['macro_parent']['f1'])} | {_pct(r['type_accuracy']['leaf'])} / {_pct(r['type_accuracy']['parent'])} | "
                     f"{_pct(r['noise']['false_case_rate'])} | {r['runtime']['mean_s']:.3f} |")
        L += ["", f"Runtime: the table's s/case is with translations cached (the diff, gate and typing only). Uncached, one case costs "
              f"**{sc['uncached_s_per_case']:.1f} s** (two full perceptions: OCR, segmentation, render-verified icon and font identification). "
              "The approach is ~100x slower than a pixel adversary; caching by image hash makes repeated targets free.", ""]
        r = reps["evaluation"]
        L += ["### Per-type (evaluation)", "", "| type | gt | pred | typed P | typed R | typed F1 | loc recall | magnitude MAE | median rel. err |", "|---|---|---|---|---|---|---|---|---|"]
        for t, d in r["per_type"].items():
            mae = f"{d['mag_mae']:.2f}" if "mag_mae" in d else "–"
            rel = f"{d['mag_rel_median']:.2f}" if "mag_rel_median" in d else "–"
            L.append(f"| `{t}` | {d['gt']} | {d['pred']} | {_pct(d['precision'])} | {_pct(d['recall'])} | {_pct(d['f1'])} | {_pct(d['loc_recall'])} | {mae} | {rel} |")
        n = r["noise"]
        L += ["", "Recall by magnitude bin: " + ", ".join(f"{b} {_pct(v)}" for b, v in r["recall_by_bin"].items()) + ". By family: "
              + ", ".join(f"{f} {_pct(v['recall'])} ({v['gt']} gt, {v['preds']} preds)" for f, v in r["by_family"].items()) + ".",
              f"Noise: {n['cases_with_false']}/{n['cases']} pure-noise cases drew a finding; " + ", ".join(f"{t} {_pct(v['false_case_rate'])} of {v['cases']}" for t, v in n["by_type"].items())
              + f". Spurious per perturbed case: {n['spurious_per_perturbed_case_clean_target']:.2f} (clean target) vs {n['spurious_per_perturbed_case_noisy_target']:.2f} (noisy target).", ""]
        cols = sorted({k for row in r["confusion"].values() for k in row}, key=lambda k: (k.startswith("("), k))
        L += ["### Confusion matrix (evaluation; rows ground truth, columns matched prediction)", "",
              "| gt \\ pred | " + " | ".join(f"`{c}`" for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for g in [t for t in TYPES if t in r["confusion"]] + (["(spurious)"] if "(spurious)" in r["confusion"] else []):
            L.append(f"| `{g}` | " + " | ".join(str(r["confusion"][g].get(c, "")) for c in cols) + " |")
        L += ["", "### Translator noise sensitivity (pure-noise cases, before the pixel gate)", "",
              "The nuisance relation measured directly on the benchmark: share of pure-noise cases whose translation changed at all, and the "
              "mean number of raw IR differences.", "", "| split | noise | cases | translation changed | mean raw differences |", "|---|---|---|---|---|"]
        for s, d in sc["noise_sensitivity"].items():
            for t, v in d.items():
                L.append(f"| {s} | {t} | {v['cases']} | {_pct(v['changed_frac'])} | {v['mean_diffs']:.1f} |")
        L += ["", "Evidence panels (green ground truth, red findings): " + ", ".join(f"`{os.path.relpath(p, M.ROOT)}`" for p in sc["examples"]), ""]
    for title, rel, kind in (("2. Relations on synthetic inputs (evaluation documents, ground truth known)", syn, "synthetic"),
                             ("3. Relations on the 8 real translated pages", rl, "real")):
        if not rel:
            continue
        st = _blind_stats(rel)
        L += [f"## {title}", "",
              f"Translator under test: `{rel['stage']}` stage (perceive + map onto Material 3; refine excluded), version `{rel['translator']}`. "
              "Counts are violations after collapsing (one per moved subtree / absent subtree). *visible* = the violation involves painted nodes; "
              "*validator-blind* = the independent pairwise validator (`dt.validate` measures, emulated per region: non-text JND fraction "
              f"> {P['adversary.meta.blind_jnd']}, flat-region ΔE2000 ≥ {P['adversary.meta.blind_region_de']}, OCR line CER > 0 or Δpos > {P['adversary.meta.blind_dpos']} px) "
              "flags the region in neither the base pair `(x, render(T(x)))` nor the follow-up pair `(t(x), render(T(t(x))))`.", ""]
        L += _rel_table(rel)
        L += ["", f"Totals: {st['n']} violations, {st['visible']} visible; **{st['blind']} ({_pct(st['blind'] / max(1, st['n']))}%) validator-blind**, "
              f"of the visible ones {st['visible_blind']} ({_pct(st['visible_blind'] / max(1, st['visible']))}%)."]
        if "gt_checked" in st:
            L += [f"Ground-truth attribution (visible violations): {st['base_side']}/{st['gt_checked']} coincide with an error of the base translation "
                  f"T(x) against the ground truth; the rest are errors only on the follow-up input t(x) (the base was right there). "
                  f"{st['base_side_blind']} base-side errors were validator-blind."]
        L += ["", "Taxonomy mapping (rows: finding type, with the aspect when it is not a plain property difference):", ""]
        L += _type_by_relation(rel)
        L += [""]
        if kind == "real" and rel.get("differential"):
            L += ["### Differential adversary on the shipped translations (target vs refined `render.png`, `ir.mapped.json`)", "",
                  "| page | findings | by type | validator visually-identical failures |", "|---|---|---|---|"]
            for d in rel["differential"]:
                bt = ", ".join(f"{k} {v}" for k, v in list(d["by_type"].items())[:6])
                gf = "; ".join((d.get("gate_failures") or [])[:3])
                L.append(f"| {d['name']} | {d['n']} | {bt} | {gf} |")
            L += [""]
    if fz:
        L += ["## 4. Robustness envelope (boundary fuzzing of perception)", "",
              "Minimal scenes (one component on md.sys.color.surface, real renderer, `perceive`). Boundary = smallest value at which the "
              "component is found and stays found for every larger grid value (grid + 2 bisection steps). *lost at* = the largest value tried "
              "where it is lost.", "", "| component | contrast: found ≥ ΔE | lost at ΔE | size: found ≥ px | lost at px | spacing: separate ≥ px | merged at px | notes |",
              "|---|---|---|---|---|---|---|---|"]
        by: dict[str, dict] = {}
        for r in fz["rows"]:
            by.setdefault(r["component"], {})[r["axis"]] = r
        for c, ax in by.items():
            cells, notes = [], []
            for a in ("contrast", "size", "spacing"):
                r = ax.get(a)
                if r is None:
                    cells += ["–", "–"]
                    continue
                if not r["nominal_ok"]:
                    cells += ["never", "nominal"]
                    notes.append(f"{a}: lost even at nominal")
                    continue
                cells += [f"{r['boundary_value']:g}" + (" (grid floor)" if r.get("below_grid") else ""), "–" if r["lost_at"] is None else f"{r['lost_at']:g}"]
                if r["islands"]:
                    notes.append(f"{a}: non-monotonic (found at {r['islands']})")
            L.append(f"| {c} | " + " | ".join(cells) + f" | {'; '.join(notes)} |")
        L += [""]
    an = _load("analysis.json")
    if an:
        for sec in an.get("sections", []):
            L += [f"## {sec['title']}", "", sec["body"].strip(), ""]
    L.append("<!-- generated by dt/adversary/metamorphic_run.py from out/adversary/metamorphic/*.json -->")
    md = "\n".join(L) + "\n"
    with open(REPORT, "w") as f:
        f.write(md)
    return REPORT


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("calibrate", "score", "synthetic", "real", "fuzz", "report", "all"):
        ap.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    bench = None
    if a.calibrate or a.score or a.all:
        bench = M.load_bench()
        M.precompute(M.bench_jobs(bench.cases), a.workers, progress=_say)
    cfg = (_load("calibration.json") or {}).get("best")
    if a.calibrate or a.all:
        cfg = calibrate(bench)["best"]
    if a.score or a.all:
        score_all(bench, cfg or {k: P[f"adversary.meta.{k}"] for k in GRID})
    if a.synthetic or a.all:
        synthetic(a.workers)
    if a.real or a.all:
        real(a.workers)
    if a.fuzz or a.all:
        fuzz(a.workers)
    print(write_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
