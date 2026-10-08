"""Run a scenario family on a set of seeds: per-case metrics, pass/fail, margins, pass rate,
worst cases with diff images.

    from dt.scenarios import registry, runner
    rep = runner.run_family(registry.get("pale_surface"), seeds=range(8), out_root="out/scenarios")
    rep["pass_rate"], rep["objective"], rep["worst"]

Outputs (``out_root`` not None): ``out/scenarios/<run_id>/<family>/report.json|md`` and
``cases/<case id>.png`` (target | render | ΔE heat, ROI and failing glyph/row boxes drawn).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from dt.params import P, register
from dt.scenarios import measures
from dt.scenarios.spec import ScenarioCase, ScenarioFamily

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_OUT_ROOT = os.path.join(ROOT, "out", "scenarios")

register("scenarios.margin_cap", 0.25,
         "criterion margins are clipped to [-1, cap] in the tuning objective: failing cases dominate, "
         "over-satisfying a criterion earns little", (0.0, 1.0))
register("scenarios.worst_k", 6, "worst cases listed (with diff images) in a scenario report", (1, 50))

_CASE_CACHE: dict[tuple, ScenarioCase] = {}
_CACHE_MAX = 256


def clip_margin(m: float) -> float:
    return float(max(-1.0, min(float(P["scenarios.margin_cap"]), m)))


def get_case(family: ScenarioFamily, seed: int) -> ScenarioCase:
    """Generated case (cached in-process: the same seed is evaluated many times while training)."""
    key = (family.name, family.version, int(seed))
    c = _CASE_CACHE.get(key)
    if c is None:
        c = family.case(int(seed))
        if len(_CASE_CACHE) >= _CACHE_MAX:
            _CASE_CACHE.pop(next(iter(_CASE_CACHE)))
        _CASE_CACHE[key] = c
    return c


def _criteria_for(family: ScenarioFamily, case: ScenarioCase):
    """Per-case criteria (mined real crops carry their own) or the family's."""
    from dt.scenarios.spec import Criterion
    cm = case.meta.get("criteria") if case.meta else None
    if cm:
        return [Criterion.from_dict(c) for c in cm]
    return family.criteria


def run_case(family: ScenarioFamily, seed: int, stage: Optional[str] = None, refine_iters: Optional[int] = None,
             overrides: Optional[dict[str, Any]] = None, out_dir: Optional[str] = None) -> dict:
    """Generate, run and judge one case. Never raises: errors fail the case with margin -1."""
    from dt.selftest.bench import param_overrides
    stage = stage or family.stage
    k = family.refine_iters if refine_iters is None else int(refine_iters)
    row: dict[str, Any] = {"family": family.name, "seed": int(seed), "id": None, "params": None, "stage": stage,
                           "passed": False, "criteria": [], "margin_min": -1.0, "margin_mean": -1.0, "metrics": {},
                           "info": {}, "error": None, "diff_image": None}
    step = "generate"
    try:
        with param_overrides(overrides):
            case = get_case(family, seed)
            row.update(id=case.id, params=case.params, fingerprint=case.fingerprint())
            if case.meta.get("provenance"):
                row["provenance"] = case.meta["provenance"]
            step = stage
            pred, rendered, info = measures.run_stage(case, stage, k)
            step = "measure"
            metrics = measures.measure(family, case, pred, rendered)
            crits = _criteria_for(family, case)
            res = [c.evaluate(measures.lookup(metrics, c.key())) for c in crits]
        row["criteria"] = res
        row["passed"] = all(r["passed"] for r in res)
        margins = [clip_margin(r["margin"]) for r in res]
        row["margin_min"], row["margin_mean"] = float(min(margins)), float(np.mean(margins))
        row["metrics"] = _jsonable(metrics)
        row["info"] = info
        if out_dir:
            row["diff_image"] = save_case_image(case, rendered, res, os.path.join(out_dir, "cases", f"{case.id}.png"))
    except Exception as e:  # noqa: BLE001 - one broken case must not lose the run
        row["error"] = f"{type(e).__name__}: {e}"
        row["error_stage"] = step
        row["traceback"] = traceback.format_exc()
    return row


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def save_case_image(case: ScenarioCase, rendered: np.ndarray, results: list[dict], path: str) -> str:
    from dt.compare import diff_map, save_diff_image
    from dt.validate.fidelity import conform_to_target
    r, _ = conform_to_target(case.target_rgb, rendered)
    boxes = []
    if case.roi is not None:
        boxes.append(case.roi)
    boxes += [b for b in case.meta.get("highlight_boxes", [])]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return save_diff_image(case.target_rgb, r, diff_map(case.target_rgb, r), path, boxes)


def _run_case_star(args: tuple) -> dict:
    fam_name, seed, stage, k, overrides, out_dir = args
    from dt.scenarios import registry
    return run_case(registry.get(fam_name), seed, stage, k, overrides, out_dir)


def run_cases(family: ScenarioFamily, seeds: Sequence[int], stage: Optional[str] = None, refine_iters: Optional[int] = None,
              overrides: Optional[dict[str, Any]] = None, out_dir: Optional[str] = None, workers: int = 1,
              verbose: bool = False) -> list[dict]:
    seeds = [int(s) for s in seeds]
    rows: list[dict] = []
    if workers > 1 and len(seeds) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        ctx = multiprocessing.get_context("spawn")
        jobs = [(family.name, s, stage, refine_iters, overrides, out_dir) for s in seeds]
        with ProcessPoolExecutor(max_workers=int(workers), mp_context=ctx) as ex:
            futs = [ex.submit(_run_case_star, j) for j in jobs]
            for s, fut in zip(seeds, futs):
                try:
                    row = fut.result()
                except Exception as e:  # noqa: BLE001 - e.g. BrokenProcessPool
                    row = {"family": family.name, "seed": s, "id": f"{family.name}_{s}", "passed": False, "criteria": [],
                           "margin_min": -1.0, "margin_mean": -1.0, "metrics": {}, "error": f"{type(e).__name__}: {e}",
                           "error_stage": "worker"}
                rows.append(row)
                if verbose:
                    _print_row(row)
    else:
        for s in seeds:
            row = run_case(family, s, stage, refine_iters, overrides, out_dir)
            rows.append(row)
            if verbose:
                _print_row(row)
    return rows


def _print_row(row: dict) -> None:
    if row.get("error"):
        msg = f"ERROR [{row.get('error_stage')}] {row['error']}"
    else:
        fails = [c["criterion"] + f" (got {c['value']:.4g})" if c["value"] is not None else c["criterion"] + " (missing)"
                 for c in row["criteria"] if not c["passed"]]
        msg = "pass" if row["passed"] else "FAIL " + "; ".join(fails)
    print(f"  [scenario] {row.get('id') or row['family']}: {msg}  margin={row.get('margin_min', -1):+.3f}",
          file=sys.stderr, flush=True)


def summarize(family: ScenarioFamily, rows: list[dict], seeds: Sequence[int], stage: str) -> dict:
    """Aggregate rows: pass rate, objective (mean clipped criterion margin), per-criterion pass
    rates and mean values, worst cases."""
    n = len(rows)
    n_pass = sum(1 for r in rows if r["passed"])
    per: dict[str, dict] = {}
    for r in rows:
        for c in r.get("criteria", []):
            d = per.setdefault(c["criterion"], {"n": 0, "passed": 0, "values": []})
            d["n"] += 1
            d["passed"] += int(c["passed"])
            if c["value"] is not None:
                d["values"].append(c["value"])
    crit = {k: {"pass_rate": v["passed"] / max(1, v["n"]), "mean": float(np.mean(v["values"])) if v["values"] else None,
                "worst": (float(max(v["values"])) if "<" in k else float(min(v["values"]))) if v["values"] else None}
            for k, v in per.items()}
    worst = sorted(rows, key=lambda r: (r["passed"], r.get("margin_min", -1.0)))[: int(P["scenarios.worst_k"])]
    return {"family": family.name, "stage": stage, "refine_iters": family.refine_iters, "seeds": [int(s) for s in seeds],
            "n": n, "n_pass": n_pass, "pass_rate": (n_pass / n) if n else 0.0,
            "objective": float(np.mean([r["margin_mean"] for r in rows])) if rows else -1.0,
            "mean_margin_min": float(np.mean([r["margin_min"] for r in rows])) if rows else -1.0,
            "n_errors": sum(1 for r in rows if r.get("error")), "criteria": crit,
            "worst": [{"id": r.get("id"), "seed": r["seed"], "passed": r["passed"], "margin_min": r.get("margin_min"),
                       "failed": [c["criterion"] for c in r.get("criteria", []) if not c["passed"]],
                       "error": r.get("error"), "diff_image": r.get("diff_image")} for r in worst]}


def run_family(family: ScenarioFamily, seeds: Iterable[int], stage: Optional[str] = None, refine_iters: Optional[int] = None,
               workers: int = 1, overrides: Optional[dict[str, Any]] = None, out_root: Optional[str] = DEFAULT_OUT_ROOT,
               run_id: Optional[str] = None, verbose: bool = False) -> dict:
    """Run ``family`` on ``seeds`` and return the report (also written under ``out_root``)."""
    seeds = [int(s) for s in seeds]
    stage = stage or family.stage
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(out_root, run_id, family.name) if out_root else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    rows = run_cases(family, seeds, stage, refine_iters, overrides, out_dir, workers, verbose)
    rep = summarize(family, rows, seeds, stage)
    rep.update(run_id=run_id, out_dir=out_dir, overrides=overrides or {}, wall_s=round(time.perf_counter() - t0, 2),
               cases=rows, created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if out_dir:
        with open(os.path.join(out_dir, "report.json"), "w") as f:
            json.dump(_jsonable(rep), f, indent=2, default=str)
        with open(os.path.join(out_dir, "report.md"), "w") as f:
            f.write(report_markdown(family, rep))
    return rep


def report_markdown(family: ScenarioFamily, rep: dict) -> str:
    L = [f"# scenario `{family.name}` — run `{rep.get('run_id')}`", "", family.description, "",
         f"stage **{rep['stage']}** (refine iters {family.refine_iters}) · {rep['n']} cases · pass rate "
         f"**{rep['pass_rate']:.3f}** ({rep['n_pass']}/{rep['n']}) · objective {rep['objective']:+.4f} · "
         f"errors {rep['n_errors']} · {rep.get('wall_s')} s", "",
         "failure refs: " + "; ".join(family.failure_refs), "", "| criterion | pass rate | mean | worst |", "|---|---|---|---|"]
    for k, v in rep["criteria"].items():
        f = (lambda x: "-" if x is None else f"{x:.4g}")
        L.append(f"| `{k}` | {v['pass_rate']:.3f} | {f(v['mean'])} | {f(v['worst'])} |")
    L += ["", "## worst cases", "", "| case | seed | margin | failed | image |", "|---|---|---|---|---|"]
    for w in rep["worst"]:
        img = os.path.relpath(w["diff_image"], rep["out_dir"]) if (w.get("diff_image") and rep.get("out_dir")) else "-"
        failed = w["error"] or ", ".join(w["failed"]) or "-"
        L.append(f"| {w['id']} | {w['seed']} | {w['margin_min']:+.3f} | {failed} | `{img}` |")
    L += ["", "## params of the worst cases", ""]
    by_id = {r.get("id"): r for r in rep["cases"]}
    for w in rep["worst"]:
        r = by_id.get(w["id"]) or {}
        L.append(f"- `{w['id']}`: `{json.dumps(r.get('params'), sort_keys=True, default=str)}`")
    return "\n".join(L) + "\n"
