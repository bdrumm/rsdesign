"""The active scenario suite (``knowledge/scenarios.json``): which families gate the pipeline,
on which TRAIN and HOLDOUT seeds, and their baseline pass rates.

    {"version": 1,
     "families": {"pale_surface": {"family_version": 1, "stage": "perceive", "refine_iters": 0,
                                   "train_seeds": [0..5], "holdout_seeds": [100000..100005],
                                   "baseline": {"train_pass_rate": .., "holdout_pass_rate": .., "train_objective": ..,
                                                "holdout_objective": .., "n_errors": 0, "created": ..,
                                                "fingerprints": {seed: pixel hash}}}}}

* ``promote(name)`` runs the family on both seed sets, records the baseline and appends a
  ``scenario_baseline`` ledger entry (revertible).
* ``gate()`` reruns every active family on its HOLDOUT seeds and fails when a family's pass rate
  drops below ``baseline - scenarios.gate.eps``. A family whose generator version changed
  fails too (its cases are no longer the baselined ones: re-promote). A family that cannot
  produce cases on this machine (e.g. real crops not mined here) is reported ``unavailable``.

Holdout seeds are never used for training (``dt train`` optimises on the train seeds only).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Iterable, Optional

from dt.learn import ledger, state
from dt.params import P, register
from dt.scenarios import registry, runner
from dt.scenarios.spec import ScenarioFamily

register("scenarios.gate.eps", 0.0, "scenario gate: allowed drop of a family's holdout pass rate below its baseline", (0.0, 0.5))

HOLDOUT_OFFSET = 100000
DEFAULT_N_TRAIN = 8
DEFAULT_N_HOLDOUT = 10


def _rel(p: Optional[str]) -> Optional[str]:
    return os.path.relpath(p, runner.ROOT) if p else None


def load(path: Optional[str] = None) -> dict:
    path = path or state.suite_path()
    if not os.path.exists(path):
        return {"version": 1, "families": {}}
    with open(path) as f:
        s = json.load(f)
    s.setdefault("families", {})
    return s


def save(s: dict, path: Optional[str] = None) -> str:
    path = path or state.suite_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    s["families"] = dict(sorted(s["families"].items()))
    with open(path, "w") as f:
        json.dump(s, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def seeds_for(family: ScenarioFamily, n_train: int = DEFAULT_N_TRAIN, n_holdout: int = DEFAULT_N_HOLDOUT,
              seed0: int = 0) -> tuple[list[int], list[int]]:
    """Disjoint train / holdout seeds. Finite families (``n_cases``) split their case indices
    (even -> train, odd -> holdout); generated families draw ``seed0..`` and ``HOLDOUT_OFFSET..``."""
    if family.n_cases is not None:
        n = int(family.n_cases())
        idx = list(range(n))
        train, hold = idx[0::2], idx[1::2]
        return train, (hold or train)
    return list(range(seed0, seed0 + n_train)), list(range(HOLDOUT_OFFSET + seed0, HOLDOUT_OFFSET + seed0 + n_holdout))


def promote(name: str, n_train: int = DEFAULT_N_TRAIN, n_holdout: int = DEFAULT_N_HOLDOUT, workers: int = 1,
            out_root: Optional[str] = runner.DEFAULT_OUT_ROOT, run_id: Optional[str] = None, record: bool = True,
            path: Optional[str] = None, verbose: bool = False) -> dict:
    """Add (or re-baseline) ``name`` in the active suite; returns the suite entry."""
    fam = registry.get(name)
    if not fam.available():
        raise RuntimeError(f"family {name} cannot produce cases on this machine")
    train, hold = seeds_for(fam, n_train, n_holdout)
    run_id = run_id or ("promote-" + time.strftime("%Y%m%d-%H%M%S"))
    rt = runner.run_family(fam, train, workers=workers, out_root=out_root, run_id=run_id + "-train", verbose=verbose)
    rh = runner.run_family(fam, hold, workers=workers, out_root=out_root, run_id=run_id + "-holdout", verbose=verbose)
    entry = {"family_version": fam.version, "stage": fam.stage, "refine_iters": fam.refine_iters,
             "train_seeds": train, "holdout_seeds": hold,
             "criteria": [c.to_dict() for c in fam.criteria],
             "baseline": {"train_pass_rate": rt["pass_rate"], "holdout_pass_rate": rh["pass_rate"],
                          "train_objective": round(rt["objective"], 6), "holdout_objective": round(rh["objective"], 6),
                          "n_errors": rt["n_errors"] + rh["n_errors"], "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                          "fingerprints": {str(r["seed"]): r.get("fingerprint") for r in rt["cases"] + rh["cases"]},
                          "reports": [_rel(rt.get("out_dir")), _rel(rh.get("out_dir"))]}}
    s = load(path)
    old = s["families"].get(name)
    s["families"][name] = entry
    save(s, path)
    if record:
        e = ledger.make_entry("scenario_baseline", "global", "scenario", [name], {"suite": {name: [old, entry]}},
                              evidence={"before": (old or {}).get("baseline", {}), "after": entry["baseline"],
                                        "gates": {}}, accepted=True)
        ledger.append(e)
        entry = dict(entry, ledger_id=e["id"])
    return entry


def gate(names: Optional[Iterable[str]] = None, workers: int = 1, overrides: Optional[dict[str, Any]] = None,
         out_root: Optional[str] = None, run_id: Optional[str] = None, eps: Optional[float] = None,
         path: Optional[str] = None, verbose: bool = False) -> dict:
    """Run the active suite (or ``names``) on HOLDOUT seeds; ``{pass, families: {...}, failures}``."""
    s = load(path)
    eps = float(P["scenarios.gate.eps"] if eps is None else eps)
    names = list(names) if names is not None else list(s["families"])
    res: dict[str, dict] = {}
    failures: list[str] = []
    run_id = run_id or ("gate-" + time.strftime("%Y%m%d-%H%M%S"))
    for name in names:
        ent = s["families"].get(name)
        if ent is None:
            failures.append(f"{name}: not in the active suite")
            res[name] = {"status": "missing", "ok": False}
            continue
        try:
            fam = registry.get(name)
        except KeyError as e:
            failures.append(f"{name}: {e}")
            res[name] = {"status": "missing", "ok": False}
            continue
        base = float(ent["baseline"]["holdout_pass_rate"])
        if not fam.available():
            res[name] = {"status": "unavailable", "ok": True, "baseline": base, "current": None}
            continue
        if int(ent.get("family_version", 1)) != fam.version:
            failures.append(f"{name}: generator version {fam.version} != baselined {ent.get('family_version')} (re-promote)")
            res[name] = {"status": "version", "ok": False, "baseline": base, "current": None}
            continue
        rep = runner.run_family(fam, ent["holdout_seeds"], workers=workers, overrides=overrides, out_root=out_root,
                                run_id=run_id, verbose=verbose)
        cur = float(rep["pass_rate"])
        ok = cur >= base - eps - 1e-9
        drift = sorted(str(r["seed"]) for r in rep["cases"]
                       if ent["baseline"].get("fingerprints", {}).get(str(r["seed"])) not in (None, r.get("fingerprint")))
        res[name] = {"status": "ok" if ok else "regressed", "ok": ok, "baseline": base, "current": cur,
                     "delta": cur - base, "objective": rep["objective"],
                     "baseline_objective": ent["baseline"].get("holdout_objective"), "n": rep["n"],
                     "n_errors": rep["n_errors"], "pixel_drift_seeds": drift, "report": rep.get("out_dir"),
                     "failed_cases": [c.get("id") for c in rep["cases"] if not c["passed"]]}
        if not ok:
            failures.append(f"{name}: holdout pass rate {cur:.3f} < baseline {base:.3f} - {eps}")
    return {"pass": not failures, "families": res, "failures": failures, "eps": eps}


def format_gate(g: dict) -> str:
    lines = [f"{'family':16s} {'baseline':>8s} {'current':>8s} {'delta':>7s}  status"]
    for name, r in g["families"].items():
        f = (lambda v: "-" if v is None else f"{v:.3f}")
        d = r.get("delta")
        lines.append(f"{name:16s} {f(r.get('baseline')):>8s} {f(r.get('current')):>8s} "
                     f"{('-' if d is None else f'{d:+.3f}'):>7s}  {r['status']}"
                     + (f" (pixel drift on seeds {r['pixel_drift_seeds']})" if r.get("pixel_drift_seeds") else ""))
    lines.append("scenario gate: " + ("PASS" if g["pass"] else "FAIL"))
    lines += ["  " + x for x in g["failures"]]
    return "\n".join(lines)


def report(path: Optional[str] = None) -> dict:
    """Suite overview: every registered family, whether it is active, and its baseline."""
    s = load(path)
    out = {}
    for name, fam in sorted(registry.families().items()):
        ent = s["families"].get(name)
        out[name] = {"active": ent is not None, "stage": fam.stage, "source": fam.source,
                     "available": bool(fam.available()), "description": fam.description,
                     "baseline": (ent or {}).get("baseline", {}) and {k: v for k, v in ent["baseline"].items()
                                                                     if k not in ("fingerprints", "reports")},
                     "n_train": len((ent or {}).get("train_seeds", [])), "n_holdout": len((ent or {}).get("holdout_seeds", []))}
    return out
