"""The self-tuning loop: tune ``dt.params`` on one failure family, keep the change only if it
generalises.

    dt train --family pale_surface [--params perceive.seg.,refine.critic.] --iters 12 [--local]
    dt train --report-generalisation [--iters 6]      # leave-one-family-out matrix (writes nothing)

1. search space = the family's ``tune_prefixes`` (or ``--params``) minus everything that defines a
   metric or a gate (``bench.``, ``compare.``, ``validate.``, ``selftest.``, ``tune.``, ``scenarios.``);
2. objective = mean clipped criterion margin over the family's TRAIN seeds
   (:func:`dt.selftest.tune.optimize`: random perturbations + coordinate refinement);
3. guards, in order (cheapest first, stop at the first failure):
   ``holdout``   -- the family's HOLDOUT pass rate must strictly improve over the current params;
   ``scenarios`` -- every other active family keeps its holdout pass rate >= baseline - eps;
   ``bench``     -- the global regression gate (``knowledge/baseline.json``) passes;
4. accepted -> written to ``dt/params.json`` (scope global) or ``$DT_HOME/params.local.json``
   (``--local``) through :func:`dt.learn.state.write_layer`, plus a ``params`` ledger entry with
   the exact diff and all the evidence (revert with ``dt learn revert <id>``). Rejected -> nothing
   is written (``--record-rejected`` logs the attempt in the ledger with ``accepted: false``).

Generalisation report (overfitting signal): for the candidate change, every other family's
holdout pass rate and objective before/after, and the bench composite delta.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable, Iterable, Optional

from dt.learn import ledger, state
from dt.params import P, register
from dt.scenarios import registry, runner, suite

register("scenarios.train.max_keys", 3,
         "dt train: keys perturbed together per random candidate (sparse changes generalise better)", (1, 20))
register("scenarios.train.min_gain", 0.01,
         "dt train: minimum train-objective gain (mean clipped margin) before a change is even gated", (0.0, 0.5))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HISTORY_PATH = os.path.join(ROOT, "out", "scenarios", "train", "history.jsonl")
EXCLUDE = ("bench.", "tune.", "compare.", "validate.", "selftest.", "scenarios.")

BenchGate = Callable[[dict[str, Any]], tuple[bool, dict]]


def default_bench_gate(workers: int = 3) -> BenchGate:
    """Global regression gate for a candidate change: bench with the overrides vs knowledge/baseline.json."""
    def gate(changed: dict[str, Any]) -> tuple[bool, dict]:
        from dt.selftest import bench
        base = bench.load_baseline()
        if base is None:
            return False, {"failures": ["no knowledge/baseline.json"]}
        rep = bench.run(overrides=changed, workers=workers, out_root=None)
        fails = bench.check_regression(rep, base)
        return not fails, {"before": {"composite": base["composite"]}, "after": {"composite": rep["composite"]},
                           "failures": fails[:10]}
    return gate


def family_seeds(name: str, n_train: Optional[int] = None, n_holdout: Optional[int] = None) -> tuple[list[int], list[int]]:
    ent = suite.load()["families"].get(name)
    if ent and n_train is None and n_holdout is None:
        return list(ent["train_seeds"]), list(ent["holdout_seeds"])
    return suite.seeds_for(registry.get(name), n_train or suite.DEFAULT_N_TRAIN, n_holdout or suite.DEFAULT_N_HOLDOUT)


def generalisation(changed: dict[str, Any], exclude: Iterable[str] = (), workers: int = 1,
                   bench_gate: Optional[BenchGate] = None, names: Optional[Iterable[str]] = None) -> dict:
    """How ``changed`` moves every (other) active family's holdout pass rate / objective, and the bench."""
    s = suite.load()
    exclude = set(exclude)
    out: dict[str, Any] = {"families": {}}
    for name in (list(names) if names is not None else list(s["families"])):
        if name in exclude:
            continue
        try:
            fam = registry.get(name)
        except KeyError:
            continue
        if not fam.available():
            out["families"][name] = {"status": "unavailable"}
            continue
        _tr, hold = family_seeds(name)
        b = runner.run_family(fam, hold, workers=workers, out_root=None)
        a = runner.run_family(fam, hold, workers=workers, overrides=changed, out_root=None)
        out["families"][name] = {"before": b["pass_rate"], "after": a["pass_rate"], "delta": a["pass_rate"] - b["pass_rate"],
                                 "objective_before": round(b["objective"], 4), "objective_after": round(a["objective"], 4)}
    if bench_gate is not None:
        ok, ev = bench_gate(changed)
        out["bench"] = {"pass": ok, **ev}
    return out


def train(family: str, prefixes: Optional[Iterable[str]] = None, iters: int = 8, local: bool = False, seed: int = 0,
          workers: int = 1, bench_gate: Optional[BenchGate] = None, bench_workers: int = 3,
          report_generalisation: bool = False, record_rejected: bool = False, dry: bool = False,
          n_train: Optional[int] = None, n_holdout: Optional[int] = None, history_path: Optional[str] = HISTORY_PATH,
          verbose: bool = True) -> dict:
    """Tune on ``family``'s train seeds; keep the change only if every guard passes (module doc)."""
    from dt.selftest import tune
    t0 = time.perf_counter()
    log = (lambda *a: print(*a, file=sys.stderr, flush=True)) if verbose else (lambda *a: None)
    fam = registry.get(family)
    if not fam.available():
        raise RuntimeError(f"family {family} cannot produce cases on this machine")
    prefixes = tuple(prefixes) if prefixes else tuple(fam.tune_prefixes)
    _import_all_stages()
    space = tune.search_space(include=prefixes or None, exclude=EXCLUDE)
    if not space:
        raise ValueError(f"empty search space for prefixes {prefixes}")
    train_seeds, hold_seeds = family_seeds(family, n_train, n_holdout)
    scope = "local" if local else "global"
    log(f"[train] {family}: {len(space)} keys ({', '.join(prefixes)}), train seeds {train_seeds}, "
        f"holdout seeds {hold_seeds}, iters {iters}, scope {scope}")

    def objective(cand: dict) -> float:
        rows = runner.run_cases(fam, train_seeds, overrides=cand, workers=workers)
        return float(sum(r["margin_mean"] for r in rows) / max(1, len(rows)))

    subset = min(1.0, max(1, int(P["scenarios.train.max_keys"])) / len(space))
    opt = tune.optimize(space, objective, iters, seed, history_path=history_path, log=log, subset_frac=subset)
    changed, best_score = prune(opt, objective, log)
    opt["best_score"] = best_score
    result: dict[str, Any] = {"family": family, "scope": scope, "space_size": len(space), "prefixes": list(prefixes),
                              "train_seeds": train_seeds, "holdout_seeds": hold_seeds,
                              "train_objective": {"before": opt["base_score"], "after": opt["best_score"]},
                              "changed": changed, "evaluations": opt["evaluations"], "accepted": False,
                              "written": False, "ledger_id": None, "guards": {}}
    if changed and best_score < opt["base_score"] + float(P["scenarios.train.min_gain"]):
        result["reason"] = (f"train gain {best_score - opt['base_score']:+.4f} below scenarios.train.min_gain "
                            f"{P['scenarios.train.min_gain']}")
        result["candidate"], result["changed"] = changed, {}
        if report_generalisation:
            result["generalisation"] = generalisation(changed, exclude=[family], workers=workers,
                                                      bench_gate=bench_gate or default_bench_gate(bench_workers))
        log(f"[train] {result['reason']}")
        _save_result(result, t0)
        return result
    if not changed:
        result["reason"] = "no candidate improved the train objective"
        log(f"[train] {result['reason']}")
        _save_result(result, t0)
        return result

    hold_eval: dict[str, dict] = {}

    def holdout_guard(ch: dict) -> tuple[bool, dict]:
        b = runner.run_family(fam, hold_seeds, workers=workers, out_root=None)
        a = runner.run_family(fam, hold_seeds, workers=workers, overrides=ch, out_root=None)
        hold_eval.update(before=b, after=a)
        return a["pass_rate"] > b["pass_rate"] + 1e-9, {
            "before": {"pass_rate": b["pass_rate"], "objective": round(b["objective"], 4)},
            "after": {"pass_rate": a["pass_rate"], "objective": round(a["objective"], 4)}}

    def scenarios_guard(ch: dict) -> tuple[bool, dict]:
        others = [n for n in suite.load()["families"] if n != family]
        if not others:
            return True, {"families": {}}
        g = suite.gate(others, workers=workers, overrides=ch)
        return g["pass"], {"families": {k: {kk: v.get(kk) for kk in ("baseline", "current", "status")}
                                        for k, v in g["families"].items()}, "failures": g["failures"]}

    bgate = bench_gate or default_bench_gate(bench_workers)
    guards = [("holdout", holdout_guard), ("scenarios", scenarios_guard), ("bench", bgate)]
    ok, ev = tune.check_guards(changed, guards, short_circuit=not report_generalisation)
    result["guards"] = ev
    gates = {k: v["pass"] for k, v in ev.items()}
    for k, _fn in guards:
        gates.setdefault(k, None)
    log("[train] guards: " + ", ".join(f"{k}={'-' if v is None else ('ok' if v else 'FAIL')}" for k, v in gates.items()))
    gen = None
    if report_generalisation:
        gen = generalisation(changed, exclude=[family], workers=workers)
        gen["bench"] = ev.get("bench")
        result["generalisation"] = gen
    result["accepted"] = bool(ok)
    evidence = {"before": {"train_objective": opt["base_score"], **(ev.get("holdout", {}).get("before") or {})},
                "after": {"train_objective": opt["best_score"], **(ev.get("holdout", {}).get("after") or {})},
                "gates": {k: v for k, v in gates.items() if v is not None},
                "guards": ev, "family": family, "train_seeds": train_seeds, "holdout_seeds": hold_seeds,
                "iters": iters, "prefixes": list(prefixes)}
    if gen is not None:
        evidence["generalisation"] = gen
    if ok and not dry:
        diff = state.write_layer(scope, changed)
        e = ledger.make_entry("params", scope, "scenario", [family], {"params": diff, "layer": scope},
                              evidence=evidence, accepted=True)
        ledger.append(e)
        result.update(written=True, ledger_id=e["id"], params_path=state.layer_path(scope))
        log(f"[train] ACCEPTED: wrote {len(diff)} params to {state.layer_path(scope)}; ledger {e['id']}")
    elif not ok and record_rejected and not dry:
        cur = P.all()
        e = ledger.make_entry("params", scope, "scenario", [family],
                              {"params": {k: [cur.get(k), v] for k, v in changed.items()}, "layer": scope, "applied": False},
                              evidence=evidence, accepted=False)
        ledger.append(e)
        result["ledger_id"] = e["id"]
    if not ok:
        result["reason"] = "guard failed: " + ", ".join(k for k, v in gates.items() if v is False)
        log(f"[train] rejected ({result['reason']}); nothing written")
    _save_result(result, t0)
    return result


def _import_all_stages() -> None:
    """Register every stage's tunables (``tune.search_space`` only imports perceive/map/compare,
    the stages ``dt tune``'s bench objective exercises; scenario families also run refine)."""
    import importlib
    for mod in ("dt.perceive", "dt.perceive.icons", "dt.mapping", "dt.refine.critic", "dt.refine.optimizer", "dt.pipeline"):
        try:
            importlib.import_module(mod)
        except Exception:  # noqa: BLE001 - an unavailable optional stage just shrinks the space
            pass


def prune(opt: dict, objective: Callable[[dict], float], log: Callable[..., None]) -> tuple[dict, float]:
    """Occam step: drop every changed key whose reversal does not lower the train objective, so
    the change that reaches the gates is the smallest one that explains the gain."""
    best, score, base = dict(opt["best"]), float(opt["best_score"]), opt["base"]
    for k in sorted(opt["changed"]):
        cand = dict(best)
        cand[k] = base[k]
        s = float(objective(cand))
        if s >= score - 1e-9:
            best, score = cand, max(score, s)
            log(f"[train] prune {k}: not needed")
    return {k: v for k, v in best.items() if v != base[k]}, score


def _save_result(result: dict, t0: float) -> None:
    result["elapsed_s"] = round(time.perf_counter() - t0, 1)
    try:
        import json
        d = os.path.join(ROOT, "out", "scenarios", "train")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{time.strftime('%Y%m%d-%H%M%S')}-{result['family']}.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
    except OSError:
        pass


def leave_one_family_out(iters: int = 6, workers: int = 1, seed: int = 0, bench_gate: Optional[BenchGate] = None,
                         names: Optional[Iterable[str]] = None, verbose: bool = True) -> dict:
    """For every active family F: tune on F (nothing written) and measure how the best change moves
    every other family and the bench. Large gains on F with losses elsewhere = overfitting."""
    s = suite.load()
    names = list(names) if names is not None else list(s["families"])
    matrix: dict[str, Any] = {}
    for f in names:
        fam = registry.get(f)
        if not fam.available():
            matrix[f] = {"status": "unavailable"}
            continue
        res = train(f, iters=iters, workers=workers, seed=seed, dry=True, report_generalisation=False,
                    bench_gate=lambda ch: (True, {"skipped": True}), verbose=verbose)
        row: dict[str, Any] = {"changed": res["changed"], "train_objective": res["train_objective"],
                               "holdout": res["guards"].get("holdout")}
        if res["changed"]:
            row["others"] = generalisation(res["changed"], exclude=[f], workers=workers, bench_gate=bench_gate)
        matrix[f] = row
    return {"matrix": matrix, "iters": iters}
