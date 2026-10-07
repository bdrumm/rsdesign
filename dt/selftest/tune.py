"""Self-tuning of ``dt.params`` thresholds against the bench (module E, phase 2).

    from dt.selftest import tune
    result = tune.search(iters=30, corpus_dirs=[...], holdout_frac=0.25, seed=0)

Search space = every registered param that has a range (``P.ranges()``), minus the keys that
define the objective itself (``bench.*``, ``tune.*``, ``compare.*``, ``validate.*``,
``selftest.*``), so the tuner cannot "improve" the score by loosening the metrics. Use
``include`` to restrict to a prefix such as ``perceive.ocr.``.

Algorithm (budget = ``iters`` bench evaluations on the train split):

1. baseline: current params on train and on holdout;
2. random phase (``tune.random_frac`` of the budget): each candidate perturbs the incumbent on
   a random subset of keys (``tune.subset_frac``) with Gaussian noise of ``tune.sigma`` x range
   (``uniform_first=True`` makes the first candidate a uniform draw over the whole space);
3. coordinate refinement (rest of the budget): cycle the keys in seeded order and try
   incumbent +/- ``tune.coord_step`` x range, accepting any train improvement.

The objective is the bench composite on the train split (``bench.run`` with ``out_root=None``).
The incumbent is then scored on the holdout split; ``dt/params.json`` (or ``params_path``) is
written only when the holdout composite beats the baseline holdout by more than
``tune.min_improve`` (train is used when there is no holdout case). Every evaluation is
appended to ``out/tune/history.jsonl``. The process' live params are restored afterwards.

CLI: ``python -m dt.selftest.tune --iters 30``; ``--dry`` only prints the search space.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Iterable, Optional, Sequence

from dt.params import P, PARAMS_PATH, register
from dt.selftest import bench

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HISTORY_PATH = os.path.join(ROOT, "out", "tune", "history.jsonl")
OBJECTIVE_PREFIXES: tuple[str, ...] = ("bench.", "tune.", "compare.", "validate.", "selftest.")
"""Params that define the objective or the corpora; never tuned."""

register("tune.random_frac", 0.5, "Fraction of the evaluation budget spent on random sampling "
         "(the rest on coordinate refinement).", (0.0, 1.0))
register("tune.subset_frac", 0.3, "Fraction of keys perturbed per random-phase candidate "
         "(after the first, fully random, candidate).", (0.05, 1.0))
register("tune.sigma", 0.2, "Gaussian perturbation scale as a fraction of each key's range.", (0.02, 1.0))
register("tune.coord_step", 0.25, "Coordinate-refinement step as a fraction of the range.", (0.02, 0.5))
register("tune.min_improve", 0.002, "Minimum holdout composite gain required to write params.json.", (0.0, 0.05))


# ----------------------------------------------------------------------------- search space
def search_space(include: Optional[Iterable[str]] = None,
                 exclude: Iterable[str] = OBJECTIVE_PREFIXES) -> dict[str, dict[str, Any]]:
    """Tunable keys -> ``{lo, hi, default, current, kind ('int'|'float'), doc}``.

    ``include``: key prefixes to keep (None = all); ``exclude``: prefixes to drop. Keys whose
    default is not numeric (or is a bool) are never tunable.
    """
    _import_stages()
    ranges, defaults, current, docs = P.ranges(), P.defaults(), P.all(), P.docs()
    inc = tuple(include) if include else None
    exc = tuple(exclude or ())
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(ranges):
        if inc and not key.startswith(inc):
            continue
        if key.startswith(exc):
            continue
        d = defaults.get(key)
        if isinstance(d, bool) or not isinstance(d, (int, float)):
            continue
        lo, hi = ranges[key]
        out[key] = {"lo": float(lo), "hi": float(hi), "default": d, "current": current.get(key, d),
                    "kind": "int" if isinstance(d, int) else "float", "doc": docs.get(key, "")}
    return out


def _import_stages() -> None:
    """Import the modules that register tunables so the space is complete."""
    for mod in ("dt.perceive", "dt.mapping", "dt.compare"):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001 - a missing optional stage just shrinks the space
            pass


def _cast(spec: dict, v: float) -> float | int:
    v = min(spec["hi"], max(spec["lo"], float(v)))
    return int(round(v)) if spec["kind"] == "int" else round(float(v), 6)


def sample_uniform(space: dict[str, dict], rng: random.Random) -> dict[str, Any]:
    """One candidate with every key drawn uniformly from its range."""
    return {k: _cast(s, rng.uniform(s["lo"], s["hi"])) for k, s in space.items()}


def perturb(space: dict[str, dict], base: dict[str, Any], rng: random.Random,
            subset_frac: Optional[float] = None, sigma: Optional[float] = None) -> dict[str, Any]:
    """Copy of ``base`` with a random subset of keys moved by Gaussian noise (fraction of range)."""
    subset_frac = float(P["tune.subset_frac"] if subset_frac is None else subset_frac)
    sigma = float(P["tune.sigma"] if sigma is None else sigma)
    keys = list(space)
    k = max(1, int(round(subset_frac * len(keys))))
    out = dict(base)
    for key in rng.sample(keys, min(k, len(keys))):
        s = space[key]
        out[key] = _cast(s, float(base[key]) + rng.gauss(0.0, sigma * (s["hi"] - s["lo"])))
    return out


def split_cases(cases: Sequence[bench.Case], holdout_frac: float, seed: int) -> tuple[list, list]:
    """Seeded train/holdout split (holdout = round(frac * n), at least 1 when 0 < frac < 1 and n >= 2)."""
    cases = list(cases)
    rng = random.Random(seed)
    rng.shuffle(cases)
    n_hold = int(round(holdout_frac * len(cases)))
    if 0 < holdout_frac < 1 and len(cases) >= 2:
        n_hold = min(len(cases) - 1, max(1, n_hold))
    return cases[n_hold:], cases[:n_hold]


# ----------------------------------------------------------------------------- objective
def objective(cases: Sequence[bench.Case], overrides: dict[str, Any], stages: Sequence[str] = bench.ALL_STAGES,
              workers: int = 1) -> float:
    """Bench composite (train objective) for ``overrides`` on ``cases``; 0 when there are no cases."""
    if not cases:
        return 0.0
    rep = bench.run(cases=cases, stages=stages, workers=workers, out_root=None, overrides=overrides)
    return float(rep["composite"])


def _append_history(path: Optional[str], entry: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def _diff(space: dict, cand: dict) -> dict:
    """Only the keys of ``cand`` that differ from the live value (for compact logs)."""
    return {k: v for k, v in cand.items() if v != space[k]["current"]}


def write_params(best: dict[str, Any], path: str = PARAMS_PATH) -> str:
    """Merge ``best`` into the params file at ``path`` (keeps unrelated overrides)."""
    data: dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:  # noqa: BLE001 - a corrupt file is replaced
            data = {}
    data.update(best)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return path


# ----------------------------------------------------------------------------- search
def search(iters: int, corpus_dirs: Iterable[str] = bench.DEFAULT_CORPORA, holdout_frac: float = 0.25, seed: int = 0,
           limit: Optional[int] = None, stages: Sequence[str] = bench.ALL_STAGES, workers: int = 1,
           include: Optional[Iterable[str]] = None, params_path: str = PARAMS_PATH,
           history_path: Optional[str] = HISTORY_PATH, cases: Optional[Sequence[bench.Case]] = None,
           uniform_first: bool = False, verbose: bool = True) -> dict:
    """Random search + coordinate refinement over the tunable params (see module docstring).

    ``uniform_first=True`` makes the first random candidate a uniform draw over the whole space
    (a global probe); by default every candidate is a local perturbation of the incumbent,
    which suits a pipeline whose params are already calibrated.

    Returns ``{space_size, n_train, n_holdout, baseline: {train, holdout}, best: {train, holdout,
    params (only changed keys)}, evaluations, written (bool), params_path, history_path,
    elapsed_s}``.
    """
    t0 = time.perf_counter()
    space = search_space(include)
    keys = list(space)
    all_cases = list(cases) if cases is not None else bench.find_cases(corpus_dirs, limit)
    train, hold = split_cases(all_cases, holdout_frac, seed)
    rng = random.Random(seed)
    run_tag = time.strftime("%Y%m%d-%H%M%S") + f"-s{seed}"
    log = (lambda *a: print(*a, file=sys.stderr, flush=True)) if verbose else (lambda *a: None)

    def score_holdout(cand: dict) -> Optional[float]:
        return objective(hold, cand, stages, workers) if hold else None

    evaluations = 0

    def evaluate(cand: dict, phase: str, i: int) -> float:
        nonlocal evaluations
        s = objective(train, cand, stages, workers)
        evaluations += 1
        _append_history(history_path, {"run": run_tag, "phase": phase, "iter": i, "train": s,
                                       "params": _diff(space, cand), "ts": time.time()})
        return s

    base = {k: space[k]["current"] for k in keys}
    base_train = evaluate(base, "baseline", 0)
    base_hold = score_holdout(base)
    log(f"[tune] {len(keys)} keys, train={len(train)} holdout={len(hold)}  baseline train={base_train:.4f} "
        f"holdout={'-' if base_hold is None else f'{base_hold:.4f}'}")
    best, best_train = dict(base), base_train

    n_random = int(round(float(P["tune.random_frac"]) * iters))
    for i in range(n_random):
        cand = sample_uniform(space, rng) if (i == 0 and uniform_first) else perturb(space, best, rng)
        s = evaluate(cand, "random", i)
        if s > best_train + 1e-9:
            best, best_train = cand, s
            log(f"[tune] random {i}: train {s:.4f} (new best)")
    n_coord = iters - n_random
    order = list(keys)
    rng.shuffle(order)
    step = float(P["tune.coord_step"])
    i = 0
    while i < n_coord and order:
        key = order[i % len(order)]
        s_ = space[key]
        delta = step * (s_["hi"] - s_["lo"]) * rng.choice((-1.0, 1.0))
        cand = dict(best)
        cand[key] = _cast(s_, float(best[key]) + delta)
        if cand[key] != best[key]:
            s = evaluate(cand, "coord", i)
            if s > best_train + 1e-9:
                best, best_train = cand, s
                log(f"[tune] coord {i} {key}={cand[key]}: train {s:.4f} (new best)")
        i += 1

    changed = {k: v for k, v in best.items() if v != base[k]}
    best_hold = base_hold if not changed else score_holdout(best)
    ref_base = base_hold if base_hold is not None else base_train
    ref_best = best_hold if best_hold is not None else best_train
    written = bool(changed) and (ref_best > ref_base + float(P["tune.min_improve"]))
    if written:
        write_params(changed, params_path)
        log(f"[tune] wrote {len(changed)} params to {params_path}")
    summary = {"run": run_tag, "phase": "summary", "space_size": len(keys), "n_train": len(train), "n_holdout": len(hold),
               "baseline": {"train": base_train, "holdout": base_hold},
               "best": {"train": best_train, "holdout": best_hold, "params": changed},
               "evaluations": evaluations, "written": written, "params_path": params_path if written else None,
               "history_path": history_path, "elapsed_s": round(time.perf_counter() - t0, 2), "ts": time.time()}
    _append_history(history_path, summary)
    log(f"[tune] best train={best_train:.4f} holdout={'-' if best_hold is None else f'{best_hold:.4f}'} "
        f"changed={len(changed)} written={written} in {summary['elapsed_s']}s")
    return summary


# ----------------------------------------------------------------------------- CLI
def format_space(space: dict[str, dict]) -> str:
    lines = [f"{len(space)} tunable params"]
    for k, s in space.items():
        lines.append(f"  {k:40s} {s['kind']:5s} [{s['lo']:g}, {s['hi']:g}]  current={s['current']!r:10}  {s['doc']}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="dt.selftest.tune", description=__doc__.split("\n\n")[0])
    ap.add_argument("--iters", type=int, default=20, help="bench evaluations on the train split")
    ap.add_argument("--corpus", nargs="*", default=list(bench.DEFAULT_CORPORA))
    ap.add_argument("--holdout", type=float, default=0.25, help="holdout fraction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="cases per corpus")
    ap.add_argument("--stages", default=",".join(bench.ALL_STAGES))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--include", nargs="*", default=None, help="key prefixes to tune (default: all)")
    ap.add_argument("--params", default=PARAMS_PATH, help="params.json to write on improvement")
    ap.add_argument("--history", default=HISTORY_PATH)
    ap.add_argument("--uniform-first", action="store_true", help="first candidate = uniform draw over the space")
    ap.add_argument("--dry", action="store_true", help="only print the search space")
    a = ap.parse_args(argv)
    if a.dry:
        print(format_space(search_space(a.include)))
        return 0
    res = search(a.iters, a.corpus, a.holdout, a.seed, limit=a.limit,
                 stages=tuple(s.strip() for s in a.stages.split(",") if s.strip()), workers=a.workers,
                 include=a.include, params_path=a.params, history_path=a.history, uniform_first=a.uniform_first)
    print(json.dumps({k: v for k, v in res.items() if k != "ts"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
