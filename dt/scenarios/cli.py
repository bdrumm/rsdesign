"""CLI for the scenario harness and the learning ledger (registered into ``dt`` by
``dt.cli.build_parser`` through :func:`register`).

    dt scenario list                          dt scenario run pale_surface --n 8 --seed 0
    dt scenario mine out/run1 out/run2        dt scenario promote pale_surface [--n-train 8 --n-holdout 10]
    dt scenario report                        dt scenario gate [--families a,b]
    dt train --family F [--params p.,q.] --iters N [--local] [--report-generalisation]
    dt train --report-generalisation [--iters N]     (leave-one-family-out matrix; writes nothing)
    dt learn log | show <id> | revert <id>
    dt bench --gate --scenarios               (also gates the active suite on its holdout seeds)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from typing import Any, Callable

EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _emit(res: Any, as_json: bool, human: Callable[[Any], str]) -> None:
    print(json.dumps(res, indent=2, default=str) if as_json else human(res))


def _csv(v: str | None) -> list[str]:
    return [x.strip() for x in (v or "").split(",") if x.strip()]


# --------------------------------------------------------------------------- scenario
def cmd_scenario(a: argparse.Namespace) -> int:
    from dt.scenarios import registry, runner, suite
    c = a.scenario_cmd
    if c == "list":
        rep = suite.report()
        errs = registry.errors()
        res = {"families": rep, "errors": errs}

        def human(r: dict) -> str:
            L = [f"{'family':16s} {'stage':9s} {'source':6s} {'active':6s} {'avail':5s} {'holdout pass':>12s}  description"]
            for n, f in r["families"].items():
                hp = (f["baseline"] or {}).get("holdout_pass_rate")
                L.append(f"{n:16s} {f['stage']:9s} {f['source']:6s} {str(f['active']):6s} {str(f['available']):5s} "
                         f"{'-' if hp is None else f'{hp:.3f}':>12s}  {f['description'][:80]}")
            L += [f"import error {k}: {v}" for k, v in r["errors"].items()]
            return "\n".join(L)
        _emit(res, a.json, human)
        return EXIT_OK
    if c == "run":
        fam = registry.get(a.family)
        if a.seeds:
            seeds = [int(s) for s in _csv(a.seeds)]
        elif a.holdout:
            seeds = suite.seeds_for(fam, a.n, a.n)[1]
        else:
            seeds = list(range(a.seed, a.seed + a.n))
        rep = runner.run_family(fam, seeds, stage=a.stage, refine_iters=a.refine_iters, workers=a.workers,
                                out_root=(a.out or None), run_id=a.run_id, verbose=not a.json)
        res = {k: v for k, v in rep.items() if k != "cases"}
        if a.cases:
            res["cases"] = rep["cases"]
        _emit(res, a.json, lambda r: f"{a.family}: pass rate {r['pass_rate']:.3f} ({r['n_pass']}/{r['n']}), objective "
                                     f"{r['objective']:+.4f}, errors {r['n_errors']} -> {r.get('out_dir')}\n"
                                     + "\n".join(f"  {k}: pass {v['pass_rate']:.3f} mean {v['mean']}" for k, v in r["criteria"].items()))
        return EXIT_OK
    if c == "mine":
        from dt.scenarios import mine
        res = mine.mine(a.run_dirs, per_run=a.per_run, dry=a.dry)
        _emit(res, a.json, lambda r: f"mined {len(r['added'])} crops ({r['n_crops']} in manifest {r['manifest']}); "
                                     f"skipped {r['skipped']}")
        return EXIT_OK
    if c == "promote":
        ent = suite.promote(a.family, a.n_train, a.n_holdout, workers=a.workers, verbose=not a.json)
        _emit(ent, a.json, lambda e: f"{a.family}: baseline train pass {e['baseline']['train_pass_rate']:.3f}, holdout pass "
                                     f"{e['baseline']['holdout_pass_rate']:.3f} -> {os.path.relpath(suite.state.suite_path())}"
                                     f" (ledger {e.get('ledger_id')})")
        return EXIT_OK
    if c == "report":
        res = suite.report()
        _emit(res, a.json, lambda r: "\n".join(
            f"{n:16s} active={f['active']!s:5s} stage={f['stage']:9s} n_train={f['n_train']} n_holdout={f['n_holdout']} "
            f"baseline={json.dumps(f['baseline'], default=str)}" for n, f in r.items()))
        return EXIT_OK
    if c == "gate":
        g = suite.gate(_csv(a.families) or None, workers=a.workers, out_root=(a.out or None), verbose=not a.json)
        _emit(g, a.json, suite.format_gate)
        return EXIT_OK if g["pass"] else EXIT_FAIL
    return EXIT_USAGE


# --------------------------------------------------------------------------- train
def cmd_train(a: argparse.Namespace) -> int:
    from dt.scenarios import train
    if a.family is None:
        if not a.report_generalisation:
            _log("dt train needs --family F (or --report-generalisation for the leave-one-family-out matrix)")
            return EXIT_USAGE
        bg = None if a.no_bench else train.default_bench_gate(a.bench_workers)
        res = train.leave_one_family_out(iters=a.iters, workers=a.workers, seed=a.seed, bench_gate=bg, verbose=not a.json)

        def human(r: dict) -> str:
            L = []
            for f, row in r["matrix"].items():
                if row.get("status"):
                    L.append(f"{f}: {row['status']}")
                    continue
                L.append(f"tuned on {f}: changed {row['changed']} train obj {row['train_objective']}")
                L.append(f"  holdout: {json.dumps(row.get('holdout'), default=str)}")
                for o, v in (row.get("others") or {}).get("families", {}).items():
                    L.append(f"  -> {o}: {json.dumps(v, default=str)}")
                if (row.get("others") or {}).get("bench"):
                    L.append(f"  -> bench: {json.dumps(row['others']['bench'], default=str)}")
            return "\n".join(L)
        _emit(res, a.json, human)
        return EXIT_OK
    bench_gate = None
    if a.no_bench:  # without the global gate nothing may be written
        bench_gate, a.dry = (lambda ch: (True, {"skipped": True})), True
        _log("[train] --no-bench: dry run, nothing will be written")
    res = train.train(a.family, prefixes=_csv(a.params) or None, iters=a.iters, local=a.local, seed=a.seed,
                      workers=a.workers, bench_gate=bench_gate, bench_workers=a.bench_workers,
                      report_generalisation=a.report_generalisation, record_rejected=a.record_rejected, dry=a.dry,
                      n_train=a.n_train, n_holdout=a.n_holdout, verbose=not a.json)
    _emit(res, a.json, lambda r: json.dumps({k: r.get(k) for k in ("family", "scope", "changed", "train_objective", "accepted",
                                                                   "written", "ledger_id", "reason", "guards", "generalisation",
                                                                   "elapsed_s") if k in r}, indent=2, default=str))
    return EXIT_OK


# --------------------------------------------------------------------------- learn
def cmd_learn(a: argparse.Namespace) -> int:
    from dt.learn import ledger
    if a.learn_cmd == "log":
        es = ledger.read()
        if a.kind:
            es = [e for e in es if e.get("kind") == a.kind]
        if a.scope:
            es = [e for e in es if e.get("scope") == a.scope]
        es = es[-a.n:] if a.n else es
        _emit(es, a.json, lambda r: "\n".join(ledger.format_entry(e) for e in r) or "(ledger empty)")
        return EXIT_OK
    if a.learn_cmd == "show":
        e = ledger.get(a.id)
        if e is None:
            _log(f"no ledger entry {a.id!r}")
            return EXIT_FAIL
        rb = ledger.reverted_by(e["id"])
        _emit(dict(e, reverted_by=rb["id"] if rb else None), a.json, lambda r: ledger.format_entry(e, verbose=True))
        return EXIT_OK
    if a.learn_cmd == "revert":
        try:
            rev = ledger.revert(a.id, force=a.force, note=a.note or "")
        except ValueError as e:
            _log(f"revert refused: {e}")
            return EXIT_FAIL
        _emit(rev, a.json, lambda r: "reverted: " + ledger.format_entry(r))
        return EXIT_OK
    return EXIT_USAGE


# --------------------------------------------------------------------------- bench --scenarios
def _wrap_bench(p: argparse.ArgumentParser) -> None:
    orig = p.get_default("fn")
    p.add_argument("--scenarios", action="store_true",
                   help="also run the active scenario suite (knowledge/scenarios.json) on its holdout seeds; "
                        "exit 1 if any family's pass rate drops below baseline - scenarios.gate.eps")
    p.add_argument("--scenario-workers", type=int, default=None, help="workers for the scenario gate (default --workers)")

    def cmd_bench_scenarios(a: argparse.Namespace) -> int:
        if not getattr(a, "scenarios", False):
            return orig(a)
        from dt.selftest import bench
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = orig(a)
        g = bench.scenario_gate(workers=a.scenario_workers or a.workers, verbose=not a.json and not a.quiet)
        out = buf.getvalue()
        if a.json:
            try:
                res = json.loads(out)
            except json.JSONDecodeError:
                res = {"bench_output": out}
            res["scenario_gate"] = g
            print(json.dumps(res, indent=2, default=str))
            _log(g["text"])
        else:
            sys.stdout.write(out)
            print(g["text"])
        return rc if (rc != EXIT_OK or g["pass"]) else EXIT_FAIL
    p.set_defaults(fn=cmd_bench_scenarios)


# --------------------------------------------------------------------------- registration
def register(add: Callable[..., argparse.ArgumentParser], sub: argparse._SubParsersAction) -> None:
    p = add("scenario", cmd_scenario, "failure-scenario harness: list | run | mine | promote | report | gate")
    sc = p.add_subparsers(dest="scenario_cmd", required=True)

    def sp(name: str, help_: str) -> argparse.ArgumentParser:
        q = sc.add_parser(name, help=help_, description=help_)
        q.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="emit one JSON object on stdout")
        return q

    sp("list", "registered families, whether active, baseline holdout pass rate")
    q = sp("run", "run a family on seeds: per-case metrics, pass rate, margins, worst cases with diff images")
    q.add_argument("family"); q.add_argument("--n", type=int, default=8); q.add_argument("--seed", type=int, default=0)
    q.add_argument("--seeds", help="explicit comma-separated seeds"); q.add_argument("--holdout", action="store_true",
                                                                                    help="use the holdout seed range")
    q.add_argument("--stage", choices=("perceive", "map", "translate")); q.add_argument("--refine-iters", type=int)
    q.add_argument("--workers", type=int, default=1); q.add_argument("--out", default="out/scenarios")
    q.add_argument("--run-id"); q.add_argument("--cases", action="store_true", help="include per-case rows in JSON")
    q = sp("mine", "translate run dirs -> real-crop cases ($DT_HOME/scenarios/real, manifest in knowledge/)")
    q.add_argument("run_dirs", nargs="+"); q.add_argument("--per-run", type=int); q.add_argument("--dry", action="store_true")
    q = sp("promote", "add a family to the active suite with its baseline pass rates (ledger: scenario_baseline)")
    q.add_argument("family"); q.add_argument("--n-train", type=int, default=8); q.add_argument("--n-holdout", type=int, default=10)
    q.add_argument("--workers", type=int, default=1)
    sp("report", "the active suite and every family's baseline")
    q = sp("gate", "run the active suite on holdout seeds; exit 1 on a pass-rate regression")
    q.add_argument("--families"); q.add_argument("--workers", type=int, default=1); q.add_argument("--out", default="")

    p = add("train", cmd_train, "self-tuning on a scenario family: train seeds objective, holdout + suite + bench gates, ledger")
    p.add_argument("--family"); p.add_argument("--params", help="comma-separated param key prefixes (default: the family's)")
    p.add_argument("--iters", type=int, default=8); p.add_argument("--local", action="store_true",
                                                                  help="write $DT_HOME/params.local.json (scope local)")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--workers", type=int, default=1)
    p.add_argument("--bench-workers", type=int, default=3)
    p.add_argument("--report-generalisation", action="store_true",
                   help="report how the change moves every other family and the bench (leave-one-family-out)")
    p.add_argument("--record-rejected", action="store_true", help="log rejected attempts in the ledger (accepted: false)")
    p.add_argument("--dry", action="store_true", help="search and gate but write nothing")
    p.add_argument("--no-bench", action="store_true", help="skip the global bench gate (never use for an accepted change)")
    p.add_argument("--n-train", type=int); p.add_argument("--n-holdout", type=int)

    p = add("learn", cmd_learn, "the shared learning ledger: log | show <id> | revert <id>")
    lc = p.add_subparsers(dest="learn_cmd", required=True)
    q = lc.add_parser("log", help="list ledger entries (global + local)")
    q.add_argument("--kind"); q.add_argument("--scope", choices=("global", "local")); q.add_argument("-n", type=int, default=0)
    q.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    q = lc.add_parser("show", help="one entry with its evidence")
    q.add_argument("id"); q.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    q = lc.add_parser("revert", help="restore the exact previous values and record a revert entry")
    q.add_argument("id"); q.add_argument("--force", action="store_true"); q.add_argument("--note")
    q.add_argument("--json", action="store_true", default=argparse.SUPPRESS)

    bench_p = sub.choices.get("bench")
    if bench_p is not None:
        _wrap_bench(bench_p)
