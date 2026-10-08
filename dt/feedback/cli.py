"""``dt feedback ...``: user-informed tuning from the command line (registered by dt/cli.py).

    dt feedback add <run_dir> corrections.json [--rating 1-5|up|down] [--share-screenshot] [--share-text]
    dt feedback from-figma <run_dir> figma_corrections.json
    dt feedback review <run_dir>                       (re)write <run_dir>/review.html
    dt feedback status | list
    dt feedback learn [--no-gate] [--workers 3]        user corpus + local matcher rules, gated, ledgered
    dt feedback eval                                   accuracy on your feedback cases -> history.jsonl
    dt feedback export -o bundle.zip [--redact-text] [--no-screenshots] [--ids fb-...]
    dt feedback import bundle.zip [--promote] [--scenarios-dir D] [--knowledge-dir D]   (maintainer side)
    dt feedback revert <ledger-id>                     restore the local rules an accepted change replaced
    dt feedback forget <bundle-id>                     delete one local bundle (and its corpus case)
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _emit(res: Any, as_json: bool, human: Callable[[Any], str]) -> None:
    print(json.dumps(res, indent=2, default=str) if as_json else human(res))


def _consent(a: argparse.Namespace) -> dict:
    return {"store_screenshot": False if getattr(a, "no_store_screenshot", False) else None,
            "share_screenshot": True if getattr(a, "share_screenshot", False) else None,
            "share_text": True if getattr(a, "share_text", False) else None}


def _bundle_human(b: dict) -> str:
    kinds: dict[str, int] = {}
    for it in b["items"]:
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    skipped = (b.get("apply_report") or {}).get("skipped") or []
    return (f"bundle {b['id']} ({b['channel']}): {len(b['items'])} items {kinds}, rating {b.get('run_rating')}, "
            f"consent {b['consent']}" + (f"\n  {len(skipped)} items could not be applied: {skipped[:3]}" if skipped else ""))


def cmd_feedback(a: argparse.Namespace) -> int:
    c = a.fb_cmd
    if c == "add":
        from dt.feedback.capture import add_corrections
        b = add_corrections(a.run_dir, a.corrections, rating=a.rating, consent=_consent(a))
        _emit(b, a.json, _bundle_human)
    elif c == "from-figma":
        from dt.feedback.figma import from_figma
        b = from_figma(a.run_dir, a.export, rating=a.rating, consent=_consent(a))
        _emit(b, a.json, _bundle_human)
    elif c == "review":
        from dt.feedback.review import write_review
        p = write_review(a.run_dir)
        _emit({"review": p}, a.json, lambda r: f"review page -> {r['review']} (open it in a browser)")
    elif c in ("status", "list"):
        from dt.feedback import list_bundles, status
        if c == "list":
            rows = [{"id": b["id"], "created": b["created"], "channel": b["channel"], "items": len(b["items"]),
                     "rating": b.get("run_rating"), "consent": b["consent"]} for b in list_bundles()]
            _emit(rows, a.json, lambda r: "\n".join(f"{x['id']}  {x['created']}  {x['channel']:9s} {x['items']:3d} items  "
                                                     f"rating={x['rating']}  {x['consent']}" for x in r) or "no feedback yet")
        else:
            s = status()
            _emit(s, a.json, lambda r: "\n".join(f"{k:14s} {v}" for k, v in r.items()))
    elif c == "learn":
        from dt.feedback.learn import learn
        r = learn(gate=not a.no_gate, workers=a.workers, bench_limit=a.limit, log=_log)
        _emit(r, a.json, lambda r: (f"{r['bundles']} bundles, {len(r['corpus'])} corpus cases, {r['rules']} rules "
                                    f"({r['active']} active); " + ("no change" if not r.get("changed") else
                                    f"{'ACCEPTED' if r['accepted'] else 'rejected'} gates={r['gates']} "
                                    f"before={r['before']} after={r['after']} ledger={r['ledger']}")))
        return EXIT_OK if r.get("accepted") in (True, None) else EXIT_FAIL
    elif c == "eval":
        from dt.feedback.learn import eval_history
        r = eval_history(log=_log)

        def human(r: dict) -> str:
            p = r.get("previous") or {}
            delta = lambda k: (f" ({r[k] - p[k]:+.4f} vs {p.get('ts')})" if r.get(k) is not None and p.get(k) is not None else "")
            return (f"model {r['model']['id']}: {r['n_cases']} cases, {r['items']} items\n"
                    f"  item accuracy {r['item_acc']}{delta('item_acc')}\n  composite     {r['composite']}{delta('composite')}\n"
                    f"  by kind       {r['by_kind']}")
        _emit(r, a.json, human)
    elif c == "export":
        from dt.feedback.share import export_bundles
        r = export_bundles(a.out, redact=a.redact_text, no_screenshots=a.no_screenshots, ids=a.ids)
        _emit(r, a.json, lambda r: f"{len(r['bundles'])} bundles -> {r['out']}\n" + "\n".join(
            f"  {b['id']}: text {b['text']}, screenshot {'yes' if b['screenshot'] else 'no (' + str(b['screenshot_excluded']) + ')'}"
            for b in r["bundles"]))
    elif c == "import":
        from dt.feedback.share import import_bundle
        r = import_bundle(a.bundle, scenarios_dir=a.scenarios_dir, knowledge_dir=a.knowledge_dir, promote_rules=a.promote,
                          gate=not a.no_gate, workers=a.workers, bench_limit=a.limit, log=_log)
        _emit(r, a.json, lambda r: (f"imported {len(r['imported'])}, rejected {len(r['rejected'])}, scenario seeds {len(r['seeds'])}, "
                                    f"candidate rules {r['candidates']} ({r['eligible']} eligible)"
                                    + (f"\npromotion: {r['promotion']}" if "promotion" in r else "")))
        if a.promote and not (r.get("promotion") or {}).get("promoted"):
            return EXIT_FAIL
    elif c == "revert":
        from dt.feedback.learn import revert
        e = revert(a.entry)
        _emit(e, a.json, lambda e: f"reverted {a.entry} -> ledger {e['id']}")
    elif c == "forget":
        from dt.feedback.store import delete_bundle
        ok = delete_bundle(a.bundle_id)
        _emit({"deleted": ok, "id": a.bundle_id}, a.json, lambda r: ("deleted " if ok else "no such bundle ") + a.bundle_id)
        return EXIT_OK if ok else EXIT_FAIL
    else:
        return EXIT_USAGE
    return EXIT_OK


def register(add: Callable[..., argparse.ArgumentParser], sub: Any) -> None:
    p = add("feedback", cmd_feedback, "user-informed tuning: record corrections, learn from them locally, share them with consent")
    fs = p.add_subparsers(dest="fb_cmd", required=True)

    def sp(name: str, help_: str) -> argparse.ArgumentParser:
        q = fs.add_parser(name, help=help_, description=help_)
        q.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="emit one JSON object on stdout")
        return q

    def consent_flags(q: argparse.ArgumentParser) -> None:
        q.add_argument("--rating", help="overall rating 1-5 (or up / down)")
        q.add_argument("--share-screenshot", action="store_true", help="consent: an export may include the screenshot")
        q.add_argument("--share-text", action="store_true", help="consent: an export may include text content")
        q.add_argument("--no-store-screenshot", action="store_true", help="do not keep a copy of the screenshot locally")

    q = sp("add", "record a corrections file (review.html download / hand-written JSON) for a translate run")
    q.add_argument("run_dir"); q.add_argument("corrections"); consent_flags(q)
    q = sp("from-figma", "diff a Figma 'Export corrections' file against the run and record the differences")
    q.add_argument("run_dir"); q.add_argument("export"); consent_flags(q)
    q = sp("review", "(re)write <run_dir>/review.html"); q.add_argument("run_dir")
    sp("status", "local feedback state: bundles, items, rules, last evaluation, ledger")
    sp("list", "list local feedback bundles")
    q = sp("learn", "user corpus + local matcher rules from your feedback (gated by your corpus and the global bench)")
    q.add_argument("--no-gate", action="store_true", help="skip the global bench gate (local rules only)")
    q.add_argument("--workers", type=int, default=3); q.add_argument("--limit", type=int, help="bench cases per corpus")
    sp("eval", "re-run all feedback cases with the current model; append to history.jsonl")
    q = sp("export", "zip your feedback for the maintainers, under each bundle's consent")
    q.add_argument("-o", "--out", required=True); q.add_argument("--redact-text", action="store_true")
    q.add_argument("--no-screenshots", action="store_true"); q.add_argument("--ids", nargs="*")
    q = sp("import", "maintainer: contributed bundle zip -> user_reported scenario seeds + candidate global rules")
    q.add_argument("bundle"); q.add_argument("--promote", action="store_true", help="gate and promote eligible rules")
    q.add_argument("--scenarios-dir"); q.add_argument("--knowledge-dir")
    q.add_argument("--no-gate", action="store_true", help=argparse.SUPPRESS)
    q.add_argument("--workers", type=int, default=3); q.add_argument("--limit", type=int)
    q = sp("revert", "restore the local rules an accepted learn replaced"); q.add_argument("entry")
    q = sp("forget", "delete one local feedback bundle"); q.add_argument("bundle_id")
