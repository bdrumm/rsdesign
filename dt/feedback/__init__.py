"""User-informed tuning: consumers' corrections become local learning, and (with consent) shared learning.

    from dt.feedback import add_corrections, learn, eval_history
    add_corrections("out/run1", "corrections.json")      # dt feedback add out/run1 corrections.json
    learn()                                               # dt feedback learn  (user corpus + local rules, gated)
    eval_history()                                        # dt feedback eval   (accuracy on *your* screens over time)

Channels: ``dt feedback add`` (corrections file, e.g. from the run's review.html), answers given through
``dt apply-decisions`` (``feedback.log_decisions``), the Figma plugin's "Export corrections" +
``dt feedback from-figma``, and the MCP tools ``submit_feedback`` / ``feedback_status``.
Everything stays under ``$DT_HOME`` (default ``~/.rsdesign``) until ``dt feedback export``; see docs/FEEDBACK.md.
"""
from dt.feedback.store import (CHANNELS, KINDS, SCHEMA, dt_home, list_bundles, load_bundle, model_version,  # noqa: F401
                               run_info, validate_bundle)


def add_corrections(*a, **k):
    from dt.feedback.capture import add_corrections as f
    return f(*a, **k)


def from_figma(*a, **k):
    from dt.feedback.figma import from_figma as f
    return f(*a, **k)


def learn(*a, **k):
    from dt.feedback.learn import learn as f
    return f(*a, **k)


def eval_history(*a, **k):
    from dt.feedback.learn import eval_history as f
    return f(*a, **k)


def export_bundles(*a, **k):
    from dt.feedback.share import export_bundles as f
    return f(*a, **k)


def import_bundle(*a, **k):
    from dt.feedback.share import import_bundle as f
    return f(*a, **k)


def status(home=None) -> dict:
    """Summary of the local feedback state (bundles, items by kind, rules, last evaluation, last ledger entries)."""
    import json
    import os
    from dt.feedback import ledger
    from dt.feedback.learn import corpus_dirs, read_rules
    from dt.feedback.store import history_path, local_rules_path
    bundles = list_bundles(home)
    kinds: dict[str, int] = {}
    for b in bundles:
        for it in b.get("items") or []:
            kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    rules = read_rules(local_rules_path(home))
    last = None
    hp = history_path(home)
    if os.path.exists(hp):
        with open(hp) as f:
            lines = [ln for ln in f if ln.strip()]
        if lines:
            e = json.loads(lines[-1])
            last = {k: e.get(k) for k in ("ts", "model", "n_cases", "items", "item_acc", "composite", "component_acc")}
    return {
        "dt_home": dt_home() if home is None else home, "bundles": len(bundles), "items": sum(kinds.values()), "by_kind": kinds,
        "by_channel": {c: sum(1 for b in bundles if b.get("channel") == c) for c in CHANNELS if any(b.get("channel") == c for b in bundles)},
        "ratings": [b["run_rating"] for b in bundles if b.get("run_rating")],
        "corpus_cases": len(corpus_dirs(home)), "rules": len(rules), "active_rules": sum(1 for r in rules if r.get("active")),
        "last_eval": last, "ledger": [{k: e.get(k) for k in ("id", "ts", "kind", "accepted")} for e in ledger.read(ledger.local_path(home))[-5:]],
        "model": model_version(home),
    }


__all__ = ["SCHEMA", "KINDS", "CHANNELS", "dt_home", "run_info", "list_bundles", "load_bundle", "validate_bundle",
           "model_version", "add_corrections", "from_figma", "learn", "eval_history", "export_bundles", "import_bundle",
           "status"]
